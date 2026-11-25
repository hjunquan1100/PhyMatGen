"""
CIF file validation for mp_20 dataset — combined fast + heavy checks.

Pipeline per file:
  1. Fast checks (multiprocessing, CPU):
     - no_atomic_overlap
     - periodic_valid
     - cn_valid
     - charge_balanced
     - bvs_valid
  2. Heavy checks (sequential, GPU, run for every file; not gated by fast checks):
     - thermo_stable
     - born_valid

overall_pass = fast_overall_pass and thermo_stable and born_valid
  -> copied to .../phymatgen_correction/data/mp20/tg/
  -> otherwise copied to .../phymatgen_correction/data/mp20/ntg/

Output CSV: .../phymatgen_correction/data/mp20/validation_combined_results.csv

Note: Run in the tomvae conda environment (chgnet must be installed).
"""

import importlib
import math
import multiprocessing as mp
import os
import shutil
import sys
import traceback
import warnings
from collections import Counter
from pathlib import Path

# project-root import
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from phymatgen_correction.src.runtime_env import (
    configure_headless_runtime,
    ensure_conda_lib_on_loader_path,
    get_correction_env,
)

ensure_conda_lib_on_loader_path()
configure_headless_runtime()

# Patch must run BEFORE networkx is imported, because networkx calls
# _set_configs_from_environment() and processes entry_points at import time.
def _patch_invalid_networkx_backends() -> None:
    """Work around stale NetworkX backend entry points with invalid names."""
    # 1. Filter entry_points whose backend name is not a valid Python identifier.
    for module_name in ("importlib.metadata", "importlib_metadata"):
        try:
            metadata_mod = importlib.import_module(module_name)
        except ImportError:
            continue

        original = getattr(metadata_mod, "entry_points", None)
        entry_points_type = getattr(metadata_mod, "EntryPoints", None)
        if original is None or getattr(
            original, "_patched_invalid_networkx_backends", False
        ):
            continue

        def _wrapped_entry_points(
            *args,
            _original=original,
            _entry_points_type=entry_points_type,
            **kwargs,
        ):
            eps = _original(*args, **kwargs)
            # When called without a group argument, importlib.metadata returns a
            # SelectableGroups (dict subclass).  Iterating over it in dict mode
            # triggers "SelectableGroups dict interface is deprecated" and yields
            # group-name strings, not EntryPoint objects, so our filter below
            # would be a no-op anyway.  Return it unchanged to avoid the warning.
            if isinstance(eps, dict):
                return eps
            try:
                filtered = [
                    ep
                    for ep in eps
                    if not (
                        getattr(ep, "group", None) == "networkx.backends"
                        and not str(getattr(ep, "name", "")).isidentifier()
                    )
                ]
            except TypeError:
                return eps

            if _entry_points_type is None:
                return filtered
            try:
                return _entry_points_type(filtered)
            except Exception:
                return filtered

        _wrapped_entry_points._patched_invalid_networkx_backends = True
        metadata_mod.entry_points = _wrapped_entry_points

    # 2. Clean up NETWORKX_* environment variables that contain backend names
    #    with hyphens or other characters that are not valid Python identifiers.
    #    Such names cause a SyntaxError in _set_configs_from_environment() when
    #    networkx dynamically generates a dataclass __init__ using exec().
    for _k in list(os.environ.keys()):
        if not _k.upper().startswith("NETWORKX_"):
            continue
        _val = os.environ[_k]
        _parts = [p.strip() for p in _val.split(",")]
        _valid = [p for p in _parts if p and p.isidentifier()]
        if len(_valid) != len(_parts):
            if _valid:
                os.environ[_k] = ",".join(_valid)
            else:
                del os.environ[_k]


_patch_invalid_networkx_backends()

import networkx as nx
import numpy as np
import pandas as pd
import pymatgen.analysis.bond_valence as _bv_mod
import torch
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.elasticity import ElasticTensor, Strain, Stress
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Element, Lattice, Species as _Species, Structure
from tqdm import tqdm

from src.eval.crystal import smact_validity
from src.eval.ehull_stability import get_ehull_subspace, get_extended_ref_energies

try:
    from chgnet.model import CHGNet
    from chgnet.model.dynamics import StructOptimizer

    HAS_CHGNET = True
except Exception:
    CHGNet = None
    StructOptimizer = None
    HAS_CHGNET = False


# paths
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "validation" / "mp20"
CIF_DIR = Path(get_correction_env("CIF_DIR", _DEFAULT_OUTPUT_DIR / "generated"))
TG_DIR = Path(get_correction_env("TG_DIR", _DEFAULT_OUTPUT_DIR / "tg"))
NTG_DIR = Path(get_correction_env("NTG_DIR", _DEFAULT_OUTPUT_DIR / "ntg"))
CSV_OUT = Path(
    get_correction_env("CSV_OUT", _DEFAULT_OUTPUT_DIR / "validation_combined_results.csv")
)

N_WORKERS = int(get_correction_env("N_WORKERS", min(16, mp.cpu_count())))
CHECKPOINT_EVERY = max(1, int(get_correction_env("CHECKPOINT_EVERY", "1")))
EHULL_THRESHOLD = 0.1
_CHGNET_RISKY = {"U", "Th", "Pa", "Np", "Pm", "B", "Ac", "Pu", "Am", "Cm"}
DEVICE_MODE = os.environ.get("CHGNET_USE_DEVICE", "auto").strip().lower()
CHECK_CUDA_MEM = os.environ.get("CHGNET_CHECK_CUDA_MEM", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_RESOLVED_CHGNET_DEVICE: str | None = None

RESULT_COLUMNS = [
    "filename",
    "no_atomic_overlap",
    "periodic_valid",
    "cn_valid",
    "charge_balanced",
    "bvs_valid",
    "fast_overall_pass",
    "thermo_stable",
    "born_valid",
    "ehull",
    "heavy_ran",
    "overall_pass",
    "fast_notes",
    "heavy_notes",
    "notes",
]


# covalent radii (from mofchecker)
COVALENT_RADII = {
    "H": 0.31,
    "He": 0.28,
    "Li": 1.28,
    "Be": 0.96,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Ne": 0.58,
    "Na": 1.66,
    "Mg": 1.41,
    "Al": 1.21,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Ar": 1.06,
    "K": 2.03,
    "Ca": 1.76,
    "Sc": 1.7,
    "Ti": 1.6,
    "V": 1.53,
    "Cr": 1.39,
    "Mn": 1.61,
    "Fe": 1.52,
    "Co": 1.5,
    "Ni": 1.24,
    "Cu": 1.32,
    "Zn": 1.22,
    "Ga": 1.22,
    "Ge": 1.2,
    "As": 1.19,
    "Se": 1.2,
    "Br": 1.2,
    "Kr": 1.16,
    "Rb": 2.2,
    "Sr": 1.95,
    "Y": 1.9,
    "Zr": 1.5,
    "Nb": 1.64,
    "Mo": 1.54,
    "Tc": 1.47,
    "Ru": 1.46,
    "Rh": 1.42,
    "Pd": 1.39,
    "Ag": 1.45,
    "Cd": 1.44,
    "In": 1.42,
    "Sn": 1.39,
    "Sb": 1.39,
    "Te": 1.38,
    "I": 1.39,
    "Xe": 1.4,
    "Cs": 2.44,
    "Ba": 2.15,
    "La": 2.07,
    "Ce": 2.04,
    "Pr": 2.03,
    "Nd": 2.01,
    "Pm": 1.99,
    "Sm": 1.98,
    "Eu": 1.98,
    "Gd": 1.96,
    "Tb": 1.94,
    "Dy": 1.92,
    "Ho": 1.92,
    "Er": 1.89,
    "Tm": 1.9,
    "Yb": 1.87,
    "Lu": 1.87,
    "Hf": 1.75,
    "Ta": 1.7,
    "W": 1.62,
    "Re": 1.51,
    "Os": 1.44,
    "Ir": 1.41,
    "Pt": 1.36,
    "Au": 1.36,
    "Hg": 1.32,
    "Tl": 1.45,
    "Pb": 1.46,
    "Bi": 1.48,
    "Po": 1.4,
    "At": 1.5,
    "Rn": 1.5,
    "Fr": 2.6,
    "Ra": 2.21,
    "Ac": 2.15,
    "Th": 2.06,
    "Pa": 2.0,
    "U": 1.96,
    "Np": 1.9,
    "Pu": 1.87,
    "Am": 1.8,
    "Cm": 1.69,
}
_CR_MEDIAN = float(np.median(list(COVALENT_RADII.values())))

_STRICT_ANION_LIKE_ELEMENTS = {"C", "N", "O", "F", "P", "S", "Se", "Cl", "Br", "I"}
_METALLOID_LIKE_ELEMENTS = {"B", "Si", "Ge", "As", "Sb", "Te"}

_EXTENDED_R0 = {
    "Au": 1.835,
    "Pt": 1.980,
    "Tc": 1.910,
    "Os": 1.940,
    "Pa": 2.150,
    "Np": 2.100,
    "Pu": 2.115,
    "Am": 2.105,
    "Pm": 2.062,
    "Ac": 2.250,
    "Ra": 2.45,
    "Po": 2.080,
    "Fr": 2.90,
    "Cm": 2.10,
    "Bk": 2.10,
    "Cf": 2.10,
    "Es": 2.10,
    "Fm": 2.10,
    "Md": 2.10,
    "No": 2.10,
    "Lr": 2.10,
}
_EXTENDED_SPECIES_BV = {
    "Au": [(1, 1.0), (3, 3.0)],
    "Pt": [(2, 2.0), (4, 4.0)],
    "Tc": [(4, 4.0), (7, 7.0)],
    "Os": [(4, 4.0), (8, 8.0)],
    "Pa": [(5, 5.0)],
    "Np": [(4, 4.0), (6, 6.0)],
    "Pu": [(3, 3.0), (4, 4.0)],
    "Am": [(3, 3.0), (4, 4.0)],
    "Pm": [(3, 3.0)],
    "Ac": [(3, 3.0)],
    "Ra": [(2, 2.0)],
    "Po": [(4, 4.0)],
    "Fr": [(1, 1.0)],
    "Cm": [(3, 3.0)],
    "Bk": [(3, 3.0)],
    "Cf": [(3, 3.0)],
    "Es": [(3, 3.0)],
    "Fm": [(3, 3.0)],
    "Md": [(3, 3.0)],
    "No": [(3, 3.0)],
    "Lr": [(3, 3.0)],
}


def _cov_radius(sym: str) -> float:
    return COVALENT_RADII.get(sym, _CR_MEDIAN)


def _append_note(notes: list[str], msg: str) -> None:
    if msg and msg not in notes:
        notes.append(msg)


def _resolve_chgnet_device() -> str:
    """
    Resolve which device CHGNet should use.

    CHGNET_USE_DEVICE supports:
      - auto  : prefer cuda, otherwise cpu
      - cuda  : require CUDA, otherwise raise
      - cpu   : force CPU
      - mps   : force Apple MPS
    """
    global _RESOLVED_CHGNET_DEVICE
    if _RESOLVED_CHGNET_DEVICE is not None:
        return _RESOLVED_CHGNET_DEVICE

    if DEVICE_MODE in {"", "auto"}:
        _RESOLVED_CHGNET_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        return _RESOLVED_CHGNET_DEVICE

    if DEVICE_MODE == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CHGNET_USE_DEVICE=cuda was requested, but torch.cuda.is_available() is False."
            )
        _RESOLVED_CHGNET_DEVICE = "cuda"
        return _RESOLVED_CHGNET_DEVICE

    if DEVICE_MODE == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "CHGNET_USE_DEVICE=mps was requested, but MPS is not available."
            )
        _RESOLVED_CHGNET_DEVICE = "mps"
        return _RESOLVED_CHGNET_DEVICE

    if DEVICE_MODE == "cpu":
        _RESOLVED_CHGNET_DEVICE = "cpu"
        return _RESOLVED_CHGNET_DEVICE

    raise ValueError(
        f"Unsupported CHGNET_USE_DEVICE={DEVICE_MODE!r}. Use auto, cuda, cpu, or mps."
    )


def _is_metallic_like_system(s: Structure) -> tuple[bool, str]:
    """Detect systems where ionic oxidation-state heuristics are unreliable."""
    try:
        elems = sorted({site.specie.symbol for site in s})
    except Exception:
        return False, ""

    if any(sym in _STRICT_ANION_LIKE_ELEMENTS for sym in elems):
        return False, ""

    if all(Element(sym).is_metal or sym in _METALLOID_LIKE_ELEMENTS for sym in elems):
        return True, "metallic_like_skip:" + ",".join(elems)

    return False, ""


def _patch_bv_params() -> None:
    """Inject extended BV parameters so BVAnalyzer covers rare elements."""
    r_o = _bv_mod.BV_PARAMS[Element("O")]["r"]
    c_o = _bv_mod.BV_PARAMS[Element("O")]["c"]

    def _r0_to_rc(r0: float, r_oxygen: float, c_oxygen: float, c_metal: float = 1.5):
        sq = (math.sqrt(c_metal) - math.sqrt(c_oxygen)) ** 2
        a = c_metal
        b = (
            c_oxygen * r_oxygen
            + c_metal * r_oxygen
            - r_oxygen * sq
            - r0 * c_metal
        )
        c = c_oxygen * r_oxygen * (r_oxygen - r0)
        disc = max(b * b - 4 * a * c, 0)
        return (-b + math.sqrt(disc)) / (2 * a), c_metal

    for sym, r0 in _EXTENDED_R0.items():
        el = Element(sym)
        if el not in _bv_mod.BV_PARAMS:
            r_metal, c_metal = _r0_to_rc(r0, r_o, c_o)
            _bv_mod.BV_PARAMS[el] = {"r": r_metal, "c": c_metal}

    for sym, ox_list in _EXTENDED_SPECIES_BV.items():
        for ox, bv_mean in ox_list:
            sp = _Species(sym, ox)
            if sp not in _bv_mod.ICSD_BV_DATA:
                _bv_mod.ICSD_BV_DATA[sp] = {
                    "mean": bv_mean,
                    "std": 0.35,
                    "n_data_pts": 50,
                }
            if sp not in _bv_mod.PRIOR_PROB:
                _bv_mod.PRIOR_PROB[sp] = 1.0 / len(ox_list)


_patch_bv_params()


def _check_atomic_overlap(
    s: Structure, tolerance: float = 0.5
) -> tuple[bool, str]:
    """True if no atomic overlap based on covalent radii."""
    try:
        dm = s.distance_matrix.copy()
        np.fill_diagonal(dm, np.inf)
        atom_types = [site.specie.symbol for site in s]
        for i in range(len(s)):
            r_i = _cov_radius(atom_types[i])
            for j in range(i + 1, len(s)):
                r_j = _cov_radius(atom_types[j])
                threshold = (1.0 - tolerance) * (r_i + r_j)
                if dm[i, j] < threshold:
                    return (
                        False,
                        f"overlap:{atom_types[i]}{i}-{atom_types[j]}{j}:{dm[i, j]:.3f}Å",
                    )
        return True, ""
    except Exception as e:
        return False, f"overlap_error:{e}"


def _check_periodicity(s: Structure) -> tuple[bool, str]:
    """True if the periodic structure graph has a single connected component."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sg = StructureGraph.from_local_env_strategy(s, CrystalNN())
        g = sg.graph
        if g.is_directed():
            g = g.to_undirected()
        components = list(nx.connected_components(g))
        if len(components) == 1:
            return True, ""
        n_floating = sum(len(c) for c in components) - max(len(c) for c in components)
        return False, f"floating_atoms:{n_floating},components:{len(components)}"
    except Exception as e:
        return False, f"periodicity_error:{e}"


def _check_carbon_cn(
    s: Structure, bond_factor: float = 1.25, cn_threshold: int = 4
) -> tuple[bool, str]:
    """True if no carbon atom exceeds the coordination threshold."""
    try:
        atom_types = [site.specie.symbol for site in s]
        c_indices = [i for i, sym in enumerate(atom_types) if sym == "C"]
        if not c_indices:
            return True, ""

        dm = s.distance_matrix
        over = []
        for i in c_indices:
            r_i = _cov_radius(atom_types[i])
            cn = sum(
                1
                for j in range(len(s))
                if j != i and dm[i, j] < bond_factor * (r_i + _cov_radius(atom_types[j]))
            )
            if cn > cn_threshold:
                over.append((i, cn))
        if not over:
            return True, ""
        return False, "over_coord_C:" + ",".join(
            f"site{i}(CN={cn})" for i, cn in over
        )
    except Exception as e:
        return False, f"cn_error:{e}"


def _check_charge_balance(s: Structure) -> tuple[bool, str]:
    """True if SMACT validity + Pauling test passes."""
    metallic_like, reason = _is_metallic_like_system(s)
    if metallic_like:
        return True, reason
    try:
        atomic_numbers = [site.specie.Z for site in s]
        counter = Counter(atomic_numbers)
        elems, counts = zip(*sorted(counter.items()))
        valid = smact_validity(elems, counts, use_pauling_test=True, include_alloys=True)
        return bool(valid), ("" if valid else "smact_invalid")
    except Exception as e:
        return False, f"charge_error:{e}"


def _check_bvs(s: Structure, std_threshold: float = 1.5) -> tuple[bool, str]:
    """True if BVAnalyzer succeeds and the valence std is below threshold."""
    metallic_like, reason = _is_metallic_like_system(s)
    if metallic_like:
        return True, reason
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            valences = BVAnalyzer().get_valences(s)
        bvs = np.array(valences, dtype=float)
        std = float(np.std(bvs))
        if std < std_threshold:
            return True, ""
        return False, f"bvs_std:{std:.3f}"
    except Exception as e:
        return False, f"bvs_error:{e}"


def _run_fast_checks(s: Structure) -> tuple[dict, str]:
    result = {
        "no_atomic_overlap": False,
        "periodic_valid": False,
        "cn_valid": False,
        "charge_balanced": False,
        "bvs_valid": False,
        "fast_overall_pass": False,
    }
    notes: list[str] = []

    ok, msg = _check_atomic_overlap(s)
    result["no_atomic_overlap"] = ok
    if not ok:
        _append_note(notes, msg)

    ok, msg = _check_periodicity(s)
    result["periodic_valid"] = ok
    if not ok:
        _append_note(notes, msg)

    ok, msg = _check_carbon_cn(s)
    result["cn_valid"] = ok
    if not ok:
        _append_note(notes, msg)

    ok, msg = _check_charge_balance(s)
    result["charge_balanced"] = ok
    if not ok or msg:
        _append_note(notes, msg)

    ok, msg = _check_bvs(s)
    result["bvs_valid"] = ok
    if not ok or msg:
        _append_note(notes, msg)

    result["fast_overall_pass"] = all(
        result[key]
        for key in (
            "no_atomic_overlap",
            "periodic_valid",
            "cn_valid",
            "charge_balanced",
            "bvs_valid",
        )
    )
    return result, "; ".join(notes) if notes else "OK"


def _fast_validate_one(cif_path: Path) -> dict:
    result = {
        "filename": cif_path.name,
        "no_atomic_overlap": False,
        "periodic_valid": False,
        "cn_valid": False,
        "charge_balanced": False,
        "bvs_valid": False,
        "fast_overall_pass": False,
        "fast_notes": "",
    }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = Structure.from_file(str(cif_path))
    except Exception as e:
        result["fast_notes"] = f"load_error:{e}"
        return result

    fast_result, fast_notes = _run_fast_checks(s)
    result.update(fast_result)
    result["fast_notes"] = fast_notes
    return result


def _fast_worker(path_str: str) -> dict:
    try:
        return _fast_validate_one(Path(path_str))
    except Exception as e:
        return {
            "filename": Path(path_str).name,
            "no_atomic_overlap": False,
            "periodic_valid": False,
            "cn_valid": False,
            "charge_balanced": False,
            "bvs_valid": False,
            "fast_overall_pass": False,
            "fast_notes": f"unhandled:{e}\n{traceback.format_exc()}",
        }


def _get_crystal_system(s: Structure) -> str:
    try:
        return s.lattice.crystal_system
    except Exception:
        return "unknown"


def _apply_strain(s: Structure, strain_voigt: list[float]) -> Structure:
    strain = Strain.from_voigt(strain_voigt)
    deformation = np.eye(3) + np.asarray(strain)
    new_lat = Lattice(np.dot(deformation, s.lattice.matrix))
    return Structure(new_lat, s.species, s.frac_coords)


def _predict_stress(model: "CHGNet", s: Structure) -> Stress:
    pred = model.predict_structure(s, task="efs")
    sv = pred["s"]
    if hasattr(sv, "numpy"):
        sv = sv.numpy()
    sv = np.array(sv, dtype=float).reshape(3, 3)
    return Stress.from_voigt([sv[0, 0], sv[1, 1], sv[2, 2], sv[1, 2], sv[0, 2], sv[0, 1]])


def _calc_elastic_tensor(model: "CHGNet", s: Structure) -> np.ndarray:
    """6-direction finite-difference elastic tensor via CHGNet stress predictions."""
    try:
        opt = StructOptimizer(
            model,
            use_device=_resolve_chgnet_device(),
        )
        res = opt.relax(s, verbose=False)
        final = res["final_structure"] if isinstance(res, dict) else res
        if isinstance(final, Structure):
            s = final
    except Exception:
        pass

    eq_stress = _predict_stress(model, s)
    mag = 0.005
    voigt_basis = [
        [mag, 0, 0, 0, 0, 0],
        [0, mag, 0, 0, 0, 0],
        [0, 0, mag, 0, 0, 0],
        [0, 0, 0, mag, 0, 0],
        [0, 0, 0, 0, mag, 0],
        [0, 0, 0, 0, 0, mag],
    ]
    strains = [Strain.from_voigt(v) for v in voigt_basis]
    stresses = [_predict_stress(model, _apply_strain(s, v)) for v in voigt_basis]
    c = ElasticTensor.from_independent_strains(strains, stresses, eq_stress=eq_stress)
    c_voigt = np.array(c.voigt, dtype=float)
    if c_voigt.shape != (6, 6):
        raise ValueError(f"Elastic tensor shape {c_voigt.shape} != (6,6)")
    return c_voigt


def _check_thermo(
    s: Structure, model: "CHGNet", ref_energies: dict
) -> tuple[bool, float, str]:
    """Return (is_stable, ehull, note) using CHGNet relaxation + e_hull."""
    els = {sp.symbol for sp in s.composition.element_composition}
    if els & _CHGNET_RISKY:
        risky = sorted(els & _CHGNET_RISKY)
        return False, float("nan"), f"risky_elements:{risky}"

    try:
        opt = StructOptimizer(
            model,
            use_device=_resolve_chgnet_device(),
        )
        res = opt.relax(s, verbose=False)
        final = res["final_structure"] if isinstance(res, dict) else res
        if not isinstance(final, Structure):
            raise ValueError("Relaxation did not return a Structure")
    except Exception as e:
        return False, float("nan"), f"relax_error:{e}"

    try:
        pred = model.predict_structure(final, task="e")
        e_per_atom = float(pred["e"].item())
        e_total = e_per_atom * len(final)
    except Exception as e:
        return False, float("nan"), f"energy_error:{e}"

    try:
        comp = final.composition
        n_total = comp.num_atoms
        missing = [el.symbol for el in comp if el.symbol not in ref_energies]
        for sym in missing:
            ref_energies[sym] = 0.0
        sum_n_mu = sum(comp[el] * ref_energies.get(el.symbol, 0.0) for el in comp)
        e_form = (e_total - sum_n_mu) / n_total
    except Exception as e:
        return False, float("nan"), f"eform_error:{e}"

    try:
        ehull, _ = get_ehull_subspace(final.composition, float(e_form))
    except Exception as e:
        return False, float("nan"), f"ehull_error:{e}"

    ehull = float(ehull)
    is_stable = ehull < EHULL_THRESHOLD
    label = "Stable" if ehull <= 1e-6 else ("Metastable" if ehull < EHULL_THRESHOLD else "Unstable")
    note = f"ehull:{ehull:.4f}({label})" if not is_stable else ""
    return is_stable, ehull, note


def _check_born(s: Structure, model: "CHGNet") -> tuple[bool, str]:
    """True if Born mechanical stability criteria are satisfied."""
    crystal_system = _get_crystal_system(s).lower()
    try:
        c = _calc_elastic_tensor(model, s)
    except Exception as e:
        return False, f"elastic_error:{e}"

    violations = []
    for i in range(6):
        if c[i, i] <= 0:
            violations.append(f"C{i+1}{i+1}={c[i, i]:.2f}<=0")

    if crystal_system == "cubic":
        c11, c12, c44 = c[0, 0], c[0, 1], c[3, 3]
        if c11 - c12 <= 0:
            violations.append(f"C11-C12={c11-c12:.2f}<=0")
        if c11 + 2 * c12 <= 0:
            violations.append(f"C11+2C12={c11 + 2 * c12:.2f}<=0")
        if c44 <= 0:
            violations.append(f"C44={c44:.2f}<=0")
    elif crystal_system == "hexagonal":
        c11, c12, c13, c33, c44 = c[0, 0], c[0, 1], c[0, 2], c[2, 2], c[3, 3]
        if c11 <= abs(c12):
            violations.append("C11<=|C12|")
        if (c11 + c12) * c33 <= 2 * c13**2:
            violations.append("(C11+C12)C33<=2C13^2")
        if c44 <= 0:
            violations.append(f"C44={c44:.2f}<=0")
    elif crystal_system == "tetragonal":
        c11, c12, c13, c33 = c[0, 0], c[0, 1], c[0, 2], c[2, 2]
        if c11 <= abs(c12):
            violations.append("C11<=|C12|")
        if (c11 + c12) * c33 <= 2 * c13**2:
            violations.append("(C11+C12)C33<=2C13^2")
    elif crystal_system == "orthorhombic":
        c11, c22, c33 = c[0, 0], c[1, 1], c[2, 2]
        c12, c13, c23 = c[0, 1], c[0, 2], c[1, 2]
        if c11 + c22 - 2 * c12 <= 0:
            violations.append(f"C11+C22-2C12={c11 + c22 - 2 * c12:.2f}<=0")
        if c11 + c33 - 2 * c13 <= 0:
            violations.append(f"C11+C33-2C13={c11 + c33 - 2 * c13:.2f}<=0")
        if c22 + c33 - 2 * c23 <= 0:
            violations.append(f"C22+C33-2C23={c22 + c33 - 2 * c23:.2f}<=0")

    try:
        eigvals = np.sort(np.linalg.eigvalsh(c))
        for idx, ev in enumerate(eigvals):
            if ev <= 0:
                violations.append(f"lambda[{idx}]={ev:.2f}<=0")
                break
    except np.linalg.LinAlgError:
        violations.append("eigval_failed")

    if violations:
        return False, "born_violations:" + "|".join(violations)
    return True, ""


def _run_heavy_checks(cif_path: Path, model: "CHGNet", ref_energies: dict) -> dict:
    result = {
        "thermo_stable": False,
        "born_valid": False,
        "ehull": float("nan"),
        "heavy_ran": True,
        "heavy_notes": "",
    }
    notes: list[str] = []

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = Structure.from_file(str(cif_path))
    except Exception as e:
        result["heavy_notes"] = f"load_error:{e}"
        return result

    stable, ehull, msg = _check_thermo(s, model, ref_energies)
    result["thermo_stable"] = stable
    result["ehull"] = ehull
    if not stable:
        _append_note(notes, msg)

    ok, msg = _check_born(s, model)
    result["born_valid"] = ok
    if not ok:
        _append_note(notes, msg)

    result["heavy_ran"] = True
    result["heavy_notes"] = "; ".join(notes) if notes else "OK"
    return result


def _write_results(rows: dict[str, dict], cif_files: list[Path]) -> pd.DataFrame:
    results = [rows[p.name] for p in cif_files]
    df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    df.sort_values("filename", inplace=True)
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = CSV_OUT.with_suffix(CSV_OUT.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(CSV_OUT)
    return df


def _copy_result_file(filename: str, overall_pass: bool) -> None:
    src = CIF_DIR / filename
    dst_dir = TG_DIR if overall_pass else NTG_DIR
    other_dir = NTG_DIR if overall_pass else TG_DIR
    dst = dst_dir / filename
    other = other_dir / filename
    if other.exists():
        other.unlink()
    shutil.copy2(src, dst)


def main() -> None:
    if not HAS_CHGNET:
        print("ERROR: chgnet is not installed. Run this script in the tomvae environment.")
        sys.exit(1)

    TG_DIR.mkdir(parents=True, exist_ok=True)
    NTG_DIR.mkdir(parents=True, exist_ok=True)

    cif_files = sorted(CIF_DIR.glob("*.cif"))
    print(f"Found {len(cif_files)} CIF files.")
    device = _resolve_chgnet_device()
    print(f"CHGNet device mode: requested={DEVICE_MODE or 'auto'}, resolved={device}")

    print(f"Running fast checks with {N_WORKERS} workers ...")
    with mp.Pool(processes=N_WORKERS) as pool:
        fast_results = list(
            tqdm(
                pool.imap(_fast_worker, [str(p) for p in cif_files], chunksize=20),
                total=len(cif_files),
                desc="Fast-validating",
            )
        )

    rows = {
        r["filename"]: {
            **r,
            "thermo_stable": False,
            "born_valid": False,
            "ehull": float("nan"),
            "heavy_ran": False,
            "heavy_notes": "heavy_pending",
            "overall_pass": False,
            "notes": r["fast_notes"],
        }
        for r in fast_results
    }

    if CSV_OUT.exists():
        try:
            previous = pd.read_csv(CSV_OUT)
            reused = 0
            for rec in previous.to_dict("records"):
                filename = rec.get("filename")
                if filename in rows and bool(rec.get("heavy_ran", False)):
                    rows[filename].update(rec)
                    reused += 1
            if reused:
                print(f"Resuming from existing CSV: reused {reused} completed heavy results.")
        except Exception as e:
            print(f"WARNING: could not read existing CSV for resume: {e}")

    heavy_candidates = [
        r["filename"] for r in fast_results if not bool(rows[r["filename"]]["heavy_ran"])
    ]

    print("Loading CHGNet model ...")
    model = CHGNet.load(
        use_device=device,
        check_cuda_mem=CHECK_CUDA_MEM,
    )
    print("Loading reference energies ...")
    ref_energies = get_extended_ref_energies()

    print(f"Running heavy checks sequentially for {len(heavy_candidates)} pending samples ...")
    if len(heavy_candidates) < len(cif_files):
        print(f"Already completed heavy checks: {len(cif_files) - len(heavy_candidates)}")

    completed_since_checkpoint = 0
    for filename in tqdm(heavy_candidates, desc="Heavy-validating"):
        cif_path = CIF_DIR / filename
        try:
            heavy_result = _run_heavy_checks(cif_path, model, ref_energies)
        except Exception as e:
            heavy_result = {
                "thermo_stable": False,
                "born_valid": False,
                "ehull": float("nan"),
                "heavy_ran": True,
                "heavy_notes": f"unhandled:{e}\n{traceback.format_exc()}",
            }

        row = rows[filename]
        row.update(heavy_result)
        row["overall_pass"] = bool(
            row["fast_overall_pass"] and row["thermo_stable"] and row["born_valid"]
        )

        all_notes = []
        if row["fast_notes"] and row["fast_notes"] != "OK":
            _append_note(all_notes, row["fast_notes"])
        if row["heavy_notes"] and row["heavy_notes"] != "OK":
            _append_note(all_notes, row["heavy_notes"])
        row["notes"] = "; ".join(all_notes) if all_notes else "OK"

        _copy_result_file(filename, row["overall_pass"])
        completed_since_checkpoint += 1
        if completed_since_checkpoint >= CHECKPOINT_EVERY:
            _write_results(rows, cif_files)
            completed_since_checkpoint = 0

    df = _write_results(rows, cif_files)

    print("Syncing output files ...")
    for out_dir in (TG_DIR, NTG_DIR):
        for old_cif in out_dir.glob("*.cif"):
            if old_cif.name not in rows:
                old_cif.unlink()
    for r in tqdm((rows[p.name] for p in cif_files), total=len(cif_files), desc="Copying"):
        if r["heavy_ran"]:
            _copy_result_file(r["filename"], r["overall_pass"])

    n_pass = int(df["overall_pass"].sum())
    fast_pass = int(df["fast_overall_pass"].sum())
    heavy_ran = int(df["heavy_ran"].sum())
    print("\n=== Combined validation complete ===")
    print(f"  Total           : {len(df)}")
    print(f"  Fast pass       : {fast_pass}")
    print(f"  Heavy ran       : {heavy_ran}")
    print(f"  Final pass      : {n_pass}  -> {TG_DIR}")
    print(f"  Final fail      : {len(df) - n_pass}  -> {NTG_DIR}")
    print(f"  CSV             : {CSV_OUT}")

    print("\nFast-stage failure counts:")
    for c in (
        "no_atomic_overlap",
        "periodic_valid",
        "cn_valid",
        "charge_balanced",
        "bvs_valid",
    ):
        print(f"  {c:<22}: {len(df) - int(df[c].sum())} failed")

    print("\nHeavy-stage failure counts (among heavy-ran samples):")
    if heavy_ran:
        print(
            f"  {'thermo_stable':<22}: "
            f"{heavy_ran - int(df.loc[df['heavy_ran'], 'thermo_stable'].sum())} failed"
        )
        print(
            f"  {'born_valid':<22}: "
            f"{heavy_ran - int(df.loc[df['heavy_ran'], 'born_valid'].sum())} failed"
        )
    else:
        print("  No samples reached heavy validation.")


if __name__ == "__main__":
    main()
