#!/usr/bin/env python
"""Stage-2 QM9 local refinement after global DiT repair.

QM9 专用二阶段局部修复。
     对 Stage-1 后仍失败的 QM9 分子，用 RDKit 尝试：
      - formula rebuild
      - UFF/MMFF 优化
      - distance bond 推断
      - 坐标/键图候选生成
        然后用项目里的 QM9 validator 重新判断，分到 stage2_tg / stage2_ntg，并输出结果和 summary。这个脚本也被 repair_ntg_with_dit.py 作为 QM9 candidate
        postprocess 调用。

The script reads QM9 molecules that failed the post-Stage-1 validator, tries a
small set of RDKit-based local/graph refinement candidates, then reuses the
same PhyMatGen correction QM9 validation rules to split outputs into stage2_tg/stage2_ntg.

For the user's 99.8% target on the full 3827-molecule Stage-1 QM9 run, at most
7 molecules may remain failed.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import math
import os
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CORRECTION_SRC = PROJECT_ROOT / "phymatgen_correction" / "src"
for path in (PROJECT_ROOT, CORRECTION_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from phymatgen_correction.src.qm9_mol_utils import (  # noqa: E402
    QM9_ALLOWED_VALENCES,
    Atom3D,
    Bond3D,
    build_rdkit_mol_from_bonds,
    build_rdkit_mol_from_coordinates,
    evaluate_molecule,
    infer_bonds,
    iter_molecule_files,
    parse_molecule_file,
)

try:  # noqa: E402
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import AllChem
    from rdkit.Geometry import Point3D
except ModuleNotFoundError as exc:  # pragma: no cover - runtime environment guard
    raise SystemExit(
        "RDKit is required. Install the project dependencies and rerun this script."
    ) from exc

RDLogger.DisableLog("rdApp.*")


DEFAULT_RUN_DIR = PROJECT_ROOT / "outputs" / "validation" / "qm9"
DEFAULT_INPUT_DIR = DEFAULT_RUN_DIR / "ntg"
DEFAULT_STAGE1_RESULTS = DEFAULT_RUN_DIR / "qm9_first3_results.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "local_refine" / "qm9"

VALENCE = {"H": 1, "C": 4, "N": 3, "O": 2, "F": 1}
TERMINAL_SYMBOLS = {"H", "F"}
PDB_SUPPORTED = {".pdb", ".xyz"}
FIELDNAMES = [
    "file",
    "source_path",
    "refined_path",
    "copied_to",
    "method",
    "before_passed",
    "before_mol_stable",
    "before_rdkit_valid",
    "before_no_atomic_overlap",
    "before_stable_atoms",
    "num_atoms",
    "stable_atoms",
    "unstable_atoms",
    "atom_stable_percent",
    "mol_stable",
    "rdkit_valid",
    "no_atomic_overlap",
    "valid",
    "passed_to_tg",
    "num_inferred_bonds",
    "bond_source",
    "canonical_smiles",
    "error",
]


@dataclass
class Candidate:
    method: str
    atoms: list[Atom3D]
    bonds: list[Bond3D] | None = None
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage1-results", type=Path, default=DEFAULT_STAGE1_RESULTS)
    parser.add_argument("--limit", type=int, default=0, help="Use <=0 for all failed molecules.")
    parser.add_argument("--target-pass-rate", type=float, default=0.998)
    parser.add_argument("--bond-margin", type=float, default=0.05)
    parser.add_argument("--overlap-tolerance", type=float, default=0.5)
    parser.add_argument("--max-iters", type=int, default=600)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--formula-rebuild-seeds",
        type=int,
        default=1,
        help="Number of ETKDG random seeds to try for formula-level rebuild candidates.",
    )
    parser.add_argument(
        "--graph-variants",
        type=int,
        default=1,
        help="Number of deterministic graph coordinate variants to try per formula rebuild candidate.",
    )
    parser.add_argument(
        "--max-candidate-errors",
        type=int,
        default=20,
        help="Maximum failed candidate diagnostics to keep in the result CSV. Use <=0 for all.",
    )
    parser.add_argument(
        "--disable-formula-rebuild",
        action="store_true",
        help=(
            "Only try geometry optimization candidates. Formula rebuild is much stronger "
            "because it can replace the failed topology while preserving atom types/counts."
        ),
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Remove old PDB/XYZ/CSV files from the stage2 output directory first.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.clear_output and args.output_dir.exists():
        clear_stage2_outputs(args.output_dir)
    paths = iter_molecule_files(args.input_dir)
    if args.limit > 0:
        paths = paths[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = args.output_dir / "best_candidates"
    tg_dir = args.output_dir / "stage2_tg"
    ntg_dir = args.output_dir / "stage2_ntg"
    for directory in (candidates_dir, tg_dir, ntg_dir):
        directory.mkdir(parents=True, exist_ok=True)

    stage1_total, stage1_passed, stage1_failed = read_stage1_counts(args.stage1_results)
    full_target_fail_max = math.floor(stage1_total * (1.0 - args.target_pass_rate)) if stage1_total else 0
    need_fixed = max(stage1_failed - full_target_fail_max, 0) if stage1_total else 0
    print_stage1_status(stage1_total, stage1_passed, stage1_failed, args.target_pass_rate, full_target_fail_max, need_fixed)
    print(f"[stage2] input_dir={args.input_dir}")
    print(f"[stage2] molecules_to_refine={len(paths)}")
    print(f"[stage2] output_dir={args.output_dir}")

    rows: list[dict[str, object]] = []
    method_counts: Counter[str] = Counter()
    passed_count = 0
    total_atoms = 0
    total_stable_atoms = 0

    for path in tqdm(paths, desc="QM9 stage2 refine"):
        row = refine_one(path, args, candidates_dir, tg_dir, ntg_dir)
        rows.append(row)
        method_counts[str(row["method"])] += 1
        passed_count += int(bool(row["passed_to_tg"]))
        total_atoms += int(row["num_atoms"])
        total_stable_atoms += int(row["stable_atoms"])

    results_csv = args.output_dir / "qm9_stage2_refine_results.csv"
    summary_csv = args.output_dir / "qm9_stage2_refine_summary.csv"
    write_results_csv(results_csv, rows)
    summary = {
        "stage1_total": stage1_total,
        "stage1_passed": stage1_passed,
        "stage1_failed": stage1_failed,
        "stage2_input_failed": len(paths),
        "stage2_refined_passed": passed_count,
        "stage2_remaining_failed": len(paths) - passed_count,
        "projected_total_passed": stage1_passed + passed_count if stage1_total else passed_count,
        "projected_total_failed": stage1_failed - passed_count if stage1_total else len(paths) - passed_count,
        "projected_pass_rate": (
            (stage1_passed + passed_count) / stage1_total if stage1_total else percent_ratio(passed_count, len(paths))
        ),
        "target_pass_rate": args.target_pass_rate,
        "target_failed_max": full_target_fail_max,
        "target_met": (
            (stage1_failed - passed_count) <= full_target_fail_max if stage1_total else passed_count == len(paths)
        ),
        "atom_stable_percent": percent(total_stable_atoms, total_atoms),
        "method_counts": dict(method_counts),
        "max_iters": args.max_iters,
        "formula_rebuild_seeds": args.formula_rebuild_seeds,
        "graph_variants": args.graph_variants,
        "results_csv": str(results_csv),
        "stage2_tg_dir": str(tg_dir),
        "stage2_ntg_dir": str(ntg_dir),
    }
    write_summary_csv(summary_csv, summary)

    print("[stage2] complete")
    print(f"[stage2] refined_passed={passed_count}/{len(paths)}")
    print(f"[stage2] projected_total_passed={summary['projected_total_passed']}/{stage1_total or len(paths)}")
    print(f"[stage2] projected_pass_rate={summary['projected_pass_rate']:.6f}")
    print(f"[stage2] target_met={summary['target_met']} target_failed_max={full_target_fail_max}")
    print(f"[stage2] results={results_csv}")
    print(f"[stage2] summary={summary_csv}")
    return 0


def refine_one(
    path: Path,
    args: argparse.Namespace,
    candidates_dir: Path,
    tg_dir: Path,
    ntg_dir: Path,
) -> dict[str, object]:
    before = quiet_evaluate_molecule(
        path,
        margin=args.bond_margin,
        overlap_tolerance=args.overlap_tolerance,
    )
    before_passed = before.mol_stable and before.rdkit_valid and before.no_atomic_overlap
    best_path = candidates_dir / path.name

    best_eval = before
    best_method = "original"
    best_atoms = before.atoms
    best_bonds = before.bonds
    best_score = score_eval(before)
    candidate_errors: list[str] = []

    if not before_passed:
        for candidate in generate_candidates(path, args):
            candidate_path = candidates_dir / f"{path.stem}.{candidate.method}.pdb"
            if candidate.error:
                candidate_errors.append(f"{candidate.method}:{candidate.error}")
                continue
            try:
                write_pdb(candidate_path, candidate.atoms, candidate.bonds)
                result = quiet_evaluate_molecule(
                    candidate_path,
                    margin=args.bond_margin,
                    overlap_tolerance=args.overlap_tolerance,
                )
            except Exception as exc:  # noqa: BLE001
                candidate_errors.append(f"{candidate.method}:{exc!r}")
                continue

            score = score_eval(result)
            if score > best_score:
                best_eval = result
                best_method = candidate.method
                best_atoms = candidate.atoms
                best_bonds = candidate.bonds if candidate.bonds is not None else result.bonds
                best_score = score

            if result.mol_stable and result.rdkit_valid and result.no_atomic_overlap:
                best_eval = result
                best_method = candidate.method
                best_atoms = candidate.atoms
                best_bonds = candidate.bonds if candidate.bonds is not None else result.bonds
                break

    write_pdb(best_path, best_atoms, best_bonds)
    final_eval = quiet_evaluate_molecule(
        best_path,
        margin=args.bond_margin,
        overlap_tolerance=args.overlap_tolerance,
    )
    final_passed = final_eval.mol_stable and final_eval.rdkit_valid and final_eval.no_atomic_overlap
    dest_dir = tg_dir if final_passed else ntg_dir
    dest_path = dest_dir / path.name
    shutil.copy2(best_path, dest_path)

    valid = final_eval.rdkit_valid and final_eval.no_atomic_overlap
    atom_count = len(final_eval.atoms)
    unstable_atoms = atom_count - final_eval.stable_atoms
    error = final_eval.error
    if candidate_errors:
        if args.max_candidate_errors > 0:
            kept_errors = candidate_errors[: args.max_candidate_errors]
            omitted = len(candidate_errors) - len(kept_errors)
            candidate_error_text = ";".join(kept_errors)
            if omitted > 0:
                candidate_error_text += f";... omitted {omitted} candidate errors"
        else:
            candidate_error_text = ";".join(candidate_errors)
        error = " | ".join(part for part in [error, "candidate_errors=" + candidate_error_text] if part)

    return {
        "file": path.name,
        "source_path": str(path),
        "refined_path": str(best_path),
        "copied_to": str(dest_path),
        "method": best_method,
        "before_passed": before_passed,
        "before_mol_stable": before.mol_stable,
        "before_rdkit_valid": before.rdkit_valid,
        "before_no_atomic_overlap": before.no_atomic_overlap,
        "before_stable_atoms": before.stable_atoms,
        "num_atoms": atom_count,
        "stable_atoms": final_eval.stable_atoms,
        "unstable_atoms": unstable_atoms,
        "atom_stable_percent": percent(final_eval.stable_atoms, atom_count),
        "mol_stable": final_eval.mol_stable,
        "rdkit_valid": final_eval.rdkit_valid,
        "no_atomic_overlap": final_eval.no_atomic_overlap,
        "valid": valid,
        "passed_to_tg": final_passed,
        "num_inferred_bonds": len(final_eval.bonds),
        "bond_source": final_eval.bond_source,
        "canonical_smiles": final_eval.canonical_smiles,
        "error": error,
    }


def generate_candidates(path: Path, args: argparse.Namespace) -> Iterable[Candidate]:
    try:
        atoms = parse_molecule_file(path)
    except Exception as exc:  # noqa: BLE001
        yield Candidate("parse_failed", [], error=repr(exc))
        return

    if not args.disable_formula_rebuild:
        yield from rebuild_formula_candidates(
            atoms,
            args.max_iters,
            args.seed,
            max(args.formula_rebuild_seeds, 1),
            max(args.graph_variants, 1),
        )
    yield from optimize_coordinate_candidate(atoms, args.max_iters, "rdkit_coords")
    bonds = infer_bonds(atoms, margin=args.bond_margin)
    yield from optimize_bond_candidate(atoms, bonds, args.max_iters, "distance_bonds")


def optimize_coordinate_candidate(atoms: list[Atom3D], max_iters: int, method: str) -> Iterable[Candidate]:
    try:
        with suppress_rdkit_stdio():
            mol = build_rdkit_mol_from_coordinates(atoms)
            Chem.SanitizeMol(mol)
        yield from optimize_mol_to_atoms(mol, atoms, max_iters, method)
    except Exception as exc:  # noqa: BLE001
        yield Candidate(method, [], error=repr(exc))


def optimize_bond_candidate(
    atoms: list[Atom3D],
    bonds: list[Bond3D],
    max_iters: int,
    method: str,
) -> Iterable[Candidate]:
    try:
        with suppress_rdkit_stdio():
            mol = build_rdkit_mol_from_bonds(atoms, bonds)
            Chem.SanitizeMol(mol)
        yield from optimize_mol_to_atoms(mol, atoms, max_iters, method)
    except Exception as exc:  # noqa: BLE001
        yield Candidate(method, [], error=repr(exc))


def rebuild_formula_candidates(
    atoms: list[Atom3D],
    max_iters: int,
    seed: int,
    num_seeds: int,
    graph_variants: int,
) -> Iterable[Candidate]:
    try:
        mol = build_valence_satisfied_mol([atom.symbol for atom in atoms])
    except Exception as exc:  # noqa: BLE001
        yield Candidate("formula_rebuild", [], error=repr(exc))
        return

    methods = [
        ("formula_rebuild_uff", "UFF"),
        ("formula_rebuild_mmff", "MMFF"),
        ("formula_rebuild_embed", "NONE"),
    ]
    for seed_i in range(num_seeds):
        trial_seed = seed + seed_i * 7919
        for method, forcefield in methods:
            method_name = f"{method}_s{seed_i}" if num_seeds > 1 else method
            try:
                trial = Chem.Mol(mol)
                embed_molecule(trial, trial_seed)
                optimize_forcefield(trial, max_iters=max_iters, forcefield=forcefield)
                translate_fragments_apart(trial, spacing=7.0)
                atoms = atoms_from_mol(trial)
                yield Candidate(method_name, atoms, bonds_from_mol(trial, atoms))
            except Exception as exc:  # noqa: BLE001
                yield Candidate(method_name, [], error=repr(exc))

    for variant_i in range(graph_variants):
        for method, forcefield in methods:
            method_name = f"{method}_graph{variant_i}"
            try:
                trial = Chem.Mol(mol)
                assign_graph_coordinates(trial, variant=variant_i)
                optimize_forcefield(trial, max_iters=max_iters, forcefield=forcefield)
                translate_fragments_apart(trial, spacing=7.0)
                atoms = atoms_from_mol(trial)
                yield Candidate(method_name, atoms, bonds_from_mol(trial, atoms))
            except Exception as exc:  # noqa: BLE001
                yield Candidate(method_name, [], error=repr(exc))


def build_valence_satisfied_mol(symbols: list[str]):
    unsupported = sorted(set(symbols) - set(QM9_ALLOWED_VALENCES))
    if unsupported:
        raise ValueError(f"unsupported QM9 symbols: {unsupported}")

    heavy = [idx for idx, symbol in enumerate(symbols) if symbol not in TERMINAL_SYMBOLS]
    terminals = [idx for idx, symbol in enumerate(symbols) if symbol in TERMINAL_SYMBOLS]

    if not heavy:
        return build_terminal_only_mol(symbols)

    heavy_valence = sum(VALENCE[symbols[idx]] for idx in heavy)
    terminal_count = len(terminals)
    if heavy_valence < terminal_count:
        raise ValueError(f"too many terminal atoms for heavy valence: {terminal_count}>{heavy_valence}")
    if (heavy_valence - terminal_count) % 2:
        raise ValueError("formula has odd heavy-heavy bond order requirement")

    target_heavy_bond_order = (heavy_valence - terminal_count) // 2
    rem = {idx: VALENCE[symbols[idx]] for idx in heavy}
    heavy_bonds = solve_heavy_bonds(symbols, heavy, rem, target_heavy_bond_order)
    if heavy_bonds is None:
        raise ValueError("could not build a valence-satisfied heavy-atom graph")

    for begin, end, order in heavy_bonds:
        rem[begin] -= order
        rem[end] -= order
    if sum(rem.values()) != terminal_count or any(value < 0 for value in rem.values()):
        raise ValueError("internal valence accounting failed")

    terminal_bonds: list[tuple[int, int, int]] = []
    for terminal in sorted(terminals, key=lambda idx: 0 if symbols[idx] == "F" else 1):
        host = choose_terminal_host(symbols, rem, symbols[terminal])
        if host is None:
            raise ValueError(f"could not attach terminal atom {symbols[terminal]}{terminal}")
        terminal_bonds.append((host, terminal, 1))
        rem[host] -= 1

    if any(value != 0 for value in rem.values()):
        raise ValueError(f"unfilled heavy valences remain: {rem}")

    return make_rdkit_mol(symbols, [*heavy_bonds, *terminal_bonds])


def build_terminal_only_mol(symbols: list[str]):
    if len(symbols) % 2:
        raise ValueError("terminal-only formula has an odd number of atoms")
    bonds = []
    unused = list(range(len(symbols)))
    while unused:
        begin = unused.pop(0)
        end = unused.pop(0)
        bonds.append((begin, end, 1))
    return make_rdkit_mol(symbols, bonds)


def solve_heavy_bonds(
    symbols: list[str],
    heavy: list[int],
    rem_by_index: dict[int, int],
    target_order: int,
) -> list[tuple[int, int, int]] | None:
    pairs = []
    for pos, begin in enumerate(heavy):
        for end in heavy[pos + 1 :]:
            pairs.append((begin, end, max_bond_order(symbols[begin], symbols[end])))
    pairs.sort(key=lambda pair: pair[2], reverse=True)

    rem = tuple(rem_by_index[idx] for idx in heavy)
    index_pos = {idx: pos for pos, idx in enumerate(heavy)}
    memo: dict[tuple[int, int, tuple[int, ...]], list[tuple[int, int, int]] | None] = {}

    def dfs(pair_pos: int, remaining: int, state: tuple[int, ...]) -> list[tuple[int, int, int]] | None:
        if remaining == 0:
            return []
        if remaining < 0 or pair_pos >= len(pairs):
            return None
        key = (pair_pos, remaining, state)
        if key in memo:
            return memo[key]
        max_possible = sum(min(order, state[index_pos[a]], state[index_pos[b]]) for a, b, order in pairs[pair_pos:])
        if max_possible < remaining:
            memo[key] = None
            return None

        begin, end, max_order = pairs[pair_pos]
        i = index_pos[begin]
        j = index_pos[end]
        allowed = min(max_order, state[i], state[j], remaining)
        for order in range(allowed, -1, -1):
            next_state = list(state)
            if order:
                next_state[i] -= order
                next_state[j] -= order
            suffix = dfs(pair_pos + 1, remaining - order, tuple(next_state))
            if suffix is not None:
                result = ([(begin, end, order)] if order else []) + suffix
                memo[key] = result
                return result
        memo[key] = None
        return None

    return dfs(0, target_order, rem)


def max_bond_order(symbol_a: str, symbol_b: str) -> int:
    pair = {symbol_a, symbol_b}
    if "O" in pair:
        return 2
    return 3


def choose_terminal_host(symbols: list[str], rem: dict[int, int], terminal_symbol: str) -> int | None:
    preferred = ["C", "N", "O"] if terminal_symbol == "H" else ["C", "N", "O"]
    for symbol in preferred:
        hosts = [idx for idx, value in rem.items() if value > 0 and symbols[idx] == symbol]
        if hosts:
            return max(hosts, key=lambda idx: rem[idx])
    hosts = [idx for idx, value in rem.items() if value > 0]
    return max(hosts, key=lambda idx: rem[idx]) if hosts else None


def make_rdkit_mol(symbols: list[str], bonds: list[tuple[int, int, int]]):
    mol = Chem.RWMol()
    for symbol in symbols:
        atom = Chem.Atom(symbol)
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)
    for begin, end, order in bonds:
        mol.AddBond(begin, end, bond_type(order))
    out = mol.GetMol()
    with suppress_rdkit_stdio():
        Chem.SanitizeMol(out)
    return out


def bond_type(order: int):
    if order == 1:
        return Chem.BondType.SINGLE
    if order == 2:
        return Chem.BondType.DOUBLE
    if order == 3:
        return Chem.BondType.TRIPLE
    raise ValueError(f"unsupported bond order: {order}")


def optimize_mol_to_atoms(mol, reference_atoms: list[Atom3D], max_iters: int, method: str) -> Iterable[Candidate]:
    for suffix, forcefield in [("uff", "UFF"), ("mmff", "MMFF"), ("none", "NONE")]:
        try:
            trial = Chem.Mol(mol)
            if trial.GetNumConformers() == 0:
                embed_molecule(trial, seed=13)
            optimize_forcefield(trial, max_iters=max_iters, forcefield=forcefield)
            translate_fragments_apart(trial, spacing=7.0)
            atoms = atoms_from_mol(trial, reference_atoms)
            yield Candidate(f"{method}_{suffix}", atoms, bonds_from_mol(trial, atoms))
        except Exception as exc:  # noqa: BLE001
            yield Candidate(f"{method}_{suffix}", [], error=repr(exc))


def embed_molecule(mol, seed: int) -> None:
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.useRandomCoords = True
    params.maxAttempts = 100
    params.enforceChirality = False
    params.useExpTorsionAnglePrefs = False
    params.useBasicKnowledge = False
    params.ignoreSmoothingFailures = True
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        params = AllChem.ETKDGv2()
        params.randomSeed = int(seed) + 7919
        params.useRandomCoords = True
        params.maxAttempts = 300
        params.enforceChirality = False
        params.useExpTorsionAnglePrefs = False
        params.useBasicKnowledge = False
        params.ignoreSmoothingFailures = True
        status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        assign_graph_coordinates(mol)


def assign_graph_coordinates(mol, variant: int = 0) -> None:
    """Deterministic 3D fallback that places bonded atoms near target distances."""
    num_atoms = mol.GetNumAtoms()
    conf = Chem.Conformer(num_atoms)
    placed: set[int] = set()
    fragments = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    directions = [
        (1.0, 0.0, 0.0),
        (-0.333, 0.943, 0.0),
        (-0.333, -0.471, 0.816),
        (-0.333, -0.471, -0.816),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ]
    if variant:
        shift = variant % len(directions)
        directions = directions[shift:] + directions[:shift]
    for frag_i, frag in enumerate(fragments):
        root = min(frag)
        origin = (float(frag_i) * 8.0, 0.0, 0.0)
        coords = {root: origin}
        placed.add(root)
        queue = [root]
        used_dirs: Counter[int] = Counter()
        while queue:
            atom_idx = queue.pop(0)
            base = coords[atom_idx]
            atom = mol.GetAtomWithIdx(atom_idx)
            for bond in atom.GetBonds():
                nbr = bond.GetOtherAtomIdx(atom_idx)
                if nbr in coords:
                    continue
                dir_i = (used_dirs[atom_idx] + variant * (atom_idx + 1)) % len(directions)
                used_dirs[atom_idx] += 1
                dx, dy, dz = directions[dir_i]
                length = target_bond_length(
                    mol.GetAtomWithIdx(atom_idx).GetSymbol(),
                    mol.GetAtomWithIdx(nbr).GetSymbol(),
                    int(round(bond.GetBondTypeAsDouble())),
                )
                coords[nbr] = (
                    base[0] + dx * length,
                    base[1] + dy * length,
                    base[2] + dz * length,
                )
                placed.add(nbr)
                queue.append(nbr)
        for atom_idx, coord in coords.items():
            conf.SetAtomPosition(atom_idx, Point3D(*coord))

    for atom_idx in range(num_atoms):
        if atom_idx not in placed:
            conf.SetAtomPosition(atom_idx, Point3D(float(atom_idx) * 8.0, 0.0, 0.0))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)


def target_bond_length(symbol_a: str, symbol_b: str, order: int) -> float:
    pair = tuple(sorted((symbol_a, symbol_b)))
    if order >= 3:
        return {
            ("C", "C"): 1.16,
            ("C", "N"): 1.13,
            ("C", "O"): 1.10,
            ("N", "N"): 1.08,
        }.get(pair, 1.12)
    if order == 2:
        return {
            ("C", "C"): 1.28,
            ("C", "N"): 1.24,
            ("C", "O"): 1.18,
            ("N", "N"): 1.20,
            ("N", "O"): 1.18,
            ("O", "O"): 1.18,
        }.get(pair, 1.24)
    return {
        ("C", "H"): 1.05,
        ("H", "N"): 0.98,
        ("H", "O"): 0.94,
        ("F", "H"): 0.90,
        ("C", "C"): 1.48,
        ("C", "N"): 1.42,
        ("C", "O"): 1.38,
        ("C", "F"): 1.32,
        ("N", "N"): 1.40,
        ("N", "O"): 1.35,
        ("F", "N"): 1.32,
        ("O", "O"): 1.43,
        ("F", "O"): 1.38,
        ("F", "F"): 1.38,
    }.get(pair, 1.40)


def optimize_forcefield(mol, max_iters: int, forcefield: str) -> None:
    if forcefield == "NONE":
        return
    try:
        with suppress_rdkit_stdio():
            if forcefield == "MMFF" and AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, maxIters=max_iters)
                return
            AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters)
    except Exception:
        if forcefield != "UFF":
            with suppress_rdkit_stdio():
                AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters)


def translate_fragments_apart(mol, spacing: float) -> None:
    if mol.GetNumConformers() == 0:
        return
    conf = mol.GetConformer()
    fragments = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    for frag_i, atom_indices in enumerate(fragments):
        shift = Point3D(float(frag_i) * spacing, 0.0, 0.0)
        for atom_idx in atom_indices:
            pos = conf.GetAtomPosition(atom_idx)
            conf.SetAtomPosition(atom_idx, Point3D(pos.x + shift.x, pos.y + shift.y, pos.z + shift.z))


def atoms_from_mol(mol, reference_atoms: list[Atom3D] | None = None) -> list[Atom3D]:
    conf = mol.GetConformer()
    atoms: list[Atom3D] = []
    for idx, atom in enumerate(mol.GetAtoms()):
        symbol = reference_atoms[idx].symbol if reference_atoms is not None and idx < len(reference_atoms) else atom.GetSymbol()
        pos = conf.GetAtomPosition(idx)
        atoms.append(Atom3D(symbol, float(pos.x), float(pos.y), float(pos.z)))
    return atoms


def bonds_from_mol(mol, atoms: list[Atom3D]) -> list[Bond3D]:
    bonds: list[Bond3D] = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        order = int(round(bond.GetBondTypeAsDouble()))
        order = max(1, min(order, 3))
        bonds.append(Bond3D(begin, end, order, molecule_distance(atoms[begin], atoms[end])))
    return bonds


def molecule_distance(a: Atom3D, b: Atom3D) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def score_eval(result) -> tuple[int, int, int, int, int, int]:
    valid = int(result.rdkit_valid and result.no_atomic_overlap)
    passed = int(result.mol_stable and valid)
    return (
        passed,
        int(result.mol_stable),
        valid,
        int(result.rdkit_valid),
        int(result.no_atomic_overlap),
        int(result.stable_atoms),
    )


def quiet_evaluate_molecule(path: Path, margin: float, overlap_tolerance: float):
    with suppress_rdkit_stdio():
        return evaluate_molecule(path, margin=margin, overlap_tolerance=overlap_tolerance)


@contextlib.contextmanager
def suppress_rdkit_stdio():
    """Suppress RDKit C++ diagnostics that bypass RDLogger."""
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)


def write_pdb(path: Path, atoms: list[Atom3D], bonds: Iterable[Bond3D] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for atom_i, atom in enumerate(atoms, start=1):
        symbol = atom.symbol
        lines.append(
            f"HETATM{atom_i:5d} {symbol:>2s}   MOL A   1"
            f"    {atom.x:8.3f}{atom.y:8.3f}{atom.z:8.3f}"
            f"  1.00  0.00          {symbol:>2s}\n"
        )
    if bonds:
        for bond in bonds:
            begin = bond.begin + 1
            end = bond.end + 1
            if begin <= 0 or end <= 0 or begin > len(atoms) or end > len(atoms):
                continue
            order = max(1, min(int(bond.order), 3))
            for _ in range(order):
                lines.append(f"CONECT{begin:5d}{end:5d}\n")
                lines.append(f"CONECT{end:5d}{begin:5d}\n")
    lines.append("END\n")
    path.write_text("".join(lines), encoding="utf-8")


def read_stage1_counts(path: Path) -> tuple[int, int, int]:
    if not path.exists():
        return 0, 0, 0
    total = 0
    passed = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total += 1
            passed += int(str(row.get("passed_to_tg", "")).lower() == "true")
    return total, passed, total - passed


def print_stage1_status(
    total: int,
    passed: int,
    failed: int,
    target_pass_rate: float,
    target_fail_max: int,
    need_fixed: int,
) -> None:
    if not total:
        print("[stage1] no stage1 results CSV found; target projection will use stage2 input only")
        return
    print(f"[stage1] total={total} passed={passed} failed={failed} pass_rate={passed / total:.6f}")
    print(
        f"[target] pass_rate={target_pass_rate:.4f} "
        f"max_failed={target_fail_max} need_fix_at_least={need_fixed}"
    )


def clear_stage2_outputs(output_dir: Path) -> None:
    for path in output_dir.rglob("*"):
        if path.is_file() and (path.suffix.lower() in PDB_SUPPORTED or path.suffix.lower() == ".csv"):
            path.unlink()


def percent(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else 100.0 * numerator / denominator


def percent_ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def write_results_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, summary: dict[str, object]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


if __name__ == "__main__":
    raise SystemExit(main())
