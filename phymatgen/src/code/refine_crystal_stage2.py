#!/usr/bin/env python
"""Stage-2 local refinement for crystal datasets (mp20 / perov5 / carbon24).

晶体专用二阶段局部修复（不依赖 DiT / VAE，只做坐标和晶格层面的局部调整）。
对全局纠正后仍失败的 ntg 晶体结构，尝试一组确定性的局部候选：

  - wrap: 把 fractional coordinates 取 mod 1.0
  - merge_duplicates: 合并距离过近的同位点
  - push_overlap: 对重叠原子对沿连线方向相互推开（带 PBC）
  - lattice_scale_X: 各向同性放大/缩小晶格
  - combos: wrap + scale 的组合

然后调用每个数据集的 `_fast_validate_one` 重新测评（不修改原评测脚本），
按 (fast_overall_pass, fast_score, 改动最小) 排序挑最优候选。

每个输入文件最终都会有一份局部纠正后的结构落到 --output-dir/jubujiuzheng/。
通过测评的复制到 --output-dir/tg/，未通过的复制到 --output-dir/btg/。
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import multiprocessing as mp
import os
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _import_validator(dataset: str):
    if dataset == "carbon24":
        from phymatgen_correction.src.validate_carbon24 import _fast_validate_one  # noqa: PLC0415
    elif dataset == "mp20":
        from phymatgen_correction.src.validate_mp20 import _fast_validate_one  # noqa: PLC0415
    elif dataset == "perov5":
        from phymatgen_correction.src.validate_perov5 import _fast_validate_one  # noqa: PLC0415
    else:
        raise ValueError(f"Unsupported crystal dataset: {dataset}")
    return _fast_validate_one


COVALENT_RADII = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76,
    "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58, "Na": 1.66, "Mg": 1.41,
    "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39,
    "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
    "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16,
    "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54,
    "Tc": 1.47, "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44,
    "In": 1.42, "Sn": 1.39, "Sb": 1.39, "Te": 1.38, "I": 1.39, "Xe": 1.40,
    "Cs": 2.44, "Ba": 2.15, "La": 2.07, "Ce": 2.04, "Pr": 2.03, "Nd": 2.01,
    "Pm": 1.99, "Sm": 1.98, "Eu": 1.98, "Gd": 1.96, "Tb": 1.94, "Dy": 1.92,
    "Ho": 1.92, "Er": 1.89, "Tm": 1.90, "Yb": 1.87, "Lu": 1.87, "Hf": 1.75,
    "Ta": 1.70, "W": 1.62, "Re": 1.51, "Os": 1.44, "Ir": 1.41, "Pt": 1.36,
    "Au": 1.36, "Hg": 1.32, "Tl": 1.45, "Pb": 1.46, "Bi": 1.48,
}
_DEFAULT_RADIUS = float(np.median(list(COVALENT_RADII.values())))


def _radius(symbol: str) -> float:
    return COVALENT_RADII.get(symbol, _DEFAULT_RADIUS)


@contextlib.contextmanager
def _suppress_stdio():
    """Silence noisy native validator diagnostics during many local trials."""
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


def _load_structure(path: Path):
    from pymatgen.core import Structure  # noqa: PLC0415

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Structure.from_file(str(path))


def _save_structure(struct, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        struct.to(str(path), fmt="cif")


def _wrap_frac(coords: np.ndarray) -> np.ndarray:
    return np.mod(coords, 1.0)


def _build_scaled(struct, scale: float):
    from pymatgen.core import Lattice, Structure  # noqa: PLC0415

    lattice = Lattice(struct.lattice.matrix * float(scale))
    return Structure(lattice, struct.species, struct.frac_coords, coords_are_cartesian=False)


def _build_wrapped(struct):
    from pymatgen.core import Structure  # noqa: PLC0415

    frac = _wrap_frac(np.asarray(struct.frac_coords, dtype=float))
    return Structure(struct.lattice, struct.species, frac, coords_are_cartesian=False)


def _merge_duplicates(struct, tol: float = 0.3):
    """Drop sites within `tol` Å of any earlier site of the same species."""
    from pymatgen.core import Structure  # noqa: PLC0415

    if len(struct) < 2:
        return struct
    keep_idx: list[int] = []
    frac = np.asarray(struct.frac_coords, dtype=float)
    species = list(struct.species)
    lattice = struct.lattice
    for i in range(len(struct)):
        is_dup = False
        for j in keep_idx:
            if species[i] != species[j]:
                continue
            d = lattice.get_all_distances(frac[i : i + 1], frac[j : j + 1])[0, 0]
            if d < tol:
                is_dup = True
                break
        if not is_dup:
            keep_idx.append(i)
    if len(keep_idx) == len(struct):
        return struct
    return Structure(
        lattice,
        [species[i] for i in keep_idx],
        frac[keep_idx],
        coords_are_cartesian=False,
    )


def _push_overlapping_atoms(struct, max_iter: int = 30, tol_scale: float = 0.85):
    """Iteratively push overlapping atom pairs apart along their separation vector.

    A pair (i, j) overlaps when `dist < tol_scale * (r_i + r_j)`. Each iteration:
      - find min-image cartesian separation for every pair
      - for overlapping pairs, move i and j away from each other by (threshold-dist)/2
      - wrap fractional coords back into the cell
    Stops early when no overlap is detected.
    """
    from pymatgen.core import Structure  # noqa: PLC0415

    if len(struct) < 2:
        return struct
    lattice = struct.lattice
    species = list(struct.species)
    radii = np.array([_radius(getattr(s, "symbol", str(s))) for s in species])
    frac = np.asarray(struct.frac_coords, dtype=float).copy()
    n = len(species)
    rng = np.random.default_rng(seed=0)

    for _ in range(int(max_iter)):
        cart = lattice.get_cartesian_coords(frac)
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                frac_diff = frac[j] - frac[i]
                frac_diff -= np.round(frac_diff)
                cart_diff = lattice.get_cartesian_coords(frac_diff[None, :])[0]
                dist = float(np.linalg.norm(cart_diff))
                if dist <= 1e-6:
                    direction = rng.standard_normal(3)
                    direction /= max(np.linalg.norm(direction), 1e-8)
                else:
                    direction = cart_diff / dist
                threshold = float(tol_scale * (radii[i] + radii[j]))
                if dist < threshold:
                    push = (threshold - dist) * 0.5
                    cart[i] = cart[i] - direction * push
                    cart[j] = cart[j] + direction * push
                    moved = True
        if not moved:
            break
        frac = _wrap_frac(lattice.get_fractional_coords(cart))

    return Structure(lattice, species, frac, coords_are_cartesian=False)


# ----------------------------------------------------------------------
# Candidate orchestration
# ----------------------------------------------------------------------


# Wider scale ladder: structures fail in both directions.
# - `floating_atoms` failures want scale DOWN (atoms closer → more bonds)
# - `overlap` failures want scale UP
CANDIDATE_LATTICE_SCALES = (
    0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.94, 0.98,
    1.02, 1.06, 1.10, 1.15, 1.20, 1.30, 1.50,
)
# Smaller, focused set for chained push+scale and shrink+push combinations
COMBO_SCALES = (0.70, 0.80, 0.90, 1.10, 1.25)
SHRINK_SCALES = (0.60, 0.70, 0.80, 0.90)
PUSH_OVERLAP_TOL_SCALES = (0.85, 0.95)


def _make_candidates(struct, candidates_dir: Path, stem: str) -> list[tuple[str, Path]]:
    """Build candidate CIFs from a source structure. Returns list of (label, path)."""

    out: list[tuple[str, Path]] = []

    def add(label: str, s) -> None:
        try:
            out.append((label, _persist(candidates_dir, stem, label, s)))
        except Exception:
            pass

    # 1) original (round-trip through pymatgen, no edit)
    add("original", struct)

    # 2) wrap — frac coords mod 1
    try:
        wrapped = _build_wrapped(struct)
    except Exception:
        wrapped = struct
    add("wrap", wrapped)

    # 3) merge duplicates (drop sites < 0.3 Å of same species)
    try:
        merged = _merge_duplicates(wrapped, tol=0.3)
        if len(merged) != len(struct):
            add("merge_dup", merged)
    except Exception:
        pass

    # 4) push_overlap variants — fixes atom overlap and over-coord
    pushed_variants: list[tuple[float, Any]] = []
    for tol_scale in PUSH_OVERLAP_TOL_SCALES:
        try:
            pushed = _push_overlapping_atoms(wrapped, tol_scale=tol_scale)
            pushed_variants.append((tol_scale, pushed))
            add(f"push_overlap_{tol_scale:.2f}", pushed)
        except Exception:
            continue

    # 5) lattice scale variants (wide ladder, on wrapped)
    for scale in CANDIDATE_LATTICE_SCALES:
        try:
            add(f"scale_{scale:.3f}", _build_scaled(wrapped, scale))
        except Exception:
            continue

    # 6) chained: push first, then scale — best when both overlap+floating present
    for tol_scale, pushed in pushed_variants:
        for s in COMBO_SCALES:
            try:
                add(f"push_{tol_scale:.2f}_scale_{s:.3f}", _build_scaled(pushed, s))
            except Exception:
                continue

    # 7) chained: shrink first (encourages bonding), then push to fix new overlaps
    for s in SHRINK_SCALES:
        try:
            shrunk = _build_scaled(wrapped, s)
            shrunk_pushed = _push_overlapping_atoms(shrunk, tol_scale=0.85)
            add(f"shrink_{s:.3f}_push", shrunk_pushed)
        except Exception:
            continue

    return out


def _persist(candidates_dir: Path, stem: str, label: str, struct) -> Path:
    path = candidates_dir / f"{stem}.{label}.cif"
    _save_structure(struct, path)
    return path


# ----------------------------------------------------------------------
# Ranking
# ----------------------------------------------------------------------


PASS_KEYS = ("no_atomic_overlap", "periodic_valid", "cn_valid", "charge_balanced", "bvs_valid")


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}


def _fast_score(row: dict[str, Any]) -> float:
    return float(sum(int(_truthy(row.get(k, False))) for k in PASS_KEYS if k in row))


def _label_change_penalty(label: str) -> float:
    """Smaller is better. Used only to break ties among equally-scoring candidates."""
    if label == "original":
        return 0.0
    if label == "wrap":
        return 0.05
    if label.startswith("merge"):
        return 0.10
    if label.startswith("push_overlap"):
        return 0.15
    if label.startswith("scale_"):
        try:
            s = float(label.split("_", 1)[1])
            return 2.0 * abs(s - 1.0)
        except Exception:
            return 1.0
    if label.startswith("push_") and "_scale_" in label:
        try:
            s = float(label.rsplit("_", 1)[1])
            return 0.15 + 2.0 * abs(s - 1.0)
        except Exception:
            return 1.0
    if label.startswith("shrink_") and label.endswith("_push"):
        try:
            s = float(label.split("_", 2)[1])
            return 0.15 + 2.0 * abs(s - 1.0)
        except Exception:
            return 1.0
    return 0.5


def _rank_key(label: str, row: dict[str, Any]) -> tuple:
    return (
        -int(_truthy(row.get("fast_overall_pass", False))),  # pass first (negate for ascending sort)
        -_fast_score(row),  # higher subscore first
        _label_change_penalty(label),  # minimal change preferred
    )


# ----------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------


def _refine_one(args_tuple: tuple) -> dict[str, Any]:
    (
        dataset,
        src_path_str,
        candidates_dir_str,
        jubu_dir_str,
        tg_dir_str,
        btg_dir_str,
        keep_candidates,
    ) = args_tuple

    _fast_validate_one = _import_validator(dataset)
    src_path = Path(src_path_str)
    candidates_dir = Path(candidates_dir_str)
    jubu_dir = Path(jubu_dir_str)
    tg_dir = Path(tg_dir_str)
    btg_dir = Path(btg_dir_str)

    row: dict[str, Any] = {
        "dataset": dataset,
        "source_path": str(src_path),
        "source_filename": src_path.name,
        "input_fast_pass": False,
        "input_fast_score": 0.0,
        "input_fast_notes": "",
        "best_label": "",
        "best_fast_pass": False,
        "best_fast_score": 0.0,
        "best_fast_notes": "",
        "num_candidates": 0,
        "refined_path": "",
        "split": "btg",
        "error": "",
    }

    if not src_path.exists():
        row["error"] = "missing_input"
        return row

    try:
        with _suppress_stdio():
            base_struct = _load_structure(src_path)
    except Exception as exc:
        row["error"] = f"load_error:{exc!r}"
        return row

    # 0) Evaluate input as-is for the manifest
    try:
        with _suppress_stdio():
            in_eval = _fast_validate_one(src_path)
        row["input_fast_pass"] = _truthy(in_eval.get("fast_overall_pass", False))
        row["input_fast_score"] = _fast_score(in_eval)
        row["input_fast_notes"] = in_eval.get("fast_notes", "")
    except Exception as exc:
        row["input_fast_notes"] = f"input_validate_error:{exc!r}"

    # 1) Build candidates
    stem = src_path.stem
    candidates_dir_for_stem = candidates_dir / stem
    candidates_dir_for_stem.mkdir(parents=True, exist_ok=True)
    try:
        with _suppress_stdio():
            cand_list = _make_candidates(base_struct, candidates_dir_for_stem, stem)
    except Exception as exc:
        row["error"] = f"candidate_build_error:{exc!r}"
        cand_list = []
    row["num_candidates"] = len(cand_list)

    # 2) Evaluate candidates
    eval_rows: list[tuple[str, Path, dict[str, Any]]] = []
    for label, cand_path in cand_list:
        try:
            with _suppress_stdio():
                eval_row = _fast_validate_one(cand_path)
        except Exception as exc:
            eval_row = {"fast_overall_pass": False, "fast_notes": f"validate_error:{exc!r}"}
        eval_rows.append((label, cand_path, eval_row))

    # 3) Pick best by rank key
    if not eval_rows:
        row["error"] = row["error"] or "no_candidates"
        return row
    eval_rows.sort(key=lambda item: _rank_key(item[0], item[2]))
    best_label, best_path, best_eval = eval_rows[0]

    # 4) Persist outputs
    jubu_dir.mkdir(parents=True, exist_ok=True)
    final_jubu = jubu_dir / f"{stem}.cif"
    shutil.copyfile(best_path, final_jubu)
    row["refined_path"] = str(final_jubu)
    row["best_label"] = best_label
    row["best_fast_pass"] = _truthy(best_eval.get("fast_overall_pass", False))
    row["best_fast_score"] = _fast_score(best_eval)
    row["best_fast_notes"] = best_eval.get("fast_notes", "")

    target_dir = tg_dir if row["best_fast_pass"] else btg_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(final_jubu, target_dir / f"{stem}.cif")
    row["split"] = "tg" if row["best_fast_pass"] else "btg"

    # 5) Cleanup intermediate candidates unless requested
    if not keep_candidates:
        shutil.rmtree(candidates_dir_for_stem, ignore_errors=True)

    return row


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["mp20", "perov5", "carbon24"])
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Folder with crystal CIFs to locally refine (e.g. .../<dataset>/ntg).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output root. Produces <out>/jubujiuzheng/, <out>/tg/, <out>/btg/.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N files (0=all).")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument(
        "--keep-candidates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep all intermediate candidate CIFs under <out>/candidates/<stem>/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out = args.output_dir
    jubu_dir = out / "jubujiuzheng"
    tg_dir = out / "tg"
    btg_dir = out / "btg"
    candidates_dir = out / "candidates"
    out.mkdir(parents=True, exist_ok=True)
    jubu_dir.mkdir(parents=True, exist_ok=True)
    tg_dir.mkdir(parents=True, exist_ok=True)
    btg_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in args.input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".cif")
    if args.limit and args.limit > 0:
        files = files[: args.limit]
    print(f"[stage2-crystal] dataset={args.dataset} input={args.input_dir} files={len(files)} workers={args.workers}")
    if not files:
        print("[stage2-crystal] no .cif inputs, nothing to do.")
        return

    task_args = [
        (
            args.dataset,
            str(p),
            str(candidates_dir),
            str(jubu_dir),
            str(tg_dir),
            str(btg_dir),
            bool(args.keep_candidates),
        )
        for p in files
    ]

    t_start = time.time()
    rows: list[dict[str, Any]] = []
    workers = max(1, int(args.workers))
    if workers == 1:
        for ta in task_args:
            rows.append(_refine_one(ta))
            if len(rows) % 50 == 0:
                print(f"  processed {len(rows)}/{len(task_args)} elapsed={time.time() - t_start:.1f}s", flush=True)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for i, row in enumerate(pool.imap_unordered(_refine_one, task_args, chunksize=4), start=1):
                rows.append(row)
                if i % 50 == 0 or i == len(task_args):
                    print(f"  processed {i}/{len(task_args)} elapsed={time.time() - t_start:.1f}s", flush=True)

    # Manifest
    rows.sort(key=lambda r: r.get("source_filename", ""))
    manifest = out / f"{args.dataset}_stage2_refine_results.csv"
    if rows:
        fieldnames = sorted({k for r in rows for k in r})
        with manifest.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    # Summary
    n = len(rows)
    n_in_pass = sum(1 for r in rows if _truthy(r.get("input_fast_pass")))
    n_out_pass = sum(1 for r in rows if _truthy(r.get("best_fast_pass")))
    n_recovered = sum(
        1 for r in rows if not _truthy(r.get("input_fast_pass")) and _truthy(r.get("best_fast_pass"))
    )
    summary = {
        "dataset": args.dataset,
        "input_dir": str(args.input_dir),
        "output_dir": str(out),
        "num_files": n,
        "input_pass": n_in_pass,
        "input_pass_rate": (n_in_pass / n) if n else 0.0,
        "stage2_pass": n_out_pass,
        "stage2_pass_rate": (n_out_pass / n) if n else 0.0,
        "stage2_recovered": n_recovered,
        "stage2_recovery_rate": (n_recovered / max(n - n_in_pass, 1)) if n > n_in_pass else 0.0,
        "elapsed_seconds": round(time.time() - t_start, 1),
        "manifest": str(manifest),
    }
    summary_csv = out / f"{args.dataset}_stage2_refine_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    if not args.keep_candidates:
        shutil.rmtree(candidates_dir, ignore_errors=True)

    print(
        f"[stage2-crystal] DONE {args.dataset}: total={n} stage2_pass={n_out_pass} "
        f"recovered={n_recovered} elapsed={summary['elapsed_seconds']}s"
    )
    print(f"  manifest: {manifest}")
    print(f"  summary:  {summary_csv}")
    print(f"  jubujiuzheng/: {jubu_dir}  tg/: {tg_dir}  btg/: {btg_dir}")


if __name__ == "__main__":
    main()
