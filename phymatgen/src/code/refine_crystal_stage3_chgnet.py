#!/usr/bin/env python
"""Stage-3 CHGNet-based local refinement for crystal btg structures.

接 refine_crystal_stage2 的产物：对 stage2 之后仍未通过的 btg 结构再做一轮
基于 CHGNet 学习势能函数的局部能量松弛。原理：

  - StructOptimizer.relax(struct) 在 CHGNet 势函数下做几何弛豫
  - 多个起点（原结构、wrap、shrink_X、push_overlap）×（fix_cell / relax_cell）
  - 再调用各数据集的 fast_validate 重新测评

通过的结构会被搬到 --tg-dir/ 并从 --btg-dir/ 里删除；未通过的留在 --btg-dir 不动。
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import os
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@contextlib.contextmanager
def _silence():
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


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}


PASS_KEYS = ("no_atomic_overlap", "periodic_valid", "cn_valid", "charge_balanced", "bvs_valid")


def _fast_score(row: dict[str, Any]) -> float:
    return float(sum(int(_truthy(row.get(k, False))) for k in PASS_KEYS if k in row))


# ----- Geometric helpers (kept tiny; we mostly rely on CHGNet) -----

def _wrap(struct):
    from pymatgen.core import Structure  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    frac = np.mod(np.asarray(struct.frac_coords, dtype=float), 1.0)
    return Structure(struct.lattice, struct.species, frac, coords_are_cartesian=False)


def _scaled(struct, scale: float):
    from pymatgen.core import Lattice, Structure  # noqa: PLC0415

    return Structure(
        Lattice(struct.lattice.matrix * float(scale)),
        struct.species,
        struct.frac_coords,
        coords_are_cartesian=False,
    )


# ----- CHGNet relaxation -----

_CHGNET_MODEL = None


def _get_chgnet(device: str):
    global _CHGNET_MODEL
    if _CHGNET_MODEL is None:
        from chgnet.model import CHGNet  # noqa: PLC0415

        with _silence():
            _CHGNET_MODEL = CHGNet.load()
    return _CHGNET_MODEL


def _chgnet_relax(struct, *, device: str, steps: int, fmax: float, relax_cell: bool):
    from chgnet.model.dynamics import StructOptimizer  # noqa: PLC0415

    model = _get_chgnet(device)
    opt = StructOptimizer(
        model=model,
        use_device=device,
        optimizer_class="FIRE",
    )
    try:
        with _silence():
            result = opt.relax(
                struct,
                verbose=False,
                steps=int(steps),
                fmax=float(fmax),
                relax_cell=bool(relax_cell),
            )
    except TypeError:
        # Older chgnet versions don't accept relax_cell kwarg
        with _silence():
            result = opt.relax(struct, verbose=False, steps=int(steps), fmax=float(fmax))
    return result["final_structure"]


# ----- Candidate orchestration -----

# Seeds: small set of starting structures, then we relax each
SEED_SCALES = (0.85, 1.00, 1.15)


def _seed_structures(struct) -> list[tuple[str, Any]]:
    seeds: list[tuple[str, Any]] = []
    try:
        wrapped = _wrap(struct)
    except Exception:
        wrapped = struct
    seeds.append(("seed_wrap", wrapped))
    for s in SEED_SCALES:
        try:
            seeds.append((f"seed_scale_{s:.2f}", _scaled(wrapped, s)))
        except Exception:
            continue
    return seeds


def _refine_one(
    *,
    dataset: str,
    src_path: Path,
    out_dir: Path,
    fast,
    device: str,
    steps: int,
    fmax: float,
    cell_modes: tuple[bool, ...],
) -> dict[str, Any]:
    from pymatgen.core import Structure  # noqa: PLC0415

    info: dict[str, Any] = {
        "source_path": str(src_path),
        "source_filename": src_path.name,
        "input_fast_pass": False,
        "input_fast_score": 0.0,
        "input_fast_notes": "",
        "best_label": "",
        "best_fast_pass": False,
        "best_fast_score": 0.0,
        "best_fast_notes": "",
        "refined_path": "",
        "n_candidates": 0,
        "elapsed_sec": 0.0,
        "error": "",
    }
    if not src_path.exists():
        info["error"] = "missing_input"
        return info

    t0 = time.time()
    try:
        with _silence():
            base = Structure.from_file(str(src_path))
    except Exception as exc:
        info["error"] = f"load_error:{exc!r}"
        return info

    # Pre-eval (baseline)
    try:
        with _silence():
            pre = fast(src_path)
        info["input_fast_pass"] = _truthy(pre.get("fast_overall_pass", False))
        info["input_fast_score"] = _fast_score(pre)
        info["input_fast_notes"] = pre.get("fast_notes", "")
    except Exception as exc:
        info["input_fast_notes"] = f"input_validate_error:{exc!r}"

    # Build seeds, relax each under each cell_mode
    cand_dir = out_dir / "candidates" / src_path.stem
    cand_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[tuple[str, Path]] = []

    for seed_label, seed_struct in _seed_structures(base):
        for relax_cell in cell_modes:
            label = f"{seed_label}__chg_{'cell' if relax_cell else 'fix'}"
            try:
                relaxed = _chgnet_relax(
                    seed_struct,
                    device=device,
                    steps=steps,
                    fmax=fmax,
                    relax_cell=relax_cell,
                )
            except Exception:
                continue
            cand_path = cand_dir / f"{label}.cif"
            try:
                with _silence():
                    relaxed.to(str(cand_path), fmt="cif")
            except Exception:
                continue
            candidates.append((label, cand_path))

    info["n_candidates"] = len(candidates)

    # Re-evaluate everyone (plus the raw seed_wrap as a sanity baseline)
    eval_rows: list[tuple[str, Path, dict[str, Any]]] = []
    for label, cand_path in candidates:
        try:
            with _silence():
                row = fast(cand_path)
        except Exception as exc:
            row = {"fast_overall_pass": False, "fast_notes": f"validate_error:{exc!r}"}
        eval_rows.append((label, cand_path, row))

    if not eval_rows:
        info["error"] = info["error"] or "no_candidates"
        info["elapsed_sec"] = round(time.time() - t0, 1)
        return info

    # Rank: pass first, score second
    eval_rows.sort(
        key=lambda item: (
            -int(_truthy(item[2].get("fast_overall_pass", False))),
            -_fast_score(item[2]),
        )
    )
    best_label, best_path, best_eval = eval_rows[0]

    # Persist best to refined/ for the manifest
    refined_dir = out_dir / "refined"
    refined_dir.mkdir(parents=True, exist_ok=True)
    refined_path = refined_dir / f"{src_path.stem}.cif"
    shutil.copyfile(best_path, refined_path)

    info["best_label"] = best_label
    info["best_fast_pass"] = _truthy(best_eval.get("fast_overall_pass", False))
    info["best_fast_score"] = _fast_score(best_eval)
    info["best_fast_notes"] = best_eval.get("fast_notes", "")
    info["refined_path"] = str(refined_path)
    info["elapsed_sec"] = round(time.time() - t0, 1)

    shutil.rmtree(cand_dir, ignore_errors=True)
    return info


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["mp20", "perov5", "carbon24"])
    parser.add_argument("--btg-dir", type=Path, required=True, help="Folder with stage2 btg .cif files.")
    parser.add_argument("--tg-dir", type=Path, required=True, help="Folder to move newly-passing files into.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Where to write refined/, manifest, summary.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="CHGNet device.")
    parser.add_argument("--steps", type=int, default=120, help="Max CHGNet relax steps per candidate.")
    parser.add_argument("--fmax", type=float, default=0.1, help="Convergence force criterion (eV/Å).")
    parser.add_argument(
        "--cell-modes",
        default="fix,cell",
        help="Comma list: fix=relax_cell False, cell=relax_cell True.",
    )
    parser.add_argument(
        "--move-passing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When True, passing files are moved (btg -> tg). When False, only copied.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fast = _import_validator(args.dataset)

    cell_mode_set: list[bool] = []
    for token in args.cell_modes.split(","):
        token = token.strip().lower()
        if token == "fix":
            cell_mode_set.append(False)
        elif token == "cell":
            cell_mode_set.append(True)
    if not cell_mode_set:
        cell_mode_set = [False, True]

    files = sorted(p for p in args.btg_dir.iterdir() if p.is_file() and p.suffix.lower() == ".cif")
    if args.limit and args.limit > 0:
        files = files[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.tg_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[stage3-chgnet] dataset={args.dataset} input={args.btg_dir} files={len(files)} "
        f"device={args.device} steps={args.steps} cell_modes={cell_mode_set}"
    )
    if not files:
        print("[stage3-chgnet] no .cif inputs, nothing to do.")
        return

    # Warm up the CHGNet model once in the parent process
    with _silence():
        _get_chgnet(args.device)

    rows: list[dict[str, Any]] = []
    t_global = time.time()
    promoted = 0
    for i, src in enumerate(files, start=1):
        info = _refine_one(
            dataset=args.dataset,
            src_path=src,
            out_dir=args.out_dir,
            fast=fast,
            device=args.device,
            steps=int(args.steps),
            fmax=float(args.fmax),
            cell_modes=tuple(cell_mode_set),
        )
        rows.append(info)

        # Promote if best now passes
        promoted_flag = False
        if _truthy(info.get("best_fast_pass", False)) and info.get("refined_path"):
            refined = Path(str(info["refined_path"]))
            if refined.exists():
                dst = args.tg_dir / f"{src.stem}.cif"
                shutil.copyfile(refined, dst)
                if args.move_passing:
                    try:
                        src.unlink()
                    except FileNotFoundError:
                        pass
                promoted += 1
                promoted_flag = True
        info["promoted_to_tg"] = promoted_flag

        if i % 10 == 0 or i == len(files):
            print(
                f"  [{i}/{len(files)}] promoted={promoted} elapsed={time.time() - t_global:.1f}s",
                flush=True,
            )

    # Manifest + summary
    manifest = args.out_dir / f"{args.dataset}_stage3_chgnet_results.csv"
    if rows:
        fieldnames = sorted({k for r in rows for k in r})
        with manifest.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    n = len(rows)
    n_pass = sum(1 for r in rows if _truthy(r.get("best_fast_pass")))
    summary = {
        "dataset": args.dataset,
        "btg_dir": str(args.btg_dir),
        "tg_dir": str(args.tg_dir),
        "out_dir": str(args.out_dir),
        "num_files": n,
        "stage3_pass": n_pass,
        "stage3_pass_rate": (n_pass / n) if n else 0.0,
        "promoted_to_tg": promoted,
        "device": args.device,
        "steps": args.steps,
        "fmax": args.fmax,
        "cell_modes": args.cell_modes,
        "elapsed_seconds": round(time.time() - t_global, 1),
        "manifest": str(manifest),
    }
    summary_csv = args.out_dir / f"{args.dataset}_stage3_chgnet_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)
    shutil.rmtree(args.out_dir / "candidates", ignore_errors=True)

    print(
        f"[stage3-chgnet] DONE {args.dataset}: total={n} stage3_pass={n_pass} "
        f"promoted_to_tg={promoted} elapsed={summary['elapsed_seconds']}s"
    )
    print(f"  manifest: {manifest}")
    print(f"  summary:  {summary_csv}")


if __name__ == "__main__":
    main()
