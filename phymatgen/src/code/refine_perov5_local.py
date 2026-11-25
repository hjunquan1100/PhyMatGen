#!/usr/bin/env python
"""perov5 局部精修（实空间，非潜空间）+ 完整 validate_perov5 测评。

对输入目录（通常是 .../perov5/ntg）中每个 CIF：

  Phase 1 (并行, CPU)：
    - 构造一组确定性的局部候选：wrap / merge_duplicates / push_overlap
      / lattice scale ladder / push+scale 组合 / shrink+push 组合。
      ⚠️ 全部在 fractional/cartesian 坐标 + 晶格层面进行，
         不依赖 DiT / VAE / latent。
    - 用 validate_perov5._fast_validate_one 评估每个候选，
      按 (fast_overall_pass, fast_score, 改动最小) 排序选最优。
    - 最优候选写到 <out>/jubujiuzheng/<stem>.cif。

  Phase 2 (串行, GPU)：
    - 对每个 jubujiuzheng/ 下的精修结果，再跑一次完整的
      validate_perov5（fast + heavy: thermo_stable & born_valid via CHGNet）。
    - overall_pass = fast_overall_pass AND thermo_stable AND born_valid
        → 通过的复制到 <out>/tg/
        → 不通过的复制到 <out>/ntg/
    - 写 manifest CSV（每个文件的 fast/heavy 全字段）和 summary CSV。
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")

import argparse
import csv
import multiprocessing as mp
import shutil
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np

# Project root so we can import `phymatgen_correction.src.*` and `src.eval.*`.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}


# ----------------------------------------------------------------------
# Phase 1 — parallel candidate building + fast-ranked selection.
# Reuses helpers from refine_crystal_stage2 so the local-edit recipes
# stay identical (real-space only).
# ----------------------------------------------------------------------


def _phase1_worker(args_tuple: tuple) -> dict[str, Any]:
    src_path_str, candidates_dir_str, jubu_dir_str, keep_candidates = args_tuple
    src_path = Path(src_path_str)
    candidates_dir = Path(candidates_dir_str)
    jubu_dir = Path(jubu_dir_str)

    row: dict[str, Any] = {
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
        "phase1_error": "",
    }

    try:
        from phymatgen_correction.src.code.refine_crystal_stage2 import (  # noqa: PLC0415
            _fast_score,
            _load_structure,
            _make_candidates,
            _rank_key,
            _suppress_stdio,
        )
        from phymatgen_correction.src.validate_perov5 import _fast_validate_one  # noqa: PLC0415

        if not src_path.exists():
            row["phase1_error"] = "missing_input"
            return row

        try:
            with _suppress_stdio():
                base_struct = _load_structure(src_path)
        except Exception as exc:
            row["phase1_error"] = f"load_error:{exc!r}"
            return row

        try:
            with _suppress_stdio():
                in_eval = _fast_validate_one(src_path)
            row["input_fast_pass"] = _truthy(in_eval.get("fast_overall_pass", False))
            row["input_fast_score"] = _fast_score(in_eval)
            row["input_fast_notes"] = in_eval.get("fast_notes", "")
        except Exception as exc:
            row["input_fast_notes"] = f"input_validate_error:{exc!r}"

        stem = src_path.stem
        candidates_dir_for_stem = candidates_dir / stem
        candidates_dir_for_stem.mkdir(parents=True, exist_ok=True)
        try:
            with _suppress_stdio():
                cand_list = _make_candidates(base_struct, candidates_dir_for_stem, stem)
        except Exception as exc:
            row["phase1_error"] = f"candidate_build_error:{exc!r}"
            cand_list = []

        cand_list = [(lbl, p) for (lbl, p) in cand_list if Path(p).exists()]
        row["num_candidates"] = len(cand_list)

        eval_rows: list[tuple[str, Path, dict[str, Any]]] = []
        for label, cand_path in cand_list:
            try:
                with _suppress_stdio():
                    eval_row = _fast_validate_one(cand_path)
            except Exception as exc:
                eval_row = {"fast_overall_pass": False, "fast_notes": f"validate_error:{exc!r}"}
            eval_rows.append((label, cand_path, eval_row))

        jubu_dir.mkdir(parents=True, exist_ok=True)
        final_jubu = jubu_dir / f"{stem}.cif"

        if eval_rows:
            eval_rows.sort(key=lambda item: _rank_key(item[0], item[2]))
            copied = False
            for best_label, best_path, best_eval in eval_rows:
                if not Path(best_path).exists():
                    continue
                try:
                    shutil.copyfile(best_path, final_jubu)
                except (FileNotFoundError, OSError) as exc:
                    row["phase1_error"] = (
                        f"{row['phase1_error']};copy_error:{exc!r}".lstrip(";")
                    )
                    continue
                row["refined_path"] = str(final_jubu)
                row["best_label"] = best_label
                row["best_fast_pass"] = _truthy(best_eval.get("fast_overall_pass", False))
                row["best_fast_score"] = _fast_score(best_eval)
                row["best_fast_notes"] = best_eval.get("fast_notes", "")
                copied = True
                break

            if not copied:
                try:
                    shutil.copyfile(src_path, final_jubu)
                    row["refined_path"] = str(final_jubu)
                    row["best_label"] = "source_fallback"
                    row["phase1_error"] = (
                        f"{row['phase1_error']};all_candidates_missing".lstrip(";")
                    )
                except Exception as exc:
                    row["phase1_error"] = (
                        f"{row['phase1_error']};fallback_copy_error:{exc!r}".lstrip(";")
                    )
        else:
            row["phase1_error"] = row["phase1_error"] or "no_candidates"
            try:
                shutil.copyfile(src_path, final_jubu)
                row["refined_path"] = str(final_jubu)
                row["best_label"] = "source_fallback"
            except Exception as exc:
                row["phase1_error"] = (
                    f"{row['phase1_error']};fallback_copy_error:{exc!r}".lstrip(";")
                )

        if not keep_candidates:
            shutil.rmtree(candidates_dir_for_stem, ignore_errors=True)

        return row
    except Exception as exc:
        row["phase1_error"] = (
            f"{row.get('phase1_error', '')};worker_unhandled:{exc!r}".lstrip(";")
        )
        return row


def run_phase1(
    files: list[Path],
    jubu_dir: Path,
    candidates_dir: Path,
    workers: int,
    keep_candidates: bool,
) -> list[dict[str, Any]]:
    task_args = [
        (str(p), str(candidates_dir), str(jubu_dir), bool(keep_candidates))
        for p in files
    ]
    rows: list[dict[str, Any]] = []
    t_start = time.time()

    if workers <= 1:
        for ta in task_args:
            rows.append(_phase1_worker(ta))
            if len(rows) % 50 == 0:
                print(
                    f"  [phase1] {len(rows)}/{len(task_args)} elapsed={time.time() - t_start:.1f}s",
                    flush=True,
                )
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for i, row in enumerate(pool.imap_unordered(_phase1_worker, task_args, chunksize=4), start=1):
                rows.append(row)
                if i % 50 == 0 or i == len(task_args):
                    print(
                        f"  [phase1] {i}/{len(task_args)} elapsed={time.time() - t_start:.1f}s",
                        flush=True,
                    )

    if not keep_candidates:
        shutil.rmtree(candidates_dir, ignore_errors=True)
    return rows


# ----------------------------------------------------------------------
# Phase 2 — sequential CHGNet heavy validation on refined CIFs.
# ----------------------------------------------------------------------


def run_phase2(
    rows: list[dict[str, Any]],
    tg_dir: Path,
    ntg_dir: Path,
) -> list[dict[str, Any]]:
    # Defer heavy imports until after phase 1 finishes (so the multiprocess
    # workers don't drag chgnet/torch into every fork).
    from phymatgen_correction.src.validate_perov5 import (  # noqa: PLC0415
        CHGNet,
        HAS_CHGNET,
        _append_note,
        _resolve_chgnet_device,
        _run_heavy_checks,
        _fast_validate_one,
        CHECK_CUDA_MEM,
    )
    from phymatgen_correction.src.validate_perov5 import (  # noqa: PLC0415
        get_extended_ref_energies,
    )

    if not HAS_CHGNET:
        print("ERROR: chgnet is not installed. Run this script in the tomvae environment.")
        sys.exit(1)

    tg_dir.mkdir(parents=True, exist_ok=True)
    ntg_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_chgnet_device()
    print(f"[phase2] CHGNet device resolved={device}")
    print("[phase2] loading CHGNet model ...")
    model = CHGNet.load(use_device=device, check_cuda_mem=CHECK_CUDA_MEM)
    print("[phase2] loading reference energies ...")
    ref_energies = get_extended_ref_energies()

    # Process only rows that have a refined_path
    pending: list[dict[str, Any]] = [r for r in rows if r.get("refined_path")]
    print(f"[phase2] heavy-validating {len(pending)} refined structures sequentially ...")

    t_start = time.time()
    for i, row in enumerate(pending, start=1):
        refined = Path(row["refined_path"])
        # 1) Fast on the refined CIF (might differ from the best-candidate
        #    snapshot if pymatgen round-trip introduces slight diffs)
        try:
            fast_res = _fast_validate_one(refined)
            row["refined_fast_pass"] = _truthy(fast_res.get("fast_overall_pass", False))
            row["refined_no_atomic_overlap"] = _truthy(fast_res.get("no_atomic_overlap", False))
            row["refined_periodic_valid"] = _truthy(fast_res.get("periodic_valid", False))
            row["refined_cn_valid"] = _truthy(fast_res.get("cn_valid", False))
            row["refined_charge_balanced"] = _truthy(fast_res.get("charge_balanced", False))
            row["refined_bvs_valid"] = _truthy(fast_res.get("bvs_valid", False))
            row["refined_fast_notes"] = fast_res.get("fast_notes", "")
        except Exception as exc:
            row["refined_fast_pass"] = False
            row["refined_fast_notes"] = f"refined_fast_error:{exc!r}"

        # 2) Heavy on refined CIF
        try:
            heavy = _run_heavy_checks(refined, model, ref_energies)
        except Exception as exc:
            heavy = {
                "thermo_stable": False,
                "born_valid": False,
                "ehull": float("nan"),
                "heavy_ran": True,
                "heavy_notes": f"unhandled:{exc!r}\n{traceback.format_exc()}",
            }
        row["thermo_stable"] = _truthy(heavy.get("thermo_stable", False))
        row["born_valid"] = _truthy(heavy.get("born_valid", False))
        row["ehull"] = heavy.get("ehull", float("nan"))
        row["heavy_ran"] = _truthy(heavy.get("heavy_ran", False))
        row["heavy_notes"] = heavy.get("heavy_notes", "")

        row["overall_pass"] = bool(
            row.get("refined_fast_pass") and row["thermo_stable"] and row["born_valid"]
        )

        all_notes: list[str] = []
        if row.get("refined_fast_notes") and row["refined_fast_notes"] != "OK":
            _append_note(all_notes, row["refined_fast_notes"])
        if row.get("heavy_notes") and row["heavy_notes"] != "OK":
            _append_note(all_notes, row["heavy_notes"])
        row["notes"] = "; ".join(all_notes) if all_notes else "OK"

        # 3) Dispatch to tg / ntg
        dst_dir = tg_dir if row["overall_pass"] else ntg_dir
        try:
            shutil.copy2(refined, dst_dir / refined.name)
            row["split"] = "tg" if row["overall_pass"] else "ntg"
        except Exception as exc:
            row["split"] = "copy_error"
            row["notes"] = f"{row.get('notes', '')}; copy_error:{exc!r}"

        if i % 20 == 0 or i == len(pending):
            print(
                f"  [phase2] {i}/{len(pending)} elapsed={time.time() - t_start:.1f}s "
                f"pass_so_far={sum(1 for r in pending[:i] if r.get('overall_pass'))}",
                flush=True,
            )

    return rows


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "validation" / "perov5" / "ntg"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "local_refine" / "perov5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0, help="Only process first N files (0 = all).")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument(
        "--keep-candidates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep all intermediate candidate CIFs (under <out>/candidates/<stem>/).",
    )
    parser.add_argument(
        "--skip-heavy",
        action="store_true",
        help="Run phase 1 only (no CHGNet). Useful for smoke testing.",
    )
    parser.add_argument(
        "--skip-phase1",
        action="store_true",
        help="Skip phase 1 and run phase 2 directly on existing <out>/jubujiuzheng/*.cif. "
             "Use this to resume after a phase-1 crash without re-running the candidates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir
    jubu_dir = out / "jubujiuzheng"
    tg_dir = out / "tg"
    ntg_dir = out / "ntg"
    candidates_dir = out / "candidates"
    for d in (out, jubu_dir, tg_dir, ntg_dir):
        d.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    if args.skip_phase1:
        refined_files = sorted(
            p for p in jubu_dir.iterdir() if p.is_file() and p.suffix.lower() == ".cif"
        )
        if args.limit and args.limit > 0:
            refined_files = refined_files[: args.limit]
        print(
            f"[refine-perov5] --skip-phase1: resuming phase2 on {len(refined_files)} files "
            f"from {jubu_dir}"
        )
        if not refined_files:
            print("[refine-perov5] no .cif files in jubujiuzheng/, nothing to do.")
            return
        rows = [
            {
                "source_path": str(p),
                "source_filename": p.name,
                "refined_path": str(p),
                "best_label": "resumed",
                "phase1_error": "",
            }
            for p in refined_files
        ]
    else:
        files = sorted(p for p in args.input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".cif")
        if args.limit and args.limit > 0:
            files = files[: args.limit]
        print(
            f"[refine-perov5] input={args.input_dir} files={len(files)} "
            f"workers={args.workers} out={out}"
        )
        if not files:
            print("[refine-perov5] no .cif inputs, nothing to do.")
            return

        print("[refine-perov5] === phase 1: local refinement (real space) + fast-ranked pick ===")
        rows = run_phase1(
            files=files,
            jubu_dir=jubu_dir,
            candidates_dir=candidates_dir,
            workers=max(1, int(args.workers)),
            keep_candidates=bool(args.keep_candidates),
        )
        n_with_refined = sum(1 for r in rows if r.get("refined_path"))
        print(
            f"[refine-perov5] phase1 done: refined={n_with_refined}/{len(rows)} "
            f"elapsed={time.time() - t_start:.1f}s"
        )

    if not args.skip_heavy:
        print("[refine-perov5] === phase 2: full perov5 validation (fast + CHGNet heavy) ===")
        rows = run_phase2(rows=rows, tg_dir=tg_dir, ntg_dir=ntg_dir)
    else:
        print("[refine-perov5] --skip-heavy set, skipping CHGNet validation; not writing tg/ntg.")

    # Manifest
    rows.sort(key=lambda r: r.get("source_filename", ""))
    manifest = out / "perov5_refine_results.csv"
    if rows:
        fieldnames = sorted({k for r in rows for k in r})
        with manifest.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    # Summary
    n = len(rows)
    n_in_pass = sum(1 for r in rows if _truthy(r.get("input_fast_pass")))
    n_phase1_fast_pass = sum(1 for r in rows if _truthy(r.get("best_fast_pass")))
    n_overall_pass = sum(1 for r in rows if _truthy(r.get("overall_pass")))
    n_recovered = sum(
        1 for r in rows
        if not _truthy(r.get("input_fast_pass")) and _truthy(r.get("overall_pass"))
    )
    summary = {
        "input_dir": str(args.input_dir),
        "output_dir": str(out),
        "num_files": n,
        "input_fast_pass": n_in_pass,
        "phase1_best_fast_pass": n_phase1_fast_pass,
        "phase2_overall_pass": n_overall_pass,
        "phase2_recovered": n_recovered,
        "phase2_pass_rate": (n_overall_pass / n) if n else 0.0,
        "phase2_recovery_rate": (n_recovered / max(n - n_in_pass, 1)) if n > n_in_pass else 0.0,
        "elapsed_seconds": round(time.time() - t_start, 1),
        "manifest": str(manifest),
        "skip_heavy": bool(args.skip_heavy),
    }
    summary_csv = out / "perov5_refine_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    print(
        f"[refine-perov5] DONE: total={n} phase1_fast_pass={n_phase1_fast_pass} "
        f"overall_pass={n_overall_pass} recovered={n_recovered} "
        f"elapsed={summary['elapsed_seconds']}s"
    )
    print(f"  manifest:        {manifest}")
    print(f"  summary:         {summary_csv}")
    print(f"  jubujiuzheng/:   {jubu_dir}")
    print(f"  tg/:             {tg_dir}")
    print(f"  ntg/:            {ntg_dir}")


if __name__ == "__main__":
    # Silence noisy pymatgen warnings before any heavy import
    warnings.filterwarnings("ignore")
    main()
