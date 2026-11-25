#!/usr/bin/env python
"""Repair ntg structures with the trained latent DiT and frozen VAE.

核心修复脚本。
  用训练好的 DiT 修复 ntg_latents.pt 里的坏样本：
  z_bad -> 加部分噪声 -> DiT denoise 到 all-pass 条件 -> VAE decode -> repaired PDB/CIF。
  它还会：

  - 根据失败指标选择 t_start
  - 每个样本生成多个 attempt
  - fast validate 候选
  - crystal 结构尝试 lattice scale / decoded atom types
  - QM9 可调用本地 stage2 refine
  - 选一个 best candidate 放到 selected_fast
  - 输出 repair_manifest.csv、selected_fast_manifest.csv、repair_summary.csv、correction_comparison.csv

The script does not overwrite the original ntg dataset. It writes repaired
structures, manifests, and summary CSVs to a timestamped output directory.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phymatgen_correction.src.runtime_env import (  # noqa: E402
    configure_headless_runtime,
    ensure_conda_lib_on_loader_path,
)

ensure_conda_lib_on_loader_path()
configure_headless_runtime()

import torch
from pymatgen.core import Lattice, Structure

from phymatgen_correction.src.code.eval_basic_cond_dit import (  # noqa: E402
    ddim_denoise,
    load_model,
)
from phymatgen_correction.src.code.eval_tg_noisy_denoise_decode import (  # noqa: E402
    decode_latents,
    get_vae_ckpt,
    save_decoded_structure,
)
from phymatgen_correction.src.code.build_latent_bank import (  # noqa: E402
    GLOBAL_METRICS,
    VariationalAutoencoderLitModule,
)
from phymatgen_correction.src.code.train_basic_cond_dit import (  # noqa: E402
    DATASET_ORDER,
    composition_to_tensor,
    make_beta_schedule,
)


@dataclass(frozen=True)
class AttemptSpec:
    attempt_i: int
    t_start: int
    repair_strength: float


@dataclass(frozen=True)
class SelectionCandidate:
    row: dict[str, Any]
    path: Path
    method: str
    cleanup: bool = False


def load_ntg_banks(data_root: Path, max_elements: int) -> dict[str, dict[str, Any]]:
    banks = {}
    for dataset in DATASET_ORDER:
        path = data_root / dataset / "latent" / "ntg_latents.pt"
        raw = torch.load(path, map_location="cpu", weights_only=False)
        comp = composition_to_tensor(raw["composition_counts"], max_elements)
        available_mask = raw["available_mask"].bool()
        banks[dataset] = {
            "z": raw["z"].float(),
            "token_mask": raw["token_mask"].bool(),
            "dataset_idx": torch.full((raw["z"].shape[0],), int(raw["dataset_idx"]), dtype=torch.long),
            "dataset_idx_value": int(raw["dataset_idx"]),
            "pbc_flag": torch.full((raw["z"].shape[0],), int(raw["pbc_flag"]), dtype=torch.long),
            "composition": comp,
            "available_mask": available_mask,
            "failure_mask": raw["failure_mask"].bool(),
            # Desired repair condition: all available metrics should pass.
            "target_pass": available_mask.float(),
            "raw_path": raw["path"],
            "filename": raw["filename"],
            "num_atoms": raw["num_atoms"],
            "metrics": raw["metrics"],
            "global_metrics": raw.get("global_metrics", GLOBAL_METRICS),
            "num_samples": int(raw["z"].shape[0]),
            "path": str(path),
        }
    return banks


def sample_indices(total: int, n: int, seed: int) -> list[int]:
    if n >= total:
        return list(range(total))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randperm(total, generator=generator)[:n].tolist()


def choose_t_start(
    dataset: str,
    failure_mask: torch.Tensor,
    metric_names: list[str],
    strategy: str,
    fixed_t_start: int | None,
) -> int:
    if fixed_t_start is not None:
        return int(fixed_t_start)

    failed = {metric_names[i] for i, flag in enumerate(failure_mask.tolist()) if bool(flag)}
    n_failed = len(failed)

    if strategy == "safe":
        # The previous conservative defaults repaired thermo/born failures but
        # introduced many new overlap/BVS/RDKit failures. Use less noise by
        # default so initially good local geometry is less likely to be erased.
        return {"qm9": 50, "mp20": 100, "perov5": 100, "carbon24": 100}[dataset]

    if strategy == "conservative":
        return {"qm9": 100, "mp20": 200, "perov5": 200, "carbon24": 200}[dataset]

    if strategy == "strong":
        return {"qm9": 400, "mp20": 400, "perov5": 400, "carbon24": 400}[dataset]

    if strategy == "target75":
        if dataset == "qm9":
            return 300 if failed & {"mol_stable", "rdkit_valid"} else 150
        if dataset in {"mp20", "perov5"}:
            if failed & {"no_atomic_overlap"}:
                return 120
            if failed & {"bvs_valid", "periodic_valid", "thermo_stable", "born_valid"} or n_failed >= 2:
                return 500
            return 300
        if dataset == "carbon24":
            if failed & {"no_atomic_overlap"}:
                return 120
            if failed & {"thermo_stable", "born_valid"} or n_failed >= 2:
                return 450
            return 250

    # balanced: stronger noise for crystal samples with multiple or heavy failures,
    # but keep QM9 and carbon24 comparatively conservative based on tg decode eval.
    if dataset == "qm9":
        return 200 if failed & {"mol_stable", "rdkit_valid"} else 100
    if dataset in {"mp20", "perov5"}:
        return 400 if (n_failed >= 2 or failed & {"thermo_stable", "born_valid"}) else 200
    if dataset == "carbon24":
        return 400 if (n_failed >= 2 and failed & {"thermo_stable", "born_valid"}) else 200
    raise ValueError(f"Unknown dataset: {dataset}")


def choose_repair_strength(
    dataset: str,
    strategy: str,
    fixed_strength: float | None,
) -> float:
    if fixed_strength is not None:
        return max(0.0, min(float(fixed_strength), 1.0))

    if strategy == "safe":
        return {"qm9": 0.25, "mp20": 0.35, "perov5": 0.35, "carbon24": 0.30}[dataset]
    if strategy == "conservative":
        return {"qm9": 0.45, "mp20": 0.55, "perov5": 0.55, "carbon24": 0.50}[dataset]
    if strategy == "balanced":
        return {"qm9": 0.65, "mp20": 0.75, "perov5": 0.75, "carbon24": 0.70}[dataset]
    if strategy == "strong":
        return 1.0
    if strategy == "target75":
        return {"qm9": 0.70, "mp20": 0.88, "perov5": 0.88, "carbon24": 0.82}[dataset]
    raise ValueError(f"Unknown strategy: {strategy}")


def make_attempt_plan(
    dataset: str,
    failure_mask: torch.Tensor,
    metric_names: list[str],
    args: argparse.Namespace,
) -> list[AttemptSpec]:
    base_t = choose_t_start(dataset, failure_mask, metric_names, args.t_start_strategy, args.fixed_t_start)
    base_strength = choose_repair_strength(dataset, args.t_start_strategy, args.repair_strength)
    attempts = max(1, int(args.attempts_per_sample))
    if attempts == 1:
        return [AttemptSpec(0, int(base_t), float(base_strength))]

    if args.fixed_t_start is not None and args.repair_strength is not None:
        return [AttemptSpec(i, int(base_t), float(base_strength)) for i in range(attempts)]

    failed = {metric_names[i] for i, flag in enumerate(failure_mask.tolist()) if bool(flag)}
    overlap_only = bool(failed) and failed <= {"no_atomic_overlap", "rdkit_valid"}
    if dataset == "qm9":
        t_values = [base_t, 200, 350, 500] if not overlap_only else [80, 120, 200, 300]
        s_values = [base_strength, 0.55, 0.80, 1.00]
    elif dataset == "carbon24":
        t_values = [base_t, 250, 400, 550] if not overlap_only else [80, 120, 200, 300]
        s_values = [base_strength, 0.60, 0.85, 1.00]
    else:
        t_values = [base_t, 300, 500, 700] if not overlap_only else [80, 120, 220, 350]
        s_values = [base_strength, 0.65, 0.90, 1.00]

    max_t = int(args.max_t_start)
    plan: list[AttemptSpec] = []
    for i in range(attempts):
        t_start = max(1, min(int(t_values[i % len(t_values)]), max_t))
        strength = max(0.0, min(float(s_values[i % len(s_values)]), 1.0))
        plan.append(AttemptSpec(i, t_start, strength))
    return plan


def batch_from_indices(
    bank: dict[str, Any],
    indices: list[int],
    device: torch.device,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    normalize: bool,
) -> dict[str, torch.Tensor]:
    idx = torch.as_tensor(indices, dtype=torch.long, device=device)
    batch = {
        "z": bank["z"][idx.cpu()].to(device),
        "token_mask": bank["token_mask"][idx.cpu()].to(device),
        "dataset_idx": bank["dataset_idx"][idx.cpu()].to(device),
        "pbc_flag": bank["pbc_flag"][idx.cpu()].to(device),
        "composition": bank["composition"][idx.cpu()].to(device),
        "target_pass": bank["target_pass"][idx.cpu()].to(device),
    }
    if normalize:
        batch["z"] = (batch["z"] - latent_mean) / latent_std.clamp(min=1e-8)
        batch["z"] = batch["z"] * batch["token_mask"][..., None]
    return batch


def repair_group(
    model,
    vae,
    bank: dict[str, Any],
    dataset: str,
    indices: list[int],
    spec: AttemptSpec,
    args: argparse.Namespace,
    device: torch.device,
    alpha_bar: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    normalize: bool,
    save_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    save_dir.mkdir(parents=True, exist_ok=True)
    t_start = int(spec.t_start)
    repair_strength = float(spec.repair_strength)

    for offset in range(0, len(indices), args.batch_size):
        batch_indices = indices[offset : offset + args.batch_size]
        batch = batch_from_indices(bank, batch_indices, device, latent_mean, latent_std, normalize)
        z0 = batch["z"]
        token_mask = batch["token_mask"]
        eps = torch.randn_like(z0) * token_mask[..., None]
        a = alpha_bar[t_start].view(1, 1, 1)
        z_t = (a.sqrt() * z0 + (1.0 - a).sqrt() * eps) * token_mask[..., None]
        z_repaired = ddim_denoise(model, z_t, t_start, alpha_bar, batch, args.denoise_steps)
        if repair_strength < 1.0:
            z_repaired = z0 + repair_strength * (z_repaired - z0)
            z_repaired = z_repaired * token_mask[..., None]
        z_decode = z_repaired * latent_std + latent_mean if normalize else z_repaired
        z_decode = z_decode * token_mask[..., None]
        out = decode_latents(vae, z_decode, token_mask)

        start = 0
        num_atoms = token_mask.sum(dim=1).long().tolist()
        for local_i, num_atom in enumerate(num_atoms):
            bank_i = int(batch_indices[local_i])
            source_path = bank["raw_path"][bank_i]
            row = save_decoded_structure(
                dataset=dataset,
                out=out,
                local_i=local_i,
                global_sample_i=bank_i,
                start=start,
                num_atom=int(num_atom),
                source_path=source_path,
                dataset_idx=int(bank["dataset_idx_value"]),
                save_dir=save_dir,
                atom_type_mode=args.atom_types,
                fix_overlap=args.fix_overlap,
                max_overlap_scale=args.max_overlap_scale,
            )
            failed_names = [
                bank["global_metrics"][j]
                for j, flag in enumerate(bank["failure_mask"][bank_i].tolist())
                if bool(flag)
            ]
            row.update(
                {
                    "dataset": dataset,
                    "ntg_index": bank_i,
                    "source_filename": bank["filename"][bank_i],
                    "attempt_i": spec.attempt_i,
                    "candidate_id": f"{dataset}_{bank_i:05d}_attempt_{spec.attempt_i:02d}",
                    "t_start": t_start,
                    "strategy": args.t_start_strategy,
                    "repair_strength": repair_strength,
                    "failed_metrics": "|".join(failed_names),
                }
            )
            rows.append(row)
            start += int(num_atom)

        print(
            f"[repair] {dataset} attempt={spec.attempt_i} t_start={t_start} "
            f"strength={repair_strength:.2f} "
            f"{min(offset + args.batch_size, len(indices))}/{len(indices)}",
            flush=True,
        )
    return rows


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}


def fast_validate_decoded(dataset: str, decoded_path: str) -> dict[str, Any]:
    path = Path(decoded_path)
    if not path.exists() or path.suffix == ".failed":
        return {"fast_pass": False, "fast_score": 0.0, "fast_notes": "decode_failed"}

    try:
        if dataset == "qm9":
            from phymatgen_correction.src.qm9_mol_utils import evaluate_molecule

            result = evaluate_molecule(path)
            checks = {
                "mol_stable": result.mol_stable,
                "rdkit_valid": result.rdkit_valid,
                "no_atomic_overlap": result.no_atomic_overlap,
            }
            checks["fast_pass"] = all(checks.values())
            checks["fast_score"] = float(sum(int(v) for v in checks.values() if isinstance(v, bool)))
            checks["fast_notes"] = result.error
            return checks

        if dataset == "carbon24":
            from phymatgen_correction.src.validate_carbon24 import _fast_validate_one
        elif dataset == "mp20":
            from phymatgen_correction.src.validate_mp20 import _fast_validate_one
        elif dataset == "perov5":
            from phymatgen_correction.src.validate_perov5 import _fast_validate_one
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        row = _fast_validate_one(path)
        checks = {
            key: _truthy(row.get(key, False))
            for key in ("no_atomic_overlap", "periodic_valid", "cn_valid", "charge_balanced", "bvs_valid")
            if key in row
        }
        checks["fast_pass"] = _truthy(row.get("fast_overall_pass", False))
        checks["fast_score"] = float(sum(int(v) for v in checks.values() if isinstance(v, bool)))
        checks["fast_notes"] = row.get("fast_notes", "")
        return checks
    except Exception as exc:  # noqa: BLE001
        return {"fast_pass": False, "fast_score": 0.0, "fast_notes": f"fast_validate_error:{exc!r}"}


@contextlib.contextmanager
def suppress_stdio():
    """Suppress noisy C/C++ validator diagnostics while trying many local candidates."""
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


def _copy_row(row: dict[str, Any], path: Path, method: str) -> dict[str, Any]:
    out = dict(row)
    out["decoded_path"] = str(path)
    out["postprocess_method"] = method
    return out


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def _filter_selected_rows(rows: list[dict[str, Any]], source_indices: set[int]) -> list[dict[str, Any]]:
    by_index: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            ntg_index = int(row.get("ntg_index", -1))
        except (TypeError, ValueError):
            continue
        if ntg_index in source_indices:
            by_index[ntg_index] = row
    return [by_index[index] for index in sorted(by_index)]


def crystal_scale_candidates(row: dict[str, Any], work_dir: Path, scales: Iterable[float]) -> list[SelectionCandidate]:
    src = Path(str(row["decoded_path"]))
    candidates: list[SelectionCandidate] = []
    if not src.exists() or src.suffix == ".failed":
        return candidates

    try:
        with suppress_stdio():
            struct = Structure.from_file(str(src))
    except Exception:
        return candidates

    work_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem
    for scale in scales:
        scale = float(scale)
        if abs(scale - 1.0) < 1e-6:
            continue
        try:
            lattice = Lattice(struct.lattice.matrix * scale)
            scaled = Structure(lattice, struct.species, struct.frac_coords, coords_are_cartesian=False)
            path = work_dir / f"{stem}.scale_{scale:.3f}.cif"
            scaled.to(str(path), fmt="cif")
            method = f"lattice_scale_{scale:.3f}"
            candidates.append(SelectionCandidate(_copy_row(row, path, method), path, method, cleanup=True))
        except Exception:
            continue
    return candidates


def _parse_atom_types(value: Any) -> list[int]:
    if value is None:
        return []
    text = str(value).replace(",", " ").strip()
    if not text:
        return []
    out: list[int] = []
    for part in text.split():
        try:
            out.append(int(float(part)))
        except ValueError:
            return []
    return out


def decoded_atom_type_candidates(row: dict[str, Any], work_dir: Path) -> list[SelectionCandidate]:
    src = Path(str(row["decoded_path"]))
    candidates: list[SelectionCandidate] = []
    if not src.exists() or src.suffix == ".failed":
        return candidates

    pred_atom_types = _parse_atom_types(row.get("pred_atom_types"))
    original_atom_types = _parse_atom_types(row.get("original_atom_types"))
    if not pred_atom_types or pred_atom_types == original_atom_types:
        return candidates

    try:
        with suppress_stdio():
            struct = Structure.from_file(str(src))
    except Exception:
        return candidates
    if len(pred_atom_types) != len(struct):
        return candidates

    try:
        decoded = Structure(struct.lattice, pred_atom_types, struct.frac_coords, coords_are_cartesian=False)
    except Exception:
        return candidates

    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / f"{src.stem}.decoded_atom_types.cif"
    try:
        decoded.to(str(path), fmt="cif")
    except Exception:
        return candidates

    candidate_row = _copy_row(row, path, "decoded_atom_types")
    candidate_row["atom_type_mode"] = "decoded"
    candidates.append(SelectionCandidate(candidate_row, path, "decoded_atom_types", cleanup=True))
    return candidates


def qm9_local_candidates(row: dict[str, Any], work_dir: Path, args: argparse.Namespace) -> list[SelectionCandidate]:
    src = Path(str(row["decoded_path"]))
    candidates: list[SelectionCandidate] = []
    if not args.qm9_local_refine or not src.exists() or src.suffix == ".failed":
        return candidates

    try:
        from phymatgen_correction.src.code.refine_qm9_stage2 import (  # noqa: PLC0415
            generate_candidates,
            quiet_evaluate_molecule,
            score_eval,
            write_pdb,
        )
    except Exception:
        return candidates

    try:
        before = quiet_evaluate_molecule(src, margin=args.qm9_bond_margin, overlap_tolerance=args.qm9_overlap_tolerance)
    except Exception:
        return candidates
    if before.mol_stable and before.rdkit_valid and before.no_atomic_overlap:
        return candidates

    work_dir.mkdir(parents=True, exist_ok=True)
    refine_args = argparse.Namespace(
        bond_margin=args.qm9_bond_margin,
        overlap_tolerance=args.qm9_overlap_tolerance,
        max_iters=args.qm9_refine_max_iters,
        seed=args.seed + int(row.get("ntg_index", 0)),
        formula_rebuild_seeds=max(1, int(args.qm9_formula_rebuild_seeds)),
        graph_variants=max(1, int(args.qm9_graph_variants)),
        disable_formula_rebuild=args.qm9_disable_formula_rebuild,
        max_candidate_errors=0,
    )

    best_score = score_eval(before)
    tried = 0
    for candidate in generate_candidates(src, refine_args):
        if tried >= int(args.qm9_max_refine_candidates):
            break
        tried += 1
        if candidate.error:
            continue
        try:
            path = work_dir / f"{src.stem}.{candidate.method}.pdb"
            write_pdb(path, candidate.atoms, candidate.bonds)
            with suppress_stdio():
                result = quiet_evaluate_molecule(
                    path,
                    margin=args.qm9_bond_margin,
                    overlap_tolerance=args.qm9_overlap_tolerance,
                )
            score = score_eval(result)
        except Exception:
            continue
        if score < best_score and not (result.mol_stable and result.rdkit_valid and result.no_atomic_overlap):
            continue
        best_score = max(best_score, score)
        candidates.append(SelectionCandidate(_copy_row(row, path, candidate.method), path, candidate.method, cleanup=True))
        if result.mol_stable and result.rdkit_valid and result.no_atomic_overlap:
            break
    return candidates


def build_selection_candidates(
    dataset: str,
    row: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> list[SelectionCandidate]:
    work_dir = output_dir / dataset / "selection_postprocess" / str(row["ntg_index"])
    if dataset == "qm9":
        return qm9_local_candidates(row, work_dir, args)
    if dataset in {"mp20", "perov5", "carbon24"}:
        candidates: list[SelectionCandidate] = []
        if args.crystal_try_decoded_atom_types:
            candidates.extend(decoded_atom_type_candidates(row, work_dir))
        overlap_scale = float(row.get("overlap_scale") or 1.0)
        needs_scale_search = overlap_scale >= float(args.crystal_scale_search_min_overlap_scale)
        if args.crystal_lattice_scale_search and needs_scale_search:
            scales = [float(item) for item in str(args.crystal_lattice_scales).split(",") if item.strip()]
            candidates.extend(crystal_scale_candidates(row, work_dir, scales))
            for atom_candidate in list(candidates):
                if atom_candidate.method != "decoded_atom_types":
                    continue
                scaled_row = dict(atom_candidate.row)
                candidates.extend(crystal_scale_candidates(scaled_row, work_dir / "decoded_scales", scales))
        return candidates
    return []


def copy_fast_pass_to_tg(dataset: str, selected_rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    tg_dir = output_dir / dataset / "tg"
    tg_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for row in selected_rows:
        if not _truthy(row.get("select_fast_pass", False)):
            continue
        src = Path(str(row.get("selected_path") or row.get("decoded_path") or ""))
        if not src.exists() or src.suffix == ".failed":
            continue
        dst = tg_dir / src.name
        if src.resolve() != dst.resolve():
            try:
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(src)
            except OSError:
                shutil.copyfile(src, dst)
        row["tg_path"] = str(dst)
        copied += 1
    return {"tg_dir": str(tg_dir), "tg_count": copied}


def _split_metric_names(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [part for part in text.split("|") if part]


def _post_metric_fields(row: dict[str, Any]) -> list[str]:
    skip = {"select_fast_pass", "select_fast_score", "select_fast_notes"}
    return sorted(
        key
        for key in row
        if key.startswith("select_") and key not in skip and isinstance(row.get(key), (bool, int, float, str))
    )


def write_correction_comparison(dataset: str, selected_rows: list[dict[str, Any]], output_dir: Path) -> str:
    path = output_dir / dataset / "correction_comparison.csv"
    comparison_rows: list[dict[str, Any]] = []
    for row in selected_rows:
        metric_fields = _post_metric_fields(row)
        post_passed_metrics = [field.removeprefix("select_") for field in metric_fields if _truthy(row.get(field))]
        post_failed_metrics = [field.removeprefix("select_") for field in metric_fields if not _truthy(row.get(field))]
        original_failed_metrics = _split_metric_names(row.get("failed_metrics"))
        fixed_original_metrics = [metric for metric in original_failed_metrics if metric in post_passed_metrics]
        still_failed_original_metrics = [metric for metric in original_failed_metrics if metric in post_failed_metrics]
        introduced_failed_metrics = [metric for metric in post_failed_metrics if metric not in original_failed_metrics]
        passed = _truthy(row.get("select_fast_pass", False))

        comparison_rows.append(
            {
                "dataset": dataset,
                "ntg_index": row.get("ntg_index", ""),
                "sample_i": row.get("sample_i", ""),
                "source_filename": row.get("source_filename", ""),
                "source_path": row.get("source_path", ""),
                "original_failed_metrics": "|".join(original_failed_metrics),
                "selected_candidate_id": row.get("candidate_id", ""),
                "selected_attempt_i": row.get("attempt_i", ""),
                "selected_t_start": row.get("t_start", ""),
                "selected_repair_strength": row.get("repair_strength", ""),
                "selected_postprocess_method": row.get("postprocess_method", "original"),
                "selected_path": row.get("selected_path", ""),
                "tg_path": row.get("tg_path", ""),
                "corrected_fast_pass": passed,
                "corrected_status": "tg" if passed else "ntg",
                "corrected_passed_metrics": "|".join(post_passed_metrics),
                "corrected_failed_metrics": "|".join(post_failed_metrics),
                "fixed_original_metrics": "|".join(fixed_original_metrics),
                "still_failed_original_metrics": "|".join(still_failed_original_metrics),
                "introduced_failed_metrics": "|".join(introduced_failed_metrics),
                "corrected_failure_reason": "" if passed else row.get("select_fast_notes", ""),
                "select_fast_score": row.get("select_fast_score", ""),
                "decode_ok": row.get("decode_ok", ""),
                "decode_error": row.get("decode_error", ""),
                "atom_type_acc": row.get("atom_type_acc", ""),
                "num_atoms": row.get("num_atoms", ""),
                "overlap_scale": row.get("overlap_scale", ""),
                "selection_candidate_count": row.get("selection_candidate_count", ""),
            }
        )

    fieldnames = [
        "dataset",
        "ntg_index",
        "sample_i",
        "source_filename",
        "source_path",
        "original_failed_metrics",
        "selected_candidate_id",
        "selected_attempt_i",
        "selected_t_start",
        "selected_repair_strength",
        "selected_postprocess_method",
        "selected_path",
        "tg_path",
        "corrected_fast_pass",
        "corrected_status",
        "corrected_passed_metrics",
        "corrected_failed_metrics",
        "fixed_original_metrics",
        "still_failed_original_metrics",
        "introduced_failed_metrics",
        "corrected_failure_reason",
        "select_fast_score",
        "decode_ok",
        "decode_error",
        "atom_type_acc",
        "num_atoms",
        "overlap_scale",
        "selection_candidate_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)
    return str(path)


def select_fast_candidates(
    dataset: str,
    rows: list[dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    selected_dir = output_dir / dataset / "selected_fast"
    selected_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / dataset / "selected_fast_manifest.csv"
    partial_path = output_dir / dataset / "selected_fast_manifest.partial.csv"
    by_source: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[int(row["ntg_index"])].append(row)
    source_indices = set(by_source)

    if selected_path.exists():
        selected_rows: list[dict[str, Any]] = _read_csv_rows(selected_path)
        selected_rows = _filter_selected_rows(selected_rows, source_indices)
        print(
            f"[select] {dataset} found complete selected manifest with "
            f"{len(selected_rows)}/{len(by_source)} sources",
            flush=True,
        )
        if len({int(row["ntg_index"]) for row in selected_rows}) < len(by_source):
            print(f"[select] {dataset} selected manifest is incomplete; resuming missing sources", flush=True)
    else:
        selected_rows = _read_csv_rows(partial_path)
        selected_rows = _filter_selected_rows(selected_rows, source_indices)
        if selected_rows:
            print(
                f"[select] {dataset} resume from partial manifest: "
                f"{len(selected_rows)}/{len(by_source)} sources already selected",
                flush=True,
            )
    done_indices = {int(row["ntg_index"]) for row in selected_rows if str(row.get("ntg_index", "")).strip()}
    last_flush = time.monotonic()
    last_print = time.monotonic()
    started = time.monotonic()

    for ntg_index, source_rows in sorted(by_source.items()):
        if ntg_index in done_indices:
            continue
        best_row: dict[str, Any] | None = None
        best_key: tuple[float, float, float, float, float] | None = None
        candidate_count = 0
        for row in source_rows:
            src = Path(str(row["decoded_path"]))
            original = SelectionCandidate(_copy_row(row, src, "original"), src, "original")
            original_checks = fast_validate_decoded(dataset, str(original.path))
            candidates = [original]
            if not original_checks.get("fast_pass", False):
                candidates.extend(build_selection_candidates(dataset, row, output_dir, args))

            for candidate_i, candidate in enumerate(candidates):
                candidate_count += 1
                candidate_row = candidate.row
                checks = original_checks if candidate_i == 0 else fast_validate_decoded(dataset, str(candidate.path))
                candidate_row.update({f"select_{key}": value for key, value in checks.items()})
                candidate_row["postprocess_cleanup"] = candidate.cleanup
                candidate_row["postprocess_source_path"] = row["decoded_path"]
                overlap_scale = float(candidate_row.get("overlap_scale") or 1.0)
                method_penalty = 0.0 if candidate.method == "original" else -1.0
                key = (
                    float(bool(checks.get("fast_pass", False))),
                    float(checks.get("fast_score", 0.0)),
                    method_penalty,
                    -abs(overlap_scale - 1.0),
                    -float(candidate_row.get("attempt_i", 0)),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_row = candidate_row
                if checks.get("fast_pass", False):
                    break
            if best_row is not None and _truthy(best_row.get("select_fast_pass", False)):
                break
        if best_row is None:
            continue

        src = Path(str(best_row["decoded_path"]))
        if src.exists() and src.suffix != ".failed":
            dst = selected_dir / src.name
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            best_row["selected_path"] = str(dst)
        else:
            best_row["selected_path"] = str(src)
        best_row["selected_fast_rank_key"] = "|".join(str(v) for v in best_key)
        best_row["selection_candidate_count"] = candidate_count
        selected_rows.append(best_row)
        done_indices.add(ntg_index)

        now = time.monotonic()
        if len(selected_rows) % max(1, int(args.select_checkpoint_interval)) == 0 or (
            now - last_flush >= float(args.select_checkpoint_seconds)
        ):
            _write_csv_rows(partial_path, selected_rows)
            last_flush = now
        if now - last_print >= float(args.select_log_interval_seconds):
            rate = (len(done_indices) / max(now - started, 1e-6)) if now > started else 0.0
            remaining = len(by_source) - len(done_indices)
            eta = remaining / rate if rate > 0 else 0.0
            print(
                f"[select] {dataset} {len(done_indices)}/{len(by_source)} "
                f"selected, last_candidates={candidate_count}, eta_min={eta / 60.0:.1f}",
                flush=True,
            )
            last_print = now

    tg_summary = copy_fast_pass_to_tg(dataset, selected_rows, output_dir)
    comparison_path = write_correction_comparison(dataset, selected_rows, output_dir)

    if selected_rows:
        _write_csv_rows(selected_path, selected_rows)
        if partial_path.exists():
            partial_path.unlink()

    fast_pass = sum(1 for row in selected_rows if _truthy(row.get("select_fast_pass", False)))
    postprocess_selected = sum(1 for row in selected_rows if row.get("postprocess_method", "original") != "original")
    postprocess_fast_pass = sum(
        1
        for row in selected_rows
        if row.get("postprocess_method", "original") != "original" and _truthy(row.get("select_fast_pass", False))
    )
    return {
        "selected_dir": str(selected_dir),
        "selected_manifest": str(selected_path),
        "num_selected_unique": len(selected_rows),
        "selected_fast_pass": fast_pass,
        "selected_fast_pass_rate": fast_pass / max(len(selected_rows), 1),
        "selected_postprocess": postprocess_selected,
        "selected_postprocess_fast_pass": postprocess_fast_pass,
        "correction_comparison": comparison_path,
        **tg_summary,
    }


def repair_dataset(
    model,
    vae,
    bank: dict[str, Any],
    dataset: str,
    args: argparse.Namespace,
    device: torch.device,
    alpha_bar: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    normalize: bool,
    output_dir: Path,
) -> dict[str, Any]:
    manifest_path = output_dir / dataset / "repair_manifest.csv"
    partial_manifest_path = output_dir / dataset / "repair_manifest.partial.csv"
    groups: dict[AttemptSpec, list[int]] = defaultdict(list)
    if args.select_only:
        load_path = manifest_path if manifest_path.exists() else partial_manifest_path
        dataset_rows = _read_csv_rows(load_path)
        if not dataset_rows:
            raise FileNotFoundError(
                f"--select-only requires a non-empty manifest: {manifest_path} "
                f"or {partial_manifest_path}"
            )
        print(f"[repair] {dataset} select-only loaded {len(dataset_rows)} candidates from {load_path}", flush=True)
    else:
        total = int(bank["num_samples"])
        n = min(int(args.samples_per_dataset), total) if args.samples_per_dataset > 0 else total
        selected = sample_indices(total, n, args.seed + len(dataset))
        metric_names = list(bank["global_metrics"])
        for idx in selected:
            for spec in make_attempt_plan(dataset, bank["failure_mask"][idx], metric_names, args):
                groups[spec].append(int(idx))

        dataset_rows = []
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        for spec, indices in sorted(groups.items(), key=lambda item: (item[0].attempt_i, item[0].t_start, item[0].repair_strength)):
            dataset_rows.extend(
                repair_group(
                    model,
                    vae,
                    bank,
                    dataset,
                    indices,
                    spec,
                    args,
                    device,
                    alpha_bar,
                    latent_mean,
                    latent_std,
                    normalize,
                    output_dir
                    / dataset
                    / f"attempt_{spec.attempt_i:02d}_t_{spec.t_start}_s_{spec.repair_strength:.2f}",
                )
            )
            _write_csv_rows(partial_manifest_path, dataset_rows)

        _write_csv_rows(manifest_path, dataset_rows)
        if partial_manifest_path.exists():
            partial_manifest_path.unlink()

    selection_summary = select_fast_candidates(dataset, dataset_rows, output_dir, args) if args.select_fast else {}

    decode_ok = sum(1 for row in dataset_rows if _truthy(row["decode_ok"]))
    summary = {
        "dataset": dataset,
        "num_sources": len({int(row["ntg_index"]) for row in dataset_rows}),
        "attempts_per_sample": max(1, int(args.attempts_per_sample)),
        "num_candidates": len(dataset_rows),
        "decode_ok_rate": decode_ok / max(len(dataset_rows), 1),
        "atom_type_acc": sum(float(row["atom_type_acc"]) for row in dataset_rows) / max(len(dataset_rows), 1),
        "manifest": str(manifest_path),
        **selection_summary,
    }
    if groups:
        for spec, indices in sorted(groups.items(), key=lambda item: (item[0].attempt_i, item[0].t_start, item[0].repair_strength)):
            summary[f"count_attempt_{spec.attempt_i:02d}_t_{spec.t_start}_s_{spec.repair_strength:.2f}"] = len(indices)
    else:
        by_attempt = defaultdict(int)
        for row in dataset_rows:
            key = (
                int(row.get("attempt_i", 0)),
                int(float(row.get("t_start", 0))),
                float(row.get("repair_strength", 0.0)),
            )
            by_attempt[key] += 1
        for attempt_i, t_start, repair_strength in sorted(by_attempt):
            summary[f"count_attempt_{attempt_i:02d}_t_{t_start}_s_{repair_strength:.2f}"] = by_attempt[
                (attempt_i, t_start, repair_strength)
            ]
    print(
        f"[summary] {dataset} sources={summary['num_sources']} "
        f"candidates={summary['num_candidates']} "
        f"decode_ok={summary['decode_ok_rate']:.3f} "
        f"atom_acc={summary['atom_type_acc']:.4f} "
        f"selected_fast_pass={summary.get('selected_fast_pass_rate', 0.0):.3f}",
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dit-ckpt", type=Path, default=None)
    parser.add_argument("--vae-ckpt", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "correction")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_ORDER), choices=list(DATASET_ORDER))
    parser.add_argument("--samples-per-dataset", type=int, default=-1, help="Use <=0 to repair all ntg samples.")
    parser.add_argument(
        "--select-only",
        action="store_true",
        help="Skip DiT/VAE sampling and resume selected_fast generation from existing repair_manifest.csv files.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--denoise-steps", type=int, default=80)
    parser.add_argument(
        "--t-start-strategy",
        choices=["safe", "conservative", "balanced", "strong", "target75"],
        default="target75",
    )
    parser.add_argument("--fixed-t-start", type=int, default=None, help="Override strategy with one t_start for all samples.")
    parser.add_argument("--max-t-start", type=int, default=800, help="Clamp adaptive t_start values to this maximum.")
    parser.add_argument(
        "--attempts-per-sample",
        type=int,
        default=4,
        help="Number of stochastic repair candidates to generate for each original ntg sample.",
    )
    parser.add_argument(
        "--select-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run fast validators on candidates and copy one best candidate per original sample to selected_fast.",
    )
    parser.add_argument(
        "--repair-strength",
        type=float,
        default=None,
        help="Blend denoised latent with original latent: z = z0 + strength * (z_denoised - z0). "
        "Use 1.0 for the old full-replacement behavior. Default is strategy/dataset specific.",
    )
    parser.add_argument("--atom-types", choices=["original", "decoded"], default="original")
    parser.add_argument(
        "--fix-overlap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For CIF outputs, uniformly expand the decoded lattice when atomic distances violate overlap thresholds.",
    )
    parser.add_argument("--max-overlap-scale", type=float, default=1.30)
    parser.add_argument(
        "--crystal-lattice-scale-search",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="During selected_fast picking, try extra uniformly scaled CIF candidates for crystal datasets.",
    )
    parser.add_argument(
        "--crystal-lattice-scales",
        default="0.94,1.00,1.06,1.12,1.20,1.30",
        help="Comma-separated lattice scale factors used by --crystal-lattice-scale-search.",
    )
    parser.add_argument(
        "--crystal-scale-search-min-overlap-scale",
        type=float,
        default=1.02,
        help="Only try extra lattice-scale candidates when decode overlap_scale is at least this value.",
    )
    parser.add_argument(
        "--crystal-try-decoded-atom-types",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="During selected_fast picking, also validate decoded atom-type CIF variants when available.",
    )
    parser.add_argument(
        "--select-checkpoint-interval",
        type=int,
        default=250,
        help="Write selected_fast_manifest.partial.csv after this many selected sources.",
    )
    parser.add_argument(
        "--select-checkpoint-seconds",
        type=float,
        default=300.0,
        help="Write selected_fast_manifest.partial.csv at least this often during selection.",
    )
    parser.add_argument(
        "--select-log-interval-seconds",
        type=float,
        default=60.0,
        help="Print selected_fast progress at least this often.",
    )
    parser.add_argument(
        "--qm9-local-refine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="During selected_fast picking, locally rebuild/refine failed QM9 molecules while preserving atom counts/types.",
    )
    parser.add_argument("--qm9-bond-margin", type=float, default=0.05)
    parser.add_argument("--qm9-overlap-tolerance", type=float, default=0.5)
    parser.add_argument("--qm9-refine-max-iters", type=int, default=300)
    parser.add_argument("--qm9-formula-rebuild-seeds", type=int, default=1)
    parser.add_argument("--qm9-graph-variants", type=int, default=1)
    parser.add_argument("--qm9-max-refine-candidates", type=int, default=8)
    parser.add_argument(
        "--qm9-disable-formula-rebuild",
        action="store_true",
        help="Only try QM9 geometry optimization candidates; leave formula rebuild disabled.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    if not args.select_only and args.dit_ckpt is None:
        parser.error("--dit-ckpt is required unless --select-only is used")
    return args


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    model = vae = cfg = ckpt = None
    vae_ckpt = str(args.vae_ckpt) if args.vae_ckpt else ""
    device = torch.device(args.device)
    if not args.select_only:
        model, cfg, ckpt = load_model(args.dit_ckpt, device)
        vae_ckpt = str(args.vae_ckpt) if args.vae_ckpt else get_vae_ckpt(args.data_root)
        vae = VariationalAutoencoderLitModule.load_from_checkpoint(vae_ckpt, map_location="cpu")
        vae.eval().requires_grad_(False).to(device)

    if args.output_dir is None:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        args.output_dir = args.data_root.parent / "debug" / timestamp
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] dit_ckpt={args.dit_ckpt}")
    print(f"[INFO] vae_ckpt={vae_ckpt or 'select-only'}")
    print(f"[INFO] output_dir={args.output_dir}")
    print(
        f"[INFO] strategy={args.t_start_strategy}, fixed_t_start={args.fixed_t_start}, "
        f"repair_strength={args.repair_strength if args.repair_strength is not None else 'auto'}, "
        f"attempts_per_sample={args.attempts_per_sample}, select_fast={args.select_fast}"
    )

    if args.select_only:
        banks = {dataset: {} for dataset in args.datasets}
        normalize = False
        latent_mean = torch.tensor([0.0], device=device)
        latent_std = torch.tensor([1.0], device=device)
        alpha_bar = torch.tensor([1.0], device=device)
    else:
        banks = load_ntg_banks(args.data_root, int(cfg["max_elements"]))
        banks = {k: v for k, v in banks.items() if k in args.datasets}

        normalize = bool(cfg.get("normalize_latents", True))
        latent_mean = torch.as_tensor(ckpt.get("latent_mean", torch.tensor([0.0])), device=device).reshape(1)
        latent_std = torch.as_tensor(ckpt.get("latent_std", torch.tensor([1.0])), device=device).reshape(1)
        alpha_bar = torch.cumprod(1.0 - make_beta_schedule(int(cfg["num_timesteps"]), device), dim=0)

    summaries = []
    for dataset in args.datasets:
        summaries.append(
            repair_dataset(
                model,
                vae,
                banks[dataset],
                dataset,
                args,
                device,
                alpha_bar,
                latent_mean,
                latent_std,
                normalize,
                args.output_dir,
            )
        )

    summary_path = args.output_dir / "repair_summary.csv"
    all_keys = sorted({key for row in summaries for key in row})
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(summaries)
    (args.output_dir / "repair_config.json").write_text(
        json.dumps(
            {
                "dit_ckpt": str(args.dit_ckpt),
                "vae_ckpt": vae_ckpt,
                "data_root": str(args.data_root),
                "datasets": args.datasets,
                "samples_per_dataset": args.samples_per_dataset,
                "select_only": args.select_only,
                "batch_size": args.batch_size,
                "denoise_steps": args.denoise_steps,
                "t_start_strategy": args.t_start_strategy,
                "fixed_t_start": args.fixed_t_start,
                "max_t_start": args.max_t_start,
                "attempts_per_sample": args.attempts_per_sample,
                "select_fast": args.select_fast,
                "repair_strength": args.repair_strength,
                "atom_types": args.atom_types,
                "fix_overlap": args.fix_overlap,
                "max_overlap_scale": args.max_overlap_scale,
                "crystal_lattice_scale_search": args.crystal_lattice_scale_search,
                "crystal_lattice_scales": args.crystal_lattice_scales,
                "crystal_scale_search_min_overlap_scale": args.crystal_scale_search_min_overlap_scale,
                "crystal_try_decoded_atom_types": args.crystal_try_decoded_atom_types,
                "select_checkpoint_interval": args.select_checkpoint_interval,
                "select_checkpoint_seconds": args.select_checkpoint_seconds,
                "select_log_interval_seconds": args.select_log_interval_seconds,
                "qm9_local_refine": args.qm9_local_refine,
                "qm9_bond_margin": args.qm9_bond_margin,
                "qm9_overlap_tolerance": args.qm9_overlap_tolerance,
                "qm9_refine_max_iters": args.qm9_refine_max_iters,
                "qm9_formula_rebuild_seeds": args.qm9_formula_rebuild_seeds,
                "qm9_graph_variants": args.qm9_graph_variants,
                "qm9_max_refine_candidates": args.qm9_max_refine_candidates,
                "qm9_disable_formula_rebuild": args.qm9_disable_formula_rebuild,
                "device": args.device,
                "seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[DONE] wrote {summary_path}")


if __name__ == "__main__":
    main()
