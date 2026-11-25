#!/usr/bin/env python
"""Run tg noisy-denoise-decode evaluation for Stage-1 DiT.

完整 decode 评估脚本。
     流程是：tg latent -> 加噪 -> DiT denoise -> VAE decode -> 写 PDB/CIF。
     用来检查 latent denoise 后真正解码成结构是否还能正常。输出：
      - 每个 dataset/t_start 的 decoded 文件

This evaluates the full latent -> VAE decoder part on sampled tg structures:

    z_good -> add noise at t_start -> DiT denoise -> z_recon -> VAE decode -> CIF/PDB

It is intentionally sample-based by default because full tg decode/validation is
expensive, especially for QM9.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.periodic_table import Element


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Importing build_latent_bank installs lightweight fallbacks needed in this env.
from phymatgen_correction.src.code.build_latent_bank import (  # noqa: E402
    VariationalAutoencoderLitModule,
    Z_TO_SYMBOL,
    load_structure_file_to_data,
)
from phymatgen_correction.src.code.eval_basic_cond_dit import (  # noqa: E402
    ddim_denoise,
    load_model,
    masked_metrics,
)
from phymatgen_correction.src.code.train_basic_cond_dit import (  # noqa: E402
    DATASET_ORDER,
    load_tg_banks,
    make_beta_schedule,
)


def get_vae_ckpt(data_root: Path) -> str:
    bank = torch.load(data_root / "qm9/latent/tg_latents.pt", map_location="cpu", weights_only=False)
    return str(bank["vae_ckpt"])


def sample_indices(num_samples: int, sample_count: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if sample_count >= num_samples:
        return torch.arange(num_samples, device=device)
    idx = torch.randperm(num_samples, generator=generator)[:sample_count]
    return idx.to(device)


def batch_from_bank(
    bank: dict[str, Any],
    idx: torch.Tensor,
    device: torch.device,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    normalize: bool,
) -> dict[str, torch.Tensor]:
    batch = {
        "z": bank["z"].to(device)[idx],
        "token_mask": bank["token_mask"].to(device)[idx],
        "dataset_idx": bank["dataset_idx"].to(device)[idx],
        "pbc_flag": bank["pbc_flag"].to(device)[idx],
        "composition": bank["composition"].to(device)[idx],
        "target_pass": bank["target_pass"].to(device)[idx],
    }
    if normalize:
        batch["z"] = (batch["z"] - latent_mean) / latent_std.clamp(min=1e-8)
        batch["z"] = batch["z"] * batch["token_mask"][..., None]
    return batch


def decode_latents(
    vae: VariationalAutoencoderLitModule,
    z_denorm: torch.Tensor,
    token_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    num_atoms = token_mask.sum(dim=1).long()
    flat_z = z_denorm[token_mask]
    batch_index = torch.repeat_interleave(torch.arange(z_denorm.shape[0], device=z_denorm.device), num_atoms)
    token_idx = (torch.cumsum(token_mask.long(), dim=1) - 1)[token_mask]
    encoded = {
        "x": flat_z,
        "num_atoms": num_atoms,
        "batch": batch_index,
        "token_idx": token_idx,
    }
    with torch.no_grad():
        out = vae.decode(encoded)
    out["frac_coords"] = out["frac_coords"] % 1.0
    out["frac_coords"] = torch.clamp(out["frac_coords"], 0.0, 0.9999)
    out["lengths"] = torch.clamp(out["lengths"], min=0.5)
    out["angles"] = torch.clamp(out["angles"], min=0.5, max=2.6)
    return out


def safe_original_atom_types(path: str, dataset_idx: int) -> np.ndarray:
    data = load_structure_file_to_data(path, dataset_idx)
    return data.atom_types.detach().cpu().numpy().astype(int)


def write_qm9_pdb(path: Path, atom_types: np.ndarray, pos: np.ndarray) -> None:
    lines = []
    for atom_i, (atomic_num, coord) in enumerate(zip(atom_types, pos), start=1):
        symbol = Z_TO_SYMBOL.get(int(atomic_num), str(int(atomic_num)))
        lines.append(
            f"HETATM{atom_i:5d} {symbol:>2s}   MOL A   1"
            f"    {float(coord[0]):8.3f}{float(coord[1]):8.3f}{float(coord[2]):8.3f}"
            f"  1.00  0.00          {symbol:>2s}\n"
        )
    lines.append("END\n")
    path.write_text("".join(lines), encoding="utf-8")


def _atomic_radius(symbol: str) -> float:
    try:
        radius = Element(symbol).atomic_radius_calculated or Element(symbol).atomic_radius
        if radius is not None:
            return float(radius)
    except Exception:  # noqa: BLE001
        pass
    return 0.8


def reduce_crystal_overlap(struct: Structure, max_scale: float) -> tuple[Structure, float]:
    """Uniformly expand the lattice if decoded atoms are closer than radius thresholds."""
    if len(struct) < 2 or max_scale <= 1.0:
        return struct, 1.0

    dm = struct.distance_matrix.copy()
    np.fill_diagonal(dm, np.inf)
    atom_types = [site.specie.symbol for site in struct]
    required_scale = 1.0
    for i in range(len(struct)):
        r_i = _atomic_radius(atom_types[i])
        for j in range(i + 1, len(struct)):
            dist = float(dm[i, j])
            if not math.isfinite(dist) or dist <= 1e-6:
                required_scale = max(required_scale, max_scale)
                continue
            threshold = 0.70 * (r_i + _atomic_radius(atom_types[j]))
            if dist < threshold:
                required_scale = max(required_scale, threshold / dist)

    scale = min(max(required_scale * 1.01, 1.0), max_scale)
    if scale <= 1.0001:
        return struct, 1.0

    lattice = Lattice.from_parameters(
        *(np.asarray(struct.lattice.abc, dtype=float) * scale).tolist(),
        *struct.lattice.angles,
    )
    return Structure(lattice, struct.species, struct.frac_coords, coords_are_cartesian=False), scale


def save_decoded_structure(
    dataset: str,
    out: dict[str, torch.Tensor],
    local_i: int,
    global_sample_i: int,
    start: int,
    num_atom: int,
    source_path: str,
    dataset_idx: int,
    save_dir: Path,
    atom_type_mode: str,
    fix_overlap: bool = False,
    max_overlap_scale: float = 1.12,
) -> dict[str, Any]:
    pred_atom_types = out["atom_types"].narrow(0, start, num_atom).argmax(dim=1).detach().cpu().numpy().astype(int)
    pred_atom_types[pred_atom_types == 0] = 1
    original_atom_types = safe_original_atom_types(source_path, dataset_idx)
    atom_acc = float((pred_atom_types[: len(original_atom_types)] == original_atom_types).mean())
    atom_types = original_atom_types if atom_type_mode == "original" else pred_atom_types

    filename = f"{dataset}_{global_sample_i:05d}"
    overlap_scale = 1.0
    try:
        if dataset == "qm9":
            pos = (out["pos"].narrow(0, start, num_atom) * 10.0).detach().cpu().numpy()
            path = save_dir / f"{filename}.pdb"
            write_qm9_pdb(path, atom_types, pos)
        else:
            frac = out["frac_coords"].narrow(0, start, num_atom).detach().cpu().numpy()
            lengths = (
                out["lengths"][local_i].detach().cpu().numpy() * float(num_atom) ** (1.0 / 3.0)
            )
            angles = torch.rad2deg(out["angles"][local_i]).detach().cpu().numpy()
            lattice = Lattice.from_parameters(*lengths.tolist(), *angles.tolist())
            struct = Structure(lattice, atom_types, frac, coords_are_cartesian=False)
            if fix_overlap:
                struct, overlap_scale = reduce_crystal_overlap(struct, max_overlap_scale)
            path = save_dir / f"{filename}.cif"
            struct.to(str(path), fmt="cif")
        ok = True
        error = ""
    except Exception as exc:  # noqa: BLE001
        path = save_dir / f"{filename}.failed"
        path.write_text(repr(exc), encoding="utf-8")
        ok = False
        error = repr(exc)

    return {
        "sample_i": global_sample_i,
        "source_path": source_path,
        "decoded_path": str(path),
        "decode_ok": ok,
        "decode_error": error,
        "atom_type_acc": atom_acc,
        "num_atoms": num_atom,
        "overlap_scale": overlap_scale if dataset != "qm9" else 1.0,
        "pred_atom_types": " ".join(str(int(v)) for v in pred_atom_types.tolist()),
        "original_atom_types": " ".join(str(int(v)) for v in original_atom_types.tolist()),
    }


def evaluate_dataset_t(
    model,
    vae,
    bank: dict[str, Any],
    dataset: str,
    t_start: int,
    args: argparse.Namespace,
    device: torch.device,
    alpha_bar: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    normalize: bool,
    output_dir: Path,
) -> dict[str, Any]:
    total = int(bank["num_samples"])
    n = min(int(args.samples_per_dataset), total)
    idx_all = sample_indices(total, n, args.seed + t_start + len(dataset), device)
    rows = []
    metric_rows = []
    save_dir = output_dir / dataset / f"t_{t_start}"
    save_dir.mkdir(parents=True, exist_ok=True)

    for offset in range(0, n, args.batch_size):
        idx = idx_all[offset : offset + args.batch_size]
        batch = batch_from_bank(bank, idx, device, latent_mean, latent_std, normalize)
        z0 = batch["z"]
        token_mask = batch["token_mask"]
        eps = torch.randn_like(z0) * token_mask[..., None]
        a = alpha_bar[t_start].view(1, 1, 1)
        z_t = (a.sqrt() * z0 + (1.0 - a).sqrt() * eps) * token_mask[..., None]
        z_recon = ddim_denoise(model, z_t, t_start, alpha_bar, batch, args.denoise_steps)
        metric_rows.append(
            {
                **masked_metrics(z_recon, z0, token_mask),
                "noise_mse": masked_metrics(z_t, z0, token_mask)["latent_mse"],
            }
        )
        z_decode = z_recon * latent_std + latent_mean if normalize else z_recon
        z_decode = z_decode * token_mask[..., None]
        out = decode_latents(vae, z_decode, token_mask)

        start = 0
        num_atoms = token_mask.sum(dim=1).long().tolist()
        for local_i, num_atom in enumerate(num_atoms):
            global_sample_i = offset + local_i
            source_path = bank["raw_path"][int(idx[local_i].item())]
            row = save_decoded_structure(
                dataset,
                out,
                local_i,
                global_sample_i,
                start,
                int(num_atom),
                source_path,
                int(bank["dataset_idx_value"]),
                save_dir,
                args.atom_types,
            )
            row["global_index"] = int(idx[local_i].item())
            row["dataset"] = dataset
            row["t_start"] = t_start
            rows.append(row)
            start += int(num_atom)

    manifest_path = save_dir / "decoded_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "dataset": dataset,
        "t_start": t_start,
        "num_samples": n,
        "decode_ok_rate": sum(1 for r in rows if r["decode_ok"]) / max(len(rows), 1),
        "atom_type_acc": sum(float(r["atom_type_acc"]) for r in rows) / max(len(rows), 1),
        "manifest": str(manifest_path),
    }
    for key in ["latent_mse", "latent_mae", "cosine", "noise_mse"]:
        result[key] = sum(m[key] for m in metric_rows) / max(len(metric_rows), 1)
    print(
        "[decode-eval] "
        f"dataset={dataset} t_start={t_start} samples={n} "
        f"decode_ok={result['decode_ok_rate']:.3f} "
        f"latent_mse={result['latent_mse']:.6f} "
        f"cosine={result['cosine']:.6f} "
        f"atom_acc={result['atom_type_acc']:.4f}",
        flush=True,
    )
    return result


def enrich_banks_with_paths(data_root: Path, banks: dict[str, dict[str, Any]]) -> None:
    for dataset, bank in banks.items():
        raw = torch.load(data_root / dataset / "latent" / "tg_latents.pt", map_location="cpu", weights_only=False)
        bank["raw_path"] = raw["path"]
        bank["dataset_idx_value"] = int(raw["dataset_idx"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dit-ckpt", type=Path, required=True)
    parser.add_argument("--vae-ckpt", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "correction")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_ORDER), choices=list(DATASET_ORDER))
    parser.add_argument("--t-starts", nargs="+", type=int, default=[100, 200, 400])
    parser.add_argument("--samples-per-dataset", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--denoise-steps", type=int, default=50)
    parser.add_argument("--atom-types", choices=["original", "decoded"], default="original")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, cfg, ckpt = load_model(args.dit_ckpt, device)
    vae_ckpt = str(args.vae_ckpt) if args.vae_ckpt else get_vae_ckpt(args.data_root)
    vae = VariationalAutoencoderLitModule.load_from_checkpoint(vae_ckpt, map_location="cpu")
    vae.eval().requires_grad_(False).to(device)

    if args.output_dir is None:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        args.output_dir = Path(cfg["output_dir"]) / f"eval_tg_noisy_denoise_decode_{timestamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] dit_ckpt={args.dit_ckpt}")
    print(f"[INFO] vae_ckpt={vae_ckpt}")
    print(f"[INFO] output_dir={args.output_dir}")
    banks = load_tg_banks(args.data_root, int(cfg["max_elements"]), device=device if device.type == "cuda" else None)
    banks = {k: v for k, v in banks.items() if k in args.datasets}
    enrich_banks_with_paths(args.data_root, banks)

    normalize = bool(cfg.get("normalize_latents", True))
    latent_mean = torch.as_tensor(ckpt.get("latent_mean", torch.tensor([0.0])), device=device).reshape(1)
    latent_std = torch.as_tensor(ckpt.get("latent_std", torch.tensor([1.0])), device=device).reshape(1)
    alpha_bar = torch.cumprod(1.0 - make_beta_schedule(int(cfg["num_timesteps"]), device), dim=0)

    summary = []
    for dataset in args.datasets:
        for t_start in args.t_starts:
            summary.append(
                evaluate_dataset_t(
                    model,
                    vae,
                    banks[dataset],
                    dataset,
                    t_start,
                    args,
                    device,
                    alpha_bar,
                    latent_mean,
                    latent_std,
                    normalize,
                    args.output_dir,
                )
            )

    summary_path = args.output_dir / "decode_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    (args.output_dir / "eval_config.json").write_text(
        json.dumps(
            {
                "dit_ckpt": str(args.dit_ckpt),
                "vae_ckpt": vae_ckpt,
                "data_root": str(args.data_root),
                "datasets": args.datasets,
                "t_starts": args.t_starts,
                "samples_per_dataset": args.samples_per_dataset,
                "batch_size": args.batch_size,
                "denoise_steps": args.denoise_steps,
                "atom_types": args.atom_types,
                "device": args.device,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[DONE] wrote {summary_path}")


if __name__ == "__main__":
    main()
