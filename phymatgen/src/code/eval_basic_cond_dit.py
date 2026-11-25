#!/usr/bin/env python
"""Evaluate a trained basic conditional DiT in latent space.

latent 空间评估脚本。
     它不解码结构，只测试 DiT 能不能把加噪后的 tg latent denoise 回原 latent。

The evaluator tests whether the model can recover tg latents after partial
noising. It does not decode structures; this is the first sanity check before
running VAE decode or ntg repair.
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

import torch
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phymatgen_correction.src.code.train_basic_cond_dit import (  # noqa: E402
    BasicConditionalDiT,
    DATASET_ORDER,
    load_tg_banks,
    make_beta_schedule,
)


@torch.no_grad()
def ddim_denoise(
    model: BasicConditionalDiT,
    z_t: torch.Tensor,
    t_start: int,
    alpha_bar: torch.Tensor,
    batch: dict[str, torch.Tensor],
    num_steps: int,
) -> torch.Tensor:
    if t_start <= 0:
        return z_t

    steps = torch.linspace(t_start, 0, min(num_steps, t_start) + 1, device=z_t.device).round().long()
    steps = torch.unique_consecutive(steps, dim=0)
    z = z_t
    token_mask = batch["token_mask"]

    for i in range(len(steps) - 1):
        t_cur = int(steps[i].item())
        t_next = int(steps[i + 1].item())
        t_batch = torch.full((z.shape[0],), t_cur, dtype=torch.long, device=z.device)
        eps_pred = model(
            z,
            t_batch,
            token_mask,
            batch["dataset_idx"],
            batch["pbc_flag"],
            batch["composition"],
            batch["target_pass"],
        )
        a_cur = alpha_bar[t_cur].view(1, 1, 1)
        pred_x0 = (z - (1.0 - a_cur).sqrt() * eps_pred) / a_cur.sqrt().clamp(min=1e-8)
        if t_next > 0:
            a_next = alpha_bar[t_next].view(1, 1, 1)
            z = a_next.sqrt() * pred_x0 + (1.0 - a_next).sqrt() * eps_pred
        else:
            z = pred_x0
        z = z * token_mask[..., None]
    return z


def masked_metrics(pred: torch.Tensor, target: torch.Tensor, token_mask: torch.Tensor) -> dict[str, float]:
    mask = token_mask[..., None].float()
    diff = (pred - target) * mask
    denom = (mask.sum() * pred.shape[-1]).clamp(min=1)
    mse = (diff.square().sum() / denom).item()
    mae = (diff.abs().sum() / denom).item()

    pred_flat = pred[token_mask]
    target_flat = target[token_mask]
    cosine = F.cosine_similarity(pred_flat, target_flat, dim=-1).mean().item()
    return {"latent_mse": mse, "latent_mae": mae, "cosine": cosine}


def load_model(ckpt_path: Path, device: torch.device) -> tuple[BasicConditionalDiT, dict[str, Any], dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = BasicConditionalDiT(
        d_x=8,
        d_model=int(cfg["d_model"]),
        num_layers=int(cfg["num_layers"]),
        nhead=int(cfg["nhead"]),
        mlp_ratio=float(cfg["mlp_ratio"]),
        max_dataset_idx=4,
        max_elements=int(cfg["max_elements"]),
        num_metrics=9,
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval().to(device)
    return model, cfg, ckpt


def pad_to(x: torch.Tensor, target_len: int, value: float | bool = 0) -> torch.Tensor:
    if x.shape[1] == target_len:
        return x
    pad_shape = list(x.shape)
    pad_shape[1] = target_len - x.shape[1]
    pad = torch.full(pad_shape, value, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=1)


def sample_dataset_batch(
    bank: dict[str, Any],
    batch_size: int,
    device: torch.device,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    normalize: bool,
) -> dict[str, torch.Tensor]:
    idx = torch.randint(0, int(bank["num_samples"]), (batch_size,), device=device)
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


def evaluate_dataset(
    model: BasicConditionalDiT,
    bank: dict[str, Any],
    dataset: str,
    args: argparse.Namespace,
    device: torch.device,
    alpha_bar: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    normalize: bool,
) -> list[dict[str, Any]]:
    rows = []
    for t_start in args.t_starts:
        per_trial = []
        for _trial in range(args.num_batches):
            batch = sample_dataset_batch(
                bank,
                args.batch_size,
                device,
                latent_mean,
                latent_std,
                normalize,
            )
            z0 = batch["z"]
            token_mask = batch["token_mask"]
            eps = torch.randn_like(z0) * token_mask[..., None]
            a = alpha_bar[t_start].view(1, 1, 1)
            z_t = (a.sqrt() * z0 + (1.0 - a).sqrt() * eps) * token_mask[..., None]
            z_recon = ddim_denoise(model, z_t, t_start, alpha_bar, batch, args.denoise_steps)
            metrics = masked_metrics(z_recon, z0, token_mask)
            metrics["noise_mse"] = masked_metrics(z_t, z0, token_mask)["latent_mse"]
            per_trial.append(metrics)

        row = {
            "dataset": dataset,
            "t_start": t_start,
            "num_samples": args.num_batches * args.batch_size,
        }
        for key in ["latent_mse", "latent_mae", "cosine", "noise_mse"]:
            row[key] = sum(m[key] for m in per_trial) / len(per_trial)
        rows.append(row)
        print(
            "[eval] "
            f"dataset={dataset} t_start={t_start} samples={row['num_samples']} "
            f"noise_mse={row['noise_mse']:.6f} "
            f"latent_mse={row['latent_mse']:.6f} "
            f"latent_mae={row['latent_mae']:.6f} "
            f"cosine={row['cosine']:.6f}",
            flush=True,
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--data-root", default=PROJECT_ROOT / "data" / "correction", type=Path)
    parser.add_argument("--output-dir", default="", type=Path)
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_ORDER), choices=list(DATASET_ORDER))
    parser.add_argument("--t-starts", nargs="+", type=int, default=[50, 100, 200, 400, 600, 800])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--denoise-steps", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, cfg, ckpt = load_model(args.ckpt, device)
    output_dir = args.output_dir
    if not output_dir:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path(cfg["output_dir"]) / f"eval_latent_denoise_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] ckpt={args.ckpt}")
    print(f"[INFO] output_dir={output_dir}")
    print("[INFO] loading tg latent banks")
    banks = load_tg_banks(args.data_root, int(cfg["max_elements"]), device=device if device.type == "cuda" else None)
    banks = {k: v for k, v in banks.items() if k in args.datasets}

    if bool(cfg.get("normalize_latents", True)):
        latent_mean = torch.as_tensor(ckpt.get("latent_mean", torch.tensor([0.0])), device=device).reshape(1)
        latent_std = torch.as_tensor(ckpt.get("latent_std", torch.tensor([1.0])), device=device).reshape(1)
    else:
        latent_mean = torch.zeros(1, device=device)
        latent_std = torch.ones(1, device=device)
    print(f"[INFO] latent_mean={latent_mean.item():.6f}, latent_std={latent_std.item():.6f}")

    alpha_bar = torch.cumprod(1.0 - make_beta_schedule(int(cfg["num_timesteps"]), device), dim=0)
    all_rows = []
    for dataset in args.datasets:
        all_rows.extend(
            evaluate_dataset(
                model,
                banks[dataset],
                dataset,
                args,
                device,
                alpha_bar,
                latent_mean,
                latent_std,
                bool(cfg.get("normalize_latents", True)),
            )
        )

    csv_path = output_dir / "latent_denoise_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    (output_dir / "eval_config.json").write_text(
        json.dumps(
            {
                "ckpt": str(args.ckpt),
                "data_root": str(args.data_root),
                "datasets": args.datasets,
                "t_starts": args.t_starts,
                "batch_size": args.batch_size,
                "num_batches": args.num_batches,
                "denoise_steps": args.denoise_steps,
                "device": args.device,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[DONE] wrote {csv_path}")
if __name__ == "__main__":
    main()
