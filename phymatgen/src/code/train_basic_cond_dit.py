#!/usr/bin/env python
"""Train the first-stage conditional DiT on tg latent banks.


训练第一阶段 conditional DiT，只用 tg_latents.pt 里的好样本。
目标是学会从加噪 latent z_t 预测噪声 eps

This script implements the minimal all-pass latent diffusion stage:

    z_t = sqrt(alpha_bar_t) * z_good + sqrt(1 - alpha_bar_t) * eps
    DiT(z_t, t, dataset_id, pbc_flag, composition, target_pass) -> eps

Only tg_latents.pt files are used. ntg_latents.pt are reserved for later repair
sampling.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "correction"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "correction"


def install_minimal_dynamo_stub() -> None:
    dynamo_mod = types.ModuleType("torch._dynamo")
    dynamo_mod.OptimizedModule = torch.nn.Module
    dynamo_mod.disable = lambda fn=None, *args, **kwargs: fn if fn is not None else (lambda wrapped: wrapped)
    dynamo_mod.graph_break = lambda *args, **kwargs: None
    dynamo_mod.is_compiling = lambda: False
    sys.modules["torch._dynamo"] = dynamo_mod


try:
    from torch._dynamo import OptimizedModule as _OptimizedModule  # noqa: F401
except Exception as exc:  # noqa: BLE001
    print(f"[WARN] Falling back to minimal torch._dynamo stub: {exc}")
    install_minimal_dynamo_stub()

DATASET_ORDER = ("qm9", "mp20", "perov5", "carbon24")
SYMBOL_TO_Z = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
}


@dataclass
class TrainConfig:
    data_root: str
    output_dir: str
    batch_size: int
    max_steps: int
    num_timesteps: int
    lr: float
    weight_decay: float
    d_model: int
    num_layers: int
    nhead: int
    mlp_ratio: float
    log_every: int
    best_window: int
    seed: int
    max_elements: int
    device: str
    normalize_latents: bool


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def token_position_embedding(mask: torch.Tensor, dim: int) -> torch.Tensor:
    idx = torch.cumsum(mask.long(), dim=1) - 1
    idx = idx.clamp(min=0)
    return timestep_embedding(idx.reshape(-1), dim).reshape(mask.shape[0], mask.shape[1], dim)


def make_beta_schedule(num_timesteps: int, device: torch.device) -> torch.Tensor:
    return torch.linspace(1e-4, 2e-2, num_timesteps, device=device)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim: int, frequency_dim: int = 256):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(t, self.frequency_dim))


class ConditionEmbedder(nn.Module):
    def __init__(self, hidden_dim: int, max_dataset_idx: int, max_elements: int, num_metrics: int):
        super().__init__()
        self.dataset_embed = nn.Embedding(max_dataset_idx + 1, hidden_dim)
        self.pbc_embed = nn.Embedding(2, hidden_dim)
        self.composition_mlp = nn.Sequential(
            nn.Linear(max_elements, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.target_mlp = nn.Sequential(
            nn.Linear(num_metrics, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(
        self,
        dataset_idx: torch.Tensor,
        pbc_flag: torch.Tensor,
        composition: torch.Tensor,
        target_pass: torch.Tensor,
    ) -> torch.Tensor:
        c = (
            self.dataset_embed(dataset_idx.long())
            + self.pbc_embed(pbc_flag.long())
            + self.composition_mlp(composition.float())
            + self.target_mlp(target_pass.float())
        )
        return self.out(c)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class DiTBlock(nn.Module):
    def __init__(self, hidden_dim: int, nhead: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_dim, nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
        )
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim))

    def forward(self, x: torch.Tensor, c: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(c).chunk(6, dim=1)
        y = modulate(self.norm1(x), shift_msa, scale_msa)
        attn = self.attn(y, y, y, key_padding_mask=~token_mask, need_weights=False)[0]
        x = x + gate_msa[:, None, :] * attn
        y = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp[:, None, :] * self.mlp(y)
        return x


class BasicConditionalDiT(nn.Module):
    def __init__(
        self,
        d_x: int = 8,
        d_model: int = 384,
        num_layers: int = 8,
        nhead: int = 6,
        mlp_ratio: float = 4.0,
        max_dataset_idx: int = 4,
        max_elements: int = 100,
        num_metrics: int = 9,
    ):
        super().__init__()
        self.d_x = d_x
        self.d_model = d_model
        self.x_embed = nn.Linear(d_x, d_model)
        self.t_embed = TimestepEmbedder(d_model)
        self.c_embed = ConditionEmbedder(d_model, max_dataset_idx, max_elements, num_metrics)
        self.blocks = nn.ModuleList([DiTBlock(d_model, nhead, mlp_ratio) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        self.final = nn.Linear(d_model, d_x)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(init)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN[-1].weight)
            nn.init.zeros_(block.adaLN[-1].bias)
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        token_mask: torch.Tensor,
        dataset_idx: torch.Tensor,
        pbc_flag: torch.Tensor,
        composition: torch.Tensor,
        target_pass: torch.Tensor,
    ) -> torch.Tensor:
        x = self.x_embed(z_t) + token_position_embedding(token_mask, self.d_model).to(z_t.device)
        c = self.t_embed(t) + self.c_embed(dataset_idx, pbc_flag, composition, target_pass)
        for block in self.blocks:
            x = block(x, c, token_mask)
        shift, scale = self.final_adaLN(c).chunk(2, dim=1)
        x = modulate(self.final_norm(x), shift, scale)
        return self.final(x) * token_mask[..., None]


def composition_to_tensor(composition_counts: list[dict[str, int]], max_elements: int) -> torch.Tensor:
    out = torch.zeros((len(composition_counts), max_elements), dtype=torch.float32)
    for row, counts in enumerate(composition_counts):
        total = sum(int(v) for v in counts.values()) or 1
        for symbol, count in counts.items():
            z = SYMBOL_TO_Z.get(symbol)
            if z is not None and 1 <= z <= max_elements:
                out[row, z - 1] = float(count) / float(total)
    return out


def load_tg_banks(data_root: Path, max_elements: int, device: torch.device | None = None) -> dict[str, dict[str, Any]]:
    banks = {}
    for dataset in DATASET_ORDER:
        path = data_root / dataset / "latent" / "tg_latents.pt"
        bank = torch.load(path, map_location="cpu", weights_only=False)
        comp = composition_to_tensor(bank["composition_counts"], max_elements)
        banks[dataset] = {
            "z": bank["z"].float(),
            "token_mask": bank["token_mask"].bool(),
            "dataset_idx": torch.full((bank["z"].shape[0],), int(bank["dataset_idx"]), dtype=torch.long),
            "pbc_flag": torch.full((bank["z"].shape[0],), int(bank["pbc_flag"]), dtype=torch.long),
            "composition": comp,
            "target_pass": bank["target_pass"].float(),
            "num_samples": bank["z"].shape[0],
            "path": str(path),
        }
        if device is not None and device.type == "cuda":
            for key in ["z", "token_mask", "dataset_idx", "pbc_flag", "composition", "target_pass"]:
                banks[dataset][key] = banks[dataset][key].to(device)
    return banks


def compute_latent_stats(banks: dict[str, dict[str, Any]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    total_sum = torch.zeros(1, device=device)
    total_sq = torch.zeros(1, device=device)
    total_count = torch.zeros(1, device=device)
    for bank in banks.values():
        z = bank["z"].to(device)
        mask = bank["token_mask"].to(device)
        vals = z[mask]
        total_sum += vals.sum()
        total_sq += (vals * vals).sum()
        total_count += vals.numel()
    mean = total_sum / total_count.clamp(min=1)
    var = total_sq / total_count.clamp(min=1) - mean * mean
    return mean.detach(), var.clamp(min=1e-8).sqrt().detach()


def pad_to(x: torch.Tensor, target_len: int, value: float | bool = 0) -> torch.Tensor:
    if x.shape[1] == target_len:
        return x
    pad_shape = list(x.shape)
    pad_shape[1] = target_len - x.shape[1]
    pad = torch.full(pad_shape, value, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=1)


def sample_balanced_batch(
    banks: dict[str, dict[str, Any]],
    batch_size: int,
    device: torch.device,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    normalize: bool,
) -> dict[str, torch.Tensor]:
    per_dataset = math.ceil(batch_size / len(DATASET_ORDER))
    chunks: dict[str, list[torch.Tensor]] = {
        "z": [],
        "token_mask": [],
        "dataset_idx": [],
        "pbc_flag": [],
        "composition": [],
        "target_pass": [],
    }
    for dataset in DATASET_ORDER:
        bank = banks[dataset]
        n = min(per_dataset, batch_size - sum(x.shape[0] for x in chunks["z"]))
        if n <= 0:
            break
        idx = torch.randint(0, int(bank["num_samples"]), (n,), device=device)
        for key in chunks:
            chunks[key].append(bank[key].to(device)[idx])

    max_tokens = max(int(chunk.shape[1]) for chunk in chunks["z"])
    z = torch.cat([pad_to(chunk, max_tokens, 0.0) for chunk in chunks["z"]], dim=0)
    token_mask = torch.cat([pad_to(chunk, max_tokens, False) for chunk in chunks["token_mask"]], dim=0)
    if normalize:
        z = (z - latent_mean) / latent_std.clamp(min=1e-8)
        z = z * token_mask[..., None]
    return {
        "z": z,
        "token_mask": token_mask,
        "dataset_idx": torch.cat(chunks["dataset_idx"], dim=0),
        "pbc_flag": torch.cat(chunks["pbc_flag"], dim=0),
        "composition": torch.cat(chunks["composition"], dim=0),
        "target_pass": torch.cat(chunks["target_pass"], dim=0),
    }


def masked_mse(pred: torch.Tensor, target: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    mask = token_mask[..., None].float()
    return ((pred - target) ** 2 * mask).sum() / (mask.sum() * pred.shape[-1]).clamp(min=1)


def save_checkpoint(
    output_dir: Path,
    name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    step: int,
    loss: float,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    best_loss: float | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(cfg),
            "step": step,
            "loss": loss,
            "best_loss": best_loss,
            "latent_mean": latent_mean.cpu(),
            "latent_std": latent_std.cpu(),
        },
        path,
    )


class FileLogger:
    def __init__(self, path: Path):
        self.path = path
        self.handle = path.open("a", encoding="utf-8")

    def log(self, message: str) -> None:
        print(message, flush=True)
        self.handle.write(message + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument("--num-timesteps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--best-window", type=int, default=100, help="Average loss window used to update best.pt.")
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--max-elements", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-normalize-latents", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else DEFAULT_OUTPUT_ROOT / f"basic_cond_dit_{timestamp}"
    )
    cfg = TrainConfig(
        data_root=args.data_root,
        output_dir=str(output_dir),
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        num_timesteps=args.num_timesteps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        d_model=args.d_model,
        num_layers=args.num_layers,
        nhead=args.nhead,
        mlp_ratio=args.mlp_ratio,
        log_every=args.log_every,
        best_window=args.best_window,
        seed=args.seed,
        max_elements=args.max_elements,
        device=args.device,
        normalize_latents=not args.no_normalize_latents,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    logger = FileLogger(output_dir / "train.log")

    logger.log(f"[INFO] output_dir={output_dir}")
    logger.log("[INFO] Loading tg latent banks")
    banks = load_tg_banks(Path(args.data_root), args.max_elements, device=device if device.type == "cuda" else None)
    for dataset, bank in banks.items():
        logger.log(f"[INFO] {dataset}: samples={bank['num_samples']}, z={tuple(bank['z'].shape)}, path={bank['path']}")

    if cfg.normalize_latents:
        logger.log("[INFO] Computing global latent mean/std over tg valid tokens")
        latent_mean, latent_std = compute_latent_stats(banks, device)
    else:
        latent_mean = torch.zeros(1, device=device)
        latent_std = torch.ones(1, device=device)
    logger.log(f"[INFO] latent_mean={latent_mean.item():.6f}, latent_std={latent_std.item():.6f}")

    sample_bank = next(iter(banks.values()))
    num_metrics = int(sample_bank["target_pass"].shape[-1])
    model = BasicConditionalDiT(
        d_x=8,
        d_model=args.d_model,
        num_layers=args.num_layers,
        nhead=args.nhead,
        mlp_ratio=args.mlp_ratio,
        max_dataset_idx=4,
        max_elements=args.max_elements,
        num_metrics=num_metrics,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    betas = make_beta_schedule(args.num_timesteps, device)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)

    model.train()
    running = 0.0
    best_running = 0.0
    best_count = 0
    best_loss = float("inf")
    best_path: Path | None = None
    start_time = time.time()
    last_log_time = start_time
    for step in range(1, args.max_steps + 1):
        batch = sample_balanced_batch(
            banks,
            args.batch_size,
            device,
            latent_mean,
            latent_std,
            cfg.normalize_latents,
        )
        z = batch["z"]
        token_mask = batch["token_mask"]
        t = torch.randint(0, args.num_timesteps, (z.shape[0],), device=device)
        eps = torch.randn_like(z) * token_mask[..., None]
        a = alpha_bar[t].view(-1, 1, 1)
        z_t = a.sqrt() * z + (1.0 - a).sqrt() * eps
        z_t = z_t * token_mask[..., None]

        eps_pred = model(
            z_t,
            t,
            token_mask,
            batch["dataset_idx"],
            batch["pbc_flag"],
            batch["composition"],
            batch["target_pass"],
        )
        loss = masked_mse(eps_pred, eps, token_mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_value = float(loss.detach().item())
        running += loss_value
        best_running += loss_value
        best_count += 1
        if step == 1 or step % args.log_every == 0:
            now = time.time()
            elapsed = now - start_time
            interval = now - last_log_time
            avg = running / (args.log_every if step % args.log_every == 0 else 1)
            running = 0.0
            last_log_time = now
            steps_per_sec = args.log_every / interval if step != 1 and interval > 0 else 0.0
            lr = optimizer.param_groups[0]["lr"]
            logger.log(
                "[train] "
                f"step={step}/{args.max_steps} "
                f"loss={loss_value:.6f} "
                f"avg_loss={avg:.6f} "
                f"best_loss={best_loss if best_loss < float('inf') else float('nan'):.6f} "
                f"lr={lr:.2e} "
                f"elapsed_min={elapsed / 60:.1f} "
                f"steps_per_sec={steps_per_sec:.3f}",
            )

        if best_count >= args.best_window:
            window_loss = best_running / best_count
            best_running = 0.0
            best_count = 0
            if window_loss < best_loss:
                best_loss = window_loss
                if best_path is not None and best_path.exists():
                    best_path.unlink()
                best_path = output_dir / f"best_loss_{best_loss:.6f}_step_{step}.pt"
                save_checkpoint(
                    output_dir,
                    best_path.name,
                    model,
                    optimizer,
                    cfg,
                    step,
                    loss_value,
                    latent_mean,
                    latent_std,
                    best_loss,
                )
                logger.log(f"[BEST] step={step} best_loss={best_loss:.6f} saved={best_path}")

    logger.log(f"[DONE] training finished. best checkpoint: {best_path}")
    logger.close()


if __name__ == "__main__":
    main()
