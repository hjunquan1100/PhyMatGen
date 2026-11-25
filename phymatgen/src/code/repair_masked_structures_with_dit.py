#!/usr/bin/env python
"""Prototype masked-structure repair with optional structure-space RePaint.

 缺失结构修复实验脚本，不是主 ntg 修复流水线。
     它会从完整结构造“残缺结构”，或者读取已有 Canque 残缺样本，然后尝试补全缺失原子。支持三种空间：
      - latent：用 DiT 在 latent 里 inpaint
      - structure：直接在结构坐标空间做 RePaint/优化
      - hybrid：先 latent 修，再结构空间 refine
        还会生成过程可视化图、attempt manifest、xiufu_mask_repair_comparison.csv。

comparison CSV.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import os
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pymatgen.core import Structure
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phymatgen_correction.src.code.build_latent_bank import (  # noqa: E402
    DATASETS,
    Z_TO_SYMBOL,
    VariationalAutoencoderLitModule,
    load_structure_file_to_data,
    split_flat_latents,
)
from phymatgen_correction.src.code.eval_basic_cond_dit import load_model  # noqa: E402
from phymatgen_correction.src.code.eval_tg_noisy_denoise_decode import (  # noqa: E402
    decode_latents,
    get_vae_ckpt,
    save_decoded_structure,
)
from phymatgen_correction.src.code.train_basic_cond_dit import (  # noqa: E402
    composition_to_tensor,
    make_beta_schedule,
)

try:  # noqa: E402
    from torch_geometric.data import Batch
except Exception:  # pragma: no cover - build_latent_bank installs a fallback.
    from torch_geometric.data import Batch


DATASET_BY_PREFIX = {
    "qm9": "qm9",
    "mp20": "mp20",
    "perov5": "perov5",
    "carbon24": "carbon24",
}


def detect_dataset(path: Path) -> str:
    stem = path.stem.lower()
    for prefix, dataset in DATASET_BY_PREFIX.items():
        if stem.startswith(prefix):
            return dataset
    if path.suffix.lower() == ".pdb":
        return "qm9"
    raise ValueError(f"Cannot infer dataset for {path}")


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}


def atom_counts(atom_types: torch.Tensor) -> dict[str, int]:
    counts = Counter(int(v) for v in atom_types.detach().cpu().tolist())
    return {Z_TO_SYMBOL.get(z, f"Z{z}"): n for z, n in sorted(counts.items())}


def choose_missing_indices(num_atoms: int, mask_ratio: float, rng: random.Random) -> list[int]:
    if num_atoms <= 1:
        return []
    missing = max(1, int(round(num_atoms * mask_ratio)))
    missing = min(missing, num_atoms - 1)
    return sorted(rng.sample(range(num_atoms), missing))


def parse_removed_indices(value: str) -> list[int]:
    missing = []
    for part in str(value).replace(";", ",").split(","):
        token = part.strip()
        if not token:
            continue
        head = token.split(":", 1)[0].strip()
        try:
            missing.append(int(head))
        except ValueError:
            continue
    return sorted(set(missing))


def load_existing_canque_manifest(source_dir: Path, canque_dir: Path) -> list[dict[str, Any]]:
    summary = canque_dir / "summary.csv"
    if not summary.exists():
        raise FileNotFoundError(f"Missing Canque summary file: {summary}")
    samples: list[dict[str, Any]] = []
    with summary.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source = source_dir / str(row["source"])
            masked = canque_dir / str(row["canque"])
            if not source.exists():
                raise FileNotFoundError(f"Missing source structure from summary: {source}")
            if not masked.exists():
                raise FileNotFoundError(f"Missing Canque structure from summary: {masked}")
            samples.append(
                {
                    "source": source,
                    "masked": masked,
                    "missing": parse_removed_indices(row.get("removed_indices_elements", "")),
                }
            )
    return samples


def write_masked_pdb(source: Path, output: Path, missing: set[int]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    kept = []
    atom_i = -1
    for line in source.read_text(encoding="utf-8").splitlines():
        record = line[:6].strip()
        if record in {"ATOM", "HETATM"}:
            atom_i += 1
            if atom_i in missing:
                continue
            kept.append(line)
        elif record == "CONECT":
            continue
        elif record == "END":
            continue
        else:
            kept.append(line)
    kept.append("END")
    output.write_text("\n".join(kept) + "\n", encoding="utf-8")


def write_masked_cif(source: Path, output: Path, missing: set[int]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    struct = Structure.from_file(str(source))
    keep = [idx for idx in range(len(struct)) if idx not in missing]
    masked = Structure(
        struct.lattice,
        [struct[idx].specie for idx in keep],
        [struct[idx].frac_coords for idx in keep],
        coords_are_cartesian=False,
    )
    masked.to(str(output), fmt="cif")


def write_masked_structure(source: Path, output: Path, missing_indices: list[int]) -> None:
    missing = set(missing_indices)
    if source.suffix.lower() == ".pdb":
        write_masked_pdb(source, output, missing)
    elif source.suffix.lower() == ".cif":
        write_masked_cif(source, output, missing)
    else:
        raise ValueError(f"Unsupported structure extension: {source.suffix}")


def encode_one(
    vae: VariationalAutoencoderLitModule,
    path: Path,
    dataset: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, Any]:
    data = load_structure_file_to_data(str(path), int(DATASETS[dataset]["dataset_idx"]))
    batch = Batch.from_data_list([data]).to(device)
    with torch.no_grad():
        encoded = vae.encode(batch)
        flat_z = encoded["posterior"].mode()
        z, token_mask = split_flat_latents(flat_z, batch.batch, batch.num_atoms)
    return z, token_mask, data


def make_condition_batch(
    z: torch.Tensor,
    token_mask: torch.Tensor,
    data: Any,
    dataset: str,
    max_elements: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    counts = atom_counts(data.atom_types)
    return {
        "z": z,
        "token_mask": token_mask,
        "dataset_idx": torch.full((1,), int(DATASETS[dataset]["dataset_idx"]), dtype=torch.long, device=device),
        "pbc_flag": torch.full((1,), int(DATASETS[dataset]["pbc_flag"]), dtype=torch.long, device=device),
        "composition": composition_to_tensor([counts], max_elements).to(device),
        "target_pass": torch.ones((1, 9), dtype=torch.float32, device=device),
    }


def decode_latents_with_grad(
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
    return vae.decode(encoded)


def coordinate_guidance_loss(
    vae: VariationalAutoencoderLitModule,
    pred_x0: torch.Tensor,
    token_mask: torch.Tensor,
    target_data: Any,
    dataset: str,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    normalize: bool,
    guidance_mask: torch.Tensor,
) -> torch.Tensor:
    z_decode = pred_x0 * latent_std + latent_mean if normalize else pred_x0
    out = decode_latents_with_grad(vae, z_decode, token_mask)
    node_mask = guidance_mask[token_mask]
    if not torch.any(node_mask):
        return pred_x0.sum() * 0.0

    if dataset == "qm9":
        pred_pos = out["pos"] * 10.0
        target_pos = target_data.pos.to(pred_pos.device)
        pred_center = pred_pos[node_mask].mean(dim=0, keepdim=True)
        target_center = target_pos[node_mask].mean(dim=0, keepdim=True)
        diff = (pred_pos - pred_center) - (target_pos - target_center)
    else:
        pred_frac = out["frac_coords"]
        target_frac = target_data.frac_coords.to(pred_frac.device)
        diff = ((pred_frac - target_frac + 0.5) % 1.0) - 0.5

    return (diff[node_mask].square().sum() / (node_mask.sum() * diff.shape[-1]).clamp(min=1.0))


def final_coordinate_refine(
    vae: VariationalAutoencoderLitModule,
    z_decode: torch.Tensor,
    token_mask: torch.Tensor,
    target_data: Any,
    dataset: str,
    refine_mask: torch.Tensor,
    steps: int,
    lr: float,
) -> torch.Tensor:
    if steps <= 0 or lr <= 0.0:
        return z_decode.detach()

    z_var = z_decode.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([z_var], lr=float(lr))
    zero = torch.zeros(1, device=z_decode.device, dtype=z_decode.dtype)
    one = torch.ones(1, device=z_decode.device, dtype=z_decode.dtype)
    best_z = z_var.detach().clone()
    best_loss = float("inf")

    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = coordinate_guidance_loss(
            vae=vae,
            pred_x0=z_var,
            token_mask=token_mask,
            target_data=target_data,
            dataset=dataset,
            latent_mean=zero,
            latent_std=one,
            normalize=False,
            guidance_mask=refine_mask,
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            z_var.mul_(token_mask[..., None])
            value = float(loss.detach().cpu().item())
            if value < best_loss:
                best_loss = value
                best_z = z_var.detach().clone()

    return best_z


def canonical_symbol(symbol: str) -> str:
    clean = "".join(ch for ch in str(symbol).strip() if ch.isalpha())
    if not clean:
        return "X"
    return clean[0].upper() + clean[1:].lower()


COVALENT_RADII = {
    "H": 0.31,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Na": 1.66,
    "Mg": 1.41,
    "Al": 1.21,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "K": 2.03,
    "Ca": 1.76,
    "Ti": 1.60,
    "Fe": 1.32,
    "Cu": 1.32,
    "Zn": 1.22,
    "Br": 1.20,
    "I": 1.39,
}


def covalent_radius(symbol: str) -> float:
    return COVALENT_RADII.get(canonical_symbol(symbol), 0.90)


def read_structure_state(path: Path, dataset: str) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    if dataset == "qm9" or path.suffix.lower() == ".pdb":
        return qm9_coords(path), [canonical_symbol(sym) for sym in qm9_symbols(path)], None
    struct = Structure.from_file(str(path))
    frac = np.asarray([site.frac_coords for site in struct], dtype=float)
    symbols = [canonical_symbol(site.specie.symbol) for site in struct]
    lattice = np.asarray(struct.lattice.matrix, dtype=float)
    return frac, symbols, lattice


def write_structure_state(source: Path, output: Path, dataset: str, coords: np.ndarray) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if dataset == "qm9" or source.suffix.lower() == ".pdb":
        output.write_bytes(source.read_bytes())
        rewrite_qm9_pdb_coords(output, coords)
        return
    struct = Structure.from_file(str(source))
    repaired = Structure(struct.lattice, struct.species, np.asarray(coords, dtype=float) % 1.0, coords_are_cartesian=False)
    repaired.to(str(output), fmt="cif")


def seed_missing_coordinates(
    reference: np.ndarray,
    symbols: list[str],
    missing: list[int],
    dataset: str,
    rng: np.random.Generator,
) -> np.ndarray:
    coords = np.array(reference, dtype=float, copy=True)
    missing_set = set(missing)
    observed = [idx for idx in range(len(coords)) if idx not in missing_set]
    if not missing:
        return coords
    if not observed:
        coords[missing] = rng.normal(0.0, 1.0, size=(len(missing), 3)) if dataset == "qm9" else rng.random((len(missing), 3))
        return coords

    obs = coords[observed]
    for local_i, atom_i in enumerate(missing):
        anchor = obs[int(rng.integers(0, len(obs)))]
        if dataset == "qm9":
            radius = covalent_radius(symbols[atom_i]) + 0.75
            direction = rng.normal(0.0, 1.0, size=3)
            norm = np.linalg.norm(direction)
            direction = direction / norm if norm > 1e-8 else np.array([1.0, 0.0, 0.0])
            coords[atom_i] = anchor + direction * radius + rng.normal(0.0, 0.15, size=3)
        else:
            coords[atom_i] = (anchor + rng.normal(0.0, 0.08, size=3) + 0.17 * local_i) % 1.0
    if dataset != "qm9":
        coords %= 1.0
    return coords


def expand_masked_observed_to_full(
    reference: np.ndarray,
    masked: np.ndarray | None,
    missing: list[int],
) -> np.ndarray:
    """Build a full-length coordinate array without reading hidden missing coordinates.

    Known atom slots are copied from the defective input in order. Missing slots are
    left as NaN so downstream seeding can initialize them from observed atoms only.
    """
    coords = np.full_like(reference, np.nan, dtype=float)
    missing_set = set(missing)
    observed = [idx for idx in range(len(reference)) if idx not in missing_set]
    if masked is None:
        coords[observed] = reference[observed]
        return coords
    n = min(len(observed), len(masked))
    if n:
        coords[observed[:n]] = masked[:n]
    if n < len(observed):
        coords[observed[n:]] = reference[observed[n:]]
    return coords


def seed_missing_coordinates_from_observed(
    observed_reference: np.ndarray,
    symbols: list[str],
    missing: list[int],
    dataset: str,
    rng: np.random.Generator,
) -> np.ndarray:
    coords = np.array(observed_reference, dtype=float, copy=True)
    n = len(coords)
    missing_set = set(missing)
    observed = [idx for idx in range(n) if idx not in missing_set and not np.any(np.isnan(coords[idx]))]
    if not missing:
        return np.nan_to_num(coords, nan=0.0)
    if not observed:
        coords[missing] = rng.normal(0.0, 1.0, size=(len(missing), 3)) if dataset == "qm9" else rng.random((len(missing), 3))
    else:
        obs = coords[observed]
        for local_i, atom_i in enumerate(missing):
            anchor = obs[int(rng.integers(0, len(obs)))]
            if dataset == "qm9":
                radius = covalent_radius(symbols[atom_i]) + 0.75
                direction = rng.normal(0.0, 1.0, size=3)
                norm = np.linalg.norm(direction)
                direction = direction / norm if norm > 1e-8 else np.array([1.0, 0.0, 0.0])
                coords[atom_i] = anchor + direction * radius + rng.normal(0.0, 0.15, size=3)
            else:
                coords[atom_i] = (anchor + rng.normal(0.0, 0.08, size=3) + 0.17 * local_i) % 1.0
    if dataset != "qm9":
        coords %= 1.0
    return np.nan_to_num(coords, nan=0.0)


def load_seed_coordinates(seed_path: Path | None, dataset: str, expected_n: int) -> np.ndarray | None:
    if seed_path is None or not seed_path.exists() or seed_path.suffix == ".failed":
        return None
    try:
        coords, _symbols, _lattice = read_structure_state(seed_path, dataset)
    except Exception:
        return None
    return coords if len(coords) == expected_n else None


def load_guidance_model(module_spec: str | None, ckpt_path: Path | None, device: torch.device) -> torch.nn.Module | None:
    if not module_spec:
        return None
    if ":" not in module_spec:
        raise ValueError("--structure-guidance-module must look like 'python.module:ClassName'")
    module_name, class_name = module_spec.split(":", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    model = cls()
    if ckpt_path is not None:
        ckpt = torch.load(str(ckpt_path), map_location=device)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state, strict=False)
    model.eval().requires_grad_(False).to(device)
    return model


def learned_guidance_loss(
    guidance_model: torch.nn.Module | None,
    coords: torch.Tensor,
    observed_mask: torch.Tensor,
    missing_mask: torch.Tensor,
    dataset: str,
    lattice: torch.Tensor | None,
    step_fraction: float,
) -> torch.Tensor:
    if guidance_model is None:
        return coords.sum() * 0.0
    dataset_idx = int(DATASETS[dataset]["dataset_idx"])
    model_input = {
        "coords": coords.unsqueeze(0),
        "observed_mask": observed_mask.unsqueeze(0),
        "missing_mask": missing_mask.unsqueeze(0),
        "dataset_idx": torch.tensor([dataset_idx], dtype=torch.long, device=coords.device),
        "t": torch.tensor([float(step_fraction)], dtype=coords.dtype, device=coords.device),
    }
    if lattice is not None:
        model_input["lattice"] = lattice.unsqueeze(0)
    try:
        pred = guidance_model(model_input)
    except TypeError:
        pred = guidance_model(**model_input)
    if isinstance(pred, dict):
        for key in ("loss", "score", "energy"):
            if key in pred:
                value = pred[key]
                break
        else:
            value = next(iter(pred.values()))
    else:
        value = pred
    value = torch.as_tensor(value, dtype=coords.dtype, device=coords.device)
    return value.mean()


def pairwise_cartesian_diffs(coords: torch.Tensor, dataset: str, lattice: torch.Tensor | None) -> torch.Tensor:
    diffs = coords[:, None, :] - coords[None, :, :]
    if dataset != "qm9" and lattice is not None:
        diffs = diffs - torch.round(diffs)
        diffs = diffs @ lattice
    return diffs


def structure_repair_loss(
    coords: torch.Tensor,
    reference: torch.Tensor,
    observed_mask: torch.Tensor,
    missing_mask: torch.Tensor,
    radii: torch.Tensor,
    dataset: str,
    lattice: torch.Tensor | None,
    min_distance_scale: float,
    anchor_radius: float,
    anchor_weight: float,
    spread_weight: float,
    compact_radius: float,
) -> torch.Tensor:
    if not torch.any(missing_mask):
        return coords.sum() * 0.0

    diffs = pairwise_cartesian_diffs(coords, dataset, lattice)
    dist = torch.sqrt((diffs * diffs).sum(dim=-1) + 1e-8)
    n = coords.shape[0]
    tri = torch.triu(torch.ones((n, n), dtype=torch.bool, device=coords.device), diagonal=1)
    mobile_pair = (missing_mask[:, None] | missing_mask[None, :]) & tri
    min_dist = (radii[:, None] + radii[None, :]) * float(min_distance_scale)
    overlap = torch.relu(min_dist - dist)
    overlap_loss = overlap[mobile_pair].square().mean() if torch.any(mobile_pair) else coords.sum() * 0.0

    anchor_loss = coords.sum() * 0.0
    nearest = None
    if torch.any(observed_mask) and anchor_weight > 0.0 and anchor_radius > 0.0:
        miss_obs_diffs = coords[missing_mask][:, None, :] - coords[observed_mask][None, :, :]
        if dataset != "qm9" and lattice is not None:
            miss_obs_diffs = miss_obs_diffs - torch.round(miss_obs_diffs)
            miss_obs_diffs = miss_obs_diffs @ lattice
        nearest = torch.sqrt((miss_obs_diffs * miss_obs_diffs).sum(dim=-1) + 1e-8).min(dim=1).values
        anchor_loss = torch.relu(nearest - float(anchor_radius)).square().mean()

    spread_loss = coords.sum() * 0.0
    if spread_weight > 0.0 and torch.any(missing_mask):
        if nearest is not None:
            spread_loss = torch.relu(float(compact_radius) - nearest).square().mean()
        elif torch.count_nonzero(missing_mask) > 1:
            miss_diffs = pairwise_cartesian_diffs(coords[missing_mask], dataset, lattice)
            miss_dist = torch.sqrt((miss_diffs * miss_diffs).sum(dim=-1) + 1e-8)
            tri_miss = torch.triu(torch.ones_like(miss_dist, dtype=torch.bool), diagonal=1)
            spread_loss = torch.relu(float(compact_radius) - miss_dist[tri_miss]).square().mean()

    known_loss = coords.sum() * 0.0
    if torch.any(observed_mask):
        diff = coords[observed_mask] - reference[observed_mask]
        if dataset != "qm9":
            diff = diff - torch.round(diff)
        known_loss = diff.square().mean()
    return overlap_loss + float(anchor_weight) * anchor_loss + float(spread_weight) * spread_loss + 10.0 * known_loss


def structure_space_repaint(
    source_path: Path,
    masked_path: Path | None,
    output_path: Path,
    dataset: str,
    missing: list[int],
    seed_path: Path | None,
    rng_seed: int,
    steps: int,
    resample_jumps: int,
    jump_length: int,
    noise_scale: float,
    step_size: float,
    guidance_scale: float,
    guidance_anneal: float,
    min_distance_scale: float,
    anchor_radius: float,
    anchor_weight: float,
    spread_weight: float,
    compact_radius: float,
    learned_guidance_model: torch.nn.Module | None,
    learned_guidance_weight: float,
    restore_mode: str,
    snapshot_count: int,
) -> tuple[Path, list[tuple[str, Path]]]:
    reference_np, symbols, lattice_np = read_structure_state(source_path, dataset)
    masked_np = None
    if masked_path is not None and masked_path.exists():
        masked_np, _masked_symbols, _masked_lattice = read_structure_state(masked_path, dataset)
    n = len(reference_np)
    missing = [idx for idx in missing if 0 <= idx < n]
    missing_set = set(missing)
    observed_np = np.asarray([idx not in missing_set for idx in range(n)], dtype=bool)
    missing_np = ~observed_np
    rng = np.random.default_rng(int(rng_seed))
    observed_reference_np = expand_masked_observed_to_full(reference_np, masked_np, missing)

    seed_np = load_seed_coordinates(seed_path, dataset, n)
    coords_np = (
        seed_np.copy()
        if seed_np is not None
        else seed_missing_coordinates_from_observed(observed_reference_np, symbols, missing, dataset, rng)
    )
    coords_np[observed_np] = observed_reference_np[observed_np]
    if restore_mode == "target":
        coords_np[missing_np] = reference_np[missing_np]
    if dataset != "qm9":
        coords_np %= 1.0

    coords = torch.as_tensor(coords_np, dtype=torch.float32)
    reference = torch.as_tensor(observed_reference_np, dtype=torch.float32)
    metric_reference = torch.as_tensor(reference_np, dtype=torch.float32)
    observed_mask = torch.as_tensor(observed_np, dtype=torch.bool)
    missing_mask = torch.as_tensor(missing_np, dtype=torch.bool)
    radii = torch.as_tensor([covalent_radius(sym) for sym in symbols], dtype=torch.float32)
    lattice = torch.as_tensor(lattice_np, dtype=torch.float32) if lattice_np is not None else None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage_paths: list[tuple[str, Path]] = []
    stage_dir = output_path.parent / "structure_stages"
    stage_dir.mkdir(parents=True, exist_ok=True)

    def save_stage(name: str, x: torch.Tensor) -> None:
        stage_path = stage_dir / f"{source_path.stem}_{name}{source_path.suffix}"
        write_structure_state(source_path, stage_path, dataset, x.detach().cpu().numpy())
        stage_paths.append((name, stage_path))

    if snapshot_count > 0:
        save_stage("structure_init", coords)
    snapshot_every = max(1, max(1, int(steps)) // max(1, snapshot_count))

    for step_i in range(max(1, int(steps))):
        repeats = 1
        if jump_length > 0 and (step_i + 1) % int(jump_length) == 0:
            repeats += max(0, int(resample_jumps))
        progress = step_i / max(1, int(steps) - 1)
        sigma = float(noise_scale) * (1.0 - progress)
        lr = float(step_size) * (0.25 + 0.75 * (1.0 - progress))
        guidance = float(guidance_scale) * ((1.0 - progress) ** float(guidance_anneal))
        for repaint_i in range(repeats):
            if sigma > 0.0 and (repaint_i > 0 or step_i == 0):
                noise = torch.randn_like(coords) * sigma
                coords = coords.clone()
                coords[missing_mask] = coords[missing_mask] + noise[missing_mask]
                if dataset != "qm9":
                    coords[missing_mask] = coords[missing_mask] % 1.0
                    coords[observed_mask] = reference[observed_mask]

            coords_var = coords.detach().clone().requires_grad_(True)
            loss = structure_repair_loss(
                coords=coords_var,
                reference=reference,
                observed_mask=observed_mask,
                missing_mask=missing_mask,
                radii=radii,
                dataset=dataset,
                lattice=lattice,
                min_distance_scale=min_distance_scale,
                anchor_radius=anchor_radius,
                anchor_weight=anchor_weight,
                spread_weight=spread_weight,
                compact_radius=compact_radius,
            )
            if learned_guidance_model is not None and learned_guidance_weight > 0.0:
                loss = loss + float(learned_guidance_weight) * learned_guidance_loss(
                    guidance_model=learned_guidance_model,
                    coords=coords_var,
                    observed_mask=observed_mask,
                    missing_mask=missing_mask,
                    dataset=dataset,
                    lattice=lattice,
                    step_fraction=progress,
                )
            grad = torch.autograd.grad(loss, coords_var, retain_graph=False)[0]
            coords = coords_var.detach()
            coords[missing_mask] = coords[missing_mask] - lr * guidance * grad[missing_mask]
            coords[observed_mask] = reference[observed_mask]
            if dataset != "qm9":
                coords[missing_mask] = coords[missing_mask] % 1.0

        if snapshot_count > 0 and (step_i + 1) % snapshot_every == 0 and len(stage_paths) < snapshot_count:
            save_stage(f"structure_repaint_{step_i + 1:03d}", coords)

    coords[observed_mask] = reference[observed_mask]
    if dataset != "qm9":
        coords[missing_mask] = coords[missing_mask] % 1.0
    write_structure_state(source_path, output_path, dataset, coords.detach().cpu().numpy())
    return output_path, stage_paths[:snapshot_count]


def masked_latent_inpaint(
    model,
    vae: VariationalAutoencoderLitModule,
    z0: torch.Tensor,
    token_mask: torch.Tensor,
    observed_mask: torch.Tensor,
    batch: dict[str, torch.Tensor],
    target_data: Any,
    dataset: str,
    alpha_bar: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    normalize: bool,
    t_start: int,
    denoise_steps: int,
    guidance_scale: float,
    coord_guidance_scale: float = 0.0,
    restore_mode: str = "prior",
    snapshot_count: int = 0,
) -> tuple[torch.Tensor, list[tuple[str, torch.Tensor]]]:
    z0_model = (z0 - latent_mean) / latent_std.clamp(min=1e-8) if normalize else z0
    z0_model = z0_model * token_mask[..., None]

    eps_obs = torch.randn_like(z0_model) * token_mask[..., None]
    eps_missing = torch.randn_like(z0_model) * token_mask[..., None]
    obs = observed_mask[..., None].float()
    miss = (token_mask[..., None].float() - obs).clamp(min=0.0, max=1.0)
    restore_target = restore_mode == "target"
    guidance_mask = token_mask[..., None].float() if restore_target else obs
    hard_mask = obs
    free_mask = (token_mask[..., None].float() - hard_mask).clamp(min=0.0, max=1.0)

    a_start = alpha_bar[t_start].view(1, 1, 1)
    z_obs_t = a_start.sqrt() * z0_model + (1.0 - a_start).sqrt() * eps_obs
    z = obs * z_obs_t + miss * eps_missing
    z = z * token_mask[..., None]

    steps = torch.linspace(t_start, 0, min(denoise_steps, t_start) + 1, device=z.device).round().long()
    steps = torch.unique_consecutive(steps, dim=0)
    snapshots: list[tuple[str, torch.Tensor]] = []
    if snapshot_count > 0:
        z_start_decode = z * latent_std + latent_mean if normalize else z
        snapshots.append((f"noise_t{t_start}", z_start_decode.detach().clone()))
        interval = max(1, (len(steps) - 1) // max(1, snapshot_count - 1))
    else:
        interval = len(steps) + 1
    for i in range(len(steps) - 1):
        t_cur = int(steps[i].item())
        t_next = int(steps[i + 1].item())
        z = z.detach().requires_grad_(True)
        t_batch = torch.full((1,), t_cur, dtype=torch.long, device=z.device)
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
        denom = (guidance_mask.sum() * pred_x0.shape[-1]).clamp(min=1.0)
        known_loss = (((pred_x0 - z0_model) * guidance_mask).square().sum() / denom)
        if coord_guidance_scale > 0.0:
            coord_loss = coordinate_guidance_loss(
                vae=vae,
                pred_x0=pred_x0,
                token_mask=token_mask,
                target_data=target_data,
                dataset=dataset,
                latent_mean=latent_mean,
                latent_std=latent_std,
                normalize=normalize,
                guidance_mask=guidance_mask.squeeze(-1).bool(),
            )
            known_loss = known_loss + float(coord_guidance_scale) * coord_loss
        grad = torch.autograd.grad(known_loss, z, retain_graph=False)[0]

        if t_next > 0:
            a_next = alpha_bar[t_next].view(1, 1, 1)
            z_next = a_next.sqrt() * pred_x0 + (1.0 - a_next).sqrt() * eps_pred
            z_obs_next = a_next.sqrt() * z0_model + (1.0 - a_next).sqrt() * eps_obs
        else:
            z_next = pred_x0
            z_obs_next = z0_model

        z_next = z_next - float(guidance_scale) * grad
        z = (hard_mask * z_obs_next + free_mask * z_next) * token_mask[..., None]
        if snapshot_count > 0 and (i + 1) % interval == 0 and t_next > 0:
            z_mid_decode = z.detach() * latent_std + latent_mean if normalize else z.detach()
            snapshots.append((f"denoise_t{t_next}", z_mid_decode.clone()))

    z = z.detach()
    z_decode = z * latent_std + latent_mean if normalize else z
    if snapshot_count > 0:
        snapshots.append(("repair_x0", z_decode.detach().clone()))
        snapshots = snapshots[: snapshot_count + 1]
    return z_decode, snapshots


def decode_and_save(
    vae: VariationalAutoencoderLitModule,
    z_decode: torch.Tensor,
    token_mask: torch.Tensor,
    dataset: str,
    source_path: Path,
    output_dir: Path,
    fix_overlap: bool,
    max_overlap_scale: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = decode_latents(vae, z_decode, token_mask)
    num_atom = int(token_mask.sum().item())
    return save_decoded_structure(
        dataset=dataset,
        out=out,
        local_i=0,
        global_sample_i=0,
        start=0,
        num_atom=num_atom,
        source_path=str(source_path),
        dataset_idx=int(DATASETS[dataset]["dataset_idx"]),
        save_dir=output_dir,
        atom_type_mode="original",
        fix_overlap=fix_overlap,
        max_overlap_scale=max_overlap_scale,
    )


def stage_decode_path(stage_dir: Path, source: Path, stage_name: str) -> Path:
    suffix = source.suffix.lower()
    return stage_dir / f"{source.stem}_{stage_name}{suffix}"


def decode_snapshot_structure(
    vae: VariationalAutoencoderLitModule,
    z_decode: torch.Tensor,
    token_mask: torch.Tensor,
    dataset: str,
    source_path: Path,
    stage_dir: Path,
    stage_name: str,
    fix_overlap: bool,
    max_overlap_scale: float,
) -> Path:
    tmp_dir = stage_dir / "_decoded_tmp"
    row = decode_and_save(
        vae=vae,
        z_decode=z_decode,
        token_mask=token_mask,
        dataset=dataset,
        source_path=source_path,
        output_dir=tmp_dir,
        fix_overlap=fix_overlap,
        max_overlap_scale=max_overlap_scale,
    )
    decoded = Path(str(row["decoded_path"]))
    final = stage_decode_path(stage_dir, source_path, stage_name)
    if decoded.exists() and decoded.suffix != ".failed":
        final.write_bytes(decoded.read_bytes())
        decoded.unlink(missing_ok=True)
        return final
    return decoded


def qm9_coords(path: Path) -> np.ndarray:
    coords = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line[:6].strip() not in {"ATOM", "HETATM"}:
            continue
        coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.asarray(coords, dtype=float)


def qm9_symbols(path: Path) -> list[str]:
    symbols = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line[:6].strip() not in {"ATOM", "HETATM"}:
            continue
        symbol = line[76:78].strip() if len(line) >= 78 else ""
        if not symbol:
            symbol = "".join(ch for ch in line[12:16].strip() if ch.isalpha())
        symbols.append(symbol or "X")
    return symbols


def rewrite_qm9_pdb_coords(path: Path, coords: np.ndarray) -> None:
    lines = []
    atom_i = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line[:6].strip() in {"ATOM", "HETATM"} and atom_i < len(coords):
            xyz = coords[atom_i]
            padded = line.ljust(80)
            line = f"{padded[:30]}{float(xyz[0]):8.3f}{float(xyz[1]):8.3f}{float(xyz[2]):8.3f}{padded[54:]}"
            atom_i += 1
        lines.append(line.rstrip())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rigid_align_to_observed(
    source: Path,
    repaired: Path,
    missing: list[int],
) -> bool:
    """Align a repaired QM9 molecule using only non-masked atoms as anchors."""
    full = qm9_coords(source)
    pred = qm9_coords(repaired)
    n = min(len(full), len(pred))
    if n == 0:
        return False
    missing_set = set(missing)
    observed = [idx for idx in range(n) if idx not in missing_set]
    if not observed:
        return False

    src_obs = full[observed]
    pred_obs = pred[observed]
    src_center = src_obs.mean(axis=0)
    pred_center = pred_obs.mean(axis=0)
    pred_centered = pred_obs - pred_center
    src_centered = src_obs - src_center

    if len(observed) >= 2:
        cov = pred_centered.T @ src_centered
        u, _s, vh = np.linalg.svd(cov)
        rotation = u @ vh
        if np.linalg.det(rotation) < 0:
            vh[-1, :] *= -1
            rotation = u @ vh
        aligned = (pred - pred_center) @ rotation + src_center
    else:
        aligned = pred - pred_center + src_center

    rewrite_qm9_pdb_coords(repaired, aligned)
    return True


def project_observed_atoms_from_source(source: Path, repaired: Path, dataset: str, missing: list[int]) -> bool:
    """Hard-copy non-masked atom coordinates from source into a repaired file."""
    if not repaired.exists() or repaired.suffix == ".failed":
        return False
    missing_set = set(missing)
    try:
        if dataset == "qm9" or repaired.suffix.lower() == ".pdb":
            source_coords = qm9_coords(source)
            repaired_coords = qm9_coords(repaired)
            n = min(len(source_coords), len(repaired_coords))
            if n == 0:
                return False
            out = repaired_coords.copy()
            observed = [idx for idx in range(n) if idx not in missing_set]
            if not observed:
                return False
            out[observed] = source_coords[observed]
            rewrite_qm9_pdb_coords(repaired, out)
            return True

        source_struct = Structure.from_file(str(source))
        repaired_struct = Structure.from_file(str(repaired))
        n = min(len(source_struct), len(repaired_struct))
        if n == 0:
            return False
        frac = np.asarray([repaired_struct[i].frac_coords for i in range(len(repaired_struct))], dtype=float)
        observed = [idx for idx in range(n) if idx not in missing_set]
        if not observed:
            return False
        source_frac = np.asarray([source_struct[i].frac_coords for i in observed], dtype=float)
        frac[observed] = source_frac
        projected = Structure(repaired_struct.lattice, repaired_struct.species, frac % 1.0, coords_are_cartesian=False)
        projected.to(str(repaired), fmt="cif")
        return True
    except Exception:
        return False


def structure_points(path: Path) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    if path.suffix.lower() == ".pdb":
        return qm9_coords(path), qm9_symbols(path), None
    struct = Structure.from_file(str(path))
    coords = np.asarray([site.coords for site in struct], dtype=float)
    symbols = [site.specie.symbol for site in struct]
    return coords, symbols, np.asarray(struct.lattice.matrix, dtype=float)


def set_axes_equal(ax) -> None:
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    ranges = [abs(lim[1] - lim[0]) for lim in (x_limits, y_limits, z_limits)]
    centers = [sum(lim) / 2.0 for lim in (x_limits, y_limits, z_limits)]
    radius = max(ranges) / 2.0 if ranges else 1.0
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def draw_lattice(ax, lattice: np.ndarray) -> None:
    if lattice is None:
        return
    origin = np.zeros(3)
    corners = [
        origin,
        lattice[0],
        lattice[1],
        lattice[2],
        lattice[0] + lattice[1],
        lattice[0] + lattice[2],
        lattice[1] + lattice[2],
        lattice[0] + lattice[1] + lattice[2],
    ]
    edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)]
    for i, j in edges:
        pts = np.vstack([corners[i], corners[j]])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="#9ca3af", linewidth=0.8)


ELEMENT_COLORS = {
    "H": "#7f7f7f",
    "C": "#1f77b4",
    "N": "#2ca02c",
    "O": "#d62728",
    "F": "#9467bd",
    "Si": "#ff7f0e",
    "P": "#8c564b",
    "S": "#bcbd22",
    "Cl": "#17becf",
    "Br": "#e377c2",
    "I": "#7b4173",
}
FALLBACK_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def color_for_symbol(symbol: str) -> str:
    clean = symbol.strip()
    if clean in ELEMENT_COLORS:
        return ELEMENT_COLORS[clean]
    idx = sum(ord(ch) for ch in clean) % len(FALLBACK_COLORS)
    return FALLBACK_COLORS[idx]


def plot_structure(
    path: Path,
    out_png: Path,
    title: str,
    missing_indices: list[int],
    show_missing: bool,
    reference_path: Path | None = None,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    coords, symbols, lattice = structure_points(path)
    ref_coords = None
    ref_symbols: list[str] = []
    if reference_path is not None and reference_path.exists():
        ref_coords, ref_symbols, _ = structure_points(reference_path)
    fig = plt.figure(figsize=(5.2, 4.6), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    missing = set(missing_indices)

    if coords.size:
        colors = [color_for_symbol(sym) for sym in symbols]
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=72, c=colors, edgecolors="black", linewidths=0.4)
        for idx, (xyz, sym) in enumerate(zip(coords, symbols)):
            text_color = "#b91c1c" if idx in missing else color_for_symbol(sym)
            ax.text(float(xyz[0]), float(xyz[1]), float(xyz[2]), f"{idx}:{sym}", fontsize=7, color=text_color)

    if ref_coords is not None and reference_path != path:
        ax.scatter(
            ref_coords[:, 0],
            ref_coords[:, 1],
            ref_coords[:, 2],
            s=110,
            facecolors="none",
            edgecolors="#6b7280",
            linewidths=1.2,
            alpha=0.75,
        )
        for idx, xyz in enumerate(ref_coords):
            sym = ref_symbols[idx] if idx < len(ref_symbols) else "?"
            ax.text(
                float(xyz[0]),
                float(xyz[1]),
                float(xyz[2]),
                f"ref {idx}:{sym}",
                fontsize=6,
                color="#6b7280",
                alpha=0.85,
            )

    if show_missing and ref_coords is not None:
        for idx in missing:
            if idx >= len(ref_coords):
                continue
            xyz = ref_coords[idx]
            ax.scatter([xyz[0]], [xyz[1]], [xyz[2]], s=120, c="#ef4444", marker="x", linewidths=2.0)
            sym = ref_symbols[idx] if idx < len(ref_symbols) else "?"
            ax.text(float(xyz[0]), float(xyz[1]), float(xyz[2]), f"mask {idx}:{sym}", fontsize=7, color="#b91c1c")

    draw_lattice(ax, lattice)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if coords.size or ref_coords is not None:
        all_coords = coords if ref_coords is None else np.vstack([coords, ref_coords])
        ax.set_xlim(all_coords[:, 0].min() - 0.5, all_coords[:, 0].max() + 0.5)
        ax.set_ylim(all_coords[:, 1].min() - 0.5, all_coords[:, 1].max() + 0.5)
        ax.set_zlim(all_coords[:, 2].min() - 0.5, all_coords[:, 2].max() + 0.5)
        set_axes_equal(ax)
    ax.view_init(elev=22, azim=35)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def make_summary_image(stage_pngs: list[Path], output: Path) -> None:
    if not stage_pngs:
        return
    images = [plt.imread(str(path)) for path in stage_pngs if path.exists()]
    if not images:
        return
    height = max(img.shape[0] for img in images)
    padded = []
    for img in images:
        if img.shape[0] < height:
            pad = np.ones((height - img.shape[0], img.shape[1], img.shape[2]), dtype=img.dtype)
            img = np.vstack([img, pad])
        padded.append(img)
    merged = np.hstack(padded)
    plt.imsave(str(output), merged)


def visualize_repair_process(
    source_path: Path,
    masked_path: Path,
    repaired_path: Path,
    stage_paths: list[tuple[str, Path]],
    visual_dir: Path,
    missing_indices: list[int],
) -> str:
    visual_dir.mkdir(parents=True, exist_ok=True)
    pngs: list[Path] = []
    full_png = visual_dir / "00_full.png"
    plot_structure(source_path, full_png, "00 full structure", missing_indices, show_missing=True, reference_path=source_path)
    pngs.append(full_png)
    masked_png = visual_dir / "01_masked.png"
    plot_structure(masked_path, masked_png, "01 masked input", missing_indices, show_missing=True, reference_path=source_path)
    pngs.append(masked_png)
    for i, (name, stage_path) in enumerate(stage_paths, start=2):
        png = visual_dir / f"{i:02d}_{name}.png"
        plot_structure(stage_path, png, f"{i:02d} {name}", missing_indices, show_missing=False, reference_path=source_path)
        pngs.append(png)
    repaired_png = visual_dir / f"{len(pngs):02d}_repaired.png"
    plot_structure(repaired_path, repaired_png, "final repaired", missing_indices, show_missing=False, reference_path=source_path)
    pngs.append(repaired_png)
    summary = visual_dir / "summary_process.png"
    make_summary_image(pngs, summary)
    return str(summary)


def compare_outputs(dataset: str, source: Path, repaired: Path, missing: list[int]) -> dict[str, float]:
    if not repaired.exists() or repaired.suffix == ".failed":
        return {"all_l2": math.nan, "missing_l2": math.nan, "observed_l2": math.nan}
    missing_set = set(missing)
    if dataset == "qm9":
        full = qm9_coords(source)
        pred = qm9_coords(repaired)
        n = min(len(full), len(pred))
        full = full[:n]
        pred = pred[:n]
        diff = pred - full
    else:
        full_s = Structure.from_file(str(source))
        pred_s = Structure.from_file(str(repaired))
        n = min(len(full_s), len(pred_s))
        full = np.asarray([full_s[i].frac_coords for i in range(n)], dtype=float)
        pred = np.asarray([pred_s[i].frac_coords for i in range(n)], dtype=float)
        diff = ((pred - full + 0.5) % 1.0) - 0.5

    if n == 0:
        return {"all_l2": math.nan, "missing_l2": math.nan, "observed_l2": math.nan}
    missing_idx = [idx for idx in range(n) if idx in missing_set]
    observed_idx = [idx for idx in range(n) if idx not in missing_set]
    return {
        "all_l2": float(np.sqrt((diff * diff).sum(axis=1).mean())),
        "missing_l2": float(np.sqrt((diff[missing_idx] * diff[missing_idx]).sum(axis=1).mean())) if missing_idx else math.nan,
        "observed_l2": float(np.sqrt((diff[observed_idx] * diff[observed_idx]).sum(axis=1).mean())) if observed_idx else math.nan,
    }


def fast_validate(dataset: str, repaired: Path) -> dict[str, Any]:
    if not repaired.exists() or repaired.suffix == ".failed":
        return {"fast_pass": False, "fast_notes": "decode_failed"}
    try:
        if dataset == "qm9":
            from phymatgen_correction.src.qm9_mol_utils import evaluate_molecule

            result = evaluate_molecule(repaired)
            return {
                "fast_pass": result.mol_stable and result.rdkit_valid and result.no_atomic_overlap,
                "fast_notes": result.error,
                "mol_stable": result.mol_stable,
                "rdkit_valid": result.rdkit_valid,
                "no_atomic_overlap": result.no_atomic_overlap,
            }
        if dataset == "carbon24":
            from phymatgen_correction.src.validate_carbon24 import _fast_validate_one
        elif dataset == "mp20":
            from phymatgen_correction.src.validate_mp20 import _fast_validate_one
        elif dataset == "perov5":
            from phymatgen_correction.src.validate_perov5 import _fast_validate_one
        else:
            return {"fast_pass": False, "fast_notes": f"unsupported_dataset:{dataset}"}
        row = _fast_validate_one(repaired)
        return {"fast_pass": truthy(row.get("fast_overall_pass")), "fast_notes": row.get("fast_notes", ""), **row}
    except Exception as exc:  # noqa: BLE001
        return {"fast_pass": False, "fast_notes": f"validate_error:{exc!r}"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=PROJECT_ROOT / "data" / "masked" / "source")
    parser.add_argument("--canque-dir", type=Path, default=PROJECT_ROOT / "data" / "masked" / "canque")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "masked_repair")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "correction")
    parser.add_argument("--dit-ckpt", type=Path, required=True)
    parser.add_argument("--vae-ckpt", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--use-existing-canque",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Read pre-built defective structures and missing indices from canque-dir/summary.csv.",
    )
    parser.add_argument("--attempts", type=int, default=6)
    parser.add_argument("--mask-ratio", type=float, default=0.30)
    parser.add_argument(
        "--repair-space",
        choices=["structure", "latent", "hybrid"],
        default="structure",
        help=(
            "structure repairs coordinates/fractional coordinates directly with RePaint-style "
            "resampling; latent keeps the old VAE latent inpainting; hybrid uses latent output "
            "as the structure-space initial guess."
        ),
    )
    parser.add_argument("--t-start", type=int, default=500)
    parser.add_argument("--denoise-steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=0.15)
    parser.add_argument("--coord-guidance-scale", type=float, default=0.5)
    parser.add_argument("--final-refine-steps", type=int, default=0)
    parser.add_argument("--final-refine-lr", type=float, default=0.0)
    parser.add_argument("--structure-steps", type=int, default=120)
    parser.add_argument("--repaint-resample-jumps", type=int, default=2)
    parser.add_argument("--repaint-jump-length", type=int, default=10)
    parser.add_argument("--structure-noise-scale", type=float, default=0.12)
    parser.add_argument("--structure-step-size", type=float, default=0.05)
    parser.add_argument("--structure-guidance-scale", type=float, default=1.0)
    parser.add_argument("--structure-guidance-anneal", type=float, default=0.5)
    parser.add_argument("--structure-min-distance-scale", type=float, default=0.70)
    parser.add_argument("--structure-anchor-radius", type=float, default=2.0)
    parser.add_argument("--structure-anchor-weight", type=float, default=0.10)
    parser.add_argument("--structure-spread-weight", type=float, default=0.05)
    parser.add_argument("--structure-compact-radius", type=float, default=0.75)
    parser.add_argument("--structure-guidance-module", default=None, help="Optional learned guidance model as python.module:ClassName.")
    parser.add_argument("--structure-guidance-ckpt", type=Path, default=None)
    parser.add_argument("--structure-learned-guidance-weight", type=float, default=0.0)
    parser.add_argument("--align-qm9-observed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--restore-mode",
        choices=["prior", "target"],
        default="prior",
        help=(
            "prior uses only observed atoms as guidance; target also anchors masked atoms "
            "to the known full structure for controlled reconstruction experiments."
        ),
    )
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visual-snapshots", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--fix-overlap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-overlap-scale", type=float, default=1.30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.canque_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    uses_latent = args.repair_space in {"latent", "hybrid"}
    model = cfg = vae = None
    guidance_model = load_guidance_model(args.structure_guidance_module, args.structure_guidance_ckpt, device)
    vae_ckpt = ""
    normalize = False
    latent_mean = torch.tensor([0.0], device=device)
    latent_std = torch.tensor([1.0], device=device)
    alpha_bar = torch.empty(0, device=device)
    if uses_latent:
        model, cfg, ckpt = load_model(args.dit_ckpt, device)
        vae_ckpt = str(args.vae_ckpt) if args.vae_ckpt else get_vae_ckpt(args.data_root)
        vae = VariationalAutoencoderLitModule.load_from_checkpoint(vae_ckpt, map_location="cpu")
        vae.eval().requires_grad_(False).to(device)

        normalize = bool(cfg.get("normalize_latents", True))
        latent_mean = torch.as_tensor(ckpt.get("latent_mean", torch.tensor([0.0])), device=device).reshape(1)
        latent_std = torch.as_tensor(ckpt.get("latent_std", torch.tensor([1.0])), device=device).reshape(1)
        alpha_bar = torch.cumprod(1.0 - make_beta_schedule(int(cfg["num_timesteps"]), device), dim=0)

    if args.use_existing_canque:
        samples = load_existing_canque_manifest(args.source_dir, args.canque_dir)[: max(1, int(args.limit))]
    else:
        files = sorted(
            path
            for path in args.source_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".pdb", ".cif"}
        )[: max(1, int(args.limit))]
        samples = [{"source": path, "masked": None, "missing": None} for path in files]
    rows: list[dict[str, Any]] = []
    for sample_i, sample in enumerate(samples):
        path = Path(sample["source"])
        dataset = detect_dataset(path)
        z0 = token_mask = data = batch = None
        if uses_latent:
            assert vae is not None and cfg is not None
            z0, token_mask, data = encode_one(vae, path, dataset, device)
            num_atoms = int(token_mask.sum().item())
        else:
            reference_coords, _symbols, _lattice = read_structure_state(path, dataset)
            num_atoms = int(len(reference_coords))
        rng = random.Random(args.seed + sample_i * 1009)
        if args.use_existing_canque:
            missing = [idx for idx in list(sample["missing"]) if 0 <= int(idx) < num_atoms]
        else:
            missing = choose_missing_indices(num_atoms, args.mask_ratio, rng)
        observed_mask = None
        if uses_latent:
            assert token_mask is not None
            observed_mask = token_mask.clone()
            if missing:
                observed_mask[0, missing] = False

        if args.use_existing_canque:
            masked_path = Path(sample["masked"])
        else:
            masked_path = args.canque_dir / f"{path.stem}_mask{len(missing)}{path.suffix}"
            write_masked_structure(path, masked_path, missing)

        if uses_latent:
            assert z0 is not None and token_mask is not None and data is not None and cfg is not None
            batch = make_condition_batch(z0, token_mask, data, dataset, int(cfg["max_elements"]), device)
        chosen: dict[str, Any] | None = None
        attempt_rows: list[dict[str, Any]] = []
        attempt_count = max(1, int(args.attempts))
        sample_attempt_dir = args.output_dir / "attempts" / path.stem
        sample_attempt_dir.mkdir(parents=True, exist_ok=True)
        for attempt_i in range(attempt_count):
            torch.manual_seed(int(args.seed) + sample_i * 10007 + attempt_i * 7919)
            attempt_out_dir = sample_attempt_dir / f"attempt_{attempt_i:02d}"
            snapshots: list[tuple[str, torch.Tensor]] = []
            structure_stage_paths: list[tuple[str, Path]] = []
            z_repaired = None
            latent_seed_path: Path | None = None

            if args.repair_space in {"latent", "hybrid"}:
                assert model is not None and vae is not None and cfg is not None
                assert z0 is not None and token_mask is not None and observed_mask is not None and batch is not None and data is not None
                z_repaired, snapshots = masked_latent_inpaint(
                    model=model,
                    vae=vae,
                    z0=z0,
                    token_mask=token_mask,
                    observed_mask=observed_mask,
                    batch=batch,
                    target_data=data,
                    dataset=dataset,
                    alpha_bar=alpha_bar,
                    latent_mean=latent_mean,
                    latent_std=latent_std,
                    normalize=normalize,
                    t_start=min(int(args.t_start), int(cfg["num_timesteps"]) - 1),
                    denoise_steps=int(args.denoise_steps),
                    guidance_scale=float(args.guidance_scale),
                    coord_guidance_scale=float(args.coord_guidance_scale),
                    restore_mode=str(args.restore_mode),
                    snapshot_count=int(args.visual_snapshots) if args.visualize and args.repair_space == "latent" else 0,
                )
                if int(args.final_refine_steps) > 0:
                    refine_mask = token_mask if args.restore_mode == "target" else observed_mask
                    z_repaired = final_coordinate_refine(
                        vae=vae,
                        z_decode=z_repaired,
                        token_mask=token_mask,
                        target_data=data,
                        dataset=dataset,
                        refine_mask=refine_mask,
                        steps=int(args.final_refine_steps),
                        lr=float(args.final_refine_lr),
                    )
                    if args.visualize and args.repair_space == "latent":
                        snapshots.append(("coord_refined_x0", z_repaired.detach().clone()))

                save_row = decode_and_save(
                    vae=vae,
                    z_decode=z_repaired,
                    token_mask=token_mask,
                    dataset=dataset,
                    source_path=path,
                    output_dir=attempt_out_dir,
                    fix_overlap=args.fix_overlap,
                    max_overlap_scale=args.max_overlap_scale,
                )
                repaired_path = Path(str(save_row["decoded_path"]))
                latent_seed_path = repaired_path
            else:
                repaired_path = attempt_out_dir / f"{path.stem}_structure_repaired{path.suffix}"

            if args.repair_space in {"structure", "hybrid"}:
                repaired_path, structure_stage_paths = structure_space_repaint(
                    source_path=path,
                    masked_path=masked_path,
                    output_path=attempt_out_dir / f"{path.stem}_structure_repaired{path.suffix}",
                    dataset=dataset,
                    missing=missing,
                    seed_path=latent_seed_path if args.repair_space == "hybrid" else None,
                    rng_seed=int(args.seed) + sample_i * 10007 + attempt_i * 7919,
                    steps=int(args.structure_steps),
                    resample_jumps=int(args.repaint_resample_jumps),
                    jump_length=int(args.repaint_jump_length),
                    noise_scale=float(args.structure_noise_scale),
                    step_size=float(args.structure_step_size),
                    guidance_scale=float(args.structure_guidance_scale),
                    guidance_anneal=float(args.structure_guidance_anneal),
                    min_distance_scale=float(args.structure_min_distance_scale),
                    anchor_radius=float(args.structure_anchor_radius),
                    anchor_weight=float(args.structure_anchor_weight),
                    spread_weight=float(args.structure_spread_weight),
                    compact_radius=float(args.structure_compact_radius),
                    learned_guidance_model=guidance_model,
                    learned_guidance_weight=float(args.structure_learned_guidance_weight),
                    restore_mode=str(args.restore_mode),
                    snapshot_count=int(args.visual_snapshots) if args.visualize else 0,
                )
            aligned_to_observed = False
            if (
                args.align_qm9_observed
                and dataset == "qm9"
                and repaired_path.exists()
                and repaired_path.suffix != ".failed"
            ):
                aligned_to_observed = rigid_align_to_observed(path, repaired_path, missing)
            projected_observed = project_observed_atoms_from_source(path, repaired_path, dataset, missing)
            validation = fast_validate(dataset, repaired_path)
            attempt_info = {
                "attempt_i": attempt_i,
                "z_repaired": z_repaired.detach().clone() if z_repaired is not None else None,
                "snapshots": snapshots,
                "structure_stage_paths": structure_stage_paths,
                "path": repaired_path,
                "validation": validation,
                "aligned_to_observed": aligned_to_observed,
                "projected_observed": projected_observed,
            }
            attempt_rows.append(
                {
                    "attempt_i": attempt_i,
                    "path": str(repaired_path),
                    "fast_pass": validation.get("fast_pass", False),
                    "fast_notes": validation.get("fast_notes", ""),
                    "aligned_to_observed": aligned_to_observed,
                    "projected_observed": projected_observed,
                }
            )
            if validation.get("fast_pass", False):
                chosen = attempt_info
                break
            chosen = attempt_info

        assert chosen is not None
        z_repaired = chosen["z_repaired"]
        snapshots = chosen["snapshots"]
        structure_stage_paths = chosen["structure_stage_paths"]
        repaired_path = Path(str(chosen["path"]))
        validation = chosen["validation"]
        aligned_to_observed = bool(chosen["aligned_to_observed"])
        projected_observed = bool(chosen["projected_observed"])
        final_path = args.output_dir / f"{path.stem}_repaired{path.suffix}"
        if repaired_path.exists() and repaired_path.suffix != ".failed":
            final_path.write_bytes(repaired_path.read_bytes())
            if repaired_path != final_path:
                repaired_path.unlink(missing_ok=True)
            repaired_path = final_path
            projected_observed = project_observed_atoms_from_source(path, repaired_path, dataset, missing) or projected_observed
        attempts_csv = sample_attempt_dir / "attempt_manifest.csv"
        with attempts_csv.open("w", newline="", encoding="utf-8") as attempt_handle:
            attempt_writer = csv.DictWriter(
                attempt_handle,
                fieldnames=["attempt_i", "path", "fast_pass", "fast_notes", "aligned_to_observed", "projected_observed"],
            )
            attempt_writer.writeheader()
            attempt_writer.writerows(attempt_rows)
        process_image = ""
        if args.visualize:
            visual_dir = args.output_dir / "visualizations" / path.stem
            stage_dir = visual_dir / "stages"
            stage_paths: list[tuple[str, Path]] = []
            if args.repair_space == "latent":
                assert vae is not None and token_mask is not None
                for stage_name, stage_z in snapshots:
                    stage_path = decode_snapshot_structure(
                        vae=vae,
                        z_decode=stage_z,
                        token_mask=token_mask,
                        dataset=dataset,
                        source_path=path,
                        stage_dir=stage_dir,
                        stage_name=stage_name,
                        fix_overlap=args.fix_overlap,
                        max_overlap_scale=args.max_overlap_scale,
                    )
                    if stage_path.exists() and stage_path.suffix != ".failed":
                        stage_paths.append((stage_name, stage_path))
            else:
                stage_paths.extend(structure_stage_paths)
            if repaired_path.exists() and repaired_path.suffix != ".failed":
                process_image = visualize_repair_process(
                    source_path=path,
                    masked_path=masked_path,
                    repaired_path=repaired_path,
                    stage_paths=stage_paths,
                    visual_dir=visual_dir,
                    missing_indices=missing,
                )
                compact_process_image = args.output_dir / f"{path.stem}_repair_process.png"
                shutil.copyfile(process_image, compact_process_image)
                process_image = str(compact_process_image)

        compare = compare_outputs(dataset, path, repaired_path, missing)
        row = {
            "sample_i": sample_i,
            "dataset": dataset,
            "source_path": str(path),
            "masked_path": str(masked_path),
            "repaired_path": str(repaired_path),
            "num_atoms": num_atoms,
            "missing_indices": " ".join(str(v) for v in missing),
            "missing_count": len(missing),
            "attempts": attempt_count,
            "selected_attempt": chosen["attempt_i"],
            "attempt_manifest": str(attempts_csv),
            "repair_space": args.repair_space,
            "mask_ratio": args.mask_ratio,
            "t_start": args.t_start,
            "denoise_steps": args.denoise_steps,
            "guidance_scale": args.guidance_scale,
            "coord_guidance_scale": args.coord_guidance_scale,
            "final_refine_steps": args.final_refine_steps,
            "final_refine_lr": args.final_refine_lr,
            "structure_steps": args.structure_steps,
            "repaint_resample_jumps": args.repaint_resample_jumps,
            "repaint_jump_length": args.repaint_jump_length,
            "structure_noise_scale": args.structure_noise_scale,
            "structure_step_size": args.structure_step_size,
            "structure_guidance_scale": args.structure_guidance_scale,
            "structure_guidance_anneal": args.structure_guidance_anneal,
            "structure_min_distance_scale": args.structure_min_distance_scale,
            "structure_anchor_radius": args.structure_anchor_radius,
            "structure_anchor_weight": args.structure_anchor_weight,
            "structure_spread_weight": args.structure_spread_weight,
            "structure_compact_radius": args.structure_compact_radius,
            "structure_guidance_module": args.structure_guidance_module or "",
            "structure_guidance_ckpt": str(args.structure_guidance_ckpt) if args.structure_guidance_ckpt else "",
            "structure_learned_guidance_weight": args.structure_learned_guidance_weight,
            "restore_mode": args.restore_mode,
            "aligned_to_observed": aligned_to_observed,
            "projected_observed": projected_observed,
            **compare,
            "fast_pass": validation.get("fast_pass", False),
            "fast_notes": validation.get("fast_notes", ""),
            "process_image": process_image,
        }
        rows.append(row)
        print(
            f"[xiufu] {path.name} dataset={dataset} missing={len(missing)}/{num_atoms} "
            f"missing_l2={row['missing_l2']:.4f} fast_pass={row['fast_pass']}",
            flush=True,
        )

    manifest = args.output_dir / "xiufu_mask_repair_comparison.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "xiufu_mask_repair_config.txt").write_text(
        "\n".join(
            [
                f"time={time.strftime('%Y-%m-%d_%H-%M-%S')}",
                f"source_dir={args.source_dir}",
                f"canque_dir={args.canque_dir}",
                f"output_dir={args.output_dir}",
                f"dit_ckpt={args.dit_ckpt}",
                f"vae_ckpt={vae_ckpt}",
                f"repair_space={args.repair_space}",
                f"mask_ratio={args.mask_ratio}",
                f"t_start={args.t_start}",
                f"denoise_steps={args.denoise_steps}",
                f"guidance_scale={args.guidance_scale}",
                f"coord_guidance_scale={args.coord_guidance_scale}",
                f"final_refine_steps={args.final_refine_steps}",
                f"final_refine_lr={args.final_refine_lr}",
                f"structure_steps={args.structure_steps}",
                f"repaint_resample_jumps={args.repaint_resample_jumps}",
                f"repaint_jump_length={args.repaint_jump_length}",
                f"structure_noise_scale={args.structure_noise_scale}",
                f"structure_step_size={args.structure_step_size}",
                f"structure_guidance_scale={args.structure_guidance_scale}",
                f"structure_guidance_anneal={args.structure_guidance_anneal}",
                f"structure_min_distance_scale={args.structure_min_distance_scale}",
                f"structure_anchor_radius={args.structure_anchor_radius}",
                f"structure_anchor_weight={args.structure_anchor_weight}",
                f"structure_spread_weight={args.structure_spread_weight}",
                f"structure_compact_radius={args.structure_compact_radius}",
                f"structure_guidance_module={args.structure_guidance_module}",
                f"structure_guidance_ckpt={args.structure_guidance_ckpt}",
                f"structure_learned_guidance_weight={args.structure_learned_guidance_weight}",
                f"align_qm9_observed={args.align_qm9_observed}",
                f"restore_mode={args.restore_mode}",
                f"visualize={args.visualize}",
                f"visual_snapshots={args.visual_snapshots}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] wrote {manifest}")


if __name__ == "__main__":
    main()
