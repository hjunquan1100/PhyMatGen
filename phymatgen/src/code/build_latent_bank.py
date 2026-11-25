#!/usr/bin/env python
"""Build padded latent banks for the repair pipeline.

编码通过和不通过测评的文件

The script reads the validation CSVs under data/correction, loads structures from
each dataset's tg/ntg folders, encodes them with a frozen VAE, and writes one
latent bank per dataset split.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phymatgen_correction.src.runtime_env import (
    configure_headless_runtime,
    ensure_conda_lib_on_loader_path,
)

ensure_conda_lib_on_loader_path()
configure_headless_runtime()


def install_minimal_pyg_stubs() -> None:
    """Install the tiny PyG subset needed for VAE encoding.

    The current interactive environment can have an ABI/version mismatch in
    torch_geometric's optional imports. The VAE path only needs Data, Batch, and
    to_dense_batch, so this fallback keeps latent extraction usable.
    """

    class Data:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def to(self, device):
            for key, value in list(self.__dict__.items()):
                if torch.is_tensor(value):
                    setattr(self, key, value.to(device))
            return self

    class Batch(Data):
        @classmethod
        def from_data_list(cls, data_list):
            out = cls()
            node_keys = ["atom_types", "pos", "frac_coords", "token_idx"]
            graph_keys = [
                "cell",
                "lattices",
                "lattices_scaled",
                "lengths",
                "lengths_scaled",
                "angles",
                "angles_radians",
                "num_atoms",
                "num_nodes",
                "dataset_idx",
                "spacegroup",
            ]
            for key in node_keys:
                setattr(out, key, torch.cat([getattr(data, key) for data in data_list], dim=0))
            for key in graph_keys:
                setattr(out, key, torch.cat([getattr(data, key) for data in data_list], dim=0))
            out.batch = torch.cat(
                [
                    torch.full((int(data.num_atoms.item()),), idx, dtype=torch.long)
                    for idx, data in enumerate(data_list)
                ],
                dim=0,
            )
            return out

    def to_dense_batch(x, batch):
        batch_size = int(batch.max().item()) + 1 if batch.numel() else 0
        counts = torch.bincount(batch, minlength=batch_size)
        max_nodes = int(counts.max().item()) if counts.numel() else 0
        dense = x.new_zeros((batch_size, max_nodes, *x.shape[1:]))
        mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool, device=x.device)
        cursor = torch.zeros(batch_size, dtype=torch.long, device=batch.device)
        for node_idx, sample_idx in enumerate(batch.tolist()):
            pos = int(cursor[sample_idx].item())
            dense[sample_idx, pos] = x[node_idx]
            mask[sample_idx, pos] = True
            cursor[sample_idx] += 1
        return dense, mask

    pyg_mod = types.ModuleType("torch_geometric")
    data_mod = types.ModuleType("torch_geometric.data")
    utils_mod = types.ModuleType("torch_geometric.utils")
    data_mod.Data = Data
    data_mod.Batch = Batch
    utils_mod.to_dense_batch = to_dense_batch
    pyg_mod.data = data_mod
    pyg_mod.utils = utils_mod
    sys.modules["torch_geometric"] = pyg_mod
    sys.modules["torch_geometric.data"] = data_mod
    sys.modules["torch_geometric.utils"] = utils_mod


try:
    from torch_geometric.data import Batch
except Exception as exc:  # noqa: BLE001
    print(f"[WARN] Falling back to minimal torch_geometric stubs: {exc}")
    install_minimal_pyg_stubs()
    from torch_geometric.data import Batch


def install_minimal_dynamo_stub() -> None:
    dynamo_mod = types.ModuleType("torch._dynamo")
    dynamo_mod.OptimizedModule = torch.nn.Module
    dynamo_mod.disable = lambda fn=None, *args, **kwargs: fn if fn is not None else (lambda wrapped: wrapped)
    dynamo_mod.is_compiling = lambda: False
    sys.modules["torch._dynamo"] = dynamo_mod


try:
    from torch._dynamo import OptimizedModule as _OptimizedModule  # noqa: F401
except Exception as exc:  # noqa: BLE001
    print(f"[WARN] Falling back to minimal torch._dynamo stub: {exc}")
    install_minimal_dynamo_stub()


def install_minimal_training_stubs() -> None:
    class LightningModule(torch.nn.Module):
        def save_hyperparameters(self, *args, **kwargs):
            return None

        @classmethod
        def load_from_checkpoint(cls, checkpoint_path, map_location=None, **kwargs):
            ckpt = torch.load(checkpoint_path, map_location=map_location or "cpu", weights_only=False)
            hparams = dict(ckpt.get("hyper_parameters", {}))
            hparams.update(kwargs)
            model = cls(**hparams)
            model.load_state_dict(ckpt["state_dict"], strict=True)
            return model

    class MeanMetric(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def update(self, *args, **kwargs):
            return None

        def reset(self):
            return None

        def compute(self):
            return torch.tensor(0.0)

    class WandbLogger:
        pass

    class Callback:
        pass

    class Logger:
        pass

    lightning_mod = types.ModuleType("lightning")
    lightning_mod.LightningModule = LightningModule
    lightning_mod.Callback = Callback
    lightning_pytorch_mod = types.ModuleType("lightning.pytorch")
    lightning_loggers_mod = types.ModuleType("lightning.pytorch.loggers")
    lightning_loggers_mod.Logger = Logger
    lightning_loggers_mod.WandbLogger = WandbLogger
    torchmetrics_mod = types.ModuleType("torchmetrics")
    torchmetrics_mod.MeanMetric = MeanMetric
    wandb_mod = types.ModuleType("wandb")
    wandb_mod.Table = type("Table", (), {"__init__": lambda self, *args, **kwargs: None})
    wandb_mod.Image = type("Image", (), {"__init__": lambda self, *args, **kwargs: None})
    wandb_mod.Molecule = type("Molecule", (), {"__init__": lambda self, *args, **kwargs: None})

    sys.modules["lightning"] = lightning_mod
    sys.modules["lightning.pytorch"] = lightning_pytorch_mod
    sys.modules["lightning.pytorch.loggers"] = lightning_loggers_mod
    sys.modules["torchmetrics"] = torchmetrics_mod
    sys.modules["wandb"] = wandb_mod


install_minimal_training_stubs()

from src.data.components.load_structure_file import load_structure_file_to_data
from src.models.vae_module import VariationalAutoencoderLitModule


GLOBAL_METRICS = [
    "mol_stable",
    "rdkit_valid",
    "no_atomic_overlap",
    "periodic_valid",
    "cn_valid",
    "charge_balanced",
    "bvs_valid",
    "thermo_stable",
    "born_valid",
]

DATASETS = {
    "qm9": {
        "dataset_idx": 1,
        "pbc_flag": 0,
        "csv": "qm9/qm9_first3_results.csv",
        "filename_col": "file",
        "exts": (".pdb", ".xyz"),
        "available": {"mol_stable", "rdkit_valid", "no_atomic_overlap"},
    },
    "mp20": {
        "dataset_idx": 0,
        "pbc_flag": 1,
        "csv": "mp20/validation_combined_results.csv",
        "filename_col": "filename",
        "exts": (".cif",),
        "available": {
            "no_atomic_overlap",
            "periodic_valid",
            "cn_valid",
            "charge_balanced",
            "bvs_valid",
            "thermo_stable",
            "born_valid",
        },
    },
    "perov5": {
        "dataset_idx": 4,
        "pbc_flag": 1,
        "csv": "perov5/validation_combined_results.csv",
        "filename_col": "filename",
        "exts": (".cif",),
        "available": {
            "no_atomic_overlap",
            "periodic_valid",
            "cn_valid",
            "charge_balanced",
            "bvs_valid",
            "thermo_stable",
            "born_valid",
        },
    },
    "carbon24": {
        "dataset_idx": 3,
        "pbc_flag": 1,
        "csv": "carbon24/validation_combined_results.csv",
        "filename_col": "filename",
        "exts": (".cif",),
        "available": {
            "no_atomic_overlap",
            "periodic_valid",
            "cn_valid",
            "thermo_stable",
            "born_valid",
        },
    },
}

Z_TO_SYMBOL = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    19: "K",
    20: "Ca",
    21: "Sc",
    22: "Ti",
    23: "V",
    24: "Cr",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
    29: "Cu",
    30: "Zn",
    31: "Ga",
    32: "Ge",
    33: "As",
    34: "Se",
    35: "Br",
    36: "Kr",
    37: "Rb",
    38: "Sr",
    39: "Y",
    40: "Zr",
    41: "Nb",
    42: "Mo",
    43: "Tc",
    44: "Ru",
    45: "Rh",
    46: "Pd",
    47: "Ag",
    48: "Cd",
    49: "In",
    50: "Sn",
    51: "Sb",
    52: "Te",
    53: "I",
    54: "Xe",
    55: "Cs",
    56: "Ba",
    57: "La",
    58: "Ce",
    59: "Pr",
    60: "Nd",
    61: "Pm",
    62: "Sm",
    63: "Eu",
    64: "Gd",
    65: "Tb",
    66: "Dy",
    67: "Ho",
    68: "Er",
    69: "Tm",
    70: "Yb",
    71: "Lu",
    72: "Hf",
    73: "Ta",
    74: "W",
    75: "Re",
    76: "Os",
    77: "Ir",
    78: "Pt",
    79: "Au",
    80: "Hg",
    81: "Tl",
    82: "Pb",
    83: "Bi",
    84: "Po",
    85: "At",
    86: "Rn",
    87: "Fr",
    88: "Ra",
    89: "Ac",
    90: "Th",
    91: "Pa",
    92: "U",
    93: "Np",
    94: "Pu",
}


def str_to_bool(value: Any) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    if text in {"", "nan", "none", "null"}:
        return None
    return None


def safe_float(value: Any) -> float:
    try:
        if value is None or str(value).strip() == "":
            return math.nan
        return float(value)
    except ValueError:
        return math.nan


def read_csv_by_filename(csv_path: Path, filename_col: str) -> dict[str, dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return {row[filename_col]: row for row in reader}


def metric_vectors(row: dict[str, str], dataset: str) -> tuple[dict[str, bool | None], list[int], list[int], list[int]]:
    info = DATASETS[dataset]
    metrics: dict[str, bool | None] = {}
    available_mask = []
    failure_mask = []
    target_pass = []
    heavy_ran = str_to_bool(row.get("heavy_ran"))

    for metric in GLOBAL_METRICS:
        is_available = metric in info["available"] and metric in row
        if metric in {"thermo_stable", "born_valid"}:
            is_available = is_available and bool(heavy_ran)
        value = str_to_bool(row.get(metric))
        metrics[metric] = value
        available = int(is_available)
        failed = int(is_available and value is False)
        available_mask.append(available)
        failure_mask.append(failed)
        target_pass.append(available)

    return metrics, available_mask, failure_mask, target_pass


def composition_from_atom_types(atom_types: torch.Tensor) -> tuple[str, dict[str, int]]:
    counts = Counter(int(z) for z in atom_types.tolist())
    symbol_counts = {Z_TO_SYMBOL.get(z, f"Z{z}"): n for z, n in sorted(counts.items())}
    formula = "".join(f"{symbol}{count if count > 1 else ''}" for symbol, count in symbol_counts.items())
    return formula, symbol_counts


def split_flat_latents(flat_z: torch.Tensor, batch_index: torch.Tensor, num_atoms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = int(num_atoms.numel())
    max_atoms = int(num_atoms.max().item())
    latent_dim = int(flat_z.shape[-1])
    z = flat_z.new_zeros((batch_size, max_atoms, latent_dim))
    token_mask = torch.zeros((batch_size, max_atoms), dtype=torch.bool, device=flat_z.device)

    for sample_idx in range(batch_size):
        sample_z = flat_z[batch_index == sample_idx]
        n = int(sample_z.shape[0])
        z[sample_idx, :n] = sample_z
        token_mask[sample_idx, :n] = True

    return z, token_mask


def cat_padded_chunks(z_chunks: list[torch.Tensor], mask_chunks: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if not z_chunks:
        return torch.empty((0, 0, 0), dtype=torch.float32), torch.empty((0, 0), dtype=torch.bool)

    max_atoms = max(int(chunk.shape[1]) for chunk in z_chunks)
    latent_dim = int(z_chunks[0].shape[-1])
    padded_z = []
    padded_masks = []

    for z, mask in zip(z_chunks, mask_chunks):
        if int(z.shape[1]) == max_atoms:
            padded_z.append(z)
            padded_masks.append(mask)
            continue

        z_pad = z.new_zeros((z.shape[0], max_atoms, latent_dim))
        mask_pad = torch.zeros((mask.shape[0], max_atoms), dtype=torch.bool)
        z_pad[:, : z.shape[1]] = z
        mask_pad[:, : mask.shape[1]] = mask
        padded_z.append(z_pad)
        padded_masks.append(mask_pad)

    return torch.cat(padded_z, dim=0), torch.cat(padded_masks, dim=0)


def list_split_files(split_dir: Path, exts: tuple[str, ...], limit: int | None) -> list[Path]:
    files = sorted(path for path in split_dir.iterdir() if path.is_file() and path.suffix.lower() in exts)
    if limit is not None:
        return files[:limit]
    return files


def build_records(data_root: Path, dataset: str, split: str, limit: int | None) -> list[tuple[Path, dict[str, str]]]:
    info = DATASETS[dataset]
    csv_rows = read_csv_by_filename(data_root / info["csv"], info["filename_col"])
    split_dir = data_root / dataset / split
    records = []
    missing_rows = 0
    for path in list_split_files(split_dir, info["exts"], limit):
        row = csv_rows.get(path.name)
        if row is None:
            missing_rows += 1
            row = {info["filename_col"]: path.name}
        records.append((path, row))
    if missing_rows:
        print(f"[WARN] {dataset}/{split}: {missing_rows} files had no CSV row; masks will be mostly unavailable.")
    return records


def encode_records(
    model: VariationalAutoencoderLitModule,
    records: list[tuple[Path, dict[str, str]]],
    dataset: str,
    split: str,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    info = DATASETS[dataset]
    z_chunks = []
    mask_chunks = []
    paths = []
    filenames = []
    num_atoms = []
    compositions = []
    composition_counts = []
    metrics_all = []
    available_masks = []
    failure_masks = []
    target_passes = []
    heavy_ran = []
    ehull = []
    notes = []
    failed_files = []

    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        data_list = []
        kept_rows = []
        kept_paths = []
        for path, row in batch_records:
            try:
                data = load_structure_file_to_data(str(path), int(info["dataset_idx"]))
            except Exception as exc:  # noqa: BLE001
                failed_files.append({"path": str(path), "error": repr(exc)})
                continue
            data_list.append(data)
            kept_rows.append(row)
            kept_paths.append(path)

        if not data_list:
            continue

        batch = Batch.from_data_list(data_list).to(device)
        with torch.no_grad():
            encoded = model.encode(batch)
            flat_z = encoded["posterior"].mode()
            z, token_mask = split_flat_latents(flat_z, batch.batch, batch.num_atoms)

        z_chunks.append(z.cpu())
        mask_chunks.append(token_mask.cpu())

        for data, path, row in zip(data_list, kept_paths, kept_rows):
            metrics, available_mask, failure_mask, target_pass = metric_vectors(row, dataset)
            formula, counts = composition_from_atom_types(data.atom_types)
            paths.append(str(path))
            filenames.append(path.name)
            num_atoms.append(int(data.num_atoms.item()))
            compositions.append(formula)
            composition_counts.append(counts)
            metrics_all.append(metrics)
            available_masks.append(available_mask)
            failure_masks.append(failure_mask)
            target_passes.append(target_pass)
            heavy_ran.append(str_to_bool(row.get("heavy_ran")))
            ehull.append(safe_float(row.get("ehull")))
            notes.append(row.get("notes") or row.get("error") or "")

        done = min(start + batch_size, len(records))
        if done == len(records) or done % max(batch_size * 20, 1) == 0:
            print(f"[{dataset}/{split}] encoded {done}/{len(records)}")

    z_all, mask_all = cat_padded_chunks(z_chunks, mask_chunks)

    return {
        "z": z_all,
        "token_mask": mask_all,
        "path": paths,
        "filename": filenames,
        "dataset": dataset,
        "split": split,
        "dataset_idx": int(info["dataset_idx"]),
        "pbc_flag": int(info["pbc_flag"]),
        "num_atoms": torch.tensor(num_atoms, dtype=torch.long),
        "composition": compositions,
        "composition_counts": composition_counts,
        "global_metrics": GLOBAL_METRICS,
        "metrics": metrics_all,
        "available_mask": torch.tensor(available_masks, dtype=torch.bool),
        "failure_mask": torch.tensor(failure_masks, dtype=torch.bool),
        "target_pass": torch.tensor(target_passes, dtype=torch.bool),
        "heavy_ran": heavy_ran,
        "ehull": ehull,
        "notes": notes,
        "failed_files": failed_files,
    }


def save_summary(bank: dict[str, Any], out_path: Path, vae_ckpt: Path) -> None:
    summary = {
        "output": str(out_path),
        "vae_ckpt": str(vae_ckpt),
        "dataset": bank["dataset"],
        "split": bank["split"],
        "num_samples": len(bank["path"]),
        "z_shape": list(bank["z"].shape),
        "token_mask_shape": list(bank["token_mask"].shape),
        "latent_dim": int(bank["z"].shape[-1]) if bank["z"].ndim == 3 else None,
        "max_atoms": int(bank["z"].shape[1]) if bank["z"].ndim == 3 else None,
        "failed_files": len(bank["failed_files"]),
        "global_metrics": bank["global_metrics"],
    }
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "correction",
        help="Root containing dataset CSVs and tg/ntg folders.",
    )
    parser.add_argument("--vae-ckpt", type=Path, required=True, help="Frozen VAE checkpoint.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--splits", nargs="+", default=["tg", "ntg"], choices=["tg", "ntg"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="Optional per dataset/split limit for dry runs.")
    parser.add_argument("--output-suffix", default="", help="Optional suffix before .pt, e.g. dryrun.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing latent banks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"[INFO] Loading VAE: {args.vae_ckpt}")
    model = VariationalAutoencoderLitModule.load_from_checkpoint(str(args.vae_ckpt), map_location="cpu")
    model.eval().requires_grad_(False).to(device)

    for dataset in args.datasets:
        for split in args.splits:
            records = build_records(args.data_root, dataset, split, args.limit)
            out_dir = args.data_root / dataset / "latent"
            out_dir.mkdir(parents=True, exist_ok=True)
            suffix = f".{args.output_suffix}" if args.output_suffix else ""
            out_path = out_dir / f"{split}_latents{suffix}.pt"
            if out_path.exists() and not args.overwrite:
                print(f"[SKIP] {out_path} exists. Pass --overwrite to rebuild.")
                continue
            print(f"[INFO] Encoding {dataset}/{split}: {len(records)} structures")
            bank = encode_records(model, records, dataset, split, args.batch_size, device)
            bank["vae_ckpt"] = str(args.vae_ckpt)
            torch.save(bank, out_path)
            save_summary(bank, out_path, args.vae_ckpt)
            print(f"[DONE] Saved {out_path} with z shape {tuple(bank['z'].shape)}")


if __name__ == "__main__":
    main()
