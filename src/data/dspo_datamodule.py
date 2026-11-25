"""DSPO 训练用 DataModule：从 pt 组文件加载，按高分阈值过滤。
若 pt 含 latents，直接产出 (latent_batch_dict, scores, log_pi_tau) 供 LDM 用潜变量训练；
否则从 paths 读结构产出 (batch_Data, scores, log_pi_tau)。
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data

from src.data.components.load_structure_file import load_structure_file_to_data
from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

# 类别 -> dataset_idx（与 LDM 一致）
CATEGORY_TO_IDX = {"mp20": 0, "qm9": 1, "carbon24": 3, "perov5": 4}

# 默认高分阈值：score >= 此值算高分
DEFAULT_HIGH_THRESHOLDS = {"mp20": 7, "perov5": 7, "carbon24": 4, "qm9": 1}
DEFAULT_MIN_HIGH_COUNTS = {"mp20": 2, "perov5": 2, "carbon24": 2, "qm9": 3}

# DSPO LDM 使用的潜变量/轨迹 token 维度，与 config diffusion_module.denoiser.d_x 一致；加载时若超出则截断
EXPECTED_LATENT_DIM = 8


class DSPOGroupsDataset(Dataset):
    """从 pt 目录读取组文件，只保留「组内既有高分又有低分」的组。"""

    def __init__(
        self,
        pt_root: str,
        high_thresholds: Optional[Dict[str, int]] = None,
        min_high_counts: Optional[Dict[str, int]] = None,
        structures_dir: Optional[str] = None,
    ):
        self.pt_root = Path(pt_root)
        self.high_thresholds = high_thresholds or DEFAULT_HIGH_THRESHOLDS
        self.min_high_counts = min_high_counts or DEFAULT_MIN_HIGH_COUNTS
        self.structures_dir = Path(structures_dir) if structures_dir else self.pt_root / "structures"

        self.groups: List[Dict] = []  # 每项: {category, paths, scores, log_pi_tau}
        self._load_groups()

    def _load_groups(self) -> None:
        pt_files = sorted(self.pt_root.glob("*.pt"))
        for pt_file in pt_files:
            d = torch.load(pt_file, map_location="cpu", weights_only=False)
            cat = d["category"]
            paths = d["paths"]
            scores = d["scores"]
            log_pi_tau = d["log_pi_tau"]
            if isinstance(log_pi_tau, torch.Tensor):
                log_pi_tau = log_pi_tau.numpy()

            thresh = self.high_thresholds.get(cat)
            if thresh is None:
                continue
            min_high = self.min_high_counts.get(cat, 1)
            high = [1 if s >= thresh else 0 for s in scores]
            n_high = sum(high)
            n_low = len(scores) - n_high
            if n_high < min_high or n_low <= 0:
                continue

            # 路径可能是绝对路径或相对 pt_root；统一为可解析路径
            resolved = []
            for p in paths:
                p = Path(p)
                if not p.is_absolute():
                    p = self.pt_root.parent / p
                if not p.exists() and self.structures_dir:
                    name = p.name
                    alt = self.structures_dir / name
                    p = alt if alt.exists() else p
                resolved.append(str(p))
            entry = {
                "category": cat,
                "paths": resolved,
                "scores": scores,
                "log_pi_tau": log_pi_tau,
            }
            if "latents" in d and len(d["latents"]) == len(paths):
                entry["latents"] = d["latents"]
                entry["dataset_idx"] = int(d.get("dataset_idx", CATEGORY_TO_IDX.get(cat, 0)))
                entry["spacegroups"] = list(d.get("spacegroups", [0] * len(paths)))
            if "trajectory_tokens" in d and "trajectory_clean" in d and len(d["trajectory_tokens"]) == len(paths):
                entry["trajectory_tokens"] = d["trajectory_tokens"]
                entry["trajectory_clean"] = d["trajectory_clean"]
                # 与 latents 分支一致：必须带 dataset_idx/spacegroups，否则 conditioning 与 batch 维不一致
                entry["dataset_idx"] = int(d.get("dataset_idx", CATEGORY_TO_IDX.get(cat, 0)))
                entry["spacegroups"] = list(d.get("spacegroups", [0] * len(paths)))
            self.groups.append(entry)
        log.info(
            f"DSPOGroupsDataset: loaded {len(self.groups)} groups from {len(pt_files)} .pt files under {self.pt_root!r}."
        )

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> Tuple[Union[Data, Dict], torch.Tensor, torch.Tensor]:
        g = self.groups[idx]
        paths = g["paths"]
        scores = np.array(g["scores"], dtype=np.float64)
        log_pi_tau = np.array(g["log_pi_tau"], dtype=np.float64)

        # 优先：完整去噪轨迹（供完整轨迹 DSPO）
        if "trajectory_tokens" in g and "trajectory_clean" in g:
            tt = g["trajectory_tokens"]
            tc = g["trajectory_clean"]
            n = len(tt)
            scores = scores[:n]
            log_pi_tau = log_pi_tau[:n]
            spacegroups = g.get("spacegroups", [0] * n)
            dataset_idx_raw = g.get("dataset_idx", 0)
            # 兼容两种 pt 格式：① 每条轨迹为 list of T 个 (N_i, d) ② 每条轨迹为单张 (N_i, T*d)
            first_tok = tt[0]
            if isinstance(first_tok, torch.Tensor) and first_tok.dim() == 2 and first_tok.shape[-1] % EXPECTED_LATENT_DIM == 0:
                d_raw = first_tok.shape[-1]
                T_steps = d_raw // EXPECTED_LATENT_DIM
                log.warning(
                    "trajectory_tokens 为 (N, %s) 单张格式，按步拆成 %s 个 (N, %s) 以与 DSPO LDM 一致。建议用 run_groups 重新生成 pt。",
                    d_raw, T_steps, EXPECTED_LATENT_DIM,
                )
                tt = [list(tt[i].reshape(tt[i].shape[0], T_steps, EXPECTED_LATENT_DIM).unbind(1)) for i in range(n)]
                tc = [list(tc[i].reshape(tc[i].shape[0], T_steps, EXPECTED_LATENT_DIM).unbind(1)) for i in range(n)]
                d_raw = EXPECTED_LATENT_DIM
            else:
                d_raw = tt[0][0].shape[-1]
            T_tokens = len(tt[0])
            T_clean = len(tc[0])
            if d_raw > EXPECTED_LATENT_DIM:
                log.warning(
                    "trajectory_tokens 最后一维 %s > %s，截断为 %s 以与 DSPO LDM 一致。建议用 run_groups 重新生成 pt。",
                    d_raw, EXPECTED_LATENT_DIM, EXPECTED_LATENT_DIM,
                )
                tt = [[t[..., :EXPECTED_LATENT_DIM].clone() for t in sample_traj] for sample_traj in tt]
                tc = [[t[..., :EXPECTED_LATENT_DIM].clone() for t in sample_traj] for sample_traj in tc]
            d = EXPECTED_LATENT_DIM if d_raw > EXPECTED_LATENT_DIM else d_raw
            N_max = max(tt[i][0].shape[0] for i in range(n))
            device = tt[0][0].device
            dtype = tt[0][0].dtype
            trajectory_tokens_padded = []
            for t in range(T_tokens):
                pad_t = torch.zeros(n, N_max, d, dtype=dtype, device=device)
                for i in range(n):
                    pad_t[i, : tt[i][t].shape[0]] = tt[i][t]
                trajectory_tokens_padded.append(pad_t)
            trajectory_clean_padded = []
            for t in range(T_clean):
                pad_t = torch.zeros(n, N_max, d, dtype=dtype, device=device)
                for i in range(n):
                    pad_t[i, : tc[i][t].shape[0]] = tc[i][t]
                trajectory_clean_padded.append(pad_t)
            token_mask = torch.zeros(n, N_max, dtype=torch.bool, device=device)
            for i in range(n):
                token_mask[i, : tt[i][0].shape[0]] = True
            dataset_idx_t = torch.full((n,), dataset_idx_raw + 1, dtype=torch.long, device=device)
            spacegroup_t = torch.tensor(spacegroups[:n], dtype=torch.long, device=device)
            full_traj_batch = {
                "use_full_trajectory": True,
                "trajectory_tokens": trajectory_tokens_padded,
                "trajectory_clean": trajectory_clean_padded,
                "token_mask": token_mask,
                "dataset_idx": dataset_idx_t,
                "spacegroup": spacegroup_t,
                "num_steps": T_clean,
            }
            return full_traj_batch, torch.tensor(scores, dtype=torch.float64), torch.tensor(log_pi_tau, dtype=torch.float64)

        if "latents" in g:
            latents = g["latents"]
            spacegroups = g.get("spacegroups", [0] * len(latents))
            dataset_idx_raw = g.get("dataset_idx", 0)
            n = len(latents)
            scores = scores[:n]
            log_pi_tau = log_pi_tau[:n]
            max_len = max(lt.shape[0] for lt in latents)
            d_raw = latents[0].shape[-1]
            if d_raw > EXPECTED_LATENT_DIM:
                log.warning(
                    "latents 最后一维 %s > %s，截断为 %s 以与 DSPO LDM 一致。建议用 run_groups 重新生成 pt。",
                    d_raw, EXPECTED_LATENT_DIM, EXPECTED_LATENT_DIM,
                )
                latents = [lt[..., :EXPECTED_LATENT_DIM].clone() for lt in latents]
            d = EXPECTED_LATENT_DIM if d_raw > EXPECTED_LATENT_DIM else d_raw
            device = latents[0].device
            x_1 = torch.zeros(n, max_len, d, dtype=latents[0].dtype, device=device)
            token_mask = torch.zeros(n, max_len, dtype=torch.bool, device=device)
            for i, lt in enumerate(latents):
                x_1[i, : lt.shape[0]] = lt
                token_mask[i, : lt.shape[0]] = True
            dataset_idx_t = torch.full((n,), dataset_idx_raw + 1, dtype=torch.long, device=device)
            spacegroup_t = torch.tensor(spacegroups[:n], dtype=torch.long, device=device)
            latent_batch = {
                "x_1": x_1,
                "token_mask": token_mask,
                "dataset_idx": dataset_idx_t,
                "spacegroup": spacegroup_t,
                "use_latent": True,
            }
            return latent_batch, torch.tensor(scores, dtype=torch.float64), torch.tensor(log_pi_tau, dtype=torch.float64)
        cat = g["category"]
        dataset_idx = CATEGORY_TO_IDX[cat]
        data_list = []
        scores_list = []
        log_pi_list = []
        for i, p in enumerate(paths):
            try:
                data_list.append(load_structure_file_to_data(p, dataset_idx))
                scores_list.append(scores[i])
                log_pi_list.append(log_pi_tau[i] if i < len(log_pi_tau) else 0.0)
            except Exception:
                continue
        if len(data_list) == 0:
            return self.__getitem__((idx + 1) % len(self.groups))
        batch_data = Batch.from_data_list(data_list)
        scores_t = torch.tensor(scores_list, dtype=torch.float64)
        log_pi_t = torch.tensor(log_pi_list, dtype=torch.float64)
        return batch_data, scores_t, log_pi_t


def _merge_latent_batches(
    items: List[Tuple[Dict, torch.Tensor, torch.Tensor]],
) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
    """把多组 use_latent 的 (batch_dict, scores, log_pi_tau) 沿 batch 维合并为一组。"""
    x_1_list, mask_list, di_list, sg_list, scores_list, log_pi_list = [], [], [], [], [], []
    n_max = 0
    d = None
    dtype = None
    device = None
    for bd, sc, lp in items:
        x_1_list.append(bd["x_1"])
        mask_list.append(bd["token_mask"])
        di_list.append(bd["dataset_idx"])
        sg_list.append(bd["spacegroup"])
        scores_list.append(sc)
        log_pi_list.append(lp)
        g, n, d = bd["x_1"].shape
        n_max = max(n_max, n)
        if dtype is None:
            dtype = bd["x_1"].dtype
            device = bd["x_1"].device
    G_total = sum(x.shape[0] for x in x_1_list)
    x_1 = torch.zeros(G_total, n_max, d, dtype=dtype, device=device)
    token_mask = torch.zeros(G_total, n_max, dtype=torch.bool, device=device)
    off = 0
    for i, x in enumerate(x_1_list):
        g, n, _ = x.shape
        x_1[off : off + g, :n] = x
        token_mask[off : off + g, :n] = mask_list[i]
        off += g
    merged = {
        "x_1": x_1,
        "token_mask": token_mask,
        "dataset_idx": torch.cat(di_list, dim=0),
        "spacegroup": torch.cat(sg_list, dim=0),
        "use_latent": True,
    }
    return merged, torch.cat(scores_list, dim=0), torch.cat(log_pi_list, dim=0)


def _merge_full_trajectory_batches(
    items: List[Tuple[Dict, torch.Tensor, torch.Tensor]],
) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
    """把多组 use_full_trajectory 的 (batch_dict, scores, log_pi_tau) 沿 batch 维合并。"""
    T = None
    for item in items:
        t = item[0]["num_steps"]
        t = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
        T = t if T is None else min(T, t)
    if T is None or T <= 0:
        return items[0]
    n_max_global = 0
    G_totals = []
    for bd, sc, _ in items:
        tok0 = bd["trajectory_tokens"][0]
        g, n, _ = tok0.shape
        n_max_global = max(n_max_global, n)
        G_totals.append(g)
    G_total = sum(G_totals)
    d = items[0][0]["trajectory_tokens"][0].shape[-1]
    dtype = items[0][0]["trajectory_tokens"][0].dtype
    device = items[0][0]["trajectory_tokens"][0].device

    trajectory_tokens = []
    trajectory_clean = []
    for t in range(T):
        tok_t = torch.zeros(G_total, n_max_global, d, dtype=dtype, device=device)
        clean_t = torch.zeros(G_total, n_max_global, d, dtype=dtype, device=device)
        off = 0
        for (bd, sc, _), g in zip(items, G_totals):
            tok_t[off : off + g, : bd["trajectory_tokens"][t].shape[1]] = bd["trajectory_tokens"][t]
            clean_t[off : off + g, : bd["trajectory_clean"][t].shape[1]] = bd["trajectory_clean"][t]
            off += g
        trajectory_tokens.append(tok_t)
        trajectory_clean.append(clean_t)

    token_mask = torch.zeros(G_total, n_max_global, dtype=torch.bool, device=device)
    off = 0
    for (bd, _, __), g in zip(items, G_totals):
        n = bd["token_mask"].shape[1]
        token_mask[off : off + g, :n] = bd["token_mask"]
        off += g

    merged = {
        "use_full_trajectory": True,
        "trajectory_tokens": trajectory_tokens,
        "trajectory_clean": trajectory_clean,
        "token_mask": token_mask,
        "dataset_idx": torch.cat([item[0]["dataset_idx"] for item in items], dim=0),
        "spacegroup": torch.cat([item[0]["spacegroup"] for item in items], dim=0),
        "num_steps": T,
    }
    scores = torch.cat([item[1] for item in items], dim=0)
    log_pi = torch.cat([item[2] for item in items], dim=0)
    return merged, scores, log_pi


def collate_dspo(batch: List[Tuple[Data, torch.Tensor, torch.Tensor]]) -> Tuple[Union[Data, Dict], torch.Tensor, torch.Tensor]:
    """每个 batch 元素是 (batch_Data, scores, log_pi_tau)。len(batch)==1 时返回该组；>1 时沿 batch 维合并多组（仅支持 use_latent / use_full_trajectory）。"""
    if len(batch) == 1:
        return batch[0]
    first = batch[0][0]
    if isinstance(first, dict):
        if first.get("use_latent"):
            return _merge_latent_batches(batch)
        if first.get("use_full_trajectory"):
            return _merge_full_trajectory_batches(batch)
    # 多组且为 PyG Data 时只取第一组（合并需 Batch.from_data_list + 合并 scores，暂不实现）
    return batch[0]


class DSPODataModule(LightningDataModule):
    """仅 DSPO 训练集；验证需配合 joint 使用。"""

    def __init__(
        self,
        pt_root: str,
        high_thresholds: Optional[Dict[str, int]] = None,
        min_high_counts: Optional[Dict[str, int]] = None,
        batch_size: int = 1,
        num_workers: int = 0,
    ):
        super().__init__()
        self.pt_root = pt_root
        self.high_thresholds = high_thresholds or DEFAULT_HIGH_THRESHOLDS
        self.min_high_counts = min_high_counts or DEFAULT_MIN_HIGH_COUNTS
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        structures_dir = os.path.join(self.pt_root, "structures")
        self.dspo_dataset = DSPOGroupsDataset(
            pt_root=os.path.join(self.pt_root, "pt"),
            high_thresholds=self.high_thresholds,
            min_high_counts=self.min_high_counts,
            structures_dir=structures_dir,
        )

    def train_dataloader(self) -> DataLoader:
        n = len(self.dspo_dataset)
        if n == 0:
            raise ValueError(
                "DSPO train dataset is empty (num_samples=0). "
                "Check that pt_root/pt/ has .pt files and each group has both high- and low-score samples "
                f"(pt_root={getattr(self, 'pt_root', '?')}, high_thresholds={getattr(self, 'high_thresholds', '?')})."
            )
        return DataLoader(
            self.dspo_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_dspo,
        )


class DSPOJointDataModule(LightningDataModule):
    """DSPO 训练 + Joint 验证：train_dataloader 来自 pt 组，val_dataloader 来自 joint 配置。"""

    def __init__(
        self,
        pt_root: str,
        high_thresholds: Optional[Dict[str, int]] = None,
        min_high_counts: Optional[Dict[str, int]] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        joint_datamodule: Any = None,
    ):
        super().__init__()
        from omegaconf import DictConfig
        self.pt_root = pt_root
        self.high_thresholds = high_thresholds or DEFAULT_HIGH_THRESHOLDS
        self.min_high_counts = min_high_counts or DEFAULT_MIN_HIGH_COUNTS
        self.batch_size = batch_size
        self.num_workers = num_workers
        if isinstance(joint_datamodule, DictConfig):
            import hydra.utils
            self.joint_datamodule = hydra.utils.instantiate(joint_datamodule)
        else:
            self.joint_datamodule = joint_datamodule

    def setup(self, stage: Optional[str] = None) -> None:
        structures_dir = os.path.join(self.pt_root, "structures")
        self.dspo_dataset = DSPOGroupsDataset(
            pt_root=os.path.join(self.pt_root, "pt"),
            high_thresholds=self.high_thresholds,
            min_high_counts=self.min_high_counts,
            structures_dir=structures_dir,
        )
        if self.joint_datamodule is not None:
            self.joint_datamodule.setup(stage)

    def train_dataloader(self) -> DataLoader:
        n = len(self.dspo_dataset)
        if n == 0:
            pt_dir = os.path.join(self.pt_root, "pt")
            raise ValueError(
                "DSPO train dataset is empty (num_samples=0). "
                "Check that pt_root/pt/ has .pt files and each group has both high- and low-score samples "
                f"(pt_root={self.pt_root!r}, pt_dir={pt_dir!r}, high_thresholds={self.high_thresholds!r})."
            )
        return DataLoader(
            self.dspo_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_dspo,
        )

    def val_dataloader(self):
        if self.joint_datamodule is not None:
            return self.joint_datamodule.val_dataloader()
        return None
