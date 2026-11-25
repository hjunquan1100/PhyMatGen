"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import copy
import os
import random
import time
from typing import Any, Dict, Literal, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from torch.nn import ModuleDict
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
from torchmetrics import MeanMetric
from tqdm import tqdm

from src.eval.crystal_generation import CrystalGenerationEvaluator
from src.eval.mof_generation import MOFGenerationEvaluator
from src.eval.molecule_generation import MoleculeGenerationEvaluator
from src.models.components.kabsch_utils import random_rotation_matrix
from src.models.vae_module import VariationalAutoencoderLitModule
from src.utils import pylogger

log = pylogger.RankedLogger(__name__)


IDX_TO_DATASET = {
    0: "mp20",
    1: "qm9",
    2: "qmof150",
    3: "carbon24",
    4: "perov5",
}
DATASET_TO_IDX = {
    "mp20": 0,  # periodic
    "qm9": 1,  # non-periodic
    "qmof150": 0,  # periodic
    "carbon24": 0,  # periodic (same idx as mp20, but identified by id prefix)
    "perov5": 0,  # periodic (same idx as mp20, but identified by dataloader_idx)
}

# joint 验证时必须跑满的四个数据集（顺序固定，保证 mp20/qm9/carbon24/perov5 都验证）
JOINT_VAL_DATASETS = ("mp20", "qm9", "carbon24", "perov5")

# 验证/采样时用的 dataset_idx：与数据里 CATEGORY_TO_IDX 一致，使 DiT 能区分 mp20 / carbon24 / perov5。
# 若 denoiser 的 num_classes >= 5 则用此映射；否则用 DATASET_TO_IDX（仅 2 类时 mp20/carbon24/perov5 共用一个条件，会混类）。
SAMPLING_DATASET_TO_IDX = {"mp20": 0, "qm9": 1, "carbon24": 3, "perov5": 4}


def _apply_dataset_atom_type_mask(out: Dict[str, Any], dataset_name: str) -> None:
    """按数据集对 atom_types logits 做掩码，与 DSPO/condition/generate.py 一致。
    - carbon24: 仅 C (Z=6)
    - qm9: 仅 H,C,N,O,F (Z=1,6,7,8,9)
    - mp20 / perov5: 无固定元素集，不掩码。
    原地修改 out['atom_types']。
    """
    if "atom_types" not in out:
        return
    logits = out["atom_types"]
    nc = logits.shape[-1]
    if dataset_name == "carbon24" and nc > 6:
        m = torch.full_like(logits, float("-inf"), device=logits.device, dtype=logits.dtype)
        m[:, 6] = logits[:, 6]
        out["atom_types"] = m
    elif dataset_name == "qm9" and nc > 9:
        allowed = (1, 6, 7, 8, 9)
        m = torch.full_like(logits, float("-inf"), device=logits.device, dtype=logits.dtype)
        for z in allowed:
            if z < nc:
                m[:, z] = logits[:, z]
        out["atom_types"] = m
    # mp20 / perov5: 不掩码


class LatentDiffusionLitModule(LightningModule):
    """LightningModule for latent diffusion generative modellling of 3D atomic systems.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        autoencoder_ckpt: str,
        denoiser: torch.nn.Module,
        interpolant: DictConfig,
        augmentations: DictConfig,
        sampling: DictConfig,
        conditioning: DictConfig,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scheduler_frequency: str,
        compile: bool,
        dspo_max_trajectory_steps_per_batch: int = 32,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # autoencoder models (first-stage model)
        self.autoencoder_ckpt = autoencoder_ckpt
        log.info(f"Loading Autoencoder ckpt: {autoencoder_ckpt}")
        self.autoencoder = VariationalAutoencoderLitModule.load_from_checkpoint(
            autoencoder_ckpt, map_location="cpu"
        )
        # freeze autoencoder
        self.autoencoder.requires_grad_(False)
        self.autoencoder.eval()
        
        # Verify VAE decoder weights are loaded correctly
        if self.global_rank == 0:
            decoder_params = sum(p.numel() for p in self.autoencoder.decoder.parameters())
            post_quant_params = sum(p.numel() for p in self.autoencoder.post_quant_conv.parameters())
            log.info(f"[VAE Check] Decoder parameters: {decoder_params:,}")
            log.info(f"[VAE Check] post_quant_conv parameters: {post_quant_params:,}")
            # Check if decoder weights are non-zero (not randomly initialized)
            decoder_weight_norm = sum(p.norm().item() for p in self.autoencoder.decoder.parameters())
            log.info(f"[VAE Check] Decoder weight norm: {decoder_weight_norm:.2f}")
            if decoder_weight_norm < 1e-6:
                log.error("[VAE Check] WARNING: Decoder weights are near zero! Checkpoint may not be loaded correctly!")

        # denoiser model (second-stage model)
        self.denoiser = denoiser

        # interpolant for diffusion or flow matching training/sampling
        self.interpolant = interpolant

        # evaluator objects for computing metrics
        self.val_generation_evaluators = {
            "mp20": CrystalGenerationEvaluator(
                dataset_cif_list=pd.read_csv(
                    os.path.join(self.hparams.sampling.data_dir, f"mp_20/raw/all.csv")
                )["cif"].tolist()
            ),
            "qm9": MoleculeGenerationEvaluator(
                dataset_smiles_list=torch.load(
                    os.path.join(self.hparams.sampling.data_dir, f"qm9/smiles.pt"),
                ),
                removeHs=self.hparams.sampling.removeHs,
                optimize_before_save=getattr(
                    self.hparams.sampling, "qm9_optimize_before_save", True
                ),
                relaxed_validity=getattr(
                    self.hparams.sampling, "qm9_relaxed_validity", False
                ),
                relaxed_bond_factor=getattr(
                    self.hparams.sampling, "qm9_relaxed_bond_factor", 2.2
                ),
                hybrid_rescue_enabled=getattr(
                    self.hparams.sampling, "qm9_hybrid_rescue_enabled", False
                ),
                hybrid_rescue_min_samples=getattr(
                    self.hparams.sampling, "qm9_hybrid_rescue_min_samples", 200
                ),
                hybrid_rescue_max_samples=getattr(
                    self.hparams.sampling, "qm9_hybrid_rescue_max_samples", 400
                ),
                hybrid_rescue_seed=getattr(
                    self.hparams.sampling, "qm9_hybrid_rescue_seed", 0
                ),
            ),
            "qmof150": MOFGenerationEvaluator(),
            "carbon24": CrystalGenerationEvaluator(
                dataset_cif_list=pd.read_csv(
                    os.path.join(self.hparams.sampling.data_dir, f"carbon_24/raw/all.csv")
                )["cif"].tolist()
            ) if os.path.exists(os.path.join(self.hparams.sampling.data_dir, f"carbon_24/raw/all.csv")) else CrystalGenerationEvaluator(dataset_cif_list=[]),
            "perov5": CrystalGenerationEvaluator(
                dataset_cif_list=pd.read_csv(
                    os.path.join(self.hparams.sampling.data_dir, f"perov_5/raw/all.csv")
                )["cif"].tolist()
            ) if os.path.exists(os.path.join(self.hparams.sampling.data_dir, f"perov_5/raw/all.csv")) else CrystalGenerationEvaluator(dataset_cif_list=[]),
        }
        self.test_generation_evaluators = copy.deepcopy(self.val_generation_evaluators)

        # metric objects for calculating and averaging across batches
        self.train_metrics = ModuleDict(
            {
                "loss": MeanMetric(),
                "x_loss": MeanMetric(),
                "x_loss t=[0,25)": MeanMetric(),
                "x_loss t=[25,50)": MeanMetric(),
                "x_loss t=[50,75)": MeanMetric(),
                "x_loss t=[75,100)": MeanMetric(),
                "t_avg": MeanMetric(),
                "dataset_idx": MeanMetric(),
            }
        )
        self.val_metrics = ModuleDict(
            {
                "mp20": ModuleDict(
                    {
                        "loss": MeanMetric(),
                        "x_loss": MeanMetric(),
                        "x_loss t=[0,25)": MeanMetric(),
                        "x_loss t=[25,50)": MeanMetric(),
                        "x_loss t=[50,75)": MeanMetric(),
                        "x_loss t=[75,100)": MeanMetric(),
                        "t_avg": MeanMetric(),
                        "valid_rate": MeanMetric(),
                        "struct_valid_rate": MeanMetric(),
                        "comp_valid_rate": MeanMetric(),
                        "unique_rate": MeanMetric(),
                        "novel_rate": MeanMetric(),
                        "sampling_time": MeanMetric(),
                    }
                ),
                "qm9": ModuleDict(
                    {
                        "loss": MeanMetric(),
                        "x_loss": MeanMetric(),
                        "x_loss t=[0,25)": MeanMetric(),
                        "x_loss t=[25,50)": MeanMetric(),
                        "x_loss t=[50,75)": MeanMetric(),
                        "x_loss t=[75,100)": MeanMetric(),
                        "t_avg": MeanMetric(),
                        "valid_rate": MeanMetric(),
                        "strict_valid_rate": MeanMetric(),
                        "unique_rate": MeanMetric(),
                        "novel_rate": MeanMetric(),
                        "mol_pred_loaded": MeanMetric(),
                        "sanitization": MeanMetric(),
                        "inchi_convertible": MeanMetric(),
                        "all_atoms_connected": MeanMetric(),
                        "bond_lengths": MeanMetric(),
                        "bond_angles": MeanMetric(),
                        "internal_steric_clash": MeanMetric(),
                        "aromatic_ring_flatness": MeanMetric(),
                        "double_bond_flatness": MeanMetric(),
                        "internal_energy": MeanMetric(),
                        "sampling_time": MeanMetric(),
                    }
                ),
                "qmof150": ModuleDict(
                    {
                        "loss": MeanMetric(),
                        "x_loss": MeanMetric(),
                        "x_loss t=[0,25)": MeanMetric(),
                        "x_loss t=[25,50)": MeanMetric(),
                        "x_loss t=[50,75)": MeanMetric(),
                        "x_loss t=[75,100)": MeanMetric(),
                        "t_avg": MeanMetric(),
                        "valid_rate": MeanMetric(),
                        "unique_rate": MeanMetric(),
                        "has_carbon": MeanMetric(),
                        "has_hydrogen": MeanMetric(),
                        "has_atomic_overlaps": MeanMetric(),
                        "has_overcoordinated_c": MeanMetric(),
                        "has_overcoordinated_n": MeanMetric(),
                        "has_overcoordinated_h": MeanMetric(),
                        "has_undercoordinated_c": MeanMetric(),
                        "has_undercoordinated_n": MeanMetric(),
                        "has_undercoordinated_rare_earth": MeanMetric(),
                        "has_metal": MeanMetric(),
                        "has_lone_molecule": MeanMetric(),
                        "has_high_charges": MeanMetric(),
                        # "is_porous": MeanMetric(),
                        "has_suspicicious_terminal_oxo": MeanMetric(),
                        "has_undercoordinated_alkali_alkaline": MeanMetric(),
                        "has_geometrically_exposed_metal": MeanMetric(),
                        # 'has_3d_connected_graph': MeanMetric(),
                        "all_checks": MeanMetric(),
                        "sampling_time": MeanMetric(),
                    }
                ),
                "carbon24": ModuleDict(
                    {
                        "loss": MeanMetric(),
                        "x_loss": MeanMetric(),
                        "x_loss t=[0,25)": MeanMetric(),
                        "x_loss t=[25,50)": MeanMetric(),
                        "x_loss t=[50,75)": MeanMetric(),
                        "x_loss t=[75,100)": MeanMetric(),
                        "t_avg": MeanMetric(),
                        "valid_rate": MeanMetric(),
                        "struct_valid_rate": MeanMetric(),
                        "comp_valid_rate": MeanMetric(),
                        "unique_rate": MeanMetric(),
                        "novel_rate": MeanMetric(),
                        "sampling_time": MeanMetric(),
                    }
                ),
                "perov5": ModuleDict(
                    {
                        "loss": MeanMetric(),
                        "x_loss": MeanMetric(),
                        "x_loss t=[0,25)": MeanMetric(),
                        "x_loss t=[25,50)": MeanMetric(),
                        "x_loss t=[50,75)": MeanMetric(),
                        "x_loss t=[75,100)": MeanMetric(),
                        "t_avg": MeanMetric(),
                        "valid_rate": MeanMetric(),
                        "struct_valid_rate": MeanMetric(),
                        "comp_valid_rate": MeanMetric(),
                        "unique_rate": MeanMetric(),
                        "novel_rate": MeanMetric(),
                        "sampling_time": MeanMetric(),
                    }
                ),
            }
        )
        self.test_metrics = copy.deepcopy(self.val_metrics)

        # Track which datasets are actually used during evaluation
        self._active_datasets = set()
        
        # Track global statistics for VAE latent normalization
        # These will be computed during training to ensure consistent normalization
        self.register_buffer("_vae_latent_mean", torch.zeros([]))
        self.register_buffer("_vae_latent_std", torch.ones([]))
        self._vae_latent_stats_initialized = False
        self._vae_latent_stats_update_count = 0

        # load bincounts for sampling
        self.num_nodes_bincount = {
            "mp20": torch.nn.Parameter(
                torch.load(
                    os.path.join(self.hparams.sampling.data_dir, f"mp_20/num_nodes_bincount.pt"),
                    map_location="cpu",
                ),
                requires_grad=False,
            ),
            "qm9": torch.nn.Parameter(
                torch.load(
                    os.path.join(self.hparams.sampling.data_dir, f"qm9/num_nodes_bincount.pt"),
                    map_location="cpu",
                ),
                requires_grad=False,
            ),
            "qmof150": torch.nn.Parameter(
                torch.load(
                    os.path.join(self.hparams.sampling.data_dir, f"qmof/num_nodes_bincount.pt"),
                    map_location="cpu",
                ),
                requires_grad=False,
            ) if os.path.exists(os.path.join(self.hparams.sampling.data_dir, f"qmof/num_nodes_bincount.pt")) else None,
            "carbon24": torch.nn.Parameter(
                torch.load(
                    os.path.join(self.hparams.sampling.data_dir, f"carbon_24/num_nodes_bincount.pt"),
                    map_location="cpu",
                ),
                requires_grad=False,
            ) if os.path.exists(os.path.join(self.hparams.sampling.data_dir, f"carbon_24/num_nodes_bincount.pt")) else None,
            "perov5": torch.nn.Parameter(
                torch.load(
                    os.path.join(self.hparams.sampling.data_dir, f"perov_5/num_nodes_bincount.pt"),
                    map_location="cpu",
                ),
                requires_grad=False,
            ) if os.path.exists(os.path.join(self.hparams.sampling.data_dir, f"perov_5/num_nodes_bincount.pt")) else None,
        }
        self.spacegroups_bincount = {
            "mp20": torch.nn.Parameter(
                torch.load(
                    os.path.join(self.hparams.sampling.data_dir, f"mp_20/spacegroups_bincount.pt"),
                    map_location="cpu",
                ),
                requires_grad=False,
            ),
            "qm9": None,
            "qmof150": None,
            "carbon24": torch.nn.Parameter(
                torch.load(
                    os.path.join(self.hparams.sampling.data_dir, f"carbon_24/spacegroups_bincount.pt"),
                    map_location="cpu",
                ),
                requires_grad=False,
            ) if os.path.exists(os.path.join(self.hparams.sampling.data_dir, f"carbon_24/spacegroups_bincount.pt")) else None,
            "perov5": torch.nn.Parameter(
                torch.load(
                    os.path.join(self.hparams.sampling.data_dir, f"perov_5/spacegroups_bincount.pt"),
                    map_location="cpu",
                ),
                requires_grad=False,
            ) if os.path.exists(os.path.join(self.hparams.sampling.data_dir, f"perov_5/spacegroups_bincount.pt")) else None,
        }

    def forward(self, batch, sample_posterior: bool = True):
        """batch 可为 PyG Data（结构）或 dict（use_latent=True 时含 x_1, token_mask, dataset_idx, spacegroup）。"""
        if isinstance(batch, dict) and batch.get("use_latent"):
            return self._forward_from_latent(batch)
        # Encode batch to latent space
        with torch.no_grad():
            encoded_batch = self.autoencoder.encode(batch)
            if sample_posterior:
                encoded_batch["x"] = encoded_batch["posterior"].sample()
            else:
                encoded_batch["x"] = encoded_batch["posterior"].mode()
            x_1 = encoded_batch["x"]
            
            # Debug: 仅训练时极少量打印，验证不刷屏
            if self.global_rank == 0 and self.training and random.random() < 0.01:
                x1_mean_before = x_1.mean().item()
                x1_std_before = x_1.std().item()
                x1_min = x_1.min().item()
                x1_max = x_1.max().item()
                log.debug(f"VAE encoded latent stats (BEFORE normalization): mean={x1_mean_before:.4f}, std={x1_std_before:.4f}, min={x1_min:.4f}, max={x1_max:.4f}")
                if abs(x1_mean_before) > 0.5 or x1_std_before < 0.5 or x1_std_before > 2.0:
                    log.debug(f"VAE encoded latent distribution deviates from standard normal! Normalizing to match standard normal distribution.")
            
            # Normalize VAE encoded latents to standard normal distribution (mean=0, std=1)
            # This is necessary because VAE's KL loss weight is very small (0.00001),
            # so the posterior distribution doesn't match standard normal.
            # By normalizing, we ensure LDM learns the correct distribution.
            # 
            # IMPORTANT: Use global statistics (computed over training) instead of per-batch
            # statistics to ensure consistent normalization across batches.
            if x_1.numel() > 0:
                if self.training:
                    # During training: update global statistics using exponential moving average
                    batch_mean = x_1.mean().detach()
                    batch_std = x_1.std().detach() + 1e-8
                    
                    if not self._vae_latent_stats_initialized:
                        # Initialize with first batch
                        self._vae_latent_mean = batch_mean
                        self._vae_latent_std = batch_std
                        self._vae_latent_stats_initialized = True
                        if self.global_rank == 0:
                            log.debug(f"Initialized VAE latent stats: mean={batch_mean.item():.4f}, std={batch_std.item():.4f}")
                    else:
                        # Update using exponential moving average (EMA)
                        # Use a small momentum (0.01) to slowly adapt to the distribution
                        momentum = 0.01
                        self._vae_latent_mean = (1 - momentum) * self._vae_latent_mean + momentum * batch_mean
                        self._vae_latent_std = (1 - momentum) * self._vae_latent_std + momentum * batch_std
                        self._vae_latent_stats_update_count += 1
                    
                    # Use global statistics for normalization
                    x_1 = (x_1 - self._vae_latent_mean) / (self._vae_latent_std + 1e-8)
                    
                    # Debug: Log global stats occasionally
                    if self.global_rank == 0 and random.random() < 0.01:
                        log.debug(f"Global VAE latent stats (after {self._vae_latent_stats_update_count} updates): mean={self._vae_latent_mean.item():.4f}, std={self._vae_latent_std.item():.4f}")
                else:
                    # During validation/test: use pre-computed global statistics
                    if self._vae_latent_stats_initialized:
                        x_1 = (x_1 - self._vae_latent_mean) / (self._vae_latent_std + 1e-8)
                    else:
                        # Fallback to per-batch normalization if stats not initialized
                        x_1_mean = x_1.mean()
                        x_1_std = x_1.std() + 1e-8
                        x_1 = (x_1 - x_1_mean) / x_1_std
                
                # Debug: 仅训练时极少量打印，验证不刷屏
                if self.global_rank == 0 and self.training and random.random() < 0.01:
                    x1_mean_after = x_1.mean().item()
                    x1_std_after = x_1.std().item()
                    log.debug(f"VAE encoded latent stats (AFTER normalization): mean={x1_mean_after:.4f}, std={x1_std_after:.4f}")
                
                if x_1.shape[0] > 0:
                    sample_tokens = x_1[:min(100, x_1.shape[0])]
                    token_std = sample_tokens.std(dim=0).mean().item()
                    if self.global_rank == 0 and self.training and random.random() < 0.01:
                        log.debug(f"VAE encoded latent per-token diversity (mean std across dims): {token_std:.4f}")
                    if token_std < 0.1:
                        if self.global_rank == 0:
                            log.debug(f"VAE encoded latent tokens have very low diversity! All tokens are similar.")

            # Convert from PyG batch to dense batch with padding
            x_1, mask = to_dense_batch(x_1, encoded_batch["batch"])
            dense_encoded_batch = {"x_1": x_1, "token_mask": mask, "diffuse_mask": mask}

        # Corrupt batch using the interpolant
        self.interpolant.device = dense_encoded_batch["x_1"].device
        noisy_dense_encoded_batch = self.interpolant.corrupt_batch(dense_encoded_batch)
        
        # Debug: Check noisy latent statistics during training
        if self.global_rank == 0 and self.training and random.random() < 0.05:
            x_t = noisy_dense_encoded_batch["x_t"]
            t = noisy_dense_encoded_batch["t"]
            log.debug(f"Noisy latent stats: mean={x_t.mean().item():.4f}, std={x_t.std().item():.4f}, t_mean={t.mean().item():.4f}")

        # Prepare conditioning inputs to forward pass
        dataset_idx = batch.dataset_idx + 1  # 0 -> null class
        # Clamp dataset_idx to valid range [1, num_datasets] to prevent index out of bounds
        # when num_datasets < actual number of datasets in the data
        max_dataset_idx = self.denoiser.dataset_embedder.num_classes
        dataset_idx = torch.clamp(dataset_idx, min=1, max=max_dataset_idx)
        # if not self.hparams.conditioning.dataset_idx:
        #     dataset_idx = torch.zeros_like(dataset_idx)
        spacegroup = batch.spacegroup
        if not self.hparams.conditioning.spacegroup:
            spacegroup = torch.zeros_like(batch.spacegroup)

        # Use self-conditioning for ~half training batches
        if (
            self.interpolant.self_condition
            and random.random() < self.interpolant.self_condition_prob
        ):
            with torch.no_grad():
                x_sc = self.denoiser(
                    x=noisy_dense_encoded_batch["x_t"],
                    t=noisy_dense_encoded_batch["t"],
                    dataset_idx=dataset_idx,
                    spacegroup=spacegroup,
                    mask=mask,
                    x_sc=None,
                )
        else:
            x_sc = None

        # Run denoiser model
        pred_x = self.denoiser(
            x=noisy_dense_encoded_batch["x_t"],
            t=noisy_dense_encoded_batch["t"],
            dataset_idx=dataset_idx,
            spacegroup=spacegroup,
            mask=mask,
            x_sc=x_sc,
        )
        
        # Debug: Check denoiser prediction statistics during training
        if self.global_rank == 0 and self.training and random.random() < 0.05:
            pred_mean = pred_x.mean().item()
            pred_std = pred_x.std().item()
            gt_mean = noisy_dense_encoded_batch["x_1"].mean().item()
            gt_std = noisy_dense_encoded_batch["x_1"].std().item()
            log.debug(f"Denoiser prediction stats: mean={pred_mean:.4f}, std={pred_std:.4f}")
            log.debug(f"Ground truth (x_1) stats: mean={gt_mean:.4f}, std={gt_std:.4f}")
            # Check if denoiser is learning the correct distribution
            std_ratio = pred_std / (gt_std + 1e-8)
            mean_diff = abs(pred_mean - gt_mean)
            if mean_diff > 0.5 or std_ratio < 0.5 or std_ratio > 2.0:
                log.debug(f"Denoiser prediction distribution doesn't match ground truth! (mean_diff={mean_diff:.4f}, std_ratio={std_ratio:.4f})")
            else:
                log.debug(f"Denoiser is learning correctly! (mean_diff={mean_diff:.4f}, std_ratio={std_ratio:.4f})")

        return pred_x, noisy_dense_encoded_batch

    def _forward_from_latent(self, batch: dict):
        """DSPO 用：直接使用 pt 中保存的潜变量 x_1，跳过 VAE 编码。"""
        x_1 = batch["x_1"].to(self.device)
        mask = batch["token_mask"].to(self.device)
        dataset_idx = batch["dataset_idx"].to(self.device)
        spacegroup = batch["spacegroup"].to(self.device)
        if not self.hparams.conditioning.spacegroup:
            spacegroup = torch.zeros_like(spacegroup, device=self.device)
        if x_1.numel() > 0 and getattr(self, "_vae_latent_stats_initialized", False):
            x_1 = (x_1 - self._vae_latent_mean.to(x_1.device)) / (self._vae_latent_std.to(x_1.device) + 1e-8)
        dense_encoded_batch = {"x_1": x_1, "token_mask": mask, "diffuse_mask": mask}
        self.interpolant.device = x_1.device
        noisy_dense_encoded_batch = self.interpolant.corrupt_batch(dense_encoded_batch)
        max_dataset_idx = self.denoiser.dataset_embedder.num_classes
        dataset_idx = torch.clamp(dataset_idx, min=1, max=max_dataset_idx)
        if (
            self.interpolant.self_condition
            and random.random() < self.interpolant.self_condition_prob
        ):
            with torch.no_grad():
                x_sc = self.denoiser(
                    x=noisy_dense_encoded_batch["x_t"],
                    t=noisy_dense_encoded_batch["t"],
                    dataset_idx=dataset_idx,
                    spacegroup=spacegroup,
                    mask=mask,
                    x_sc=None,
                )
        else:
            x_sc = None
        pred_x = self.denoiser(
            x=noisy_dense_encoded_batch["x_t"],
            t=noisy_dense_encoded_batch["t"],
            dataset_idx=dataset_idx,
            spacegroup=spacegroup,
            mask=mask,
            x_sc=x_sc,
        )
        return pred_x, noisy_dense_encoded_batch

    def criterion(
        self,
        noisy_dense_encoded_batch: Dict[str, torch.Tensor],
        pred_x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Compute MSE loss w/ masking for padded tokens
        gt_x_1 = noisy_dense_encoded_batch["x_1"]
        norm_scale = 1 - torch.min(noisy_dense_encoded_batch["t"].unsqueeze(-1), torch.tensor(0.9))
        x_error = (gt_x_1 - pred_x) / norm_scale
        loss_mask = (
            noisy_dense_encoded_batch["token_mask"] * noisy_dense_encoded_batch["diffuse_mask"]
        )
        loss_denom = torch.sum(loss_mask, dim=-1) * pred_x.size(-1)
        x_loss = torch.sum(x_error**2 * loss_mask[..., None], dim=(-1, -2)) / loss_denom
        loss_dict = {"loss": x_loss.mean(), "x_loss": x_loss}

        # add diffusion loss stratified across t
        num_bins = 4
        flat_losses = x_loss.detach().cpu().numpy().flatten()
        flat_t = noisy_dense_encoded_batch["t"].detach().cpu().numpy().flatten()
        bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins + 1)
        bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
        t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
        t_binned_n = np.bincount(bin_idx)
        for t_bin in np.unique(bin_idx).tolist():
            bin_start = bin_edges[t_bin]
            bin_end = bin_edges[t_bin + 1]
            t_range = f"x_loss t=[{int(bin_start*100)},{int(bin_end*100)})"
            range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
            loss_dict[t_range] = range_loss
        loss_dict["t_avg"] = np.mean(flat_t)

        return loss_dict

    #####################################################################################################

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """加载 checkpoint 后打印 denoiser 输出维度，便于排查 8 vs 10：数据是 8 维，若这里是 10 则问题在 checkpoint/模型。"""
        d_x_attr = getattr(self.denoiser, "d_x", None)
        out_dim = None
        if hasattr(self.denoiser, "final_layer") and hasattr(self.denoiser.final_layer, "linear"):
            w = self.denoiser.final_layer.linear.weight
            if w is not None:
                out_dim = int(w.shape[0])
        if self.global_rank == 0:
            log.info(
                "[DSPO 维度检查] 加载 checkpoint 后: denoiser.d_x=%s, final_layer.linear.weight.shape[0]=%s (应与 pt 轨迹 8 维一致)",
                d_x_attr, out_dim,
            )
            if out_dim is not None and out_dim != 8:
                log.warning("[DSPO 维度检查] denoiser 实际输出 %s 维，与数据 8 维不一致，将用代码切片兜底；建议检查该 checkpoint 是否用 d_x=8 训练。", out_dim)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        for dataset in self.val_metrics.keys():
            for metric in self.val_metrics[dataset].values():
                metric.reset()

    def on_train_epoch_start(self) -> None:
        """Lightning hook that is called when a training epoch starts."""
        for metric in self.train_metrics.values():
            metric.reset()
        # 每轮开始时打一条训练进度说明，避免终端只看到 Sampling
        if self.global_rank == 0:
            try:
                dl = self.trainer.train_dataloader
                n_batches = len(dl) if hasattr(dl, "__len__") else "?"
                log.info(f"[Epoch {self.current_epoch}] Training: {n_batches} batches this epoch.")
            except Exception:
                log.info(f"[Epoch {self.current_epoch}] Training started.")

    def _move_dspo_batch_to_device(self, batch):
        """把 DSPO 的 batch 显式移到当前 rank 的 device，避免 DDP 下 Lightning 未转移嵌套 dict/list 导致多卡报错。"""
        if isinstance(batch, dict):
            if batch.get("use_full_trajectory"):
                batch["trajectory_tokens"] = [t.to(self.device) for t in batch["trajectory_tokens"]]
                batch["trajectory_clean"] = [t.to(self.device) for t in batch["trajectory_clean"]]
                batch["token_mask"] = batch["token_mask"].to(self.device)
                batch["dataset_idx"] = batch["dataset_idx"].to(self.device)
                batch["spacegroup"] = batch["spacegroup"].to(self.device)
            elif batch.get("use_latent"):
                batch["x_1"] = batch["x_1"].to(self.device)
                batch["token_mask"] = batch["token_mask"].to(self.device)
                batch["dataset_idx"] = batch["dataset_idx"].to(self.device)
                batch["spacegroup"] = batch["spacegroup"].to(self.device)
            return batch
        if isinstance(batch, Data):
            return batch.to(self.device, non_blocking=True)
        return batch

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step. Supports DSPO: when batch is (batch_Data, scores, log_pi_tau), use advantage-weighted loss."""
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            batch_data, scores, _log_pi_tau = batch
            batch_data = self._move_dspo_batch_to_device(batch_data)
            scores = scores.to(self.device)
            return self._training_step_dspo(batch_data, scores, batch_idx)
        return self._training_step_diffusion(batch, batch_idx)

    def _compute_dspo_advantage(
        self,
        scores: torch.Tensor,
        dataset_idx: torch.Tensor,
        *,
        log_prefix: str,
    ) -> torch.Tensor:
        """按 dataset_idx 分组标准化 scores，避免不同数据集的 reward 量纲互相污染。"""
        scores = scores.reshape(-1)
        dataset_idx = dataset_idx.to(scores.device).reshape(-1)
        if scores.numel() != dataset_idx.numel():
            raise ValueError(
                f"DSPO advantage shape mismatch: scores{tuple(scores.shape)} vs dataset_idx{tuple(dataset_idx.shape)}"
            )

        adv = torch.empty_like(scores)
        degenerate_groups = []
        for ds in torch.unique(dataset_idx):
            mask = dataset_idx == ds
            group_scores = scores[mask]
            if group_scores.numel() <= 1:
                adv[mask] = 1.0
                degenerate_groups.append((int(ds.item()), int(group_scores.numel())))
                continue

            group_std = group_scores.std(unbiased=False)
            if torch.isnan(group_std) or group_std < 1e-6:
                adv[mask] = 1.0
                degenerate_groups.append((int(ds.item()), int(group_scores.numel())))
                continue

            adv[mask] = (group_scores - group_scores.mean()) / (group_std + 1e-8)

        if degenerate_groups and self.global_rank == 0:
            logged = getattr(self, "_dspo_group_norm_logged", False)
            if not logged:
                detail = ", ".join(
                    f"dataset_idx={ds} (n={n})" for ds, n in degenerate_groups
                )
                log.warning(
                    "%s: per-dataset score std is too small for %s; falling back to weight=1.0 for those groups.",
                    log_prefix,
                    detail,
                )
                self._dspo_group_norm_logged = True

        return adv

    def _training_step_dspo(
        self, batch, scores: torch.Tensor, batch_idx: int
    ) -> torch.Tensor:
        """DSPO: advantage-weighted denoising loss。支持 use_latent（仅终点）或 use_full_trajectory（完整轨迹）。"""
        if isinstance(batch, dict) and batch.get("use_full_trajectory"):
            return self._training_step_dspo_full_trajectory(batch, scores, batch_idx)
        if not (isinstance(batch, dict) and batch.get("use_latent")):
            with torch.no_grad():
                sample_is_periodic = batch.dataset_idx != DATASET_TO_IDX["qm9"]
                node_is_periodic = sample_is_periodic[batch.batch]
                if self.hparams.augmentations.frac_coords == True and node_is_periodic.any():
                    random_translation = (
                        torch.normal(
                            torch.abs(batch.lengths.mean(dim=0)),
                            torch.abs(batch.lengths.std(dim=0).nan_to_num(1e-8)),
                        )
                        / 2
                    )
                    pos_aug = batch.pos + random_translation
                    batch.pos = pos_aug
                    cell_per_node_inv = torch.linalg.inv(batch.cell[batch.batch][node_is_periodic])
                    frac_coords_aug = torch.einsum(
                        "bi,bij->bj", batch.pos[node_is_periodic], cell_per_node_inv
                    )
                    frac_coords_aug = frac_coords_aug % 1.0
                    batch.frac_coords[node_is_periodic] = frac_coords_aug
                if self.hparams.augmentations.pos == True:
                    rot_mat = random_rotation_matrix(validate=True, device=self.device)
                    batch.pos = batch.pos @ rot_mat.T
                    batch.cell = batch.cell @ rot_mat.T
        pred_x, noisy_dense_encoded_batch = self.forward(batch)
        loss_dict = self.criterion(noisy_dense_encoded_batch, pred_x)
        x_loss = loss_dict["x_loss"]  # (B,)
        scores = scores.to(x_loss.device, dtype=x_loss.dtype)
        dataset_idx = batch["dataset_idx"] if isinstance(batch, dict) else batch.dataset_idx
        adv = self._compute_dspo_advantage(
            scores,
            dataset_idx,
            log_prefix="DSPO",
        )
        loss = (adv * x_loss).mean()
        self.log("train/dspo_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def _training_step_dspo_full_trajectory(
        self, batch: dict, scores: torch.Tensor, batch_idx: int
    ) -> torch.Tensor:
        """完整轨迹 DSPO：对轨迹每一步算 denoiser 预测与 target 的 MSE，再按 advantage 加权求和。"""
        traj_tokens = batch["trajectory_tokens"]
        traj_clean = batch["trajectory_clean"]
        token_mask = batch["token_mask"].to(self.device)
        dataset_idx = batch["dataset_idx"].to(self.device)
        spacegroup = batch["spacegroup"].to(self.device)
        if not self.hparams.conditioning.spacegroup:
            spacegroup = torch.zeros_like(spacegroup, device=self.device)
        T = batch["num_steps"]
        if isinstance(T, torch.Tensor):
            T = int(T.item())
        G = token_mask.shape[0]
        d = traj_tokens[0].shape[-1]
        d_x = self.denoiser.d_x  # LDM 按 config 固定为 8 维，与 VAE 潜空间一致
        if d != d_x:
            log.warning(
                "DSPO 轨迹 token 维 %s != denoiser.d_x=%s，将截断为 %s 维。建议用 run_groups 重新生成 pt 以保持 8 维。", d, d_x, d_x
            )
        scores = scores.to(self.device, dtype=traj_tokens[0].dtype)
        adv = self._compute_dspo_advantage(
            scores,
            dataset_idx,
            log_prefix="DSPO full_trajectory",
        )
        min_t = getattr(self.interpolant, "min_t", 1e-2)
        ts = torch.linspace(min_t, 1.0, T, device=self.device, dtype=traj_tokens[0].dtype)
        max_dataset_idx = self.denoiser.dataset_embedder.num_classes
        dataset_idx = torch.clamp(dataset_idx, min=1, max=max_dataset_idx)
        # 子采样时间步：只对 K 步建图 backward，避免 T=250 步大图导致 OOM（梯度仍为无偏估计）
        max_steps_per_batch = getattr(self.hparams, "dspo_max_trajectory_steps_per_batch", 32)
        K = min(T, max_steps_per_batch)
        step_indices = torch.randperm(T, device=self.device)[:K] if K < T else torch.arange(T, device=self.device)
        step_indices = step_indices.sort().values
        loss_sum = 0.0
        for i in range(K):
            t = int(step_indices[i].item())
            x_t = traj_tokens[t].to(self.device)
            target_x1 = traj_clean[t].to(self.device)
            if x_t.dim() == 4:
                x_t = x_t.reshape(x_t.size(0), x_t.size(1), -1)[..., :d_x].contiguous()
            elif x_t.shape[-1] != d_x:
                x_t = x_t[..., :d_x].contiguous()
            if target_x1.dim() == 4:
                target_x1 = target_x1.reshape(target_x1.size(0), target_x1.size(1), -1)[..., :d_x].contiguous()
                if not getattr(self, "_dspo_4d_target_logged", False) and self.global_rank == 0:
                    log.warning("DSPO: pt 中 traj_clean 为 4 维 (B,N,8,8)，已 reshape 为 (B,N,%s)。建议检查 run_groups 保存格式。", d_x)
                    self._dspo_4d_target_logged = True
            elif target_x1.shape[-1] != d_x:
                target_x1 = target_x1[..., :d_x].contiguous()
            t_val = ts[t].item()
            t_batch = torch.full((G, 1), t_val, device=self.device, dtype=x_t.dtype)
            pred_x1 = self.denoiser(x_t, t_batch, dataset_idx, spacegroup, token_mask, x_sc=None)
            B_out = pred_x1.shape[0]
            N_pred = pred_x1.shape[1]
            N_mask = token_mask.shape[1]
            N_target = target_x1.shape[1]
            N_use = min(N_pred, N_mask, N_target)
            d_target = int(target_x1.size(-1))
            d_pred = int(pred_x1.size(-1))
            d_use = min(d_target, d_pred)
            if not getattr(self, "_dspo_dim_logged", False) and (d_target != d_x or d_pred != d_x):
                log.warning(
                    "DSPO 维度不一致：denoiser.d_x=%s, target_x1.shape[-1]=%s, pred_x1.shape[-1]=%s → 使用 d_use=%s。LDM 应为 8 维。",
                    d_x, d_target, d_pred, d_use,
                )
                self._dspo_dim_logged = True
            t_slice = target_x1[:B_out, :N_use, :d_use].clone()
            p_slice = pred_x1[:B_out, :N_use, :d_use].clone()
            assert t_slice.shape == p_slice.shape, (
                f"DSPO dim mismatch: t_slice{t_slice.shape} vs p_slice{p_slice.shape}"
            )
            norm_scale = 1.0 - min(t_val, 0.9)
            x_error = (t_slice - p_slice) / norm_scale
            m = token_mask[:B_out, :N_use]
            if m.dim() == 3:
                m = m.any(dim=-1)
            loss_mask = m.unsqueeze(-1).to(dtype=x_error.dtype, device=x_error.device)
            loss_denom = (m.float().sum(dim=1) * d_use + 1e-8).to(x_error.device)
            x_loss_t = (x_error ** 2 * loss_mask).sum(dim=(1, 2)) / loss_denom
            loss_sum = loss_sum + (adv[:B_out] * x_loss_t).mean()
        loss = loss_sum / K
        self.log("train/dspo_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def _training_step_diffusion(self, batch: Data, batch_idx: int) -> torch.Tensor:
        """Standard diffusion training step."""
        with torch.no_grad():
            # save masks used to apply augmentations
            sample_is_periodic = batch.dataset_idx != DATASET_TO_IDX["qm9"]
            node_is_periodic = sample_is_periodic[batch.batch]

            if self.hparams.augmentations.frac_coords == True:
                if node_is_periodic.any():
                    # sample random translation vector from batch length distribution / 2
                    random_translation = (
                        torch.normal(
                            torch.abs(batch.lengths.mean(dim=0)),
                            torch.abs(batch.lengths.std(dim=0).nan_to_num(1e-8)),
                        )
                        / 2
                    )
                    # apply same random translation to all Cartesian coordinates
                    pos_aug = batch.pos + random_translation
                    batch.pos = pos_aug
                    # compute new fractional coordinates for samples which are periodic
                    cell_per_node_inv = torch.linalg.inv(batch.cell[batch.batch][node_is_periodic])
                    frac_coords_aug = torch.einsum(
                        "bi,bij->bj", batch.pos[node_is_periodic], cell_per_node_inv
                    )
                    frac_coords_aug = frac_coords_aug % 1.0
                    batch.frac_coords[node_is_periodic] = frac_coords_aug

            if self.hparams.augmentations.pos == True:
                rot_mat = random_rotation_matrix(validate=True, device=self.device)
                pos_aug = batch.pos @ rot_mat.T
                batch.pos = pos_aug
                cell_aug = batch.cell @ rot_mat.T
                batch.cell = cell_aug
                # fractional coordinates are rotation invariant
                # assert torch.allclose(
                #     batch.frac_coords,
                #     torch.einsum("bi,bij->bj", pos_aug, torch.linalg.inv(cell_aug)[batch.batch]) % 1.0,
                #     rtol=1e-3,
                #     atol=1e-3,
                # )

        # forward pass
        pred_x, noisy_dense_encoded_batch = self.forward(batch)

        # calculate loss
        loss_dict = self.criterion(noisy_dense_encoded_batch, pred_x)

        # log relative proportions of datasets in batch
        loss_dict["dataset_idx"] = batch.dataset_idx.detach().flatten()

        # update and log train metrics
        for k, v in loss_dict.items():
            self.train_metrics[k](v)
            self.log(
                f"train/{k}",
                self.train_metrics[k],
                on_step=True,
                on_epoch=False,
                prog_bar=False if k != "loss" else True,
            )

        # return loss or backpropagation will fail
        return loss_dict["loss"]

    #####################################################################################################

    def on_validation_epoch_start(self) -> None:
        self.on_evaluation_epoch_start(stage="val")

    def validation_step(self, batch: Data, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.evaluation_step(batch, batch_idx, dataloader_idx, stage="val")

    def on_validation_epoch_end(self) -> None:
        self.on_evaluation_epoch_end(stage="val")

    #####################################################################################################

    def on_test_epoch_start(self) -> None:
        self.on_evaluation_epoch_start(stage="test")

    def test_step(self, batch: Data, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.evaluation_step(batch, batch_idx, dataloader_idx, stage="test")

    def on_test_epoch_end(self) -> None:
        self.on_evaluation_epoch_end(stage="test")

    #####################################################################################################

    def on_evaluation_epoch_start(self, stage: Literal["val", "test"]) -> None:
        "Lightning hook that is called when a validation/test epoch starts."
        if stage not in ["val", "test"]:
            raise ValueError("stage must be 'val' or 'test'.")
        # Reset active datasets tracking
        self._active_datasets.clear()
        metrics = getattr(self, f"{stage}_metrics")
        for dataset in metrics.keys():
            for metric in metrics[dataset].values():
                metric.reset()
        generation_evaluators = getattr(self, f"{stage}_generation_evaluators")
        for dataset in generation_evaluators.keys():
            generation_evaluators[dataset].clear()  # clear lists for next epoch

    def evaluation_step(
        self,
        batch: Data,
        batch_idx: int,
        dataloader_idx: int,
        stage: Literal["val", "test"],
    ) -> None:
        """Perform a single evaluation step on a batch of data from the validation/test set."""

        if stage not in ["val", "test"]:
            raise ValueError("stage must be 'val' or 'test'.")
        
        # First, try to determine dataset from batch.id (for carbon24, perov5, etc.)
        dataset_name = None
        if hasattr(batch, 'id') and batch.id is not None:
            try:
                # Get first id in batch to determine dataset
                if hasattr(batch.id, '__len__') and len(batch.id) > 0:
                    if hasattr(batch.id, '__getitem__'):
                        first_id = str(batch.id[0])
                    elif hasattr(batch.id, 'item'):
                        first_id = str(batch.id.item())
                    else:
                        first_id = str(batch.id)
                else:
                    first_id = str(batch.id)
                
                # Check for dataset indicators in id
                if first_id.startswith("C-"):
                    dataset_name = "carbon24"
                elif first_id.startswith("perov5_") or first_id.startswith("perov5-"):
                    dataset_name = "perov5"
                elif first_id.startswith("mp-"):
                    dataset_name = "mp20"
                elif 'qm9' in first_id.lower() or first_id.startswith("qm9_"):
                    dataset_name = "qm9"
                elif 'qmof' in first_id.lower():
                    dataset_name = "qmof150"
            except Exception:
                pass
        
        # If not determined from id, use dataloader_idx as fallback (for perov5 which also uses mp- prefix)
        if dataset_name is None:
            # Use dataloader_idx mapping similar to VAE module
            dataloader_to_dataset = {
                0: "mp20",
                1: "qm9", 
                2: "qmof150",
                3: "carbon24",
                4: "perov5",
            }
            dataset_name = dataloader_to_dataset.get(dataloader_idx)
            
            # If still not found, try dataset_idx mapping
            if dataset_name is None:
                batch_dataset_idx = batch.dataset_idx[0].item() if hasattr(batch, 'dataset_idx') and len(batch.dataset_idx) > 0 else dataloader_idx
                dataset_name = IDX_TO_DATASET.get(batch_dataset_idx, IDX_TO_DATASET.get(dataloader_idx, "mp20"))
        
        # Track which datasets are actually used
        self._active_datasets.add(dataset_name)
        
        # Check if dataset_name exists in metrics and evaluators
        stage_metrics = getattr(self, f"{stage}_metrics")
        stage_evaluators = getattr(self, f"{stage}_generation_evaluators")
        
        if dataset_name not in stage_metrics:
            log.warning(f"Dataset {dataset_name} not found in {stage}_metrics. Available: {list(stage_metrics.keys())}")
            return
        if dataset_name not in stage_evaluators:
            log.warning(f"Dataset {dataset_name} not found in {stage}_generation_evaluators. Available: {list(stage_evaluators.keys())}")
            return
        
        metrics = stage_metrics[dataset_name]
        generation_evaluator = stage_evaluators[dataset_name]
        generation_evaluator.device = metrics["loss"].device

        # forward pass
        pred_x, noisy_dense_encoded_batch = self.forward(batch)

        # calculate loss
        loss_dict = self.criterion(noisy_dense_encoded_batch, pred_x)

        # update and log per-step val metrics
        for k, v in loss_dict.items():
            metrics[k](v)
            self.log(
                f"{stage}_{dataset_name}/{k}",
                metrics[k],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                add_dataloader_idx=False,
            )

    def on_evaluation_epoch_end(self, stage: Literal["val", "test"]) -> None:
        """Lightning hook that is called when a validation/test epoch ends."""

        if stage not in ["val", "test"]:
            raise ValueError("stage must be 'val' or 'test'.")
        metrics = getattr(self, f"{stage}_metrics")
        generation_evaluators = getattr(self, f"{stage}_generation_evaluators")

        # 固定对 joint 四类（mp20, qm9, carbon24, perov5）做验证采样，不依赖 _active_datasets
        active_datasets = [
            d for d in JOINT_VAL_DATASETS
            if d in metrics
            and d in generation_evaluators
            and self.num_nodes_bincount.get(d) is not None
        ]
        missing = [d for d in JOINT_VAL_DATASETS if d not in active_datasets]
        if missing and self.global_rank == 0:
            log.warning(
                "Validation sampling skipped for %s (missing in metrics/evaluators or num_nodes_bincount). "
                "Ensure data_dir has %s/num_nodes_bincount.pt for joint four.",
                missing,
                " and ".join(missing),
            )
        agg_valid_rates: list[float] = []
        for dataset in active_datasets:
            generation_evaluators[dataset].device = metrics[dataset]["loss"].device
            t_start = time.time()
            for samples_so_far in tqdm(
                range(0, self.hparams.sampling.num_samples, self.hparams.sampling.batch_size),
                desc=f"    Sampling",
            ):
                # 采样时用唯一 dataset_idx，使 DiT 区分 mp20 / carbon24 / perov5（需 num_datasets>=5）
                num_classes = getattr(
                    self.denoiser.dataset_embedder, "num_classes", 0
                )
                sidx = (
                    SAMPLING_DATASET_TO_IDX[dataset]
                    if num_classes >= 5
                    else DATASET_TO_IDX[dataset]
                )
                cfg_scale = getattr(
                    self.hparams.sampling, "cfg_scale_qm9", self.hparams.sampling.cfg_scale
                ) if dataset == "qm9" else self.hparams.sampling.cfg_scale
                num_timesteps_qm9 = getattr(
                    self.hparams.sampling, "num_timesteps_qm9", None
                )
                if dataset == "qm9" and self.global_rank == 0:
                    log.info(
                        f"[QM9 sampling] cfg_scale={cfg_scale}, num_timesteps={num_timesteps_qm9 or self.interpolant.num_timesteps}"
                    )
                out, batch, samples = self.sample_and_decode(
                    num_nodes_bincount=self.num_nodes_bincount[dataset],
                    spacegroups_bincount=self.spacegroups_bincount.get(dataset),
                    batch_size=self.hparams.sampling.batch_size,
                    cfg_scale=cfg_scale,
                    dataset_idx=sidx,
                    num_timesteps=num_timesteps_qm9 if dataset == "qm9" else None,
                )
                # 按数据集的元素约束对 atom_types logits 做掩码（与 DSPO 脚本一致）
                _apply_dataset_atom_type_mask(out, dataset)
                # Save predictions for metrics and visualisation
                start_idx = 0
                for idx_in_batch, num_atom in enumerate(batch["num_atoms"].tolist()):
                    _atom_types = (
                        out["atom_types"].narrow(0, start_idx, num_atom).argmax(dim=1)
                    )  # take argmax
                    _atom_types[_atom_types == 0] = 1  # atom type 0 -> 1 (H) to prevent crash
                    _pos = out["pos"].narrow(0, start_idx, num_atom) * 10.0  # nm to A
                    _frac_coords = out["frac_coords"].narrow(0, start_idx, num_atom)
                    _lengths = out["lengths"][idx_in_batch] * float(num_atom) ** (
                        1 / 3
                    )  # unscale lengths
                    _angles = torch.rad2deg(out["angles"][idx_in_batch])  # convert to degrees
                    generation_evaluators[dataset].append_pred_array(
                        {
                            "atom_types": _atom_types.detach().cpu().numpy(),
                            "pos": _pos.detach().cpu().numpy(),
                            "frac_coords": _frac_coords.detach().cpu().numpy(),
                            "lengths": _lengths.detach().cpu().numpy(),
                            "angles": _angles.detach().cpu().numpy(),
                            "sample_idx": samples_so_far
                            + self.global_rank * len(batch["num_atoms"])
                            + idx_in_batch,
                        }
                    )
                    start_idx = start_idx + num_atom
            t_end = time.time()

            # Compute generation metrics
            gen_metrics_dict = generation_evaluators[dataset].get_metrics(
                save=self.hparams.sampling.visualize,
                save_dir=self.hparams.sampling.save_dir + f"/{dataset}_{stage}_{self.global_rank}",
            )
            gen_metrics_dict["sampling_time"] = t_end - t_start
            
            # 每个数据集打一行 valid_rate，便于在终端看到（含 QM9）
            if self.global_rank == 0:
                vr = gen_metrics_dict.get("valid_rate")
                if isinstance(vr, torch.Tensor):
                    vr = vr.float().mean().item() if vr.numel() > 1 else vr.item()
                else:
                    vr = float(vr) if vr is not None else None
                line = f"  {dataset} valid_rate: {vr:.4f}" if vr is not None else f"  {dataset}: (no valid_rate)"
                if dataset == "qm9" and "strict_valid_rate" in gen_metrics_dict:
                    svr = gen_metrics_dict["strict_valid_rate"]
                    if isinstance(svr, torch.Tensor):
                        svr = svr.float().mean().item() if svr.numel() > 1 else svr.item()
                    else:
                        svr = float(svr)
                    line += f", strict_valid_rate: {svr:.4f}"
                log.info(line)
            
            # 详细指标改为 debug 级别，默认不打印；需要时设置 logging 为 DEBUG
            if self.global_rank == 0:
                log.debug(f"\n{'='*60}")
                log.debug(f"Generation metrics for {dataset} at epoch {self.current_epoch}:")
                for k, v in gen_metrics_dict.items():
                    if isinstance(v, torch.Tensor):
                        if v.dtype == torch.bool:
                            valid_count = v.sum().item()
                            total_count = v.numel()
                            log.debug(f"  {k}: {valid_count}/{total_count} ({valid_count/total_count*100:.2f}%)")
                        elif v.numel() > 1:
                            log.debug(f"  {k}: mean={v.float().mean().item():.4f}, min={v.float().min().item():.4f}, max={v.float().max().item():.4f}")
                        else:
                            log.debug(f"  {k}: {v.item():.4f}")
                    else:
                        log.debug(f"  {k}: {v}")
                log.debug(f"{'='*60}\n")
            
            for k, v in gen_metrics_dict.items():
                # Convert boolean tensors to float for MeanMetric
                if isinstance(v, torch.Tensor) and v.dtype == torch.bool:
                    v = v.float()
                # If tensor has multiple values, compute mean first
                if isinstance(v, torch.Tensor) and v.numel() > 1:
                    v = v.mean()
                if k == "valid_rate":
                    agg_valid_rates.append(
                        v.item() if isinstance(v, torch.Tensor) else float(v)
                    )
                metrics[dataset][k](v)
                self.log(
                    f"{stage}_{dataset}/{k}",
                    metrics[dataset][k],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False if k != "valid_rate" else True,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

            if self.hparams.sampling.visualize and type(self.logger) == WandbLogger:
                pred_table = generation_evaluators[dataset].get_wandb_table(
                    current_epoch=self.current_epoch,
                    save_dir=self.hparams.sampling.save_dir
                    + f"/{dataset}_{stage}_{self.global_rank}",
                )
                self.logger.experiment.log(
                    {f"{dataset}_{stage}_samples_table_device{self.global_rank}": pred_table}
                )

        # 聚合各数据集的 valid_rate（均值），供 jointtrain 等 callback 监控
        if agg_valid_rates:
            agg = sum(agg_valid_rates) / len(agg_valid_rates)
            if self.global_rank == 0:
                log.info(f"  {stage} valid_rate (avg over {len(agg_valid_rates)} datasets): {agg:.4f}")
            self.log(
                f"{stage}_valid_rate",
                agg,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                add_dataloader_idx=False,
            )
        # DDP：等所有 rank 都跑完验证（含 get_metrics 写盘/算指标），再一起进入 Lightning 的 metric sync，避免 NCCL 集合超时
        if stage == "val" and torch.distributed.is_initialized():
            torch.distributed.barrier()
        # 验证结束后释放 GPU 缓存，避免下一轮训练步 OOM（验证采样会占满显存）
        if stage == "val" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    #####################################################################################################

    def sample_and_decode(
        self,
        num_nodes_bincount,
        spacegroups_bincount,
        batch_size,
        cfg_scale=4.0,
        dataset_idx=0,
        num_timesteps=None,
        return_log_pi=False,
        return_trajectory=False,
    ):
        dataset_idx_cond = dataset_idx  # 原始 0/1/3/4，供 DSPO 存轨迹用
        # sample random lengths from distribution: (B, 1)
        sample_lengths = torch.multinomial(
            num_nodes_bincount.float(),
            batch_size,
            replacement=True,
        ).to(self.device)

        # create dataset_idx tensor
        # NOTE 0 -> null class within DiT, while 0 -> MP20 elsewhere, so increment by 1
        dataset_idx = torch.full(
            (batch_size,), dataset_idx + 1, dtype=torch.int64, device=self.device
        )

        # create spacegroup tensor
        if not self.hparams.conditioning.spacegroup or spacegroups_bincount is None:
            # null spacegroup
            spacegroup = torch.zeros(batch_size, dtype=torch.int64, device=self.device)
        else:
            # sample random spacegroups from distribution: (B, 1)
            spacegroup = torch.multinomial(
                spacegroups_bincount.float(),
                batch_size,
                replacement=True,
            ).to(self.device)

        # create token mask for visualization
        token_mask = torch.zeros(
            batch_size,
            max(sample_lengths),
            dtype=torch.bool,
            device=self.device,
        )
        for idx, length in enumerate(sample_lengths):
            token_mask[idx, :length] = True

        # create new samples from interpolant (num_timesteps 可大于默认以提升质量，如 QM9)
        samples = self.interpolant.sample_with_classifier_free_guidance(
            batch_size=batch_size,
            num_tokens=max(sample_lengths),
            emb_dim=self.denoiser.d_x,
            model=self.denoiser,
            dataset_idx=dataset_idx,
            spacegroup=spacegroup,
            cfg_scale=cfg_scale,
            num_timesteps=num_timesteps,
            token_mask=token_mask,
        )
        # log π(τ): 轨迹的 log prob，用初始噪声 x_0 的 log p(x_0) 作为 DSPO 所需
        log_pi_tau = None
        if return_log_pi:
            import math
            x_0 = samples["tokens_traj"][0]  # (2B, N, d) when CFG else (B, N, d)
            if x_0.shape[0] == 2 * batch_size:
                x_0 = x_0[:batch_size]
            d = x_0.shape[-1]
            log_2pi = math.log(2 * math.pi)
            log_p_per_token = -0.5 * ((x_0 ** 2).sum(dim=-1) + d * log_2pi)
            log_p_per_token = log_p_per_token * token_mask.float()
            log_pi_tau = log_p_per_token.sum(dim=1)
            n_valid = token_mask.sum(dim=1)
            log_pi_tau = torch.where(n_valid > 0, log_pi_tau, torch.zeros_like(log_pi_tau))
        # get final samples and remove padding (to PyG format)
        x = samples["clean_traj"][-1][token_mask]

        # 完整去噪轨迹：每条样本的 tokens_traj（含 x_0..x_1）与 clean_traj（每步 pred），供完整轨迹 DSPO
        traj_tokens_per_sample = None
        traj_clean_per_sample = None
        if return_trajectory:
            d_x_out = self.denoiser.d_x
            traj_tokens_per_sample = []
            traj_clean_per_sample = []
            for i in range(batch_size):
                tok_list = []
                for t in range(len(samples["tokens_traj"])):
                    xt = samples["tokens_traj"][t]
                    if xt.shape[0] == 2 * batch_size:
                        xt = xt[:batch_size]
                    val = xt[i][token_mask[i]].detach().cpu().clone()
                    # 保证每步为 2D (N_i, d_x)，供 run_groups 存 pt 与 DSPO 训练一致
                    if val.dim() > 2:
                        val = val.reshape(val.size(0), -1)[:, :d_x_out].contiguous()
                    elif val.size(-1) > d_x_out:
                        val = val[..., :d_x_out].contiguous()
                    tok_list.append(val)
                traj_tokens_per_sample.append(tok_list)
                clean_list = []
                for t in range(len(samples["clean_traj"])):
                    xc = samples["clean_traj"][t]
                    if xc.shape[0] == 2 * batch_size:
                        xc = xc[:batch_size]
                    val = xc[i][token_mask[i]].detach().cpu().clone()
                    if val.dim() > 2:
                        val = val.reshape(val.size(0), -1)[:, :d_x_out].contiguous()
                    elif val.size(-1) > d_x_out:
                        val = val[..., :d_x_out].contiguous()
                    clean_list.append(val)
                traj_clean_per_sample.append(clean_list)

        # Debug: Check latent variable statistics and normalize if needed
        x_mean = x.mean().item()
        x_std = x.std().item()
        x_min = x.min().item()
        x_max = x.max().item()
        if self.global_rank == 0:
            log.debug(f"LDM generated latent stats: mean={x_mean:.4f}, std={x_std:.4f}, min={x_min:.4f}, max={x_max:.4f}")
            # Expected: mean≈0, std≈1 (if matching standard normal distribution)
        
        # Check if latent variables are in reasonable range
        # VAE expects latent variables to be approximately standard normal (mean≈0, std≈1)
        # If they deviate significantly, normalize them
        if abs(x_mean) > 0.5 or x_std < 0.5 or x_std > 2.0:
            if self.global_rank == 0:
                log.debug(f"Latent variables deviate from standard normal! Normalizing...")
            # Normalize to standard normal distribution
            x = (x - x.mean()) / (x.std() + 1e-8)
            if self.global_rank == 0:
                log.debug(f"After normalization: mean={x.mean().item():.4f}, std={x.std().item():.4f}")
        
        # CRITICAL FIX: If latent variables are too narrow (std < 0.1), add noise to increase diversity
        # This happens when LDM generates very similar latent codes for all tokens
        if x_std < 0.1:
            if self.global_rank == 0:
                log.debug(f"Latent variables too narrow (std={x_std:.4f})! Adding noise to increase diversity...")
            # Add noise proportional to the expected standard deviation
            # Scale noise based on how narrow the distribution is
            noise_scale = max(0.5, 1.0 - x_std)  # More noise if std is smaller
            x = x + torch.randn_like(x) * noise_scale
            if self.global_rank == 0:
                log.debug(f"After adding noise: mean={x.mean().item():.4f}, std={x.std().item():.4f}")

        batch = {
            "x": x,
            "num_atoms": sample_lengths,
            "batch": torch.repeat_interleave(
                torch.arange(len(sample_lengths), device=self.device), sample_lengths
            ),
            "token_idx": (torch.cumsum(token_mask, dim=-1, dtype=torch.int64) - 1)[token_mask],
        }
        # decode samples to crystal structures using frozen decoder
        out = self.autoencoder.decode(batch)
        
        # Post-process decoder output to fix issues
        # CRITICAL FIX: If fractional coordinates are too concentrated, add random noise
        frac_coords_before = out["frac_coords"]
        frac_coords_std_before = frac_coords_before.std().item()
        
        if frac_coords_std_before < 0.1:  # If coordinates are too concentrated
            if self.global_rank == 0:
                log.debug(f"Fractional coordinates too concentrated (std={frac_coords_std_before:.4f})! Adding noise...")
            # Add random noise to break up the concentration
            # Use uniform noise in a reasonable range
            noise = (torch.rand_like(out["frac_coords"]) - 0.5) * 0.5  # Uniform noise in [-0.25, 0.25]
            out["frac_coords"] = out["frac_coords"] + noise
        
        # 1. Normalize fractional coordinates to [0, 1) range
        out["frac_coords"] = out["frac_coords"] % 1.0
        out["frac_coords"] = torch.clamp(out["frac_coords"], 0.0, 0.9999)
        
        # 2. Ensure minimum lattice lengths
        out["lengths"] = torch.clamp(out["lengths"], min=2.0)
        
        # 3. Ensure angles are in valid range (already in radians from decoder)
        out["angles"] = torch.clamp(out["angles"], min=0.5, max=2.6)  # ~30-150 degrees in radians
        
        # Debug: Check decoder output statistics
        if self.global_rank == 0:
            # Check fractional coordinates distribution
            frac_coords_mean = out["frac_coords"].mean().item()
            frac_coords_std = out["frac_coords"].std().item()
            frac_coords_min = out["frac_coords"].min().item()
            frac_coords_max = out["frac_coords"].max().item()
            log.debug(f"Decoder frac_coords stats (after post-processing): mean={frac_coords_mean:.4f}, std={frac_coords_std:.4f}, min={frac_coords_min:.4f}, max={frac_coords_max:.4f}")
            # Expected: frac_coords should be in [0, 1) range
            
            # Check if all atoms have the same fractional coordinates (overlap issue)
            # Sample a few structures to check
            start_idx = 0
            overlap_count = 0
            for idx_in_batch, num_atom in enumerate(batch["num_atoms"][:10].tolist()):  # Check first 10 structures
                _frac_coords = out["frac_coords"].narrow(0, start_idx, num_atom)
                # Check if all atoms have similar coordinates (indicating overlap)
                coord_std = _frac_coords.std(dim=0)
                if coord_std.max().item() < 0.01:  # All atoms very close together
                    overlap_count += 1
                    if idx_in_batch < 3:  # Only log first 3
                        log.debug(f"Structure {idx_in_batch}: All atoms have very similar frac_coords (std={coord_std.max().item():.6f}), indicating overlap!")
                start_idx += num_atom
            if overlap_count > 0:
                log.debug(f"{overlap_count}/10 structures have overlapping atoms (coord_std < 0.01)")
        
        if return_log_pi:
            # dataset_idx_cond 为传入的 0/1/3/4，spacegroup 为 (B,) 供 DSPO 存轨迹用
            # return_trajectory 时额外返回 traj_tokens_per_sample, traj_clean_per_sample（每条样本完整轨迹）
            if return_trajectory and traj_tokens_per_sample is not None and traj_clean_per_sample is not None:
                return out, batch, samples, log_pi_tau, dataset_idx_cond, spacegroup, traj_tokens_per_sample, traj_clean_per_sample
            return out, batch, samples, log_pi_tau, dataset_idx_cond, spacegroup
        return out, batch, samples

    #####################################################################################################

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        # Verify VAE decoder weights are loaded correctly (check here when global_rank is available)
        if self.global_rank == 0:
            decoder_params = sum(p.numel() for p in self.autoencoder.decoder.parameters())
            post_quant_params = sum(p.numel() for p in self.autoencoder.post_quant_conv.parameters())
            log.info(f"[VAE Check] Decoder parameters: {decoder_params:,}")
            log.info(f"[VAE Check] post_quant_conv parameters: {post_quant_params:,}")
            # Check if decoder weights are non-zero (not randomly initialized)
            decoder_weight_norm = sum(p.norm().item() for p in self.autoencoder.decoder.parameters())
            log.info(f"[VAE Check] Decoder weight norm: {decoder_weight_norm:.2f}")
            if decoder_weight_norm < 1e-6:
                log.error("[VAE Check] WARNING: Decoder weights are near zero! Checkpoint may not be loaded correctly!")
            
            # Check a sample weight to verify it's not all zeros
            sample_weight = next(iter(self.autoencoder.decoder.parameters()))
            log.info(f"[VAE Check] Sample decoder weight stats: mean={sample_weight.mean().item():.6f}, std={sample_weight.std().item():.6f}, min={sample_weight.min().item():.6f}, max={sample_weight.max().item():.6f}")
        
        try:
            # Clear cache for Equiformer SO3 embeddings
            self.autoencoder.encoder.mappingReduced.device = self.device
            self.autoencoder.encoder.mappingReduced.mask_indices_cache = None
            self.autoencoder.encoder.mappingReduced.rotate_inv_rescale_cache = None
            for rotation_module in self.autoencoder.encoder.SO3_rotation:
                rotation_module.mapping.device = self.device
                rotation_module.mapping.mask_indices_cache = None
                rotation_module.mapping.rotate_inv_rescale_cache = None
            log.info("Clear Equiformer checkpoint SO3 rotation mapping cache.")
        except AttributeError:
            pass

        if self.hparams.compile and stage == "fit":
            self.autoencoder = torch.compile(self.autoencoder)
            self.denoiser = torch.compile(self.denoiser)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            # 对于 diffusion，通常我们关心的是晶体/材料数据集上的生成质量。
            # 如果存在 perov5 验证指标，则优先使用 `val_perov5/valid_rate` 作为调度器监控指标；
            # 否则退回到 mp20。
            monitor_metric = "val_mp20/valid_rate"
            if hasattr(self, "val_metrics") and "perov5" in self.val_metrics:
                monitor_metric = "val_perov5/valid_rate"

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": monitor_metric,
                    "interval": "epoch",
                    "frequency": self.hparams.scheduler_frequency,
                },
            }
        return {"optimizer": optimizer}
