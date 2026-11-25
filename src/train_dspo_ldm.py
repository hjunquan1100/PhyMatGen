"""DSPO 微调 LDM：用动态过滤的 pt 组（高/低分）做 advantage-weighted 训练，验证用 joint 四类。"""
import logging
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from src.runtime_env import configure_runtime, ensure_conda_lib_on_loader_path
except ModuleNotFoundError:
    from runtime_env import configure_runtime, ensure_conda_lib_on_loader_path

ensure_conda_lib_on_loader_path()
configure_runtime()

import torch

try:
    from src.torch_compat import ensure_torch_dynamo_compat
except ModuleNotFoundError:
    from torch_compat import ensure_torch_dynamo_compat

ensure_torch_dynamo_compat()

import hydra
import lightning as L
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def _add_file_log_handler(log_dir: str, filename: str = "train.log") -> None:
    """把 root logger 的日志同时写入运行目录下的文件，便于保留完整训练记录。"""
    log_path = Path(log_dir) / filename
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        root = logging.getLogger()
        for h in getattr(root, "handlers", []):
            if getattr(h, "baseFilename", None) == str(log_path):
                return
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(root.level if root.level else logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        root.addHandler(fh)
        log.info("Training log will also be written to %s", log_path)
    except Exception as e:
        log.warning("Could not add file log handler to %s: %s", log_path, e)


def _save_metrics_json(trainer: Trainer, path: str) -> None:
    """将当前 epoch 的验证指标（含四类）追加到 JSON。"""
    import json
    cb = trainer.checkpoint_callback
    metrics = {"epoch": trainer.current_epoch, "global_step": trainer.global_step}
    for k, v in trainer.callback_metrics.items():
        if isinstance(v, torch.Tensor) and v.numel() == 1:
            metrics[k] = v.item()
        elif isinstance(v, (int, float)):
            metrics[k] = v
    if cb and getattr(cb, "best_model_path", None):
        metrics["best_model_path"] = str(cb.best_model_path)
    out = Path(path)
    records = []
    if out.exists():
        try:
            with open(out) as f:
                records = json.load(f)
        except Exception:
            records = []
    records.append(metrics)
    with open(out, "w") as f:
        json.dump(records, f, indent=2)


class DSPOMetricsJSONCallback(Callback):
    """每轮验证结束后将四类指标写入 run 目录下 metrics.json。"""

    def __init__(self, save_path: Optional[str] = None):
        self.save_path = save_path

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.global_rank != 0:
            return
        path = self.save_path or (trainer.log_dir and str(Path(trainer.log_dir) / "metrics.json"))
        if path:
            _save_metrics_json(trainer, path)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # 把训练日志同时写入运行目录，方便查看完整记录（不被进度条覆盖）
    out_dir = OmegaConf.select(cfg, "hydra.runtime.output_dir") or ""
    if out_dir:
        _add_file_log_handler(out_dir)

    log.info(f"Instantiating datamodule <{cfg.data.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=True
    )

    log.info(f"Instantiating diffusion model <{cfg.diffusion_module._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.diffusion_module)

    # 仅加载预训练权重（不恢复 epoch/step/optimizer），用于从 epoch=0 重新 finetune
    pretrained_ckpt_path = cfg.get("pretrained_ckpt_path")
    if pretrained_ckpt_path:
        log.info(f"Loading pretrained weights (weights-only) from: {pretrained_ckpt_path}")
        # PyTorch>=2.6: torch.load 默认 weights_only=True（安全模式），会拒绝反序列化自定义类
        # 这里 ckpt 来自本项目训练，视为可信，显式关闭 weights_only 以读取 state_dict
        try:
            ckpt = torch.load(pretrained_ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(pretrained_ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            log.warning(
                "Missing keys when loading pretrained weights (showing up to 20): %s",
                missing[:20],
            )
        if unexpected:
            log.warning(
                "Unexpected keys when loading pretrained weights (showing up to 20): %s",
                unexpected[:20],
            )

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(
        cfg.get("callbacks"), using_logger=(len(logger) > 0)
    )
    if cfg.get("train"):
        out_dir = OmegaConf.select(cfg, "hydra.runtime.output_dir") or ""
        if out_dir:
            callbacks.append(DSPOMetricsJSONCallback(save_path=str(Path(out_dir) / "metrics.json")))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    # 实例化时排除 trainer 下的自定义键，避免传给 Trainer() 导致 TypeError
    trainer_cfg = OmegaConf.create(OmegaConf.to_container(cfg.trainer, resolve=True))
    if "ddp_timeout_seconds" in trainer_cfg:
        del trainer_cfg["ddp_timeout_seconds"]
    trainer: Trainer = hydra.utils.instantiate(trainer_cfg, callbacks=callbacks, logger=logger)

    # DDP：延长 NCCL 集合超时，避免验证结束时 barrier 因某 rank 在 get_metrics（如 QM9）中过久而超时
    ddp_timeout_seconds = OmegaConf.select(cfg, "trainer.ddp_timeout_seconds")
    if ddp_timeout_seconds is not None and hasattr(trainer.strategy, "timeout"):
        trainer.strategy.timeout = timedelta(seconds=int(ddp_timeout_seconds))
        log.info(f"DDP strategy timeout set to {ddp_timeout_seconds}s")

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }
    if logger:
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting DSPO training!")
        # 如果是 finetune（只加载权重），不要把 ckpt_path 传给 fit，否则会 resume 到旧 epoch
        fit_ckpt_path = cfg.get("ckpt_path")
        if pretrained_ckpt_path:
            fit_ckpt_path = None
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=fit_ckpt_path)

    metric_dict = trainer.callback_metrics
    if cfg.get("test"):
        ckpt_path = getattr(trainer.checkpoint_callback, "best_model_path", None) or ""
        if not ckpt_path:
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        metric_dict = {**metric_dict, **trainer.callback_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_dspo_ldm.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    extras(cfg)
    metric_dict, _ = train(cfg)
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric", "val_valid_rate")
    )
    return metric_value


if __name__ == "__main__":
    try:
        import lovely_tensors as lt
        lt.monkey_patch()
    except ImportError:
        pass
    main()
