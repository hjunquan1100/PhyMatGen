"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from lightning.pytorch.callbacks import TQDMProgressBar


class EpochAwareTQDMProgressBar(TQDMProgressBar):
    """TQDM 进度条，在每轮第一个 batch 时把描述设为当前 epoch，避免验证结束后仍显示上一轮。"""

    def on_train_batch_start(self, trainer, *args, **kwargs):
        super().on_train_batch_start(trainer, *args, **kwargs)
        # 训练条在父类里多为 main_progress_bar，在第一个 batch 时刷新为当前 epoch
        bar = getattr(self, "main_progress_bar", None) or getattr(
            self, "_train_tqdm", None
        )
        if bar is not None:
            bar.set_description(f"Epoch {trainer.current_epoch}")
