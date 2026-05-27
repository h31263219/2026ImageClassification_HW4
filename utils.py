"""Utility functions: PSNR, checkpointing, seeding."""
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def psnr_torch(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Per-image PSNR. pred/target are (B, 3, H, W) in [0, 1]. Returns (B,)."""
    diff = pred.clamp(0.0, max_val) - target.clamp(0.0, max_val)
    mse = diff.pow(2).mean(dim=(1, 2, 3))
    # Avoid log(0)
    mse = mse.clamp(min=1e-12)
    return 10.0 * torch.log10((max_val ** 2) / mse)


def psnr_numpy(pred: np.ndarray, target: np.ndarray, max_val: float = 255.0) -> float:
    """PSNR for uint8 arrays in (C, H, W) or (H, W, C). pred and target same shape."""
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    mse = np.mean((pred - target) ** 2)
    if mse < 1e-12:
        return 99.0
    return 10.0 * float(np.log10((max_val ** 2) / mse))


def save_ckpt(path, model, optimizer=None, scheduler=None, scaler=None, epoch=0, best_psnr=0.0, extra=None):
    state = {
        "model": model.state_dict(),
        "epoch": epoch,
        "best_psnr": best_psnr,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    if extra is not None:
        state["extra"] = extra
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_ckpt(path, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu"):
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    return state.get("epoch", 0), state.get("best_psnr", 0.0)


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(1, self.count)
