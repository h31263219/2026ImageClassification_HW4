"""Paired (degraded, clean) dataset for rain/snow image restoration."""
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _list_pairs(root: Path):
    """Return list of (degraded_path, clean_path, degradation_type) tuples."""
    degraded_dir = root / "degraded"
    clean_dir = root / "clean"
    pairs = []
    for fname in sorted(os.listdir(degraded_dir)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        # fname like "rain-1.png" or "snow-1.png"
        stem, ext = os.path.splitext(fname)
        deg_type, idx = stem.split("-", 1)
        clean_name = f"{deg_type}_clean-{idx}{ext}"
        clean_path = clean_dir / clean_name
        if not clean_path.is_file():
            raise FileNotFoundError(f"Missing clean image for {fname}: {clean_path}")
        pairs.append((str(degraded_dir / fname), str(clean_path), deg_type))
    return pairs


def split_pairs(pairs, val_per_class=50, seed=42):
    """Class-balanced train/val split."""
    by_type = {}
    for p in pairs:
        by_type.setdefault(p[2], []).append(p)
    rng = random.Random(seed)
    train_pairs, val_pairs = [], []
    for deg_type, lst in by_type.items():
        idxs = list(range(len(lst)))
        rng.shuffle(idxs)
        val_idx = set(idxs[:val_per_class])
        for i, item in enumerate(lst):
            (val_pairs if i in val_idx else train_pairs).append(item)
    rng.shuffle(train_pairs)
    return train_pairs, val_pairs


def _to_chw_float(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))  # CHW


class PairedRestorationDataset(Dataset):
    """Paired degraded/clean dataset with optional cropping and flipping augmentation.

    Returns float tensors in [0, 1] of shape (3, H, W).
    """

    def __init__(self, pairs, patch_size=128, augment=True, full_image=False):
        self.pairs = pairs
        self.patch_size = patch_size
        self.augment = augment
        self.full_image = full_image  # if True, return whole image without crop/flip

    def __len__(self):
        return len(self.pairs)

    def _augment(self, degraded: np.ndarray, clean: np.ndarray):
        # Random crop
        _, h, w = degraded.shape
        ps = self.patch_size
        if h > ps and w > ps:
            top = random.randint(0, h - ps)
            left = random.randint(0, w - ps)
            degraded = degraded[:, top:top + ps, left:left + ps]
            clean = clean[:, top:top + ps, left:left + ps]
        # Random horizontal flip
        if random.random() < 0.5:
            degraded = degraded[:, :, ::-1].copy()
            clean = clean[:, :, ::-1].copy()
        # Random vertical flip
        if random.random() < 0.5:
            degraded = degraded[:, ::-1, :].copy()
            clean = clean[:, ::-1, :].copy()
        # Random 90-degree rotation (k=0,1,2,3)
        k = random.randint(0, 3)
        if k:
            degraded = np.rot90(degraded, k=k, axes=(1, 2)).copy()
            clean = np.rot90(clean, k=k, axes=(1, 2)).copy()
        return degraded, clean

    def __getitem__(self, idx):
        deg_path, clean_path, deg_type = self.pairs[idx]
        degraded = _to_chw_float(Image.open(deg_path))
        clean = _to_chw_float(Image.open(clean_path))

        if not self.full_image and self.augment:
            degraded, clean = self._augment(degraded, clean)

        return (
            torch.from_numpy(degraded),
            torch.from_numpy(clean),
            deg_type,
        )


class TestDataset(Dataset):
    """Test set: degraded images only, no ground truth."""

    def __init__(self, test_dir):
        self.test_dir = Path(test_dir)
        self.files = sorted(
            [f for f in os.listdir(self.test_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = Image.open(self.test_dir / fname)
        arr = _to_chw_float(img)
        return fname, torch.from_numpy(arr)
