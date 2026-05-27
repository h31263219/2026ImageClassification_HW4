"""Inference with Test-Time Augmentation (TTA) -> pred.npz.

TTA averages the model output over the 8-symmetry dihedral group D4:
  identity, hflip, vflip, hvflip, rot90, rot90+hflip, rot180, rot270.
Each transform is applied to the input, the model runs, and the inverse
transform is applied to the output before averaging.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TestDataset
from model import build_promptir
from utils import load_ckpt


# 8 dihedral group transforms and their inverses for TTA.
# Each tuple: (forward_fn, inverse_fn, name)
def _flip_h(x): return torch.flip(x, dims=[-1])
def _flip_v(x): return torch.flip(x, dims=[-2])
def _flip_hv(x): return torch.flip(x, dims=[-1, -2])
def _rot90(x): return torch.rot90(x, k=1, dims=(-2, -1))
def _rot180(x): return torch.rot90(x, k=2, dims=(-2, -1))
def _rot270(x): return torch.rot90(x, k=3, dims=(-2, -1))
def _rot90_then_h(x): return torch.flip(torch.rot90(x, k=1, dims=(-2, -1)), dims=[-1])

# Inverse rotations: rot90^-1 = rot270, rot180^-1 = rot180, rot270^-1 = rot90
def _identity(x): return x


TTA_8 = [
    (_identity, _identity, "id"),
    (_flip_h, _flip_h, "flipH"),
    (_flip_v, _flip_v, "flipV"),
    (_flip_hv, _flip_hv, "flipHV"),
    (_rot90, _rot270, "rot90"),
    (_rot180, _rot180, "rot180"),
    (_rot270, _rot90, "rot270"),
    # rot90 then hflip -> inverse is hflip then rot270
    (_rot90_then_h, lambda x: torch.rot90(torch.flip(x, dims=[-1]), k=3, dims=(-2, -1)), "rot90H"),
]

TTA_4 = [
    (_identity, _identity, "id"),
    (_flip_h, _flip_h, "flipH"),
    (_flip_v, _flip_v, "flipV"),
    (_flip_hv, _flip_hv, "flipHV"),
]

TTA_1 = [(_identity, _identity, "id")]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint.")
    p.add_argument("--model-config", type=str, default="light",
                   choices=["light", "medium", "standard", "large"],
                   help="Must match the checkpoint's architecture.")
    p.add_argument("--test-dir", type=str,
                   default="hw4_realse_dataset/test/degraded")
    p.add_argument("--output", type=str, default="pred.npz")
    p.add_argument("--tta", type=str, default="8", choices=["1", "4", "8"],
                   help="TTA group: 1 (no TTA), 4 (D2 flips), or 8 (D4 dihedral).")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    return p.parse_args()


@torch.no_grad()
def predict(model, degraded, tta_set, use_amp=True):
    """Return prediction averaged over TTA, shape (B, 3, H, W) in [0, 1]."""
    accum = None
    for fwd, inv, _name in tta_set:
        x = fwd(degraded)
        with torch.amp.autocast("cuda", enabled=use_amp):
            y = model(x)
        y = inv(y).float()
        accum = y if accum is None else accum + y
    return (accum / len(tta_set)).clamp_(0.0, 1.0)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tta_set = {"1": TTA_1, "4": TTA_4, "8": TTA_8}[args.tta]
    print(f"Device: {device}, TTA: {args.tta}-way")

    model = build_promptir(args.model_config).to(device)
    epoch, best_psnr = load_ckpt(args.ckpt, model, map_location=device)
    print(f"Loaded ckpt: {args.ckpt} (epoch={epoch}, best_psnr={best_psnr:.3f})")
    model.eval()

    test_set = TestDataset(args.test_dir)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    out_dict: dict[str, np.ndarray] = {}
    for fnames, degraded in tqdm(test_loader, desc="infer"):
        degraded = degraded.to(device, non_blocking=True)
        pred = predict(model, degraded, tta_set, use_amp=args.amp)
        pred_u8 = (pred * 255.0 + 0.5).clamp_(0, 255).to(torch.uint8).cpu().numpy()
        for i, fname in enumerate(fnames):
            out_dict[fname] = pred_u8[i]  # shape (3, H, W)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out_dict)
    print(f"Saved {len(out_dict)} predictions to {out_path}")


if __name__ == "__main__":
    main()
