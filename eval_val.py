"""Evaluate val PSNR for single model vs ensemble — for sanity-checking ensemble gains.

Mirrors inference.py's TTA-8 averaging but on val set (with GT available).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import PairedRestorationDataset, _list_pairs, split_pairs
from inference import TTA_8, predict
from model import build_promptir
from utils import load_ckpt, psnr_torch


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--use-prompt", nargs="+", type=int, required=True)
    p.add_argument("--model-config", type=str, default="light",
                   help="Single config applied to all ckpts (overridden by --configs).")
    p.add_argument("--configs", nargs="+", default=None,
                   help="Per-checkpoint configs (light/medium/standard/large). Length must match --ckpts.")
    p.add_argument("--data-root", type=str, default="hw4_realse_dataset/train")
    p.add_argument("--no-tta", action="store_true")
    args = p.parse_args()
    if args.configs is None:
        args.configs = [args.model_config] * len(args.ckpts)
    assert len(args.configs) == len(args.ckpts) == len(args.use_prompt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = _list_pairs(Path(args.data_root))
    _, val_pairs = split_pairs(pairs, val_per_class=50, seed=42)
    val_set = PairedRestorationDataset(val_pairs, augment=False, full_image=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)

    models = []
    for ckpt, up, cfg in zip(args.ckpts, args.use_prompt, args.configs):
        m = build_promptir(cfg, use_prompt=bool(up)).to(device)
        ep, bp = load_ckpt(ckpt, m, map_location=device)
        m.eval()
        print(f"  loaded {ckpt} (config={cfg}, use_prompt={bool(up)}, ep={ep}, val_psnr_ckpt={bp:.3f})")
        models.append(m)

    tta = [(lambda x: x, lambda x: x, "id")] if args.no_tta else TTA_8

    psnrs = []
    by_type = {"rain": [], "snow": []}
    for degraded, clean, dt in val_loader:
        degraded = degraded.to(device)
        clean = clean.to(device)
        preds = [predict(m, degraded, tta, use_amp=True) for m in models]
        avg = torch.stack(preds).mean(dim=0).clamp_(0, 1)
        ps = psnr_torch(avg, clean)
        for i, t in enumerate(dt):
            v = ps[i].item()
            psnrs.append(v)
            by_type[t].append(v)
    print(f"\n  Val PSNR (avg): {np.mean(psnrs):.4f} dB ({len(psnrs)} imgs)")
    for t, vs in by_type.items():
        print(f"    {t}: {np.mean(vs):.4f} dB ({len(vs)} imgs)")


if __name__ == "__main__":
    main()
