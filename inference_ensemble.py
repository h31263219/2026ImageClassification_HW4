"""Ensemble inference: average TTA-8 predictions from multiple checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TestDataset
from inference import TTA_8, predict
from model import build_promptir
from utils import load_ckpt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True,
                   help="One or more checkpoint paths to ensemble.")
    p.add_argument("--use-prompt", nargs="+", type=int, required=True,
                   help="For each ckpt: 1 if it was trained with PGB, 0 if --no-prompt.")
    p.add_argument("--model-config", type=str, default="light")
    p.add_argument("--configs", nargs="+", default=None,
                   help="Per-checkpoint configs. Length must match --ckpts. Overrides --model-config.")
    p.add_argument("--test-dir", type=str, default="hw4_realse_dataset/test/degraded")
    p.add_argument("--output", type=str, default="pred_ensemble.npz")
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()
    if args.configs is None:
        args.configs = [args.model_config] * len(args.ckpts)
    assert len(args.configs) == len(args.ckpts) == len(args.use_prompt)
    return args


@torch.no_grad()
def main():
    args = parse_args()
    assert len(args.ckpts) == len(args.use_prompt), "--ckpts and --use-prompt length mismatch"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Ensembling {len(args.ckpts)} checkpoints on {device}")

    models = []
    for ckpt, up, cfg in zip(args.ckpts, args.use_prompt, args.configs):
        m = build_promptir(cfg, use_prompt=bool(up)).to(device)
        ep, bp = load_ckpt(ckpt, m, map_location=device)
        m.eval()
        print(f"  - {ckpt} (config={cfg}, use_prompt={bool(up)}, ep={ep}, val_psnr={bp:.3f})")
        models.append(m)

    test_set = TestDataset(args.test_dir)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    out_dict = {}
    for fnames, degraded in tqdm(test_loader, desc="ensemble"):
        degraded = degraded.to(device, non_blocking=True)
        preds = []
        for m in models:
            preds.append(predict(m, degraded, TTA_8, use_amp=True))
        avg = torch.stack(preds).mean(dim=0).clamp_(0, 1)
        pred_u8 = (avg * 255.0 + 0.5).clamp_(0, 255).to(torch.uint8).cpu().numpy()
        for i, fname in enumerate(fnames):
            out_dict[fname] = pred_u8[i]

    out_path = Path(args.output)
    np.savez(out_path, **out_dict)
    print(f"Saved {len(out_dict)} ensemble predictions to {out_path}")


if __name__ == "__main__":
    main()
