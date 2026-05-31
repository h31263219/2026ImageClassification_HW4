"""Generate all figures used in the HW4 report.

Inputs:
  output/train_log.jsonl              (light + L1, original baseline)
  output_no_prompt/train_log.jsonl    (light WITHOUT PGB, §4 ablation)
  output_medium_charb/train_log.jsonl (medium + Charbonnier patch128)
  output_medium_p192/train_log.jsonl  (medium + Charbonnier + patch192 — SUBMITTED)
  output_medium_p192/best.pt          (best ckpt of submitted model)

Outputs (saved to figures/):
  fig_training_curves.png    — main model (medium-p192) loss + per-type val PSNR
  fig_iterative_gains.png    — light → medium → +patch192 val PSNR trajectories
  fig_ablation_compare.png   — §4 PGB on/off (bigger subplots, user request)
  fig_per_type_gap.png       — rain vs snow PGB gap
  fig_qualitative.png        — degraded / predicted (medium-p192) / clean
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import _list_pairs, split_pairs, PairedRestorationDataset
from model import build_promptir
from utils import load_ckpt

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


def load_log(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def fig_training_curves(log: list[dict]) -> None:
    """Submitted model: train loss + val PSNR (avg + per type). Larger figure."""
    eps = [r["epoch"] for r in log]
    train_loss = [r["train_loss"] for r in log]
    val_psnr = [r.get("val_psnr") for r in log]
    val_rain = [r["val_by_type"]["rain"] for r in log if "val_by_type" in r]
    val_snow = [r["val_by_type"]["snow"] for r in log if "val_by_type" in r]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    ax1.plot(eps, train_loss, color="#1f77b4", linewidth=1.5)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Training Charbonnier loss", fontsize=12)
    ax1.set_title("Training loss (PromptIR-medium, patch192, Charbonnier)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=11)

    ax2.plot(eps, val_psnr, label="val PSNR (avg)", color="black", linewidth=2.2)
    ax2.plot(eps, val_rain, label="val PSNR (rain)", color="#1f77b4", alpha=0.85, linewidth=1.4)
    ax2.plot(eps, val_snow, label="val PSNR (snow)", color="#d62728", alpha=0.85, linewidth=1.4)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Validation PSNR (dB)", fontsize=12)
    ax2.set_title("Validation PSNR by degradation type", fontsize=12)
    ax2.legend(loc="lower right", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=11)

    plt.tight_layout()
    out = FIG_DIR / "fig_training_curves.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def fig_iterative_gains(light_log, med_log, p192_log) -> None:
    """Three-model overlay showing the iterative improvement story."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 7.5))

    for log, label, color in [
        (light_log, "light (14.5M) + L1 + patch128 — test 29.75", "#999999"),
        (med_log,   "medium (24.88M) + Charbonnier + patch128 — test 30.10", "#ff7f0e"),
        (p192_log,  "medium + Charbonnier + patch192 — test 30.78  ← submitted", "#1f77b4"),
    ]:
        eps = [r["epoch"] for r in log]
        vp = [r["val_psnr"] for r in log]
        ax.plot(eps, vp, label=label, color=color, linewidth=2.2)

    ax.axhline(29.75, color="#999999", linestyle=":", alpha=0.6, linewidth=1.2)
    ax.axhline(30.10, color="#ff7f0e", linestyle=":", alpha=0.6, linewidth=1.2)
    ax.axhline(30.78, color="#1f77b4", linestyle=":", alpha=0.6, linewidth=1.2)
    ax.set_xlabel("Epoch", fontsize=16)
    ax.set_ylabel("Validation PSNR (dB, single-pass)", fontsize=16)
    ax.set_title("Iterative improvement on the validation set\n(each design change isolates one architectural lever)",
                 fontsize=17)
    ax.legend(loc="lower right", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=14)

    plt.tight_layout()
    out = FIG_DIR / "fig_iterative_gains.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def fig_ablation_compare(main_log: list[dict], abl_log: list[dict]) -> None:
    """§4 PGB on/off ablation — same data as before but BIGGER (user requested)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    def series(log):
        return [r["epoch"] for r in log], [r["val_psnr"] for r in log]

    em, ym = series(main_log)
    ea, ya = series(abl_log)

    ax1.plot(em, ym, label="With PGB (14.50M params)", color="#1f77b4", linewidth=1.7)
    ax1.plot(ea, ya, label="Without PGB (8.47M params)", color="#d62728", linestyle="--", linewidth=1.7)
    ax1.set_xlabel("Epoch", fontsize=13)
    ax1.set_ylabel("Validation PSNR (dB)", fontsize=13)
    ax1.set_title("Val PSNR vs epoch (PromptIR-light ablation)", fontsize=13)
    ax1.legend(loc="lower right", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=12)

    # Smoothed gap
    n = min(len(ym), len(ya))
    gap = np.array(ym[:n]) - np.array(ya[:n])
    window = 5
    if n >= window:
        smooth = np.convolve(gap, np.ones(window) / window, mode="valid")
        ax2.plot(em[window - 1:n], smooth, color="#2ca02c", linewidth=1.7)
    else:
        ax2.plot(em[:n], gap, color="#2ca02c", linewidth=1.7)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("Epoch", fontsize=13)
    ax2.set_ylabel("Gap (dB) — with PGB minus without", fontsize=13)
    ax2.set_title("PGB contribution per epoch (5-epoch smoothed)", fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=12)

    plt.tight_layout()
    out = FIG_DIR / "fig_ablation_compare.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def fig_per_type_gap(main_log: list[dict], abl_log: list[dict]) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))

    def series(log, key):
        return ([r["epoch"] for r in log if "val_by_type" in r],
                [r["val_by_type"][key] for r in log if "val_by_type" in r])

    er_m, vr_m = series(main_log, "rain")
    er_a, vr_a = series(abl_log, "rain")
    es_m, vs_m = series(main_log, "snow")
    es_a, vs_a = series(abl_log, "snow")

    ax.plot(er_m, vr_m, label="rain — with PGB", color="#1f77b4", linewidth=1.7)
    ax.plot(er_a, vr_a, label="rain — no PGB", color="#1f77b4", linestyle="--", alpha=0.7, linewidth=1.4)
    ax.plot(es_m, vs_m, label="snow — with PGB", color="#d62728", linewidth=1.7)
    ax.plot(es_a, vs_a, label="snow — no PGB", color="#d62728", linestyle="--", alpha=0.7, linewidth=1.4)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation PSNR (dB)", fontsize=12)
    ax.set_title("Per-degradation val PSNR (rain vs snow, with/without PGB) — light config", fontsize=12)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    out = FIG_DIR / "fig_per_type_gap.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


@torch.no_grad()
def fig_qualitative(n_per_type: int = 3) -> None:
    """Side-by-side using the SUBMITTED model (medium-p192)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = _list_pairs(Path("hw4_realse_dataset/train"))
    _, val_pairs = split_pairs(pairs, val_per_class=50, seed=42)
    pick = {"rain": [], "snow": []}
    for p in val_pairs:
        dt = p[2]
        if len(pick[dt]) < n_per_type:
            pick[dt].append(p)
    sel = pick["rain"] + pick["snow"]

    model = build_promptir("medium", use_prompt=True).to(device)
    load_ckpt("output_medium_p192/best.pt", model, map_location=device)
    model.eval()

    val_set = PairedRestorationDataset(sel, augment=False, full_image=True)
    rows = len(sel)
    fig, axes = plt.subplots(rows, 3, figsize=(10, 3.2 * rows))
    if rows == 1:
        axes = np.array([axes])

    for i in range(rows):
        item = val_set[i]
        degraded, clean, dt = item
        x = degraded.unsqueeze(0).to(device)
        with torch.amp.autocast("cuda", enabled=True):
            pred = model(x).clamp(0, 1)
        pred_np = pred[0].float().cpu().numpy().transpose(1, 2, 0)
        deg_np = degraded.numpy().transpose(1, 2, 0)
        clean_np = clean.numpy().transpose(1, 2, 0)

        def psnr(a, b):
            mse = ((a - b) ** 2).mean()
            return 99.0 if mse < 1e-10 else 10 * np.log10(1.0 / mse)

        axes[i, 0].imshow(deg_np)
        axes[i, 0].set_title(f"degraded ({dt})  PSNR={psnr(deg_np, clean_np):.2f}", fontsize=11)
        axes[i, 1].imshow(pred_np)
        axes[i, 1].set_title(f"predicted  PSNR={psnr(pred_np, clean_np):.2f}", fontsize=11)
        axes[i, 2].imshow(clean_np)
        axes[i, 2].set_title("clean (GT)", fontsize=11)
        for j in range(3):
            axes[i, j].axis("off")

    plt.tight_layout()
    out = FIG_DIR / "fig_qualitative.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def main():
    light_log = load_log("output/train_log.jsonl")
    abl_log = load_log("output_no_prompt/train_log.jsonl")
    med_log = load_log("output_medium_charb/train_log.jsonl")
    p192_log = load_log("output_medium_p192/train_log.jsonl")
    print(f"Loaded epochs: light={len(light_log)}, abl={len(abl_log)}, "
          f"medium-p128={len(med_log)}, medium-p192={len(p192_log)}")

    fig_training_curves(p192_log)              # submitted model curves
    fig_iterative_gains(light_log, med_log, p192_log)
    fig_ablation_compare(light_log, abl_log)   # §4 ablation (light config)
    fig_per_type_gap(light_log, abl_log)
    fig_qualitative()                          # uses medium-p192 best.pt


if __name__ == "__main__":
    main()
