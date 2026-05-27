"""Training script for PromptIR on rain+snow image restoration."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PairedRestorationDataset, _list_pairs, split_pairs
from model import build_promptir
from utils import AverageMeter, load_ckpt, psnr_torch, save_ckpt, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str,
                   default="hw4_realse_dataset/train",
                   help="Train root containing degraded/ and clean/ subfolders.")
    p.add_argument("--output-dir", type=str, default="output")
    p.add_argument("--model-config", type=str, default="light",
                   choices=["light", "medium", "standard", "large"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--val-per-class", type=int, default=50,
                   help="Validation images per degradation type.")
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", default=True,
                   help="Use mixed precision (fp16) training.")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from.")
    p.add_argument("--max-iters-per-epoch", type=int, default=0,
                   help="If >0, limit batches per epoch (for smoke testing).")
    p.add_argument("--save-every", type=int, default=10,
                   help="Save 'latest' checkpoint every N epochs.")
    p.add_argument("--no-prompt", action="store_true", default=False,
                   help="Ablation: disable Prompt Generation Blocks (degenerates to Restormer).")
    p.add_argument("--loss", type=str, default="l1", choices=["l1", "charbonnier"],
                   help="Pixel loss: 'l1' (default) or 'charbonnier' = sqrt(x^2 + eps^2).")
    p.add_argument("--compile", action="store_true", default=False,
                   help="Wrap model in torch.compile() for ~1.3-2x speedup on transformer-heavy archs.")
    return p.parse_args()


def make_loaders(args):
    pairs = _list_pairs(Path(args.data_root))
    train_pairs, val_pairs = split_pairs(pairs, val_per_class=args.val_per_class, seed=args.seed)
    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    train_set = PairedRestorationDataset(train_pairs, patch_size=args.patch_size, augment=True)
    val_set = PairedRestorationDataset(val_pairs, augment=False, full_image=True)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False, num_workers=2, pin_memory=True,
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    psnr_meter = AverageMeter()
    psnr_by_type: dict[str, AverageMeter] = {}
    for degraded, clean, deg_type in loader:
        degraded = degraded.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        pred = model(degraded)
        ps = psnr_torch(pred, clean)
        for i, t in enumerate(deg_type):
            v = ps[i].item()
            psnr_meter.update(v)
            psnr_by_type.setdefault(t, AverageMeter()).update(v)
    return psnr_meter.avg, {k: m.avg for k, m in psnr_by_type.items()}


def train_one_epoch(model, loader, optimizer, scaler, device, criterion, max_iters=0, use_amp=True):
    model.train()
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    pbar = tqdm(loader, desc="train", leave=False)
    for it, (degraded, clean, _deg_type) in enumerate(pbar):
        if max_iters and it >= max_iters:
            break
        degraded = degraded.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(degraded)
            loss = criterion(pred, clean)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        with torch.no_grad():
            ps = psnr_torch(pred.detach().float(), clean).mean().item()
        loss_meter.update(loss.item(), n=degraded.size(0))
        psnr_meter.update(ps, n=degraded.size(0))
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", psnr=f"{psnr_meter.avg:.2f}")
    return loss_meter.avg, psnr_meter.avg


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, AMP: {args.amp}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save args
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    train_loader, val_loader = make_loaders(args)

    model = build_promptir(args.model_config, use_prompt=not args.no_prompt).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tag = f"{args.model_config}{' (no-prompt)' if args.no_prompt else ''}{' (compiled)' if args.compile else ''}"
    print(f"Model: {tag} | params: {n_params / 1e6:.2f}M")
    if args.compile:
        model = torch.compile(model)

    # AdamW with cosine schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    if args.loss == "charbonnier":
        eps = 1e-6
        def criterion(pred, target):
            return torch.sqrt((pred - target) ** 2 + eps).mean()
    else:
        criterion = nn.L1Loss()

    start_epoch = 0
    best_psnr = 0.0
    if args.resume:
        start_epoch, best_psnr = load_ckpt(args.resume, model, optimizer, scheduler, scaler, map_location=device)
        print(f"Resumed from {args.resume} @ epoch {start_epoch}, best PSNR {best_psnr:.3f}")
        start_epoch += 1

    log_path = out_dir / "train_log.jsonl"
    log_fp = open(log_path, "a", encoding="utf-8")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]
        tr_loss, tr_psnr = train_one_epoch(
            model, train_loader, optimizer, scaler, device, criterion,
            max_iters=args.max_iters_per_epoch, use_amp=args.amp,
        )
        scheduler.step()

        val_record = {}
        if (epoch + 1) % args.val_every == 0:
            val_psnr, val_by_type = evaluate(model, val_loader, device)
            val_record = {"val_psnr": val_psnr, "val_by_type": val_by_type}
            print(
                f"[Epoch {epoch+1}/{args.epochs}] lr={lr_now:.2e} "
                f"loss={tr_loss:.4f} train_psnr={tr_psnr:.2f} "
                f"val_psnr={val_psnr:.3f} ({val_by_type}) "
                f"time={time.time()-t0:.1f}s"
            )
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_ckpt(out_dir / "best.pt", model, optimizer, scheduler, scaler,
                          epoch=epoch, best_psnr=best_psnr,
                          extra={"model_config": args.model_config})
                print(f"  -> new best: {best_psnr:.3f} (saved best.pt)")
        else:
            print(
                f"[Epoch {epoch+1}/{args.epochs}] lr={lr_now:.2e} "
                f"loss={tr_loss:.4f} train_psnr={tr_psnr:.2f} "
                f"time={time.time()-t0:.1f}s"
            )

        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            save_ckpt(out_dir / "latest.pt", model, optimizer, scheduler, scaler,
                      epoch=epoch, best_psnr=best_psnr,
                      extra={"model_config": args.model_config})

        record = {
            "epoch": epoch + 1,
            "lr": lr_now,
            "train_loss": tr_loss,
            "train_psnr": tr_psnr,
            "time_s": time.time() - t0,
            **val_record,
        }
        log_fp.write(json.dumps(record) + "\n")
        log_fp.flush()

    log_fp.close()
    print(f"\nDone. Best val PSNR: {best_psnr:.3f}")


if __name__ == "__main__":
    main()
