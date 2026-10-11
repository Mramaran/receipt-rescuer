"""Train the restoration network on synthetically faded receipt patches.

Example:
    python train.py --data data/sroie --name run1 --iters 30000

Writes runs/<name>/config.json (all settings and the seed), log.csv, and the
weights. The exponential-moving-average weights are the ones used at inference.
"""
import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from rescuer.data import ReceiptPatches, list_images, split
from rescuer.model import RescueNet, count_params


def charbonnier(pred, target, eps=1e-3):
    return torch.sqrt((pred - target) ** 2 + eps * eps).mean()


def psnr(a, b):
    mse = torch.mean((a.clamp(0, 1) - b) ** 2).item()
    return 99.0 if mse == 0 else -10 * np.log10(mse)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    out_psnr, in_psnr, n = 0.0, 0.0, 0
    for lq, gt in loader:
        lq, gt = lq.to(device), gt.to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            pred = model(lq)
        out_psnr += psnr(pred.float(), gt)
        in_psnr += psnr(lq, gt)
        n += 1
    return float(out_psnr / n), float(in_psnr / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sroie")
    ap.add_argument("--name", default="run1")
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patch", type=int, default=192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-patches", type=int, default=200)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run = Path("runs") / args.name
    run.mkdir(parents=True, exist_ok=True)

    paths = list_images(args.data)
    if not paths:
        raise SystemExit(f"no images under {args.data}; run scripts/download_data.py first")
    train_paths, val_paths = split(paths)
    (run / "val_files.txt").write_text("\n".join(p.name for p in val_paths), encoding="utf-8")

    model = RescueNet(width=args.width).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    cfg = vars(args) | {"params": count_params(model), "device": str(device),
                        "train_images": len(train_paths), "val_images": len(val_paths),
                        "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (run / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(json.dumps(cfg))

    train_ds = ReceiptPatches(train_paths, patch=args.patch, length=args.iters * args.batch)
    val_ds = ReceiptPatches(val_paths, patch=256, length=args.val_patches, fixed=True, seed=args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch, num_workers=args.workers,
                          persistent_workers=args.workers > 0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=8, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.9))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters, eta_min=1e-6)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    log = (run / "log.csv").open("w", newline="", encoding="utf-8")
    writer = csv.writer(log)
    writer.writerow(["iter", "loss", "lr", "val_psnr", "val_psnr_input", "elapsed_s"])

    torch.backends.cudnn.benchmark = True
    t0, best, running, count = time.time(), -1.0, 0.0, 0
    model.train()
    for it, (lq, gt) in enumerate(train_dl, start=1):
        lq, gt = lq.to(device, non_blocking=True), gt.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            loss = charbonnier(model(lq).float(), gt)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - args.ema)
        running += loss.item()
        count += 1
        avg = running / count

        if it % 100 == 0:
            print(f"it {it:6d}  loss {avg:.4f}  {it / (time.time() - t0):.2f} it/s", flush=True)
            if it % args.val_every:
                writer.writerow([it, f"{avg:.5f}", f"{sched.get_last_lr()[0]:.2e}", "", "",
                                 f"{time.time() - t0:.0f}"])
        if it % args.val_every == 0 or it == args.iters:
            v, vin = validate(ema, val_dl, device)
            writer.writerow([it, f"{avg:.5f}", f"{sched.get_last_lr()[0]:.2e}",
                             f"{v:.3f}", f"{vin:.3f}", f"{time.time() - t0:.0f}"])
            log.flush()
            print(f"   val PSNR {v:.2f} dB (faded input {vin:.2f} dB)", flush=True)
            state = {"model": ema.state_dict(), "width": args.width, "iter": it, "val_psnr": float(v)}
            torch.save(state, run / "last.pt")
            if v > best:
                best = v
                torch.save(state, run / "best.pt")
            model.train()
        if it % 100 == 0:
            running, count = 0.0, 0
        if it >= args.iters:
            break
    log.close()
    print(f"done: best val PSNR {best:.2f} dB, weights in {run / 'best.pt'}")


if __name__ == "__main__":
    main()
