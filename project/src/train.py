"""训练回归模型：CNN 基线与 Transformer 主模型。"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import FEATURE_DIR, PROCESSED_DIR, TABLE_DIR, load_config, setup_logger
from dataset import WeightCXR
from metrics import regression_metrics
from models import build_model, count_parameters, normalization_for


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, optimizer, scaler, device, train: bool, amp: bool, head: str = "point",
              target_mean: float = 0.0, target_std: float = 1.0, criterion=None):
    model.train(train)
    losses, preds, trues = [], [], []
    point_loss = criterion if criterion is not None else nn.SmoothL1Loss(beta=1.0)
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y_raw = batch["weight"].to(device, non_blocking=True)
        y = (y_raw - target_mean) / target_std
        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=amp):
                out = model(x).squeeze(-1)
                if head == "hetero":
                    mu, logvar = out[..., 0], out[..., 1]
                    loss = 0.5 * (torch.exp(-logvar) * (y - mu) ** 2 + logvar).mean()
                    pred = mu
                else:
                    loss = point_loss(out, y)
                    pred = out
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        losses.append(float(loss.detach().cpu()))
        preds.append((pred.detach().float().cpu().numpy() * target_std) + target_mean)
        trues.append(y_raw.detach().float().cpu().numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    metrics = regression_metrics(y_true, y_pred)
    metrics["loss"] = float(np.mean(losses))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--split", default="random", choices=["random", "temporal"])
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--head", default="point", choices=["point", "hetero"])
    ap.add_argument("--loss", default="huber", choices=["huber", "l1", "mse"])
    ap.add_argument("--augment-profile", default="default", choices=["default", "geom_off"])
    ap.add_argument("--split-file", default=None, help="切分文件名（默认按 split 参数选择）")
    ap.add_argument("--split-col", default=None, help="切分列名（默认 split_random/split_temporal）")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--no-test", action="store_true",
                    help="跳过训练结束时的测试集评价（模型搜索阶段使用）")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = load_config()
    log = setup_logger(f"train_{args.backbone}_{args.split}_s{args.seed}")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s backbone=%s size=%d split=%s seed=%d", device, args.backbone, args.size, args.split, args.seed)

    mean, std = normalization_for(args.backbone)
    common_ds = dict(split_kind=args.split, size=args.size, mean=mean, std=std,
                     split_file=args.split_file, split_col=args.split_col)
    train_ds = WeightCXR(args.train_split, train=True, augment_profile=args.augment_profile, **common_ds)
    val_ds = WeightCXR(args.val_split, train=False, **common_ds)
    test_ds = WeightCXR("test", train=False, **common_ds)
    log.info("sizes: train=%d val=%d test=%d", len(train_ds), len(val_ds), len(test_ds))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    target_mean = float(train_ds.df["weight_kg"].mean())
    target_std = float(train_ds.df["weight_kg"].std())
    log.info("target standardization: mean=%.2f kg, std=%.2f kg", target_mean, target_std)

    model = build_model(args.backbone, out_dim=2 if args.head == "hetero" else 1,
                        img_size=args.size).to(device)
    log.info("trainable params: %.2fM", count_parameters(model) / 1e6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    criterion = {"huber": nn.SmoothL1Loss(beta=1.0), "l1": nn.L1Loss(), "mse": nn.MSELoss()}[args.loss]

    ckpt_dir = FEATURE_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.backbone}_{args.split}_s{args.seed}"
    best_path = ckpt_dir / f"{tag}.pt"
    history, best = [], {"val_mae": float("inf"), "epoch": -1}
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, optimizer, scaler, device, True, args.amp, args.head,
                       target_mean, target_std, criterion)
        va = run_epoch(model, val_loader, optimizer, scaler, device, False, args.amp, args.head,
                       target_mean, target_std, criterion)
        scheduler.step()
        row = {"epoch": epoch, "lr": scheduler.get_last_lr()[0],
               "train_mae": tr["mae"], "train_loss": tr["loss"],
               "val_mae": va["mae"], "val_rmse": va["rmse"],
               "val_hit10": va["hit_within_10pct"], "seconds": time.time() - t0}
        history.append(row)
        log.info("epoch %d | train_mae %.3f | val_mae %.3f | val_hit10 %.3f | %.0fs",
                 epoch, tr["mae"], va["mae"], va["hit_within_10pct"], row["seconds"])
        if va["mae"] < best["val_mae"]:
            best = {"val_mae": va["mae"], "epoch": epoch, "val_metrics": va}
            torch.save({"model": model.state_dict(), "args": vars(args), "best": best,
                        "target_mean": target_mean, "target_std": target_std}, best_path)
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                log.info("early stop at epoch %d", epoch)
                break

    pd.DataFrame(history).to_csv(TABLE_DIR / f"history_{tag}.csv", index=False)
    if args.no_test:
        test = None
        log.info("skipping test evaluation (--no-test)")
    else:
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        test = run_epoch(model, test_loader, optimizer, scaler, device, False, args.amp, args.head,
                         target_mean, target_std, criterion)
    payload = {"tag": tag, "args": vars(args), "best": best, "test": test,
               "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
               "target_mean": target_mean, "target_std": target_std}
    with open(TABLE_DIR / f"metrics_{tag}.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    if test is not None:
        log.info("TEST %s", json.dumps(test, ensure_ascii=False))


if __name__ == "__main__":
    main()
