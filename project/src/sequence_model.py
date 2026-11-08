"""时间注意力聚合（研究方案 8.3）：对同一住院多张影像做 Transformer 聚合。"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import FEATURE_DIR, PROCESSED_DIR, TABLE_DIR, save_json, setup_logger
from metrics import regression_metrics


class SeqModel(nn.Module):
    def __init__(self, in_dim: int = 768, d_model: int = 128, nhead: int = 4, layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True,
                                         dim_feedforward=256, dropout=0.1)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x, mask=None):
        h = self.proj(x)
        h = self.encoder(h, src_key_padding_mask=mask)
        # 取最后一个有效时间步
        lengths = (~mask).sum(dim=1) - 1 if mask is not None else torch.full(
            (h.shape[0],), h.shape[1] - 1, device=h.device)
        last = h[torch.arange(h.shape[0]), lengths]
        return self.head(last).squeeze(-1)


def build_sequences(df: pd.DataFrame, feats: np.ndarray, max_len: int = 8):
    xs, ys, stays, subjects, last_dicom = [], [], [], [], []
    for stay_id, g in df.groupby("stay_id"):
        g = g.sort_values("study_datetime")
        if len(g) < 2:
            continue
        g = g.iloc[-max_len:]
        idx = g.index.values
        seq = feats[idx].astype(np.float32)
        xs.append(seq)
        ys.append(float(g["weight_kg"].values[-1]))
        stays.append(stay_id)
        subjects.append(int(g["subject_id"].values[-1]))
        last_dicom.append(g["dicom_id"].values[-1])
    return xs, np.array(ys), stays, subjects, last_dicom


def pad_batch(seqs):
    max_len = max(len(s) for s in seqs)
    dim = seqs[0].shape[1]
    x = np.zeros((len(seqs), max_len, dim), dtype=np.float32)
    mask = np.ones((len(seqs), max_len), dtype=bool)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = s
        mask[i, :len(s)] = False
    return torch.from_numpy(x), torch.from_numpy(mask)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="sequence_features.npy")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    log = setup_logger("sequence_model")
    feats = np.load(FEATURE_DIR / args.features).astype(np.float32)
    meta = pd.read_csv(PROCESSED_DIR / "sequence_features_index.csv")
    splits = pd.read_csv(PROCESSED_DIR / "splits_random.csv")
    df = meta.merge(splits, on="subject_id", how="left").reset_index(drop=True)
    df["split_random"] = df["split_random"].fillna("train")

    tr = df[df["split_random"] == "train"]
    va = df[df["split_random"] == "val"]
    te = df[df["split_random"] == "test"]
    tr_x, tr_y, _, _, _ = build_sequences(tr, feats)
    va_x, va_y, _, _, _ = build_sequences(va, feats)
    te_x, te_y, _, te_subj, te_last_dicom = build_sequences(te, feats)
    log.info("sequences: train=%d val=%d test=%d", len(tr_x), len(va_x), len(te_x))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SeqModel(in_dim=feats.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fn = nn.SmoothL1Loss(beta=1.0)
    best = {"val_mae": float("inf"), "state": None}
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(len(tr_x))
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            bx, bm = pad_batch([tr_x[j] for j in idx])
            by = torch.from_numpy(tr_y[idx]).float()
            bx, bm, by = bx.to(device), bm.to(device), by.to(device)
            pred = model(bx, bm)
            loss = loss_fn(pred, by)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            bx, bm = pad_batch(va_x)
            pred = model(bx.to(device), bm.to(device)).cpu().numpy()
        m = regression_metrics(va_y, pred)
        if m["mae"] < best["val_mae"]:
            best = {"val_mae": m["mae"], "state": {k: v.clone() for k, v in model.state_dict().items()}}
        if epoch % 5 == 0:
            log.info("epoch %d val_mae=%.3f val_hit10=%.3f", epoch, m["mae"], m["hit_within_10pct"])
    model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        bx, bm = pad_batch(te_x)
        pred = model(bx.to(device), bm.to(device)).cpu().numpy()
    out = {"n_test_sequences": len(te_x), "best_val_mae": best["val_mae"],
           "test": regression_metrics(te_y, pred)}
    pd.DataFrame({"subject_id": te_subj, "last_dicom_id": te_last_dicom,
                  "weight_kg": te_y, "pred": pred}).to_csv(
        PROCESSED_DIR / "sequence_test_predictions.csv", index=False)
    save_json(TABLE_DIR / "sequence_model.json", out)
    log.info(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
