"""评价指标与患者级聚类 bootstrap。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import warnings


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred) / y_true) * 100.0)


def hit_rate(y_true: np.ndarray, y_pred: np.ndarray, rel: float = 0.10) -> float:
    return float(np.mean(np.abs(y_true - y_pred) <= rel * y_true))


def hit_rate_abs(y_true: np.ndarray, y_pred: np.ndarray, tol: float = 5.0) -> float:
    return float(np.mean(np.abs(y_true - y_pred) <= tol))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 2 and np.std(y_pred) > 0 else float("nan")
        spearman = float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")) if np.std(y_pred) > 0 else float("nan")
    return {
        "n": int(len(y_true)),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape_pct": mape(y_true, y_pred),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "pearson_r": pearson,
        "spearman_r": spearman,
        "hit_within_10pct": hit_rate(y_true, y_pred, 0.10),
        "hit_within_5kg": hit_rate_abs(y_true, y_pred, 5.0),
        "hit_within_10kg": hit_rate_abs(y_true, y_pred, 10.0),
        "bias": float(np.mean(y_pred - y_true)),
        "loa_lower": float(np.mean(y_pred - y_true) - 1.96 * np.std(y_pred - y_true, ddof=1)),
        "loa_upper": float(np.mean(y_pred - y_true) + 1.96 * np.std(y_pred - y_true, ddof=1)),
    }


def cluster_bootstrap_ci(
    df: pd.DataFrame,
    metric_fn,
    cluster_col: str = "subject_id",
    n_boot: int = 1000,
    seed: int = 20261004,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """患者级聚类 bootstrap：返回 (point, lower, upper)。"""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby(cluster_col, sort=False)]
    n_groups = len(groups)
    point = metric_fn(df)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_groups, n_groups)
        sample = pd.concat([groups[i] for i in idx], ignore_index=True)
        stats.append(metric_fn(sample))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def metric_fn_factory(kind: str, pred_col: str = "pred", true_col: str = "weight_kg"):
    if kind == "mae":
        return lambda d: mae(d[true_col].values, d[pred_col].values)
    if kind == "rmse":
        return lambda d: rmse(d[true_col].values, d[pred_col].values)
    if kind == "hit10pct":
        return lambda d: hit_rate(d[true_col].values, d[pred_col].values, 0.10)
    raise ValueError(kind)


def bootstrap_ci_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    clusters: np.ndarray,
    metric: str = "mae",
    n_boot: int = 1000,
    seed: int = 20261004,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """快速患者级聚类 bootstrap（纯 numpy，避免 DataFrame 拼接开销）。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    sorted_inv = inv[order]
    boundaries = np.searchsorted(sorted_inv, np.arange(len(uniq) + 1))
    groups = [order[boundaries[i]:boundaries[i + 1]] for i in range(len(uniq))]
    n_groups = len(groups)

    def _metric(idx: np.ndarray) -> float:
        yt, yp = y_true[idx], y_pred[idx]
        if metric == "mae":
            return float(np.mean(np.abs(yt - yp)))
        if metric == "rmse":
            return float(np.sqrt(np.mean((yt - yp) ** 2)))
        if metric == "hit10pct":
            return float(np.mean(np.abs(yt - yp) <= 0.10 * yt))
        raise ValueError(metric)

    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    all_idx = np.arange(len(y_true))
    for b in range(n_boot):
        pick = rng.integers(0, n_groups, n_groups)
        stats[b] = _metric(np.concatenate([groups[p] for p in pick]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return _metric(all_idx), float(lo), float(hi)
