"""精度模拟（研究方案 9.5）：患者级聚类下 MAE 的 95% CI 半宽。"""

from __future__ import annotations

import json

import numpy as np

from common import TABLE_DIR, save_json, setup_logger


def simulate(n_stays: int, n_patients: int, sd: float, icc: float,
             n_boot: int = 2000, seed: int = 20261004) -> dict:
    rng = np.random.default_rng(seed)
    mean_stays = n_stays / n_patients
    de = 1.0 + (mean_stays - 1.0) * icc
    # 患者随机效应 + 半正态误差
    patient_sd = sd * np.sqrt(max(icc, 0.0))
    resid_sd = sd * np.sqrt(max(1.0 - icc, 0.0))
    widths = []
    for _ in range(n_boot):
        counts = rng.multinomial(n_stays, np.full(n_patients, 1.0 / n_patients))
        counts = counts[counts > 0]
        n = len(counts)
        u = rng.normal(0.0, patient_sd, n)
        err = np.abs(rng.normal(0.0, resid_sd, n))
        # 患者级平均绝对误差（保守近似：患者效应取绝对值 + 残差半正态）
        per_patient = np.abs(u) + err
        mae = per_patient.mean()
        se = per_patient.std(ddof=1) / np.sqrt(n)
        widths.append(1.96 * se)
    return {
        "n_stays": n_stays, "n_patients": n_patients, "sd_kg": sd, "icc": icc,
        "design_effect": de,
        "mae_ci_halfwidth_mean": float(np.mean(widths)),
        "mae_ci_halfwidth_p05": float(np.quantile(widths, 0.05)),
        "mae_ci_halfwidth_p95": float(np.quantile(widths, 0.95)),
    }


def main() -> None:
    log = setup_logger("precision_simulation")
    rows = []
    for sd in (6.0, 10.0, 15.0):
        for icc in (0.3, 0.5):
            rows.append(simulate(8767, 7166, sd, icc))
    save_json(TABLE_DIR / "precision_simulation.json", {"rows": rows})
    for r in rows:
        log.info("sd=%.0f icc=%.1f DE=%.2f -> CI halfwidth mean=%.3f kg (p05-p95 %.3f-%.3f)",
                 r["sd_kg"], r["icc"], r["design_effect"],
                 r["mae_ci_halfwidth_mean"], r["mae_ci_halfwidth_p05"], r["mae_ci_halfwidth_p95"])


if __name__ == "__main__":
    main()
