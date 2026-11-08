"""等待 ResNet-50 @308 三个种子训练完成 → 自动评测 → 运行消融分析 → 输出判定结论。

判定规则（由项目负责人指定，事先固定）
    若 ResNet-50 @308 三种子集成的 MAE 低于 Swin-T @308 三种子集成（8.15 kg），
    则主模型改为 ResNet-50 @308；否则维持 Swin-T @308 作为主模型。
    两种情形均同时报告配对差的 95% CI 与 Wilcoxon p 值。

产出
    project/results/tables/attention_ablation_decision.json
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from common import PROCESSED_DIR, TABLE_DIR, save_json, setup_logger

PY = sys.executable
TAGS = ["resnet50_308_s1", "resnet50_308_s2", "resnet50_308_s3"]
CKPT = Path("project/data/features/checkpoints")


def trained(tag: str) -> bool:
    return (CKPT / f"{tag}.pt").exists() and (TABLE_DIR / f"metrics_{tag}.json").exists()


def main() -> None:
    log = setup_logger("finalize_attention_ablation")
    log.info("等待 ResNet-50 @308 三个种子训练完成 ……")
    while True:
        done = [t for t in TAGS if trained(t)]
        if len(done) == len(TAGS):
            break
        log.info("已完成 %d/%d（%s）", len(done), len(TAGS), ", ".join(done) or "-")
        time.sleep(120)
    log.info("三个种子均已训练完成，开始逐种子评测")

    for tag in TAGS:
        out_csv = PROCESSED_DIR / f"test_predictions_{tag}.csv"
        if out_csv.exists():
            continue
        cmd = [PY, "project/src/evaluate.py", "--tag", tag,
               "--checkpoints", str(CKPT / f"{tag}.pt"),
               "--backbone", "resnet50", "--size", "308"]
        log.info("evaluate: %s", " ".join(cmd))
        subprocess.run(cmd, check=False)

    log.info("运行消融分析")
    subprocess.run([PY, "project/src/attention_ablation.py"], check=False)
    ab = json.loads((TABLE_DIR / "attention_ablation.json").read_text(encoding="utf-8"))
    swin = ab["ensembles"].get("swin-T @308 (3-seed ensemble)", {}).get("mae")
    resnet = ab["ensembles"].get("resnet50 @308 (3-seed ensemble)", {}).get("mae")
    cmp_ = ab["ensembles"].get("ensemble attention vs ensemble convolution", {})
    decision = {
        "swin_ensemble_mae": swin,
        "resnet_ensemble_mae": resnet,
        "paired_diff_attention_minus_convolution": cmp_.get("diff"),
        "paired_ci95": [cmp_.get("ci_low"), cmp_.get("ci_high")],
        "wilcoxon_p": cmp_.get("p_value"),
        "rule": "ResNet-50 @308 三种子集成 MAE 更低 → 主模型改为 ResNet-50 @308；否则维持 Swin-T @308",
    }
    if swin is not None and resnet is not None:
        decision["switch_main_model"] = bool(resnet < swin)
        decision["conclusion"] = ("改用 ResNet-50 @308 作为主模型" if resnet < swin
                                  else "维持 Swin-T @308 作为主模型（消融写入补充材料）")
    save_json(TABLE_DIR / "attention_ablation_decision.json", decision)
    log.info("判定结果：%s", json.dumps(decision, ensure_ascii=False))


if __name__ == "__main__":
    main()
