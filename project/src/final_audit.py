"""完成度审计：把研究方案的步骤映射到实际产物，输出缺失清单。"""

from __future__ import annotations

import json
from pathlib import Path

from common import (MAIN_SEED_TAG, MAIN_TAG, PROCESSED_DIR, PROJECT, TABLE_DIR, save_json,
                    setup_logger)

ROOT = PROJECT

CHECKS = [
    # (阶段, 要求, 证据路径/glob, 必需?)
    ("Stage1", "队列 A/B 与索引影像", ["data/processed/cohort_a_index.csv",
                                        "data/processed/cohort_a_all_images.csv",
                                        "data/processed/cohort_b_index.csv"], True),
    ("Stage1", "切分文件", ["data/processed/splits_random.csv",
                            "data/processed/splits_temporal.csv"], True),
    ("Stage1", "纳排计数与标签清洗", ["data/processed/cohort_flow.json",
                                      "data/processed/qc_summary.json"], True),
    ("Stage1", "数据字典", ["docs/data_dictionary.md"], True),
    ("Stage1", "分析计划冻结", ["docs/analysis_plan_frozen.md"], True),
    ("Stage2", "DICOM 清单", ["data/processed/image_manifest_index.csv",
                              "data/processed/image_manifest_all.csv"], True),
    ("Stage2", "影像缓存与 QC", ["data/processed/qc_images_index_320.csv",
                                 "results/tables/qc_image_summary.json"], True),
    ("Stage2", "人工复核样表", ["data/processed/qc_manual_review_sample.csv",
                                "results/figures/fig_qc_review_sheet.png"], False),
    ("Stage2", "人工复核工具包（QC）", ["docs/human_review/qc_review_template.csv",
                                        "docs/human_review/qc_review_sheet_numbered.png",
                                        "docs/human_review/README.md"], True),
    ("Stage2", "自动质控敏感性分析", ["results/tables/qc_sensitivity.json"], True),
    ("Stage2", "QC 一致性或豁免记录", ["docs/human_review/WAIVER.md"], True),
    ("Stage3", "基线结果", ["results/tables/baselines.json"], True),
    ("Stage3", "精度模拟", ["results/tables/precision_simulation.json"], True),
    ("Stage3", "CNN 基线训练", ["results/tables/metrics_resnet50_224_s1.json"], True),
    ("Stage3", "CNN 基线（目标标准化版）", ["results/tables/metrics_resnet50_224_s1_std.json"], True),
    ("Stage4", "主模型 3 seeds", [f"results/tables/metrics_{MAIN_SEED_TAG}_s1.json",
                                  f"results/tables/metrics_{MAIN_SEED_TAG}_s2.json",
                                  f"results/tables/metrics_{MAIN_SEED_TAG}_s3.json"], True),
    ("Stage4", "架构消融（卷积 vs 注意力）与主模型切换记录",
     ["results/tables/attention_ablation.json", "results/tables/attention_ablation_decision.json",
      "results/figures/figS_attention_ablation.png"], True),
    ("Stage4", "域预训练对照", ["results/tables/metrics_rad_dino_224_s1.json"], False),
    ("Stage5", "测试集评价与成功标准", [f"results/tables/evaluation_{MAIN_TAG}.json"], True),
    ("Stage5", "不确定性", [f"results/tables/uncertainty_{MAIN_TAG}.json"], True),
    ("Stage5", "构念效度", ["results/tables/construct_validity.json"], True),
    ("Stage5", "序列聚合", ["results/tables/sequence_model.json"], False),
    ("Stage5", "注意力/遮挡", ["results/tables/attention_occlusion_swin_ensemble.json"], False),
    ("Stage5", "CAM 掩膜人工复核工具包", ["docs/human_review/cam_review_template.csv",
                                          "docs/human_review/cam_review_sheet_numbered.png"], True),
    ("Stage5", "CAM 复核或豁免记录", ["docs/human_review/WAIVER.md"], True),
    ("Stage5", "CAM 修复记录", ["docs/cam_fix_record.md"], True),
    ("Stage5", "时间外验证", ["results/tables/metrics_swin_t_308_temporal_s1.json"], True),
    ("Stage6", "剂量模拟", [f"results/tables/dose_simulation_{MAIN_TAG}.json"], True),
    ("Stage6", "图表", ["results/figures/fig1_cohort_flow.png",
                        "results/figures/fig2_weight_distribution.png",
                        "results/figures/fig3_baselines.png",
                        f"results/figures/fig4_bland_altman_{MAIN_TAG}.png",
                        f"results/figures/fig5_subgroup_weight_{MAIN_TAG}.png"], True),
    ("Stage6", "结果分析报告", ["docs/final_report.md"], True),
    ("Stage6", "公式法分层失真分析", ["results/tables/formula_stratified_analysis.json",
                                      "results/figures/formula_stratified_analysis.png",
                                      "docs/formula_stratified_analysis.md"], True),
    ("Stage6", "事后补充分析（探索性）",
     ["results/tables/flagging_model_v2.json", "docs/unreliable_flagging_v2.md",
      "results/tables/no_reliable_record_analysis.json", "docs/no_reliable_record_analysis.md",
      "results/tables/multimodal_weight.json", "results/tables/multimodal_embed_mlp.json",
      "docs/multimodal_weight.md", "docs/submission_positioning.md",
      "docs/anthropometry_availability_check.md", "docs/circumference_future_analysis.md",
      "results/tables/journal_tiers_2025.json", "docs/journal_tiers_2025.md"], False),
    ("Stage6", "四本 2 区期刊投稿路线图", ["docs/submission_roadmap_2q.md"], False),
    ("Stage6", "亚组分析（方案冻结 + 执行）",
     ["docs/subgroup_analysis_plan.md", "results/tables/subgroup_analysis.json",
      "results/tables/subgroup_analysis_strata.csv", "results/figures/figS2_subgroups.png"], True),
    ("Stage6", "标签质量敏感性分析",
     ["docs/label_quality_sensitivity.md", "results/tables/label_quality_analysis.json"], True),
    ("Stage6", "决策曲线与剂量策略比较",
     ["docs/decision_curve_analysis.md", "results/tables/decision_curve_analysis.json",
      "results/tables/dose_strategy_comparison.csv",
      "results/figures/figS3_decision_curve.png"], True),
    ("Stage6", "已发表体检模型复刻（方案 A）",
     ["docs/vgg16_replication.md", "results/tables/vgg16_two_stage.json",
      "results/figures/vgg16_two_stage.png"], False),
    ("Stage7", "Insights into Imaging 投稿包（主干与合规清单）",
     ["../Insights_into_Imaging/README.md",
      "../Insights_into_Imaging/01_story_and_framework.md",
      "../Insights_into_Imaging/02_submission_compliance.md",
      "../Insights_into_Imaging/03_external_model_availability.md"], False),
    ("Stage7", "内部评审记录", ["docs/report_review_r1.md"], False),
    ("Stage7", "补充材料（ESM 1–3）",
     ["../Insights_into_Imaging/supplementary/ESM1_architecture_ablation.md",
      "../Insights_into_Imaging/supplementary/ESM2_subgroup_analysis.md",
      "../Insights_into_Imaging/supplementary/ESM3_model_attribution.md",
      "../Insights_into_Imaging/figures/figS1_architecture_ablation.png",
      "../Insights_into_Imaging/figures/figS2_subgroups.png",
      "../Insights_into_Imaging/figures/figS3_decision_curve.png",
      "../Insights_into_Imaging/figures/figS4_attention_regions.png"], True),
]


def main() -> None:
    log = setup_logger("final_audit")
    rows = []
    missing_required = []
    for stage, requirement, paths, required in CHECKS:
        status = {}
        for p in paths:
            full = ROOT / p
            status[p] = full.exists()
        ok = all(status.values())
        rows.append({"stage": stage, "requirement": requirement, "required": required,
                     "ok": ok, "paths": status})
        if required and not ok:
            missing_required.append(requirement)
    payload = {"n_checks": len(rows), "n_ok": sum(r["ok"] for r in rows),
               "missing_required": missing_required, "rows": rows}
    save_json(TABLE_DIR / "final_audit.json", payload)
    log.info("checks=%d ok=%d missing_required=%s", payload["n_checks"], payload["n_ok"],
             missing_required)
    for r in rows:
        if not r["ok"]:
            log.info("MISSING [%s] %s -> %s", r["stage"], r["requirement"],
                     [k for k, v in r["paths"].items() if not v])


if __name__ == "__main__":
    main()
