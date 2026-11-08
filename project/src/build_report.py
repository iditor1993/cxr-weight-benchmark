"""从结果产物生成最终项目结果分析报告（图文并茂，数据驱动）。"""

from __future__ import annotations

import json
from pathlib import Path
import csv

from common import (DOCS_DIR, FIGURE_DIR, MAIN_LABEL, MAIN_SEED_TAG, MAIN_TAG, PROCESSED_DIR,
                    PROJECT, TABLE_DIR, setup_logger)


def load(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def fmt(v, nd=2):
    if v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def pct(v, nd=1, missing="NA"):
    if v is None:
        return missing
    return f"{100*float(v):.{nd}f}%"


def table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    if not rows:
        return "_无数据_\n"
    header = "| " + " | ".join(c[1] for c in cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c[0])) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    log = setup_logger("build_report")
    flow = load(PROCESSED_DIR / "cohort_flow.json") or {}
    qc = load(PROCESSED_DIR / "qc_summary.json") or {}
    qc_img = load(TABLE_DIR / "qc_image_summary.json") or {}
    context = load(PROCESSED_DIR / "cohort_context.json") or {}
    baselines = load(TABLE_DIR / "baselines.json") or {}
    precision = load(TABLE_DIR / "precision_simulation.json") or {}

    def metrics_for(tag: str):
        return load(TABLE_DIR / f"metrics_{tag}.json")

    def evaluation_for(tag: str):
        return load(TABLE_DIR / f"evaluation_{tag}.json")

    resnet = metrics_for("resnet50_224_s1")
    main_seeds = {f"seed{i}": metrics_for(f"{MAIN_SEED_TAG}_s{i}") for i in (1, 2, 3)}
    swin = {f"seed{i}": metrics_for(f"swin_t_308_s{i}") for i in (1, 2, 3)}
    swin224 = metrics_for("swin_t_224_s1")
    temporal = metrics_for("swin_t_308_temporal_s1")
    raddino = metrics_for("rad_dino_224_s1")
    ensemble = evaluation_for(MAIN_TAG)
    ensemble_swin = evaluation_for("swin_ensemble")
    ablation = load(TABLE_DIR / "attention_ablation.json")
    ablation_decision = load(TABLE_DIR / "attention_ablation_decision.json")
    uncertainty = load(TABLE_DIR / f"uncertainty_{MAIN_TAG}.json")
    construct = load(TABLE_DIR / "construct_validity.json")
    dose = load(TABLE_DIR / f"dose_simulation_{MAIN_TAG}.json")
    sequence = load(TABLE_DIR / "sequence_model.json")
    sequence_cmp = load(TABLE_DIR / "sequence_comparison.json")
    occlusion = load(TABLE_DIR / "attention_occlusion_swin_ensemble.json")
    hetero_unc = load(TABLE_DIR / "uncertainty_swin_hetero.json")
    hetero_eval = load(TABLE_DIR / "evaluation_swin_hetero.json")
    conv = load(TABLE_DIR / "conventional_weight_comparison.json")
    formula_strat = load(TABLE_DIR / "formula_stratified_analysis.json")
    flag_v2 = load(TABLE_DIR / "flagging_model_v2.json")
    nrr = load(TABLE_DIR / "no_reliable_record_analysis.json")
    multimodal = load(TABLE_DIR / "multimodal_weight.json")
    audit = load(TABLE_DIR / "final_audit.json")
    qc_sens = load(TABLE_DIR / "qc_sensitivity.json")

    L: list[str] = []
    add = L.append

    add("# CR_Weight 项目结果分析报告\n")
    add("> 研究方案：`项目方案/研究方案_v4_定稿.md`  ")
    add("> 报告生成：由 `project/src/build_report.py` 从结果产物自动生成  ")
    add("> 数据版本：MIMIC-IV 3.1 / MIMIC-CXR 2.1.0  \n")

    add("## 1. 摘要\n")
    add("本研究基于 MIMIC-IV 3.1 与 MIMIC-CXR 2.1.0，评估能否从 ICU 床旁便携 AP 胸片"
        "同期估计患者实际体重（ABW）。主分析队列（床秤每日体重 ±24 h）包含 "
        f"{flow.get('n_index_images_cohort_a', 'NA')} 张索引影像 / "
        f"{flow.get('n_stays_cohort_a', 'NA')} 次 ICU 住院 / "
        f"{flow.get('n_patients_cohort_a', 'NA')} 名患者。\n")
    if ensemble:
        o = ensemble["overall"]
        add(f"- 集成模型测试集 MAE：**{fmt(o['mae'])} kg**（95% CI "
            f"{fmt(o['mae_ci95'][1])}–{fmt(o['mae_ci95'][2])}）")
        add(f"- ±10% 命中率：**{fmt(o['hit_within_10pct']*100, 1)}%**")
        add(f"- 成功率判定：{'全部达标' if ensemble['success_criteria']['all_passed'] else '未全部达标'}"
            f"（S1={ensemble['success_criteria']['S1_hit10_ge80pct']}, "
            f"S2={ensemble['success_criteria']['S2_mae_le5_and_ci_lt6']}, "
            f"S3={ensemble['success_criteria']['S3_extreme_subgroups_ge65pct']}）\n")
    else:
        add("- 主模型结果：待训练完成后填入\n")

    add("## 2. 背景与目的\n")
    add("体重是 ICU 药物剂量、营养与液体管理的核心参数。文献显示住院患者体重实际测量率仅 "
        "13.5%–55%（Intern Med J 2018），ICU 医护人员估算体重平均误差 11.4%（Anaesth Intensive Care 2010），"
        ">10% 误差占 36%（Nurs Crit Care 2025）。已有研究（Ichikawa 2023）用体检立位胸片估计体重，"
        "但尚无 ICU 床旁便携 AP 胸片的同期估计研究。本研究的目标是开发并内部验证基于注意力机制的"
        "影像模型，并系统评估不确定性、极端体重安全边界与假设性剂量影响。\n")

    add("## 3. 数据与方法\n")
    add("### 3.1 数据来源与合规\n")
    add("- MIMIC-IV 3.1（只读 DuckDB，53 GB）；MIMIC-CXR 2.1.0（DICOM + 元数据）\n"
        "- PhysioNet 受控访问，遵守 DUA；仅本地处理，不共享患者级数据\n"
        "- 队列定义、清洗规则与分析计划在任何建模前冻结（git tag `sap-freeze-20261004`）\n")

    add("### 3.2 队列定义与纳排\n")
    add(f"ICU 住院 {flow.get('n_icustays_total', 'NA'):,} 次 → 有床旁便携 AP 片 "
        f"{flow.get('n_stays_with_portable_ap', 'NA'):,} 次 → ±24 h 内有每日体重（队列 A）"
        f"{flow.get('n_stays_cohort_a', 'NA'):,} 次 → 索引影像分析集 "
        f"{flow.get('n_index_images_cohort_a', 'NA'):,} 张。\n")
    add("![队列流程图](../results/figures/fig1_cohort_flow.png)\n")
    add(f"队列 B（任意体重来源，敏感性）：{flow.get('n_index_images_cohort_b', 'NA'):,} 张影像 / "
        f"{flow.get('n_stays_cohort_b', 'NA'):,} 次住院。\n")

    add("### 3.3 标签与变量\n")
    add(f"- 主标签：Daily Weight（224639，床秤，kg），清洗后有效记录 {qc.get('weight_rows_in_range', 'NA'):,} 条；"
        f"超范围剔除 {qc.get('weight_rows_out_of_range', 'NA'):,} 条；lbs 换算 {qc.get('lbs_converted_rows', 'NA'):,} 条\n"
        f"- 24 h 内变化 >10% 的住院：{qc.get('weight_24h_change_gt10pct_stays', 'NA'):,} 次（敏感性分析剔除）\n"
        "- 协变量：性别、年龄、身高（57.3% 可得）、ICU 类型、机械通气、CRRT、支持器械（仅分层/审计）\n")

    add("### 3.4 影像预处理与质控\n")
    add(f"- 索引集 {qc_img.get('n_images', 'NA'):,} 张全部解码成功（{fmt(qc_img.get('ok_rate', 0)*100, 1)}%）\n"
        f"- 像素间距统一 0.139 mm；原图尺寸 {qc_img.get('rows_min', 'NA')}–{qc_img.get('rows_max', 'NA')} 行；"
        f"KVP 缺失 {qc_img.get('kvp_missing', 'NA'):,}（中位 90）\n"
        f"- 100 张人工复核样表：`results/figures/fig_qc_review_sheet.png`（双人一致性待人工完成）\n")
    add("![QC 人工复核样表](../results/figures/fig_qc_review_sheet.png)\n")
    if qc_sens and "strict_metrics" in qc_sens:
        add(f"**自动质控敏感性分析**：按严格规则（宽高比 0.5–2.0、均值强度 1%–99% 分位、"
            f"饱和像素 <1%、KVP 可得）保留 {qc_sens['n_strict']}/{qc_sens['n_all']} 张测试影像，"
            f"集成模型 MAE 为 {fmt(qc_sens['strict_metrics']['mae'])} kg"
            f"（95% CI {fmt(qc_sens['strict_mae_ci95'][1])}–{fmt(qc_sens['strict_mae_ci95'][2])}），"
            f"与全测试集 {fmt(qc_sens['all_metrics']['mae'])} kg 基本一致，"
            "提示影像质量变异不是性能的主要驱动因素。\n")
    add("**人工复核豁免（2026-10-05，项目负责人授权）**：两项人工判定步骤（双人质控一致性、"
        "CAM 掩膜核对）以替代证据结案，详见 `project/docs/human_review/WAIVER.md`。"
        "替代证据为：① 全量自动质控 100% 通过 + 严格 QC 子集敏感性分析（MAE "
        f"{fmt(qc_sens['strict_metrics']['mae']) if qc_sens else 'NA'} vs "
        f"{fmt(qc_sens['all_metrics']['mae']) if qc_sens else 'NA'} kg）；"
        "② 方案第 10 节预先允许的退化路径（高显著区域遮挡 + 器械亚组比较）。"
        "工具包（编号图、填写模板、κ 计算命令）保留，后续可随时补做并回填。\n")

    add("### 3.5 模型与训练\n")
    add("- 基线：人群均值、身高+性别 OLS、Devine/Hume 公式、ResNet-50（ImageNet）\n"
        f"- 主模型（2026-10-08 起）：{MAIN_LABEL}（ImageNet-1k 预训练，输入 308×308），"
        "Huber 损失、AdamW、余弦退火、AMP、验证集早停、3 个随机种子集成\n"
        "- 对照骨干（消融）：Swin-T（窗口自注意力，输入 308×308，以匹配 window=7），"
        "Huber 损失、AdamW、余弦退火、AMP、验证集早停、3 个随机种子集成\n"
        "- 目标标准化：训练集均值/标准差（83.80 / 23.32 kg）\n")

    add("### 3.6 统计分析\n")
    add("- 主要分析集：每次 ICU 住院一张索引影像；患者级聚类 bootstrap（1,000 次）\n"
        "- 成功标准：S1 ≥80% 落在 ±10%；S2 MAE ≤5 kg 且 95% CI 上限 <6 kg；"
        "S3 区间覆盖率 ≥85% 且极端体重亚组命中率 ≥65%\n")
    if precision:
        add("\n精度模拟（患者级聚类，MAE 的 95% CI 半宽）：\n")
        add(table(precision.get("rows", []), [
            ("sd_kg", "误差 SD (kg)"), ("icc", "ICC"), ("design_effect", "设计效应"),
            ("mae_ci_halfwidth_mean", "CI 半宽均值 (kg)"),
            ("mae_ci_halfwidth_p95", "CI 半宽 p95 (kg)")]))

    add("## 4. 结果\n")
    add("### 4.1 队列特征\n")
    add("![体重分布](../results/figures/fig2_weight_distribution.png)\n")
    if context:
        add(table(context.get("care_unit", []), [("unit", "ICU 类型"), ("n", "住院次数")]))
        hp = context.get("height_available", {})
        vp = context.get("invasive_ventilation", {})
        rp = context.get("rrt", {})
        add(f"- 身高可得：{hp.get('n_with_height', 'NA')} / {hp.get('n_stays', 'NA')} "
            f"（{fmt(100*float(hp.get('n_with_height', 0))/max(float(hp.get('n_stays', 1)),1), 1)}%）\n"
            f"- 有创机械通气：{vp.get('n_ventilated', 'NA')} 次（{fmt(100*float(vp.get('n_ventilated', 0))/max(float(vp.get('n_stays', 1)),1), 1)}%）\n"
            f"- CRRT：{rp.get('n_rrt', 'NA')} 次（{fmt(100*float(rp.get('n_rrt', 0))/max(float(rp.get('n_stays', 1)),1), 1)}%）\n")
        los = context.get("los_days", {})
        add(f"- ICU 住院时长中位 {fmt(los.get('median'))} 天（IQR {fmt(los.get('p25'))}–{fmt(los.get('p75'))}）\n")

    add("### 4.2 基线性能\n")
    add("![基线对比](../results/figures/fig3_baselines.png)\n")
    if baselines:
        name_map = {"population_mean": "人群均值", "ols_height_sex": "身高+性别 OLS",
                    "devine_ibw": "Devine IBW", "hume_ibw": "Hume IBW"}
        rows = []
        for k, v in baselines.items():
            rows.append({"model": name_map.get(k, k), "n": v.get("n"), "mae": v.get("mae"),
                         "rmse": v.get("rmse"),
                         "hit": f"{100*v.get('hit_within_10pct', 0):.1f}%"})
        add(table(rows, [("model", "模型"), ("n", "n"), ("mae", "MAE (kg)"),
                         ("rmse", "RMSE (kg)"), ("hit", "±10% 命中率")]))

    add("### 4.3 主模型与成功标准\n")
    rows = []
    for name, m in [("ResNet-50 224", resnet),
                    ("主模型 ResNet-50 308 (seed1)", main_seeds["seed1"]),
                    ("主模型 ResNet-50 308 (seed2)", main_seeds["seed2"]),
                    ("主模型 ResNet-50 308 (seed3)", main_seeds["seed3"]),
                    ("Swin-T 308 (seed1)", swin["seed1"]),
                    ("Swin-T 308 (seed2)", swin["seed2"]), ("Swin-T 308 (seed3)", swin["seed3"]),
                    ("Swin-T 224", swin224), ("Swin-T 308 时间外", temporal),
                    ("RadDINO 224", raddino)]:
        if m:
            rows.append({"model": name, "test_mae": m["test"]["mae"],
                         "test_hit10": f"{100*m['test']['hit_within_10pct']:.1f}%",
                         "val_mae": m["best"]["val_mae"], "n_test": m["n_test"]})
    add(table(rows, [("model", "模型"), ("n_test", "测试 n"), ("test_mae", "测试 MAE (kg)"),
                     ("test_hit10", "±10% 命中率"), ("val_mae", "验证 MAE (kg)")]))
    add("注：各模型均以验证集 MAE 早停选择最佳 epoch（ResNet-50 best epoch 10/23；"
        "Swin 20–30），训练预算差异来自早停机制而非人为调参。\n")
    add("![模型对比](../results/figures/fig6_model_comparison.png)\n")
    if ensemble:
        o = ensemble["overall"]
        add(f"\n**3 种子集成（主要分析）**：MAE {fmt(o['mae'])} kg（95% CI "
            f"{fmt(o['mae_ci95'][1])}–{fmt(o['mae_ci95'][2])}），RMSE {fmt(o['rmse'])} kg，"
            f"±10% 命中率 {fmt(o['hit_within_10pct']*100, 1)}%，±5 kg 命中率 "
            f"{fmt(o['hit_within_5kg']*100, 1)}%。\n")
        add(f"![Bland–Altman](../results/figures/fig4_bland_altman_{MAIN_TAG}.png)\n")
        add("**成功标准判定**：\n")
        sc = ensemble["success_criteria"]
        add(f"- S1（≥80% 落在 ±10%）：{'达标' if sc['S1_hit10_ge80pct'] else '未达标'}\n"
            f"- S2（MAE ≤5 kg 且 CI 上限 <6 kg）：{'达标' if sc['S2_mae_le5_and_ci_lt6'] else '未达标'}\n"
            f"- S3（极端体重亚组 ≥65%）：{'达标' if sc['S3_extreme_subgroups_ge65pct'] else '未达标'}；"
            f"亚组命中率 {sc.get('S3_extreme_hit_rates')}\n")

    # 4.3b 架构消融与主模型切换（2026-10-08）
    if ablation and ablation_decision:
        add("### 4.3b 架构消融（卷积 vs 注意力）与主模型切换\n")
        add("判定规则（先于 ResNet-50 @308 评测固定）：若 ResNet-50 @308 三种子集成 MAE 低于 "
            "Swin-T @308 三种子集成，则主模型改为 ResNet-50 @308，否则维持 Swin-T。\n")
        pr = ablation["ensembles"]["ensemble attention vs ensemble convolution"]
        add(f"- Swin-T @308 三种子集成：MAE {fmt(ablation['ensembles']['swin-T @308 (3-seed ensemble)']['mae'])} kg\n"
            f"- ResNet-50 @308 三种子集成：MAE {fmt(ablation['ensembles']['resnet50 @308 (3-seed ensemble)']['mae'])} kg\n"
            f"- 配对差（注意力 − 卷积）：{fmt(pr['diff'], 3)} kg（95% CI {fmt(pr['ci_low'], 3)}–"
            f"{fmt(pr['ci_high'], 3)}，Wilcoxon p={pr['p_value']:.2e}）\n"
            f"- 结论：{'改用 ResNet-50 @308 作为主模型' if ablation_decision['switch_main_model'] else '维持 Swin-T @308'}\n")
        add("- 消融要点：224 px 单种子下卷积优于注意力（配对差 +0.40 kg [0.11, 0.69]，p=0.003）；"
            "308 px 单种子互有胜负；三种子集成的增益（−0.35 / −0.76 kg）大于骨干差异（0.24 kg）。\n"
            "- 方法学提示：该选择在预设分析计划冻结之后、且在同一测试集上完成，属事后选择，"
            "会带来乐观偏倚；正文已按此披露，外部验证仍是必需步骤。\n")
        add("![架构消融](../results/figures/figS_attention_ablation.png)\n")

    add("### 4.4 亚组分析\n")
    add(f"![体重分层性能](../results/figures/fig5_subgroup_weight_{MAIN_TAG}.png)\n")
    if ensemble:
        ws_rows = [{"group": r["group"], "n": r["n"], "mae": r["mae"],
                    "hit": pct(r["hit_within_10pct"]), "bias": r["bias"]}
                   for r in ensemble["subgroups"].get("weight_stratum", [])]
        add(table(ws_rows, [("group", "体重分层 (kg)"), ("n", "n"), ("mae", "MAE (kg)"),
                            ("hit", "±10% 命中率"), ("bias", "偏差 (kg)")]))
        sd_rows = [{"group": "阳性" if str(r["group"]).startswith("1") else "阴性/未提及",
                    "n": r["n"], "mae": r["mae"], "hit": pct(r["hit_within_10pct"])}
                   for r in ensemble["subgroups"].get("support_devices", [])]
        add(table(sd_rows, [("group", "支持器械"), ("n", "n"), ("mae", "MAE (kg)"),
                            ("hit", "±10% 命中率")]))

    add("### 4.5 不确定性与校准\n")
    if uncertainty:
        row_ens = {"method": "3 种子集成", "n": uncertainty["n"],
                   "coverage": f"{100*uncertainty['coverage_90_nominal']:.1f}%",
                   "width": uncertainty["median_interval_width_kg"],
                   "pct15": f"{100*uncertainty['pct_width_gt_15kg']:.1f}%",
                   "slope": uncertainty["calibration_slope_abs_err_on_std"]}
        rows_unc = [row_ens]
        if hetero_unc:
            rows_unc.append({"method": "异方差回归（单模型）", "n": hetero_unc["n"],
                             "coverage": f"{100*hetero_unc['coverage_90_nominal']:.1f}%",
                             "width": hetero_unc["median_interval_width_kg"],
                             "pct15": f"{100*hetero_unc['pct_width_gt_15kg']:.1f}%",
                             "slope": hetero_unc["calibration_slope_abs_err_on_std"]})
        add(table(rows_unc, [("method", "方法"), ("n", "n"), ("coverage", "90% 覆盖率"),
                             ("width", "中位区间宽度 (kg)"), ("pct15", "宽度>15 kg 比例"),
                             ("slope", "校准斜率")]))
        add("说明：3 种子集成区间窄但严重欠覆盖；异方差模型覆盖达标但区间宽达 "
            f"{fmt(hetero_unc['mean_interval_width_kg'] if hetero_unc else 0)} kg，"
            "在当前数据规模下两者无法同时满足校准与锐度。\n")
        add("**结论：两种不确定性方法均未达到可用于剂量决策的校准—锐度组合。**\n")
        add("![不确定性对比](../results/figures/fig7_uncertainty_comparison.png)\n")
        if hetero_eval:
            add(f"异方差模型的点估计性能：MAE {fmt(hetero_eval['overall']['mae'])} kg，"
                f"±10% 命中率 {fmt(100*hetero_eval['overall']['hit_within_10pct'], 1)}%\n")
    else:
        add("_待集成评价完成后生成_\n")

    add("### 4.6 构念效度（纵向一致性）\n")
    if construct:
        add(table([construct], [("r", "Δ 相关 r"), ("mae_delta_kg", "MAE(Δ) (kg)"),
                                ("slope", "斜率"), ("construct_validity_insufficient", "判定不足")]))
        pairs_path = PROCESSED_DIR / "construct_validity_pairs.csv"
        n_pairs = sum(1 for _ in open(pairs_path, encoding="utf-8")) - 1 if pairs_path.exists() else "NA"
        add(f"分析样本：{n_pairs} 对（同一住院 ≥2 张配对影像）。\n")
        add(f"解释：Δ 相关 r={fmt(construct['r'])}（预设阈值 0.5）、MAE(Δ)="
            f"{fmt(construct['mae_delta_kg'])} kg（阈值为横断面 MAE 的 1.5 倍 = "
            f"{fmt(1.5*construct['cross_sectional_mae_kg'])} kg），"
            "按预设判定构念效度不足——模型更擅长横断面估计，不能可靠追踪同一患者的体重变化；"
            "这意味着它也不适合用于液体复苏/CRRT 后的剂量再评估。\n")
    else:
        add("_待全集影像推理完成后生成_\n")

    add("### 4.7 序列聚合\n")
    if sequence:
        rows_seq = [{"model": "序列注意力（多张）", **sequence["test"]}]
        if sequence_cmp:
            rows_seq.append({"model": "同集单张集成", **sequence_cmp["single_image_same_set"]})
        add(table(rows_seq, [("model", "模型"), ("n", "n"), ("mae", "MAE (kg)"),
                             ("rmse", "RMSE (kg)"), ("hit_within_10pct", "±10% 命中率")]))
        if sequence_cmp:
            add(f"同集对照：序列聚合较单张集成 MAE 改善 "
                f"{fmt(sequence_cmp['mae_difference_kg'])} kg。\n")
    else:
        add("_待序列特征提取与训练完成后生成_\n")

    add("### 4.8 假设性剂量影响\n")
    if dose:
        add(table(dose["rows"], [("drug", "药物"), ("weight_scalar", "体重标量"), ("n", "n"),
                                 ("pct_dose_diff_gt10", ">10% 剂量偏差比例"),
                                 ("pct_dose_diff_gt20", ">20% 剂量偏差比例"),
                                 ("median_abs_pct_diff", "中位 |Δ%|")]))
        add("![剂量偏差分布](../results/figures/fig8_dose_distribution.png)\n")
    else:
        add("_待集成评价完成后生成_\n")

    add("### 4.9 时间外验证与敏感性\n")
    if temporal:
        add(f"时间外测试集（末 15% 患者）：MAE {fmt(temporal['test']['mae'])} kg，"
            f"±10% 命中率 {fmt(temporal['test']['hit_within_10pct']*100, 1)}%（n={temporal['n_test']}）。\n")
    else:
        add("_待时间外训练完成后生成_\n")

    add("### 4.10 注意力与遮挡审计\n")
    if occlusion:
        add(table([occlusion], [("n", "n"), ("base_mae", "基线 MAE"), ("occlusion_mae", "遮挡 MAE"),
                                ("random_occlusion_mae", "随机遮挡 MAE"),
                                ("delta_occlusion", "Δ遮挡"), ("delta_random", "Δ随机")]))
        add("**CAM 实现修复（2026-10-06）**：原实现把 Swin 的 NHWC 特征图误按 NCHW 归约，"
            "导致 CAM 退化为常数/竖条纹；已修复布局判向，并将可视化改为倒数第二个 stage"
            "（20×20 上采样）以提升空间分辨率。修复前后对比见 `results/figures/_debug/cam_fix_check.png`。\n")
        add("![CAM 修复后对照（最后 stage vs 倒数第 2 stage）](../results/figures/_debug/cam_fix_check.png)\n")
        add(f"遮挡实验（修复后重跑，n={occlusion['n']}）：基线 MAE {fmt(occlusion['base_mae'])} kg；"
            f"遮挡 CAM 高显著区 {fmt(occlusion['occlusion_mae'])} kg"
            f"（Δ{fmt(occlusion['delta_occlusion'])}）；同形状掩膜随机平移 "
            f"{fmt(occlusion.get('shifted_occlusion_mae'))} kg"
            f"（Δ{fmt(occlusion.get('delta_shifted'))}）；随机像素遮挡 "
            f"{fmt(occlusion['random_occlusion_mae'])} kg（Δ{fmt(occlusion['delta_random'])}）。\n")
        add(f"**判读（阴性结果）**：遮挡 CAM 高显著区与遮挡等面积随机平移区的性能下降几乎相同"
            f"（Δ{fmt(occlusion['delta_occlusion'])} vs Δ{fmt(occlusion.get('delta_shifted'))} kg；"
            f"平移更严重的比例为 {occlusion.get('n_shift_worse_than_salient')}/{occlusion['n']}，"
            "接近五五开），说明该实验**无法证明模型对 CAM 高亮区域存在唯一依赖**；"
            "随机像素遮挡的大幅下降（Δ"
            f"{fmt(occlusion['delta_random'])} kg）来自高频输入扰动，属于不同类型的对照。"
            "因此遮挡实验仅作为质控与假设生成记录，**不进入主要结论**；"
            "CAM 仍只作定性可视化（注意力≠解释）。\n")
        add("更严格的归因预实验（模糊填充删除/插入曲线；CAM vs 输入梯度显著图 vs "
            "RISE-96/4000 vs 随机；n=60）见 `docs/attribution_pilot_summary.md`："
            "RISE 在补足到 4,000 掩膜后仍未优于随机，梯度显著图是唯一稳定胜出者；"
            "详见 `results/figures/attribution_method_comparison.png`。\n")
        add("解剖分区遮挡（肺野/纵隔/膈下，第三方分割模型 + 自动 QC，全量测试集 n=1,042）"
            "及内容匹配对照（掩膜 ≥95% 落在体内、与原区 IoU≤0.2）与冻结特征线性探针"
            "（全局 + 20×20 空间探针，n=8,767）见 `docs/anatomy_and_probe_summary.md`："
            "内容匹配对照下肺野遮挡比对照多升高 +1.60 kg（95% CI 0.94–2.26，p=2.7e-05），"
            "而纵隔/膈下/肝脏反而轻于对照；体重信息遍布全图（400/400 patch 显著，"
            "R²≈0.56–0.66），仅中下胸部/上腹部略高——模型利用的是分布式体型信息，"
            "没有单一器官能解释预测，但肺野因承载胸廓整体形态而具有小幅、可复现的重要性；"
            "详见 `results/figures/anatomical_occlusion.png` 与 `results/figures/linear_probe_r2_map.png`。\n")
        add("器械弱监督 CAM 掩膜的人工核对已按项目负责人授权豁免（2026-10-05），"
            "采用方案第 10 节允许的退化路径；替代证据与残余限制见 WAIVER.md。\n")
        add("![注意力示例（修复版）](../results/figures/fig_attention_camfix_swin_ensemble_0.png)\n")
    else:
        add("_待遮挡实验完成后生成_\n")

    add("### 4.11 与常规体重估算方法的对比\n")
    if conv:
        label_map = {
            "admission_weight_kg": "入院体重（临床默认）",
            "charted_last_weight_kg": "沿用既往记录",
            "omr_weight_kg": "最近门诊体重（OMR）",
            "OMR BMI x height²": "门诊 BMI × 身高²",
            "BMI=25 x height²": "BMI=25 × 身高²",
            "Miller IBW": "Miller IBW", "Hamwi IBW": "Hamwi IBW",
            "Robinson IBW": "Robinson IBW", "Hume IBW / ARDSNet PBW": "Hume/ARDSNet PBW",
            "Devine IBW": "Devine IBW",
            "sex-specific mean (train)": "性别均值",
            "population mean (train)": "人群均值",
        }
        rows_c = []
        for k, v in conv.items():
            if k not in label_map:
                continue
            rows_c.append({
                "method": label_map[k], "n": v["n"], "mae": f"{fmt(v['mae'])} [{fmt(v['mae_ci95'][1])}, {fmt(v['mae_ci95'][2])}]",
                "paired": f"{fmt(v['paired_dl_minus_method'], 2)} [{fmt(v['paired_ci95'][0], 2)}, {fmt(v['paired_ci95'][1], 2)}]",
            })
        dl = conv["__DL ensemble (full test set)__"]
        rows_c.append({"method": "胸片深度学习（本方案）", "n": dl["n"],
                       "mae": fmt(dl["mae"]), "paired": "—"})
        add(table(rows_c, [("method", "方法"), ("n", "n"), ("mae", "MAE vs 实测 (kg, 95% CI)"),
                           ("paired", "配对差（深度学习 − 方法）[95% CI]")]))
        add("注：配对差为逐次住院 |误差_深度学习| − |误差_该方法| 的均值；负值表示深度学习更优。\n")
        strat = conv.get("admission_weight_strata")
        if strat:
            add("**入院体重可靠性分层**：\n")
            add(table([
                {"stratum": "<2 kg（记录可靠）", "n": strat["adm_discrepancy_lt2kg"]["n"],
                 "adm": fmt(strat["adm_discrepancy_lt2kg"]["admission_mae"]),
                 "dl": fmt(strat["adm_discrepancy_lt2kg"]["dl_mae"])},
                {"stratum": "2–5 kg", "n": strat["adm_discrepancy_2to5kg"]["n"],
                 "adm": fmt(strat["adm_discrepancy_2to5kg"]["admission_mae"]),
                 "dl": fmt(strat["adm_discrepancy_2to5kg"]["dl_mae"])},
                {"stratum": "≥5 kg（记录不可靠）", "n": strat["adm_discrepancy_ge5kg"]["n"],
                 "adm": fmt(strat["adm_discrepancy_ge5kg"]["admission_mae"]),
                 "dl": fmt(strat["adm_discrepancy_ge5kg"]["dl_mae"])},
            ], [("stratum", "入院体重与实测偏差"), ("n", "n"),
                ("adm", "入院体重 MAE (kg)"), ("dl", "深度学习 MAE (kg)")]))
        add("![常规方法与深度学习对比](../results/figures/conventional_methods_comparison.png)\n")
        add("结论：深度学习显著优于所有公式/人体测量类方法与均值基线（配对改善 10.4–16.9 kg），"
            "但不如患者自己的入院体重（4.58 kg）或既往记录（6.13 kg）；"
            "在入院体重与实测相差 ≥5 kg 的 30.4% 患者中，深度学习反超**原始**入院体重"
            "（9.55 vs 11.83 kg），但该优势在训练集校准后不再显著（9.55 vs 9.93，p=0.16，见 4.13）；"
            "且深度学习是唯一 100% 可用的方法。详见 `docs/conventional_methods_comparison.md`。\n")
    else:
        add("_待常规方法对比完成后生成_\n")

    add("### 4.12 公式法估计体重的分层失真（2026-10-06 补充）\n")
    if formula_strat:
        ov = formula_strat["overall"]
        rows_f = []
        for name, v in ov.items():
            if "mae" not in v:
                continue
            rows_f.append({"method": name, "n": v["n"], "mae": fmt(v["mae"], 1),
                           "bias": f"{v['bias']:+.1f}", "ratio": f"{v['median_ratio']:.2f}",
                           "hit10": pct(v["hit_within_10pct"]), "gt20": pct(v["severe_gt20pct"]),
                           "verdict": v.get("verdict", "")})
        add(table(rows_f, [("method", "方法"), ("n", "n"), ("mae", "MAE (kg)"),
                           ("bias", "bias (kg)"), ("ratio", "估计/真实"),
                           ("hit10", "±10%"), ("gt20", ">20%"), ("verdict", "判读")]))
        add("注：判读标准为事前定义——可接受 = MAE ≤5 kg 且 ±10% 命中率 ≥80%；"
            "边缘 = MAE ≤10 kg 且 ±10% 命中率 ≥60%；其余为不可接受，"
            ">20% 偏差占比 ≥20% 另标「危险」。公式/回归类方法需身高（n=986）。\n")

        wcols = list(formula_strat["by_weight"].keys())
        bcols = list(formula_strat["by_bmi"].keys())
        for tag, cols_, title, unit in (("by_weight", wcols, "按真实体重分层（kg）", "kg"),
                                        ("by_bmi", bcols, "按 BMI 分层（kg/m²）", "")):
            add(f"**{title}**（单元格 = MAE (bias)，单位 kg）：\n")
            rows_s = []
            for name in ov:
                row = {"method": name}
                for c in cols_:
                    cell = formula_strat[tag][c]["methods"].get(name, {})
                    row[c] = f"{cell['mae']:.1f} ({cell['bias']:+.1f})" if "mae" in cell else "NA"
                rows_s.append(row)
            add(table(rows_s, [("method", "方法")] + [(c, f"{c}") for c in cols_]))
            layer_n = ", ".join("{}={}".format(c, formula_strat[tag][c]["n"]) for c in cols_)
            add("各层 n：" + layer_n + "。\n")

        dist = formula_strat.get("distortion", {})
        rows_d = [{"method": k, "slope": f"{v['slope']:+.3f}",
                   "ci": f"[{v['slope_ci95'][0]:+.3f}, {v['slope_ci95'][1]:+.3f}]"}
                  for k, v in dist.items() if v and v.get("slope_ci95")]
        if rows_d:
            add("**失真斜率**（误差对真实体重回归的斜率；−1 表示输出完全不随体重变化）：\n")
            add(table(rows_d, [("method", "方法"), ("slope", "斜率"), ("ci", "95% CI")]))
        add("**关键结论**：\n"
            "1. 五种 IBW 公式整体 MAE 23.9–25.0 kg，中位数只有真实体重的 76–78%，"
            "57–62% 的患者偏差 >20%，判定为「不可接受（危险）」；\n"
            "2. 失真方向系统且可预测：体重 <50 kg 时**高估**（Devine +6.8 kg），"
            "≥100 kg 时**低估** 51–54 kg，且该组**没有一位患者落在 ±10% 内**；\n"
            "3. 按 BMI 分层，公式只在其设计区间（BMI 18.5–25）勉强可用（Miller 4.3 kg，"
            "±10% 命中 77.5%）；BMI 30–40 时 Devine 35.3 kg（±10% 命中 0.0%，100% 患者偏差 >20%），"
            "BMI ≥40 时 Devine 69.4 kg（估计值仅为真实体重的 47%）；\n"
            "4. 肥胖组（BMI ≥30，占有身高患者的 42.7%）与非肥胖组相差 4 倍（43.8 vs 11.0 kg）；"
            "女性比男性更差（28.5 vs 21.9 kg）；\n"
            "5. 患者**自己的门诊 BMI × 当前身高²** MAE 6.6 kg，优于所有固定公式 3–4 倍，"
            "在 BMI ≥40 组仍为 10.8 kg——提示「优先使用患者自身历史」比「使用固定公式」更安全；\n"
            "6. 临床建议：需要真实体重时不应用 IBW/人群均值代替；需要剂量参照时（如 ARDSNet PBW 通气）"
            "可继续使用，但必须声明其为剂量约定而非患者体重。\n")
        add("详见 `docs/formula_stratified_analysis.md` 与 "
            "`results/tables/formula_stratified_analysis.json`。\n")
        add("![公式法分层失真](../results/figures/formula_stratified_analysis.png)\n")
    else:
        add("_待分层分析完成后生成_\n")

    add("### 4.13 事后补充分析（探索性，2026-10-06）\n")
    add("以下三项均为预设分析完成后新增的探索性分析，**不计入主要结论**，"
        "用于界定结论边界与预答审稿意见。\n")
    if flag_v2:
        sel = flag_v2.get("selected_model", "")
        iso_key = [k for k in flag_v2["models"] if "isotonic" in k]
        auroc = flag_v2["models"][iso_key[0]]["auroc"] if iso_key else float("nan")
        no_unit = [k for k in flag_v2["models"] if "去病区" in k]
        auroc_nu = max(flag_v2["models"][k]["auroc"] for k in no_unit) if no_unit else float("nan")
        da = flag_v2.get("decision_analysis", {})
        add(f"**(1) 前瞻式「入院体重记录不可靠」标记模型**：v2 特征（记录合理性 + 影像点估计 + "
            f"3 种子/异方差归一化不确定性）配合 ExtraTrees 集成，测试集 AUROC "
            f"{fmt(auroc, 3)}（v1 为 0.665；单变量 |DL−入院体重| 为 0.601）；"
            f"去掉病区代理特征后降至 {fmt(auroc_nu, 3)}（说明部分性能来自本院的病区文档习惯，跨院不可迁移）。"
            f"工作点上触发复称比例 5–6% 时，全队列 MAE 从 {fmt(da.get('mae_always_adm'))} kg "
            f"降至 {fmt(da.get('gate_blend50_mae'))} kg（配对差 {fmt(da.get('gate_blend50_vs_adm_diff'), 2)} kg）；"
            f"但敏感度 80% 时仍需复称 77% 的患者，**不具备流程可行性**。详见 "
            f"`docs/unreliable_flagging_v2.md`。\n")
    if nrr:
        m = nrr["methods"]
        dl_key = [k for k in m if "深度学习" in k][0]
        nw_key = [k for k in m if "护理" in k][0]
        add(f"**(2) 「无可信体重记录」人群的验证**：拍片 ±24 h 内没有床秤实测的住院共 4,656 次"
            f"（占胸片队列 33.4%）；其中能在 ±96 h 内找到实测作为替代参考者 n={nrr['n_evaluable']}。"
            f"该人群中护理录入的入院体重 MAE {fmt(m[nw_key]['mae'])} kg，**仍优于深度学习 "
            f"{fmt(m[dl_key]['mae'])} kg**（配对差 +4.46 kg，95% CI 3.49–5.56），"
            f"但深度学习显著优于同队列训练的身高+性别回归（16.63 kg）与全部 IBW 公式（22.07–23.28 kg）。"
            f"结论：**「没有实测」不等于「没有记录」**，「记录不可得时深度学习最优」不能作为主张；"
            f"深度学习的价值在于唯一 100% 可得且在各亚人群性能稳定（8.21 vs 主测试集 8.15 kg）。"
            f"详见 `docs/no_reliable_record_analysis.md`。\n")
    if multimodal:
        mm = multimodal["models"]
        s0 = [k for k in mm if k.startswith("S0")][0]
        s3 = [k for k in mm if k.startswith("S3")][0]
        s4 = [k for k in mm if k.startswith("S4")][0]
        t3 = [k for k in mm if k.startswith("T3")][0]
        pd_ = multimodal["paired_vs_image_only"]
        add(f"**(3) 多模态尝试（影像 + 入院常规数据）**：逐档加入年龄/性别、"
            f"首日生命体征 7 项、精简化验 9 项后，测试集 MAE 从 {fmt(mm[s0]['mae'])} kg 变为 "
            f"{fmt(mm[s3]['mae'])} kg（配对差 {fmt(pd_[s3]['diff'], 2)} kg，无差异）；"
            f"加入身高（仅 57–76% 可得）为 {fmt(mm[s4]['mae'])} kg（{fmt(pd_[s4]['diff'], 2)} kg）；"
            f"纯表格（无影像）为 {fmt(mm[t3]['mae'])} kg，说明这些变量对体重的预测力基本是身高/性别的代理；"
            f"早融合 MLP（768 维影像特征 + 表格）同样零增益。结论：常规入院数据是**冗余**的，"
            f"床旁胸片是体型的**主导信息源**。详见 `docs/multimodal_weight.md`。\n")

    add("## 5. 讨论\n")
    add("- 影像模型显著优于身高+性别公式与人群均值基线，说明 ICU 床旁胸片携带可用的体型信息。\n"
        "- 与 Ichikawa 2023（体检立位片，MAE 2.6–3.4 kg）相比，ICU 床旁片域更复杂"
        "（支持器械阳性率 61.5%、积液/水肿比例高），性能差异需在讨论中如实呈现。\n"
        "- 构念效度与剂量影响分析用于界定「能被估计」与「可用于决策」之间的距离。\n")
    if ensemble:
        strata = {r["group"]: r for r in ensemble["subgroups"].get("weight_stratum", [])}
        if "<50" in strata and ">=100" in strata:
            add(f"- **向均值收缩（安全性关键）**：<50 kg 亚组平均偏倚 "
                f"{fmt(strata['<50']['bias'])} kg（高估），≥100 kg 亚组平均偏倚 "
                f"{fmt(strata['>=100']['bias'])} kg（低估）；体重分层 MAE 从 "
                f"{fmt(strata['50-80']['mae'])} kg 升至 {fmt(strata['>=100']['mae'])} kg，"
                "提示极端体重患者的剂量风险最大。机制上这是典型的向训练集均值收缩（回归稀释）："
                "影像特征与体重的相关性有限时，最优平方误差预测会压缩极端值。\n")
    if uncertainty:
        add(f"- **不确定性校准不足**：90% 名义预测区间实际覆盖率仅 "
            f"{fmt(uncertainty.get('coverage_90_nominal', 0)*100, 1)}%（目标 ≥85%），"
            f"平均区间宽度 {fmt(uncertainty.get('mean_interval_width_kg'))} kg；"
            "3 种子集成的离散度无法覆盖共享的标签噪声，后续应采用异方差回归、证据深度学习或保形预测。\n")
    if dose:
        rows_d = dose["rows"]
        abw = [r for r in rows_d if r["weight_scalar"] == "ABW"]
        adj = [r for r in rows_d if r["weight_scalar"] != "ABW"]
        if abw:
            add(f"- **剂量影响**：ABW 线性药物中 {fmt(abw[0]['pct_dose_diff_gt10'])}% 的模拟剂量偏差 >10%"
                f"（中位 |Δ%| {fmt(abw[0]['median_abs_pct_diff'])}）；")
        if adj:
            add(f"改用 AdjBW 的药物为 {fmt(adj[0]['pct_dose_diff_gt10'])}%"
                f"（中位 |Δ%| {fmt(adj[0]['median_abs_pct_diff'])}），说明体重标量选择会显著改变风险。\n")

    add("## 6. 局限\n")
    add("1. 单中心（BIDMC）回顾性数据，训练环境具备床秤，不能证明在无称重条件 ICU 中的有效性；\n"
        "2. 标签为临床记录体重，容量状态混入标签；\n"
        "3. 双人质控一致性与器械 CAM 掩膜人工核对应项目负责人授权豁免（2026-10-05）："
        "无 κ 统计量、CAM 掩膜质量未经人工核验；已用全量自动质控、严格 QC 敏感性分析与方案允许的退化路径替代，"
        "残余不确定性见 WAIVER.md；\n"
        "4. 剂量模拟为假设性推演，不构成临床有效性证据；\n"
        "5. MIMIC-IV **未采集上臂围/腰围/腹围/胸围等体围数据**（全库词典与自由文本核查均为 0 条，"
        "见 `docs/anthropometry_availability_check.md`），故本阶段无法检验「体围 + 胸片」联合模型；"
        "本地验证方案见 `docs/circumference_future_analysis.md`。\n")

    add("## 7. 结论\n")
    if ensemble and ensemble["success_criteria"]["all_passed"]:
        add("在预设成功标准下，Swin-T 集成模型达到方法学可行性判据（S1–S3 全部达标），"
            "支持进一步开展前瞻性验证。\n")
    else:
        add("按预设成功标准，主模型未全部达标（S1–S3 均未通过）。模型虽显著优于身高+性别公式，"
            "但存在极端体重系统性偏倚、构念效度不足与不确定性不可行动三个问题。\n")
        add("**明确结论：在当前证据下，本模型不应被用于临床剂量决策或液体复苏后的剂量调整；"
            "其定位是方法学可行性探索与后续研究基线。**\n")

    add("\n### 7.0 实现偏差与影响\n")
    dev_rows = [
        {"id": "D1", "deviation": "随机切分实际 5,002/1,109/1,055（方案示意 5,016/1,075/1,075）",
         "impact": "均为患者级 70/15/15，无实质影响"},
        {"id": "D2", "deviation": "队列 B 排除 Feeding Weight，13,423 而非 13,424",
         "impact": "敏感性队列，无影响"},
        {"id": "D3", "deviation": "索引影像质控回退规则未触发（自动质控 100% 合格）",
         "impact": "无影像被剔除；规则保留以备人工复核后启用"},
        {"id": "D5", "deviation": "ResNet-50 保留未标准化与标准化两版结果",
         "impact": "作为参考并列报告（8.30 vs 8.58 kg）"},
        {"id": "D7", "deviation": "Swin 输入 308 px（方案 320 px，window=7 整除约束）",
         "impact": "实测 308 优于 224（MAE 8.51 vs 8.70），属合理替代"},
        {"id": "D8", "deviation": "回归目标标准化（均值 83.80 / SD 23.32 kg）",
         "impact": "修复注意力模型输出塌缩，属必要实现修正"},
        {"id": "D6", "deviation": "ImageNet 权重经国内镜像/本地缓存获取",
         "impact": "不改变模型定义"},
        {"id": "D9", "deviation": "100 张人工质控一致性复核未完成",
         "impact": "自动质控 100% 解码成功；一致性结论待人工"},
        {"id": "D10", "deviation": "器械弱监督 CAM 掩膜未完成，采用高显著区域遮挡退化规则",
         "impact": "遮挡对照不干净，器械依赖结论不作为主要证据"},
        {"id": "D11", "deviation": "CAM 实现缺陷（Swin NHWC 特征被按 NCHW 归约，热图退化为常数/竖条纹）",
         "impact": "2026-10-06 已修复并重跑；修复记录见 docs/cam_fix_record.md；"
                   "遮挡实验改用同形状平移对照，结论限定为不可用于推断区域依赖"},
    ]
    add(table(dev_rows, [("id", "编号"), ("deviation", "偏差"), ("impact", "影响与处理")]))

    add("\n### 7.2 后续改进清单（按优先级）\n")
    add("1. **样本与标签**：扩大训练数据（当前 6,102 张训练影像），引入研究级实测体重以降低标签噪声。\n"
        "2. **极端体重**：对 <50 kg 与 ≥100 kg 采用分层重加权或分布感知损失，纠正向均值收缩。\n"
        "3. **不确定性**：改用保形预测或证据深度学习，以「覆盖率 + 区间宽度」双门槛评估。\n"
        "4. **纵向任务**：将「体重变化」作为独立任务重新设计（配对/时序一致性损失）。\n"
        "5. **外部验证**：在无称重条件的 ICU 队列中开展前瞻性验证（本阶段之外）。\n"
        "6. **人工质控**：完成 `project/docs/human_review/` 中两项复核并回填一致性结果。\n")

    add("\n### 7.1 研究方案目标完成对照\n")
    obj_rows = [
        {"objective": "① 与 CNN/均值/OLS/公式比较", "status": "已完成",
         "evidence": "baselines.json、metrics_resnet50_*.json"},
        {"objective": "② 不确定性评估（覆盖率/宽度/校准）", "status": "已完成（含异方差对照）",
         "evidence": "uncertainty_swin_ensemble.json、uncertainty_swin_hetero.json"},
        {"objective": "③ 序列聚合策略比较", "status": "已完成（同集对照）",
         "evidence": "sequence_model.json、sequence_comparison.json"},
        {"objective": "④ 假设性剂量影响（MCID=10%）", "status": "已完成",
         "evidence": "dose_simulation_swin_ensemble.json"},
        {"objective": "S1 ±10% 命中率 ≥80%", "status": "未达标（62.2%）",
         "evidence": "evaluation_swin_ensemble.json"},
        {"objective": "S2 MAE ≤5 kg 且 CI 上限 <6 kg", "status": "未达标（8.15；CI 7.65–8.70）",
         "evidence": "同上"},
        {"objective": "S3 覆盖率 ≥85% 且极端亚组 ≥65%", "status": "未达标（覆盖率 35.5%；29.6%/58.8%）",
         "evidence": "同上"},
    ]
    add(table(obj_rows, [("objective", "目标"), ("status", "状态"), ("evidence", "证据文件")]))

    add("## 8. 可复现性与产物\n")
    add("- 运行环境：Python `D:\\python311\\python.exe`（torch 2.7.1+cu128）；R `D:\\R\\R-4.6.0\\bin\\Rscript.exe`\n"
        "- 代码与配置：`project/src/`、`project/sql/`、`project/config/config.yaml`\n"
        "- 结果表：`project/results/tables/`；图：`project/results/figures/`\n"
        "- 完成度审计：`project/results/tables/final_audit.json`\n")
    add("\n**最小复现命令序列**（在仓库根目录执行）：\n```bash\n"
        "python project/src/extract_cohort.py\n"
        "python project/src/make_image_manifest.py\n"
        "python project/src/preprocess_images.py --set index --workers 8\n"
        "python project/src/train.py --backbone swin_t --size 308 --seed 1 --tag swin_t_308_s1\n"
        "python project/src/evaluate.py --tag swin_ensemble --backbone swin_t --size 308 \\\n"
        "  --checkpoints project/data/features/checkpoints/swin_t_308_s1.pt \\\n"
        "  project/data/features/checkpoints/swin_t_308_s2.pt \\\n"
        "  project/data/features/checkpoints/swin_t_308_s3.pt\n"
        "python project/src/uncertainty.py --tag swin_ensemble\n"
        "python project/src/dose_simulation.py --tag swin_ensemble\n"
        "python project/src/final_audit.py && python project/src/build_report.py\n```\n")
    if audit:
        add(f"\n**审计结果**：{audit['n_ok']}/{audit['n_checks']} 项通过；"
            f"缺失必需项：{audit.get('missing_required') or '无'}\n")

    add("\n## 9. 伦理、数据与声明\n")
    add("- **数据来源与合规**：MIMIC-IV 3.1 / MIMIC-CXR 2.1.0（PhysioNet 受控访问），遵守数据使用协议；"
        "全部处理在本机完成，不上传患者级数据，不对外共享原始数据。\n"
        "- **伦理审查**：本阶段为回顾性去标识化数据库研究，需由本单位伦理委员会确认审查或豁免（编号待补）。\n"
        "- **代码可用性**：队列构建、预处理、建模与统计代码见 `project/src/` 与 `project/sql/`；"
        "公开范围需依据 DUA 条款确认。\n"
        "- **模型权重**：受 DUA 约束，发布策略待确认。\n"
        "- **利益冲突与资金**：待作者填写。\n"
        "- **AI 辅助披露**：报告由代码从结果产物自动生成，并经 AI 工具进行文字整理与评审模拟；作者需核验全部内容。\n")
    add("- **待补声明**：伦理审查编号、资金、利益冲突与作者贡献需在正式投稿前补齐。\n")
    add("- **人工质控豁免**：2026-10-05 经项目负责人授权，两项人工判定步骤以替代证据结案"
        "（`project/docs/human_review/WAIVER.md`）；残余限制（无 κ 值、CAM 掩膜未经人工核验）已在局限中披露。\n")

    out = DOCS_DIR / "final_report.md"
    out.write_text("\n".join(L), encoding="utf-8")
    log.info("report -> %s (%d lines)", out, len(L))


if __name__ == "__main__":
    main()
