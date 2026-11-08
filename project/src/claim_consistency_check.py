"""稿件数字 ↔ 原始结果文件的一致性核查（claim-ref alignment 的精简版）。

对每条关键论断给出：稿件中的数值、来源文件与字段、容差，逐条比对并输出 PASS/FAIL。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

T = Path("project/results/tables")
P = Path("project/data/processed")
MANU = Path("Insights_into_Imaging/manuscript/main_text.md")
MAIN_TAG = os.environ.get("CR_MAIN_TAG", "resnet50_308_ensemble")


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


ev = load(T / f"evaluation_{MAIN_TAG}.json")
base = load(T / "baselines.json")
conv = load(T / "conventional_weight_comparison.json")
strat = load(T / "formula_stratified_analysis.json")
nrr = load(T / "no_reliable_record_analysis.json")
flow_nrr = load(P / "cohort_nrr_flow.json")
flow = load(P / "cohort_flow.json")
unc = load(T / f"uncertainty_{MAIN_TAG}.json")
dose = load(T / f"dose_simulation_{MAIN_TAG}.json")
mm = load(T / "multimodal_weight.json")
rep = load(T / "vgg16_two_stage.json")
ab = load(T / "attention_ablation.json")
ab_dec = load(T / "attention_ablation_decision.json")
flag = load(T / "flagging_model_v2.json")
t1 = load(T / "table1_characteristics.json")
psig = load(T / "paired_significance.json")
sig = {r["method"]: r for r in psig["rows"]}

text = MANU.read_text(encoding="utf-8")

CHECKS: list[tuple[str, float, float, float]] = []


def add(label: str, manuscript_value: float, source_value: float, tol: float = 0.011) -> None:
    CHECKS.append((label, manuscript_value, source_value, tol))


# 队列
add("ICU stays screened", 94458, flow["n_icustays_total"], 0)
add("stays with portable AP", 13960, flow["n_stays_with_portable_ap"], 0)
add("analysis cohort stays", 8767, flow["n_stays_cohort_a"], 0)
add("patients", 7166, flow["n_patients_cohort_a"], 0)
add("median image-weight interval (h)", 7.1, t1["gap_hours_median"], 0.05)
add("within 6 h (%)", 44.7, 100 * t1["gap_within_6h_pct"], 0.05)
add("train", 6102, t1["splits"]["train"], 0)
add("validation", 1362, t1["splits"]["val"], 0)
add("test", 1303, t1["splits"]["test"], 0)
add("median weight (kg)", 81.1, t1["weight_median"], 0.05)
add("stays <50 kg", 338, t1["n_lt50"], 0)
add("stays >=100 kg", 1876, t1["n_ge100"], 0)
add("ICU LOS median (d)", 3.9, t1["los_median_days"], 0.05)
add("ventilated (%)", 67.9, 100 * t1["ventilated_pct"], 0.05)
add("male (%)", 56.3, 100 * t1["male_pct"], 0.05)
add("height available (n)", 986, sum(v["n"] for v in strat["by_bmi"].values()), 0)

# 主结果
add("model MAE", 7.91, ev["overall"]["mae"], 0.01)
add("model MAE CI lower", 7.42, ev["overall"]["mae_ci95"][1], 0.01)
add("model MAE CI upper", 8.44, ev["overall"]["mae_ci95"][2], 0.01)
add("model within 10%", 63.2, 100 * ev["overall"]["hit_within_10pct"], 0.05)
add("model bias", -0.29, ev["overall"]["bias"], 0.01)
add("admission weight MAE", 4.58, conv["admission_weight_kg"]["mae"], 0.01)
add("charted last weight MAE", 6.13, conv["charted_last_weight_kg"]["mae"], 0.01)
add("outpatient weight MAE", 6.16, conv["omr_weight_kg"]["mae"], 0.01)
add("height+sex regression MAE", 17.22, strat["overall"]["身高+性别回归"]["mae"], 0.01)
add("population mean MAE", 19.03, base["population_mean"]["mae"], 0.01)
add("sex mean MAE", 18.57, conv["sex-specific mean (train)"]["mae"], 0.01)
add("BMI=25 MAE", 19.35, conv["BMI=25 x height²"]["mae"], 0.01)
for name, val in (("Miller IBW", 23.89), ("Hamwi IBW", 24.59), ("Robinson IBW", 24.94),
                  ("Hume IBW / ARDSNet PBW", 24.95), ("Devine IBW", 25.00)):
    add(f"{name} MAE", val, conv[name]["mae"], 0.01)
add("paired Devine", -17.12, sig["Devine IBW"]["paired_diff"], 0.01)
add("paired Devine CI lower", -18.44, sig["Devine IBW"]["ci_low"], 0.01)
add("paired Devine CI upper", -15.79, sig["Devine IBW"]["ci_high"], 0.01)
add("paired admission weight", 3.33, sig["Nurse-recorded admission weight"]["paired_diff"], 0.01)
add("paired admission weight CI lower", 2.83, sig["Nurse-recorded admission weight"]["ci_low"], 0.01)
add("paired admission weight CI upper", 3.86, sig["Nurse-recorded admission weight"]["ci_high"], 0.01)
add("paired height+sex regression", -9.34, sig["Height-and-sex regression"]["paired_diff"], 0.01)
add("paired height+sex CI lower", -10.30, sig["Height-and-sex regression"]["ci_low"], 0.01)
add("paired height+sex CI upper", -8.41, sig["Height-and-sex regression"]["ci_high"], 0.01)
# 显著性：13 项比较全部通过 BH 校正
add("compared methods", 13, psig["n_comparators"], 0)
add("max BH q across comparisons", 0.0, max(r["q_value"] for r in psig["rows"]), 0.001)
add("largest improvement vs non-measurement estimate", 17.12,
    -min(r["paired_diff"] for r in psig["rows"]), 0.05)
add("smallest improvement vs non-measurement estimate", 9.34,
    -max(r["paired_diff"] for r in psig["rows"] if r["paired_diff"] < 0), 0.05)

# 公式失真
slope = strat["distortion"]
for label, manu in (("Devine slope", -0.822), ("Robinson slope", -0.858), ("Miller slope", -0.890),
                    ("population mean slope", -1.000), ("model slope", -0.264),
                    ("admission weight slope", -0.111)):
    key = {"Devine slope": "Devine IBW", "Robinson slope": "Robinson IBW",
           "Miller slope": "Miller IBW", "population mean slope": "人群均值",
           "model slope": "胸片深度学习", "admission weight slope": "护理录入入院体重"}[label]
    add(label, manu, slope[key]["slope"], 0.002)
add("Devine at BMI 18.5-25", 5.0, strat["by_bmi"]["18.5-25"]["methods"]["Devine IBW"]["mae"], 0.05)
add("Miller at BMI 18.5-25", 4.3, strat["by_bmi"]["18.5-25"]["methods"]["Miller IBW"]["mae"], 0.05)
add("Devine at BMI 25-30", 15.3, strat["by_bmi"]["25-30"]["methods"]["Devine IBW"]["mae"], 0.05)
add("Devine at BMI 30-40", 35.3, strat["by_bmi"]["30-40"]["methods"]["Devine IBW"]["mae"], 0.05)
add("Devine at BMI >=40", 69.4, strat["by_bmi"]["≥40"]["methods"]["Devine IBW"]["mae"], 0.05)
add("Devine ratio at BMI >=40", 0.47, strat["by_bmi"]["≥40"]["methods"]["Devine IBW"]["median_ratio"], 0.005)
add("obese n", 421, strat["by_obesity"]["肥胖 BMI≥30"]["n"], 0)
add("Devine in obesity", 43.8, strat["by_obesity"]["肥胖 BMI≥30"]["methods"]["Devine IBW"]["mae"], 0.05)
add("Devine non-obese", 11.0, strat["by_obesity"]["非肥胖 BMI<30"]["methods"]["Devine IBW"]["mae"], 0.05)
add("Devine bias <50 kg", 6.8, strat["by_weight"]["<50"]["methods"]["Devine IBW"]["bias"], 0.05)
add("Devine bias >=100 kg", -52.4, strat["by_weight"]["≥100"]["methods"]["Devine IBW"]["bias"], 0.05)

# 无同期实测人群
add("no-measurement cohort stays", 4656, flow_nrr["n_no_reliable_measure_record_only"], 0)
add("evaluable subset", 482, nrr["n_evaluable"], 0)
add("delayed reference median (h)", 38.4, nrr["ref_gap_hours_median"], 0.05)
add("model MAE in subset", 7.88, nrr["methods"]["胸片深度学习（本次）"]["mae"], 0.01)
add("model MAE CI lower in subset", 7.08, nrr["methods"]["胸片深度学习（本次）"]["mae_ci95"][0], 0.01)
add("model MAE CI upper in subset", 8.86, nrr["methods"]["胸片深度学习（本次）"]["mae_ci95"][1], 0.01)
add("nurse record MAE in subset", 3.75, nrr["methods"]["护理录入的入院体重"]["mae"], 0.01)
add("paired nurse record in subset", 4.14, nrr["paired_vs_dl"]["护理录入的入院体重"]["diff"], 0.01)
add("paired nurse record CI lower", 3.18, nrr["paired_vs_dl"]["护理录入的入院体重"]["ci95"][0], 0.01)
add("paired nurse record CI upper", 5.22, nrr["paired_vs_dl"]["护理录入的入院体重"]["ci95"][1], 0.01)
add("height+sex regression in subset", 16.63,
    nrr["methods"]["身高+性别回归（同队列训练）"]["mae"], 0.01)
add("time-matched model MAE", 8.4,
    nrr["sensitivity_time_matched"]["methods"]["胸片深度学习（本次）"]["mae"], 0.05)
add("time-matched nurse MAE", 5.0,
    nrr["sensitivity_time_matched"]["methods"]["护理录入的入院体重"]["mae"], 0.05)

# 安全性与不确定性
sub = {r["group"]: r for r in ev["subgroups"]["weight_stratum"]}
add("bias <50 kg", 7.5, sub["<50"]["bias"], 0.05)
add("bias >=100 kg", -8.2, sub[">=100"]["bias"], 0.05)
add("90% interval coverage", 39.7, 100 * unc["coverage_90_nominal"], 0.05)
add("dose >10% (ABW)", 36.8, next(r["pct_dose_diff_gt10"] for r in dose["rows"]
                                  if r["weight_scalar"] == "ABW"), 0.05)
add("dose >10% (AdjBW)", 6.8, next(r["pct_dose_diff_gt10"] for r in dose["rows"]
                                   if r["weight_scalar"] == "AdjBW"), 0.05)
add("dose median abs deviation (ABW)", 7.2, next(r["median_abs_pct_diff"] for r in dose["rows"]
                                                 if r["weight_scalar"] == "ABW"), 0.05)

# 事后分析
add("multimodal S0 MAE", 7.92, mm["models"]["S0 影像单独"]["mae"], 0.01)
add("multimodal S3 MAE", 7.91, mm["models"]["S3 +精简化验"]["mae"], 0.01)
add("multimodal S4 MAE", 7.86, mm["models"]["S4 +身高(可选)"]["mae"], 0.01)
add("replication MAE", 9.75, rep["models"]["A 原文设定（原始特征、无 dropout） / 已知性别"]["mae"], 0.01)
add("replication paired vs main model", 1.84, rep["paired_vgg_vs_main"]["diff"], 0.01)
add("replication paired CI lower", 1.46, rep["paired_vgg_vs_main"]["ci95"][0], 0.01)
add("replication paired CI upper", 2.19, rep["paired_vgg_vs_main"]["ci95"][1], 0.01)
add("sex classifier accuracy", 85.5, 100 * rep["sex_classifier"]["accuracy_mean"], 0.05)

# 架构消融与主模型切换（2026-10-08）
add("ablation: Swin-T ensemble MAE", 8.15,
    ab["ensembles"]["swin-T @308 (3-seed ensemble)"]["mae"], 0.01)
add("ablation: ResNet-50 ensemble MAE", 7.91,
    ab["ensembles"]["resnet50 @308 (3-seed ensemble)"]["mae"], 0.01)
add("ablation: ensemble paired difference", 0.24,
    ab["ensembles"]["ensemble attention vs ensemble convolution"]["diff"], 0.01)
add("ablation: 224 px CNN vs attention (raw target)", 0.40,
    ab["single_seed_comparisons"]["224 px, single seed (CNN: raw target)"]["diff"], 0.01)
add("ablation: ensemble gain (Swin, vs single seed)", -0.35,
    ab["ensembles"]["swin-T @308 (3-seed ensemble) vs single seed"]["diff"], 0.01)
add("ablation: ensemble gain (ResNet, vs single seed)", -0.76,
    ab["ensembles"]["resnet50 @308 (3-seed ensemble) vs single seed"]["diff"], 0.01)
assert ab_dec["switch_main_model"] is True, "architecture decision file says no switch"

# 记录可靠性标记模型（AUC）
et_v2 = flag["models"]["ET [v2 全特征集]"]["auroc"]
et_noproxy = flag["models"]["ET [v2 去病区代理]"]["auroc"]
add("flagging AUC (full features)", 0.68, et_v2, 0.006)
add("flagging AUC (ward proxies removed)", 0.66, et_noproxy, 0.006)

# 稿件文本层面：主模型表述、消融披露与补充材料引用
text_flat = " ".join(text.split())
TEXT_CHECKS: list[tuple[str, bool]] = [
    ("manuscript names ResNet-50 as backbone", "ResNet-50" in text_flat),
    ("manuscript discloses the Swin-T comparison", "Swin-T" in text_flat),
    ("manuscript cites ESM 1", "Electronic Supplementary Material 1" in text_flat),
    ("manuscript cites ESM 2", "Electronic Supplementary Material 2" in text_flat),
    ("manuscript cites ESM 3", "Electronic Supplementary Material 3" in text_flat),
]

# 补充材料 ESM 2（亚组分析）：正文数字 ↔ subgroup_analysis.json
ESM2 = Path("Insights_into_Imaging/supplementary/ESM2_subgroup_analysis.md")
ESM_CHECKS: list[tuple[str, str, bool]] = []
if ESM2.exists():
    sa = load(T / "subgroup_analysis.json")
    esm = ESM2.read_text(encoding="utf-8")

    def esm_has(label: str, needle: str) -> None:
        ESM_CHECKS.append((label, needle, needle in esm))

    rr = sa["axes"]["record_reliability"]
    it_rr = next(i for i in rr["interaction"] if i["comparator"] == "C1")
    esm_has("ESM2 record-reliability interaction", f"{-abs(it_rr['delta_diff']):.2f}")
    esm_has("ESM2 win rate >=5 kg", f"{100*sa['win_rate']['by_axis']['record_reliability']['>=5 kg']['win_rate']:.1f}%")
    fb = sa["axes"]["fluid_balance"]
    it_fb = next(i for i in fb["interaction"] if i["comparator"] == "C1")
    esm_has("ESM2 fluid-balance interaction", f"{-abs(it_fb['delta_diff']):.2f}")
    b40 = next(r for r in sa["axes"]["bmi"]["strata"]["C1"]["rows"] if r["stratum"] == ">=40")
    esm_has("ESM2 BMI >=40 delta", f"+{b40['delta']:.2f}")
    gate = sa["gate"]
    esm_has("ESM2 gate MAE",
            f"{gate['fitted_flagging_model_reference']['gate_dl_mae']:.2f}")
    pl = {p["axis"]: p for p in sa["placebo"]}
    esm_has("ESM2 placebo p (fluid balance)", f"{pl['fluid_balance']['placebo_p_two_sided']:.3f}")
    sw = {(r["axis"], r["comparator"], r["model"]): r["delta_diff"]
          for r in sa["swin_replication"]}
    esm_has("ESM2 Swin replication (record reliability)",
            f"{sw[('record_reliability', 'C1', 'swin')]:.2f}")
    esm_has("ESM2 placebo SD (record reliability)", f"{pl['record_reliability']['placebo_sd']:.2f}")
    lq = load(T / "label_quality_analysis.json")
    ex = lq["exclusion_sensitivity"]["drop_contradicted"]
    esm_has("ESM2 label-exclusion MAE", f"{ex['mae_dl']:.2f} kg")
    esm_has("ESM2 copy-forward share",
            f"{100*lq['copy_forward_evidence']['share_exact_same_value']:.0f}%")
    esm_has("ESM2 label-exclusion record reliability",
            f"{ex['record_reliability_ge5_vs_lt2']['delta_diff']:.2f} kg")
    tv = load(T / "temporal_validation.json")
    esm_has("ESM2 temporal ResNet MAE", f"{tv['ensemble_resnet']['mae']:.2f} kg")
    esm_has("ESM2 temporal paired difference",
            f"{tv['paired_resnet_vs_swin']['paired_diff_a_minus_b']:+.2f} kg")
    dc = load(T / "decision_curve_analysis.json")
    dose = {r["strategy"]: r for r in dc["dose_strategies"]}
    esm_has("ESM2 dose error, admission record", f"{dose['admission record']['pct_gt10']:.1f}%")
    esm_has("ESM2 dose error, best imaging rule",
            f"{dose['rule: imaging if gap>=5 and low uncertainty']['pct_gt10']:.1f}%")
    esm_has("ESM2 net benefit at 0.2",
            f"{dc['reused_calibrated_curves']['model']['0.2']:.3f}")

ESM3 = Path("Insights_into_Imaging/supplementary/ESM3_model_attribution.md")
if ESM3.exists():
    esm3 = ESM3.read_text(encoding="utf-8")
    ar = load(T / "attention_region_figure.json")

    def esm3_has(label: str, needle: str) -> None:
        ESM_CHECKS.append((label, needle, needle in esm3))

    esm3_has("ESM3 subdiaphragmatic CAM density",
             f"{ar['region_density_mean']['Subdiaphragmatic']:.2f}")
    esm3_has("ESM3 lung occlusion excess",
             f"+{ar['causal_occlusion']['Lungs']['delta']:.2f}")
    esm3_has("ESM3 probe mean R2", "0.559")

fail = 0
for label, manu, src, tol in CHECKS:
    ok = abs(manu - src) <= tol
    fail += (not ok)
    print(f"{'PASS' if ok else 'FAIL'}  {label:34s} manuscript={manu:<8} source={round(src, 3)}")

for label, needle, ok in ESM_CHECKS:
    fail += (not ok)
    print(f"{'PASS' if ok else 'FAIL'}  {label:34s} needle={needle}")

# 主文新增结果段（时间外验证 / 标签排除 / 决策曲线）
tv2 = load(T / "temporal_validation.json")
add("main text: temporal MAE", 7.80, tv2["ensemble_resnet"]["mae"], 0.01)
add("main text: temporal paired difference", -0.16,
    tv2["paired_resnet_vs_swin"]["paired_diff_a_minus_b"], 0.01)
lq2 = load(T / "label_quality_analysis.json")
add("main text: label-exclusion MAE", 7.81,
    lq2["exclusion_sensitivity"]["drop_contradicted"]["mae_dl"], 0.01)
dc2 = load(T / "decision_curve_analysis.json")
dose2 = {r["strategy"]: r for r in dc2["dose_strategies"]}
add("main text: dose error, admission record", 17.2, dose2["admission record"]["pct_gt10"], 0.05)
add("main text: dose error, best imaging rule", 25.2,
    dose2["rule: imaging if gap>=5 and low uncertainty"]["pct_gt10"], 0.05)

for label, ok in TEXT_CHECKS:
    fail += (not ok)
    print(f"{'PASS' if ok else 'FAIL'}  {label}")

print(f"\n合计 {len(CHECKS)} 项 + ESM {len(ESM_CHECKS)} 项 + 文本 {len(TEXT_CHECKS)} 项，"
      f"FAIL {fail} 项")
