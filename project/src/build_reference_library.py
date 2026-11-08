"""为 Insights into Imaging 论文建立**经过核验**的参考文献库（50–60 篇，以近 5 年为主）。

做法
    对每条候选（已知标题或主题检索式）调用 Europe PMC REST 检索，
    取最佳匹配并按标题相似度确认，仅保留能在 Europe PMC 命中的真实文献；
    输出 JSON / BibTeX / Markdown 三种形式，附 DOI、PMID、期刊、年份。

产出
    Insights_into_Imaging/references/references.json
    Insights_into_Imaging/references/references.bib
    Insights_into_Imaging/references/references_table.md
"""

from __future__ import annotations

import json
import re
import ssl
import time
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

CTX = ssl.create_default_context()
UA = {"User-Agent": "Mozilla/5.0"}
OUT = Path("Insights_into_Imaging/references")
OUT.mkdir(parents=True, exist_ok=True)


def epmc(query: str, size: int = 8, relevance: bool = True) -> list[dict]:
    url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query="
           + urllib.parse.quote(query)
           + f"&format=json&pageSize={size}&resultType=core")
    if not relevance:                     # 精确标题检索时才用被引排序
        url += "&sort=" + urllib.parse.quote("CITED desc")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=45, context=CTX) as r:
        d = json.loads(r.read().decode("utf-8"))
    return d.get("resultList", {}).get("result", [])


def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, re.sub(r"\W+", " ", a.lower()),
                           re.sub(r"\W+", " ", b.lower())).ratio()


# 候选：(类型, 检索串, 必须命中的关键词列表)
CANDIDATES: list[tuple[str, str, tuple[str, ...]]] = [
    # —— 目标期刊近作（Insights into Imaging，约 3 篇）
    ("T", "Radiologists' memory as a data protection risk: a worst-case stress test for chest radiograph re-identification", ()),
    ("T", "Development of an expert-annotated chest X-ray dataset to support AI validation in tuberculosis diagnosis", ()),
    ("T", "Enhancing radiology workflows through collaborative AI-assisted chest X-ray reporting using large vision-language models", ()),
    # —— 直接前作：影像估重
    ("T", "Prediction of body weight from chest radiographs using deep learning with a convolutional neural network", ()),
    ("T", "A deep-learning method using computed tomography scout images for estimating patient body weight", ()),
    ("T", "Deep learning-based body weight from scout images can be an alternative to actual body weight in CT radiation dose management", ()),
    ("T", "Estimating Body Weight From Measurements From Different Single-Slice Computed Tomography Levels", ()),
    ("T", "Chest X-ray-based opportunistic screening of sarcopenia using deep learning", ()),
    # —— 数据集
    ("Q", "MIMIC-IV freely accessible electronic health record dataset", ("mimic-iv",)),
    ("Q", "MIMIC-CXR de-identified publicly available database chest radiographs free-text reports",
     ("mimic-cxr",)),
    # —— IBW 公式溯源与评价
    ("T", "The origin of the ideal body weight equations", ()),
    ("T", "Determination of ideal body weight for drug dosage calculations", ()),
    ("T", "Prediction of lean body mass from height and weight", ()),
    ("T", "Ventilation with lower tidal volumes as compared with traditional tidal volumes for acute lung injury and the acute respiratory distress syndrome", ()),
    ("T", "Universal equation for estimating ideal body weight and body weight at any BMI", ()),
    # —— ICU 体重测量实践与误差
    ("Q", 'accuracy of recorded body weight versus measured body weight in critically ill patients',
     ("body weight", "measure")),
    ("Q", 'nurses estimation of patient weight accuracy intensive care',
     ("estimat", "weight")),
    ("Q", 'body weight measurement practices intensive care unit barriers weighing patients',
     ("weigh", "intensive care")),
    ("Q", 'weight-based drug dosing in critically ill obese patients pharmacokinetics',
     ("weight", "dosing")),
    # —— 胸片与 ICU 影像 AI
    ("Q", 'portable chest radiograph deep learning intensive care unit',
     ("portable", "chest radiograph")),
    ("Q", 'deep learning chest radiograph intensive care mortality prediction',
     ("chest radiograph", "mortality")),
    ("T", "Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs", ()),
    ("T", "Artificial intelligence in radiology", ()),
    ("T", "Key challenges for delivering clinical impact with artificial intelligence", ()),
    # —— 不确定性、校准、可解释性批判
    ("T", "Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead", ()),
    ("Q", 'calibration of prediction models clinical prediction uncertainty',
     ("calibration", "prediction")),
    ("Q", 'conformal prediction medical imaging uncertainty quantification',
     ("conformal",)),
    # —— 域偏移与泛化
    ("Q", 'distribution shift external validation clinical machine learning model performance degradation',
     ("external validation",)),
    ("Q", 'domain generalization medical imaging deep learning review',
     ("domain", "imaging")),
    # —— 报告规范与清单
    ("T", "TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods", ()),
    ("T", "Checklist for Artificial Intelligence in Medical Imaging (CLAIM)", ()),
    ("T", "The Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) statement", ()),
    ("T", "Reporting guideline for the early-stage clinical evaluation of decision support systems driven by artificial intelligence: DECIDE-AI", ()),
    # —— 剂量与体重的关系（临床影响）
    ("Q", 'vancomycin dosing body weight obesity clinical outcomes',
     ("vancomycin", "weight")),
    ("Q", 'vasopressor dose body weight intensive care',
     ("vasopressor", "weight")),
    ("Q", 'nutrition energy protein requirements body weight critically ill guideline',
     ("nutrition", "critically ill")),
    ("Q", 'fluid balance weight change critically ill outcomes',
     ("fluid", "weight")),
    # —— 体重与预后
    ("Q", 'obesity body mass index intensive care outcomes association',
     ("obesity", "intensive care")),
    ("Q", 'antimicrobial pharmacokinetics altered body composition critically ill dosing',
     ("pharmacokinetic", "critically ill")),
    # —— 其他影像估重/身体参数
    ("Q", 'deep learning estimation of body height from medical images',
     ("height", "deep learning")),
    ("Q", 'body composition from chest radiograph artificial intelligence',
     ("body composition", "radiograph")),
    ("Q", 'emergency department weight estimation accuracy adults',
     ("weight estimation",)),
    # —— AI 落地与漂移
    ("Q", 'clinical deployment monitoring of artificial intelligence models radiology drift',
     ("deployment", "artificial intelligence")),
    ("Q", 'artificial intelligence prospective randomized clinical trial radiology workflow',
     ("prospective", "artificial intelligence")),
    # —— MIMIC 方法学/可复现性
    ("Q", 'reproducibility machine learning studies critical care databases',
     ("reproducib",)),
    ("Q", 'reporting quality machine learning critical care studies review',
     ("reporting", "machine learning")),
    # —— 危重病评分/严重度（背景）
    ("T", "APACHE II: a severity of disease classification system", ()),
    ("T", "The SOFA (Sepsis-related Organ Failure Assessment) score to describe organ dysfunction/failure", ()),
    # —— 立位/卧位胸片与体型
    ("Q", 'chest radiograph cardiac silhouette body size correlation',
     ("radiograph", "body")),
    ("Q", 'radiographic measurement of thoracic width body habitus',
     ("thoracic",)),
]


def main() -> None:
    rows, seen = [], set()
    for kind, text, must in CANDIDATES:
        query = f'TITLE:"{text}"' if kind == "T" else text
        try:
            hits = epmc(query, relevance=(kind == "Q" or not must))
        except Exception as e:                                      # noqa: BLE001
            print("ERR", text[:60], type(e).__name__, str(e)[:80])
            hits = []
        best, best_score = None, 0.0
        for h in hits:
            title = h.get("title") or ""
            abstract = h.get("abstractText") or ""
            blob = (title + " " + abstract).lower()
            if must and not all(tok.lower() in blob for tok in must):
                continue
            if kind == "T":
                score = sim(text, title)
            else:
                hits_n = sum(1 for tok in must if tok.lower() in (title or "").lower())
                score = 0.6 + 0.1 * hits_n
            if score > best_score:
                best, best_score = h, score
        if not best or (kind == "T" and best_score < 0.6):
            print("MISS", kind, text[:70])
            time.sleep(0.4)
            continue
        key = (best.get("doi") or best.get("pmid") or best.get("id") or "").lower()
        if not key or key in seen:
            time.sleep(0.4)
            continue
        seen.add(key)
        rows.append({
            "query": text, "kind": kind,
            "title": best.get("title"), "authors": best.get("authorString"),
            "journal": (best.get("journalInfo", {}).get("journal", {}) or {}).get("title")
            or best.get("journalTitle"),
            "year": best.get("pubYear"), "volume": best.get("journalInfo", {}).get("volume"),
            "issue": best.get("journalInfo", {}).get("issue"), "pages": best.get("pageInfo"),
            "doi": best.get("doi"), "pmid": best.get("pmid"), "pmcid": best.get("pmcid"),
            "cited": best.get("citedByCount"), "match": round(best_score, 2),
        })
        print(f"OK  [{best.get('pubYear')}] {str(best.get('title'))[:95]} | {best.get('journalTitle')}")
        time.sleep(0.4)

    (OUT / "references.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    # BibTeX
    def bibkey(r, i):
        first = (r["authors"] or "anon").split(",")[0].split()[0].lower()
        first = re.sub(r"\W", "", first)
        return f"{first}{r['year'] or ''}{i}"

    lines = []
    for i, r in enumerate(rows, 1):
        lines.append(f"@article{{{bibkey(r, i)},")
        lines.append(f"  author  = {{{r['authors']}}},")
        lines.append(f"  title   = {{{r['title']}}},")
        lines.append(f"  journal = {{{r['journal']}}},")
        lines.append(f"  year    = {{{r['year']}}},")
        if r.get("volume"):
            lines.append(f"  volume  = {{{r['volume']}}},")
        if r.get("issue"):
            lines.append(f"  number  = {{{r['issue']}}},")
        if r.get("pages"):
            lines.append(f"  pages   = {{{r['pages']}}},")
        if r.get("doi"):
            lines.append(f"  doi     = {{{r['doi']}}},")
        lines.append("}")
        lines.append("")
    (OUT / "references.bib").write_text("\n".join(lines), encoding="utf-8")
    # Markdown 表
    md = ["# 参考文献库（Europe PMC 核验）\n",
          "| # | 年 | 期刊 | 标题 | DOI | PMID |",
          "|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        md.append(f"| {i} | {r['year']} | {r['journal']} | {str(r['title'])[:110]} | "
                  f"{r['doi'] or ''} | {r['pmid'] or ''} |")
    (OUT / "references_table.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n共 {len(rows)} 篇；近 5 年（>=2021）{sum(1 for r in rows if int(r['year'] or 0) >= 2021)} 篇；"
          f"目标期刊 {sum(1 for r in rows if 'insights into imaging' in (r['journal'] or '').lower())} 篇")


if __name__ == "__main__":
    main()
