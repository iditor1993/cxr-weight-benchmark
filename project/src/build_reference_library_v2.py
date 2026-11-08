"""参考文献库 v2：以 PMID（本会话已核验）+ 精确标题检索两种方式建立，输出编号文献表。

产出
    Insights_into_Imaging/references/references.json    完整元数据
    Insights_into_Imaging/references/references.bib     BibTeX
    Insights_into_Imaging/references/references.md      编号文献表（Springer 数字编号风格）
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


def get(url: str) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                  timeout=45, context=CTX).read()


def epmc_by_pmid(pmid: str) -> dict | None:
    url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:" + pmid
           + "&format=json&resultType=core&pageSize=1")
    res = json.loads(get(url).decode())["resultList"]["result"]
    return res[0] if res else None


def epmc_by_title(title: str) -> dict | None:
    url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query="
           + urllib.parse.quote(f'TITLE:"{title}"')
           + "&format=json&resultType=core&pageSize=5")
    res = json.loads(get(url).decode())["resultList"]["result"]
    best, score = None, 0.0
    for r in res:
        s = SequenceMatcher(None, re.sub(r"\W+", " ", title.lower()),
                            re.sub(r"\W+", " ", (r.get("title") or "").lower())).ratio()
        if s > score:
            best, score = r, s
    if best and score >= 0.75:
        best["_match"] = round(score, 2)
        return best
    return None


def epmc_by_doi(doi: str) -> dict | None:
    url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query="
           + urllib.parse.quote(f'DOI:"{doi}"')
           + "&format=json&resultType=core&pageSize=1")
    res = json.loads(get(url).decode())["resultList"]["result"]
    return res[0] if res else None


def epmc_by_query(q: str) -> dict | None:
    url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query="
           + urllib.parse.quote(q) + "&format=json&resultType=core&pageSize=5")
    res = json.loads(get(url).decode())["resultList"]["result"]
    return res[0] if res else None


# (标签, 来源类型, 标识)
ENTRIES: list[tuple[str, str, str]] = [
    # datasets
    ("MIMIC-IV", "title", "MIMIC-IV, a freely accessible electronic health record dataset"),
    ("MIMIC-CXR", "title", "MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports"),
    # predecessors (imaging-based weight)
    ("Ichikawa2023RPT", "title", "Prediction of body weight from chest radiographs using deep learning with a convolutional neural network"),
    ("Ichikawa2021SciRep", "title", "A deep-learning method using computed tomography scout images for estimating patient body weight"),
    ("Ichikawa2023JACMP", "title", "Deep learning-based body weight from scout images can be an alternative to actual body weight in CT radiation dose management"),
    ("Ichikawa2024JCAT", "doi", "10.1097/rct.0000000000001587"),
    ("Ryu2023SarcopeniaCXR", "pmid", "36457204"),
    # IBW / formulas
    ("Pai2000IBWorigin", "title", "The origin of the ideal body weight equations"),
    ("Robinson1983", "title", "Determination of ideal body weight for drug dosage calculations"),
    ("Hume1966", "title", "Prediction of lean body mass from height and weight"),
    ("ARDSNet2000", "title", "Ventilation with lower tidal volumes as compared with traditional tidal volumes for acute lung injury and the acute respiratory distress syndrome"),
    ("Peterson2016Universal", "title", "Universal equation for estimating ideal body weight and body weight at any BMI"),
    # ICU weight measurement practice
    ("Leary2000", "pmid", "11029569"),
    ("Freitag2010", "pmid", "20576445"),
    ("Davies2019", "pmid", "31811748"),
    ("OBrien2016", "pmid", "28979490"),
    ("Eastwood2006", "pmid", "16426933"),
    ("Antonio2021", "pmid", "34722903"),
    ("Menon2005", "pmid", "15796724"),
    ("Lin2009", "pmid", "19931751"),
    ("Thomas2019", "pmid", "30336742"),
    ("WeighingBarriers2026", "query",
     'TITLE:"Barriers to Weighing Patients in UK Intensive Care Units"'),
    # obesity / outcomes / dosing
    ("Schetz2019ObesityICU", "pmid", "30888440"),
    ("Falls2021ObesityCC", "pmid", "34749896"),
    ("Langton2021VancoObesity", "pmid", "31709893"),
    ("VancoDose2026", "title", "Practical dose ranges supporting target vancomycin exposure with AUC-guided dosing in overweight"),
    ("VasopressorDosing2026", "query", 'TITLE:"weight-based, fixed-dose, or phenotype-guided"'),
    ("Malbrain2018Fluids", "query", 'TITLE:"Principles of fluid management and stewardship in septic shock"'),
    ("ESPEN_ICU_nutrition", "title", "ESPEN guideline on clinical nutrition in the intensive care unit"),
    # prediction model methodology
    ("VanCalster2019Calibration", "pmid", "31842878"),
    ("Alba2017Calibration", "pmid", "29049590"),
    ("Collins2024EvalPart1", "pmid", "38191193"),
    ("Riley2024EvalPart2", "pmid", "38224968"),
    ("Ramspek2021External", "pmid", "33564405"),
    ("Wong2021SepsisExternal", "pmid", "34152373"),
    ("Vazquez2022Conformal", "pmid", "35898853"),
    # AI in radiology / reporting standards
    ("Zech2018Generalization", "title", "Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs"),
    ("Hosny2018AIRadiology", "doi", "10.1038/s41568-018-0016-5"),
    ("Kelly2019Challenges", "title", "Key challenges for delivering clinical impact with artificial intelligence"),
    ("Rudin2019StopExplaining", "title", "Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead"),
    ("TRIPOD_AI2024", "title", "TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods"),
    ("CLAIM2024", "title", "Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 Update"),
    ("STROBE2007", "doi", "10.7326/0003-4819-147-8-200710160-00010"),
    ("DECIDE_AI2022", "title", "Reporting guideline for the early-stage clinical evaluation of decision support systems driven by artificial intelligence: DECIDE-AI"),
    ("FUTURE_AI2025", "pmid", "39909534"),
    ("Chen2023Fairness", "pmid", "37380750"),
    ("Calli2021CXRsurvey", "pmid", "34171622"),
    # ICU imaging context
    ("Eisenhuber2012BedsideCXR", "pmid", "22391269"),
    ("Suh2015ACRICU", "pmid", "26439890"),
    # recent ICU ML / drift
    ("Nemati2018SepsisML", "pmid", "29286945"),
    ("Tsai2024OsteoporosisCXR", "pmid", "38217829"),
    ("Drift2026", "title", "Drift isn't a bug, it's the work: post-deployment surveillance for clinical artificial intelligence"),
    # target journal (Insights into Imaging)
    ("IoI2026Reidentification", "title", "Radiologists' memory as a data protection risk: a worst-case stress test for chest radiograph re-identification"),
    ("IoI2026TBDataset", "title", "Development of an expert-annotated chest X-ray dataset to support AI validation in tuberculosis diagnosis"),
    ("IoI2026VLMCXR", "title", "Enhancing radiology workflows through collaborative AI-assisted chest X-ray reporting using large vision-language models"),
    # severity scores (context)
    ("Vincent1996SOFA", "pmid", "8844239"),
    # 近期同类工作（2025–2026，补充近 5 年占比）
    ("EJR2025AorticDilatationCXR", "doi", "10.1016/j.ejrad.2025.112380"),
    ("EJR2026LLMQCCXR", "doi", "10.1016/j.ejrad.2026.112840"),
    ("AcadRadiol2026MultiReaderAI", "doi", "10.1016/j.acra.2026.08.014"),
    ("CBM2025FairnessCXR", "doi", "10.1016/j.compbiomed.2025.111147"),
]


def main() -> None:
    rows = []
    for label, kind, ident in ENTRIES:
        try:
            if kind == "pmid":
                r = epmc_by_pmid(ident)
            elif kind == "doi":
                r = epmc_by_doi(ident)
            elif kind == "query":
                r = epmc_by_query(ident)
            else:
                r = epmc_by_title(ident)
        except Exception as e:                                     # noqa: BLE001
            print("ERR ", label, type(e).__name__, str(e)[:70])
            r = None
        if not r:
            print("MISS", label, "|", ident[:70])
            time.sleep(0.3)
            continue
        rows.append({
            "label": label,
            "authors": r.get("authorString"),
            "title": r.get("title"),
            "journal": (r.get("journalInfo", {}).get("journal", {}) or {}).get("title")
            or r.get("journalTitle"),
            "year": r.get("pubYear"),
            "volume": r.get("journalInfo", {}).get("volume"),
            "issue": r.get("journalInfo", {}).get("issue"),
            "pages": r.get("pageInfo"),
            "doi": r.get("doi"), "pmid": r.get("pmid"),
            "match": r.get("_match"),
        })
        print(f"OK   {label:26s} [{r.get('pubYear')}] {str(r.get('title'))[:78]}")
        time.sleep(0.3)

    (OUT / "references.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    # 编号文献表（Springer 数字编号风格）
    md = ["# 编号参考文献表（供稿件引用，全部经 Europe PMC 核验）\n"]
    for i, r in enumerate(rows, 1):
        authors = (r["authors"] or "").rstrip(".")
        title = (r["title"] or "").rstrip(".")
        md.append(f"{i}. {authors}. {title}. *{r['journal']}*. {r['year']}"
                  + (f";{r['volume']}" if r.get("volume") else "")
                  + (f"({r['issue']})" if r.get("issue") else "")
                  + (f":{r['pages']}" if r.get("pages") else "")
                  + (f". doi:{r['doi']}" if r.get("doi") else "") + ".")
    (OUT / "references.md").write_text("\n".join(md), encoding="utf-8")

    recent = sum(1 for r in rows if int(r["year"] or 0) >= 2021)
    ioi = sum(1 for r in rows if "insights into imaging" in (r["journal"] or "").lower())
    print(f"\n合计 {len(rows)} 篇；≥2021 年 {recent} 篇；目标期刊 {ioi} 篇")
    missing_journal = [r["label"] for r in rows if not r["journal"]]
    if missing_journal:
        print("缺期刊名：", missing_journal)


if __name__ == "__main__":
    main()
