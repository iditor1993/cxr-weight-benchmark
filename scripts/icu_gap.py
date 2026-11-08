"""Check whether the ICU/critical-care angle is already covered, plus height and China searches."""

import json
import time

import requests

S = requests.Session()
S.headers.update({"User-Agent": "paper-lookup/1.0 (mailto:research@example.org)"})
BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def pm(term, retmax=40):
    for attempt in range(3):
        r = S.get(f"{BASE}/esearch.fcgi", params={
            "db": "pubmed", "retmode": "json", "retmax": retmax,
            "sort": "relevance", "term": term}, timeout=60)
        if r.status_code == 200:
            break
        time.sleep(2)
    ids = r.json().get("esearchresult", {})
    idlist = ids.get("idlist", [])
    records = []
    if idlist:
        time.sleep(0.5)
        summ = S.get(f"{BASE}/esummary.fcgi", params={
            "db": "pubmed", "retmode": "json", "id": ",".join(idlist)}, timeout=60).json()
        res = summ.get("result", {})
        for pmid in idlist:
            item = res.get(pmid, {})
            doi = ""
            for aid in item.get("articleids", []):
                if aid.get("idtype") == "doi":
                    doi = aid.get("value", "")
            records.append({"pmid": pmid, "title": item.get("title", ""),
                            "journal": item.get("fulljournalname") or item.get("source", ""),
                            "year": (item.get("pubdate") or "")[:4], "doi": doi,
                            "authors": [a.get("name", "") for a in item.get("authors", [])][:5]})
    return ids.get("count"), records


QUERIES = {
    "icu_cxr_weight": '("chest radiograph*"[tiab] OR "chest x-ray*"[tiab] OR CXR[tiab]) AND ("body weight"[tiab] OR "weight"[tiab]) AND ("intensive care"[tiab] OR "critically ill"[tiab] OR "ICU"[tiab] OR "mechanical ventilation"[tiab] OR "ventilated"[tiab])',
    "icu_weight_dosing": '("body weight"[tiab] OR "weight-based"[tiab] OR "weight based dosing"[tiab]) AND ("intensive care"[tiab] OR "critically ill"[tiab]) AND ("estimat*"[tiab] OR "prediction"[tiab] OR "deep learning"[tiab] OR "machine learning"[tiab])',
    "portable_cxr_weight": '("portable chest radiograph*"[tiab] OR "bedside chest radiograph*"[tiab] OR "supine chest radiograph*"[tiab]) AND ("weight"[tiab] OR "body size"[tiab] OR "habitus"[tiab])',
    "height_estimation_imaging": '("height estimation"[tiab] OR "estimate height"[tiab] OR "height prediction"[tiab] OR "predicting height"[tiab]) AND ("radiograph*"[tiab] OR "x-ray"[tiab] OR "computed tomography"[tiab] OR "deep learning"[tiab])',
    "pediatric_cxr_weight": '("chest radiograph*"[tiab] OR "chest x-ray*"[tiab]) AND (child*[tiab] OR pediatric[tiab] OR paediatric[tiab] OR infant*[tiab]) AND ("body weight"[tiab] OR "weight estimation"[tiab] OR "weight prediction"[tiab])',
    "weight_ai_icu": '("weight"[ti] AND ("estimation"[ti] OR "prediction"[ti] OR "estimated"[ti])) AND (radiograph*[tiab] OR "chest x-ray"[tiab] OR "imaging"[tiab] OR "deep learning"[tiab]) AND ("intensive care"[tiab] OR "critical care"[tiab] OR "emergency department"[tiab] OR "critically ill"[tiab])',
}

out = {}
for tag, q in QUERIES.items():
    total, records = pm(q)
    out[tag] = {"query": q, "total_hits": total, "records": records}
    print(tag, total, flush=True)
    time.sleep(0.5)

with open("lit_search/icu_gap.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)

for tag, payload in out.items():
    print("=" * 100)
    print(tag, "total=", payload["total_hits"])
    for r in payload["records"][:20]:
        print(f"  ({r['year']}) {r['title'][:170]}")
        print(f"      {r['journal']} | {r['doi']}")
