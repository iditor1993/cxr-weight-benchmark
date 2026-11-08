"""Evidence base: how unreliable are recorded/estimated ICU weights?"""

import json
import time

import requests

S = requests.Session()
S.headers.update({"User-Agent": "paper-lookup/1.0 (mailto:research@example.org)"})
BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def pm(term, retmax=40):
    r = S.get(f"{BASE}/esearch.fcgi", params={"db": "pubmed", "retmode": "json",
                                             "retmax": retmax, "sort": "relevance",
                                             "term": term}, timeout=60).json()
    ids = r.get("esearchresult", {}).get("idlist", [])
    total = r.get("esearchresult", {}).get("count")
    recs = []
    if ids:
        time.sleep(0.5)
        summ = S.get(f"{BASE}/esummary.fcgi", params={"db": "pubmed", "retmode": "json",
                                                      "id": ",".join(ids)}, timeout=60).json()
        for pmid in ids:
            item = summ.get("result", {}).get(pmid, {})
            doi = ""
            for aid in item.get("articleids", []):
                if aid.get("idtype") == "doi":
                    doi = aid.get("value", "")
            recs.append({"title": item.get("title", ""),
                         "journal": item.get("fulljournalname") or item.get("source", ""),
                         "year": (item.get("pubdate") or "")[:4], "doi": doi, "pmid": pmid})
    return total, recs


QUERIES = {
    "icu_weight_accuracy": '("body weight"[tiab] OR "weight"[tiab]) AND ("intensive care"[tiab] OR "critically ill"[tiab] OR "ICU"[tiab]) AND ("accuracy"[tiab] OR "accurate"[tiab] OR "inaccurate"[tiab] OR "error"[tiab] OR "reliability"[tiab] OR "agreement"[tiab] OR "discrepancy"[tiab]) AND ("recorded"[tiab] OR "documented"[tiab] OR "estimated"[tiab] OR "measurement"[tiab])',
    "nurse_estimate_weight": '("weight estimation"[tiab] OR "estimated weight"[tiab] OR "estimate body weight"[tiab]) AND (nurse*[tiab] OR physician*[tiab] OR clinician*[tiab] OR staff[tiab]) AND (accuracy[tiab] OR error[tiab] OR agreement[tiab] OR reliability[tiab])',
    "weight_dosing_error": '("body weight"[tiab] OR "weight-based dosing"[tiab] OR "weight based dosing"[tiab]) AND ("medication error"[tiab] OR "dosing error"[tiab] OR "drug dosing"[tiab] OR "dose adjustment"[tiab]) AND ("intensive care"[tiab] OR "critically ill"[tiab] OR "emergency"[tiab])',
    "weight_measurement_practice": '(weighing[tiab] OR "bed scale"[tiab] OR "weight measurement"[tiab]) AND ("intensive care unit"[tiab] OR "ICU"[tiab]) AND (practice[tiab] OR survey[tiab] OR guideline*[tiab] OR protocol[tiab])',
    "china_icu_weight": '("intensive care"[tiab] OR "critically ill"[tiab] OR "ICU"[tiab]) AND ("body weight"[tiab]) AND (China[tiab] OR Chinese[tiab]) AND (estimat*[tiab] OR record*[tiab] OR measur*[tiab] OR dos*[tiab])',
}

out = {}
for tag, q in QUERIES.items():
    total, recs = pm(q)
    out[tag] = {"query": q, "total": total, "records": recs}
    print("=" * 95)
    print(tag, "total=", total)
    for r in recs[:15]:
        print(f"  ({r['year']}) {r['title'][:165]}")
        print(f"      {r['journal']} | {r['doi']} | pmid={r['pmid']}")
    time.sleep(0.5)

with open("lit_search/weight_accuracy_lit.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)
print("\nsaved lit_search/weight_accuracy_lit.json")
