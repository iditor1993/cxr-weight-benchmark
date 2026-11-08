"""Is anyone already predicting weight/BMI from MIMIC-CXR?"""

import json
import time

import requests

S = requests.Session()
S.headers.update({"User-Agent": "paper-lookup/1.0 (mailto:research@example.org)"})
BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def pm(term, retmax=50):
    r = S.get(f"{BASE}/esearch.fcgi", params={
        "db": "pubmed", "retmode": "json", "retmax": retmax, "sort": "relevance",
        "term": term}, timeout=60).json()
    ids = r.get("esearchresult", {}).get("idlist", [])
    total = r.get("esearchresult", {}).get("count")
    recs = []
    if ids:
        time.sleep(0.5)
        summ = S.get(f"{BASE}/esummary.fcgi", params={
            "db": "pubmed", "retmode": "json", "id": ",".join(ids)}, timeout=60).json()
        for pmid in ids:
            item = summ.get("result", {}).get(pmid, {})
            doi = ""
            for aid in item.get("articleids", []):
                if aid.get("idtype") == "doi":
                    doi = aid.get("value", "")
            recs.append({"title": item.get("title", ""),
                         "journal": item.get("fulljournalname") or item.get("source", ""),
                         "year": (item.get("pubdate") or "")[:4], "doi": doi})
    return total, recs


out = {}
QUERIES = {
    "mimic_weight": '("MIMIC-CXR"[tiab] OR "MIMIC IV"[tiab] OR "MIMIC-IV"[tiab]) AND (weight[tiab] OR "body mass index"[tiab] OR BMI[tiab] OR anthropometr*[tiab])',
    "mimic_imaging_anthropometry": '("MIMIC-CXR"[tiab] OR "MIMIC-IV"[tiab]) AND ("deep learning"[tiab] OR "machine learning"[tiab] OR "artificial intelligence"[tiab] OR "neural network*"[tiab]) AND (weight[tiab] OR BMI[tiab] OR "body composition"[tiab])',
    "cxr_physique": '("chest radiograph*"[tiab] OR "chest x-ray*"[tiab] OR CXR[tiab]) AND ("body size"[tiab] OR "body habitus"[tiab] OR "body composition"[tiab] OR "adiposity"[tiab] OR "obesity"[tiab]) AND ("deep learning"[tiab] OR "artificial intelligence"[tiab] OR "neural network*"[tiab])',
}

for tag, q in QUERIES.items():
    total, recs = pm(q)
    out[tag] = {"query": q, "total": total, "records": recs}
    print("=" * 95)
    print(tag, "total=", total)
    for r in recs[:25]:
        print(f"  ({r['year']}) {r['title'][:165]}")
        print(f"      {r['journal']} | {r['doi']}")
    time.sleep(0.5)

# OpenAlex too
def oa(filt, per=50):
    r = S.get("https://api.openalex.org/works", params={
        "filter": filt, "per_page": per, "mailto": "research@example.org"}, timeout=60).json()
    return r.get("meta", {}).get("count"), [
        {"title": w.get("display_name"), "year": w.get("publication_year"),
         "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
         "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name")}
        for w in r.get("results", [])]

for tag, filt in {
    "oa_mimic_weight": "title_and_abstract.search:(MIMIC-CXR OR MIMIC-IV) AND (body weight OR body mass index OR anthropometric)",
    "oa_mimic_bmi_dl": "title_and_abstract.search:(MIMIC-CXR) AND (deep learning OR convolutional) AND (BMI OR body weight OR body composition)",
}.items():
    total, recs = oa(filt)
    out[tag] = {"filter": filt, "total": total, "records": recs}
    print("=" * 95)
    print(tag, "total=", total)
    for r in recs[:25]:
        print(f"  ({r['year']}) {r['title'][:165]} | {r['venue']} | {r['doi']}")

with open("lit_search/mimic_lit_check.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)
