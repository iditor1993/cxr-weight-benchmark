import json
import time

import requests

S = requests.Session()
S.headers.update({"User-Agent": "paper-lookup/1.0 (mailto:research@example.org)"})
BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def pm(term, retmax=40):
    r = S.get(f"{BASE}/esearch.fcgi", params={"db": "pubmed", "retmode": "json", "retmax": retmax,
                                             "sort": "relevance", "term": term}, timeout=60).json()
    ids = r.get("esearchresult", {})
    idlist = ids.get("idlist", [])
    recs = []
    if idlist:
        time.sleep(0.6)
        summ = S.get(f"{BASE}/esummary.fcgi", params={"db": "pubmed", "retmode": "json",
                                                      "id": ",".join(idlist)}, timeout=60).json()
        for pmid in idlist:
            item = summ.get("result", {}).get(pmid, {})
            doi = ""
            for aid in item.get("articleids", []):
                if aid.get("idtype") == "doi":
                    doi = aid.get("value", "")
            recs.append({"title": item.get("title", ""), "year": (item.get("pubdate") or "")[:4],
                         "journal": item.get("fulljournalname") or item.get("source", ""), "doi": doi})
    return ids.get("count"), recs


QUERIES = {
    "cxr_bmi_precise": '("chest radiograph*"[ti] OR "chest x-ray"[ti] OR "chest radiography"[ti] OR CXR[ti]) AND ("body mass index"[tiab] OR BMI[tiab] OR obesity[tiab]) AND ("deep learning"[tiab] OR "machine learning"[tiab] OR "artificial intelligence"[tiab] OR "neural network*"[tiab])',
    "cxr_weight_title": '("body weight"[ti] OR weight[ti]) AND ("chest radiograph*"[ti] OR "chest x-ray"[ti] OR "chest radiography"[ti])',
    "icu_weight_title_est": '(weight[ti] OR "body weight"[ti]) AND (estimat*[ti] OR predict*[ti]) AND ("critically ill"[tiab] OR "intensive care"[tiab] OR "ICU"[tiab] OR "emergency"[tiab])',
}

out = {}
for tag, q in QUERIES.items():
    total, recs = pm(q)
    out[tag] = {"query": q, "total": total, "records": recs}
    print("=" * 95)
    print(tag, "total=", total)
    for r in recs[:20]:
        print(f"  ({r['year']}) {r['title'][:160]} | {r['journal']} | {r['doi']}")
    time.sleep(0.5)

# Crossref metadata for the two key DOIs
for doi in ["10.1007/s12194-023-00697-3", "10.1038/s41598-021-95170-9"]:
    cr = S.get(f"https://api.crossref.org/works/{doi}", timeout=45).json().get("message", {})
    out.setdefault("crossref_meta", {})[doi] = {
        "volume": cr.get("volume"), "issue": cr.get("issue"), "page": cr.get("page"),
        "published": cr.get("published", {}).get("date-parts"),
        "journal": (cr.get("container-title") or [""])[0],
        "title": (cr.get("title") or [""])[0],
    }
    print(doi, out["crossref_meta"][doi])

with open("lit_search/last_checks.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)
