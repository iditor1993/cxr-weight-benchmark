"""Citation trail and post-2023 papers on the same idea."""

import json
import time

import requests

MAILTO = "research@example.org"
S = requests.Session()
S.headers.update({"User-Agent": f"paper-lookup/1.0 (mailto:{MAILTO})"})


def j(url, params=None):
    for attempt in range(4):
        try:
            r = S.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503):
                time.sleep(3 * (attempt + 1))
                continue
            return {"_error": f"HTTP {r.status_code}"}
        except Exception as exc:  # noqa: BLE001
            time.sleep(2)
            print("ERR", exc)
    return {}


out = {}

# 1. Who cites the 2023 CXR->weight paper
for wid, label in [("W4315798299", "ichikawa_2023_cxr_weight"),
                   ("W4392467010", "ichikawa_2024_ct_slice_weight")]:
    data = j("https://api.openalex.org/works", {
        "filter": f"cites:{wid}", "per_page": 100, "mailto": MAILTO,
    })
    out[f"cites_{label}"] = {
        "count": data.get("meta", {}).get("count"),
        "works": [
            {"title": w.get("display_name"), "year": w.get("publication_year"),
             "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
             "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
             "authors": [a["author"]["display_name"] for a in w.get("authorships", [])][:5]}
            for w in data.get("results", [])
        ],
    }
    print(label, out[f"cites_{label}"]["count"], flush=True)
    time.sleep(1)

# 2. Any work mentioning chest radiograph + body weight, 2023 onwards
for tag, filt in {
    "oa_recent_cxr_weight": "title_and_abstract.search:(chest radiograph OR chest x-ray OR chest radiography) AND (body weight OR weight estimation OR weight prediction),from_publication_date:2023-01-01",
    "oa_recent_weight_dl": "title_and_abstract.search:(body weight estimation OR weight prediction) AND (deep learning OR convolutional neural network),from_publication_date:2023-01-01",
}.items():
    data = j("https://api.openalex.org/works", {
        "filter": filt, "per_page": 100, "mailto": MAILTO, "sort": "relevance_score:desc",
    })
    out[tag] = {
        "count": data.get("meta", {}).get("count"),
        "works": [
            {"title": w.get("display_name"), "year": w.get("publication_year"),
             "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
             "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
             "authors": [a["author"]["display_name"] for a in w.get("authorships", [])][:5]}
            for w in data.get("results", [])
        ],
    }
    print(tag, out[tag]["count"], flush=True)
    time.sleep(1)

# 3. PubMed: CXR + weight, filtered to 2023+
base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
term = ('("chest radiograph*"[tiab] OR "chest x-ray*"[tiab] OR "chest radiography"[tiab] OR "thoracic radiograph*"[tiab]) '
        'AND ("body weight"[tiab] OR "weight estimation"[tiab] OR "weight prediction"[tiab] OR "body mass index"[tiab]) '
        'AND ("deep learning"[tiab] OR "machine learning"[tiab] OR "neural network*"[tiab] OR "artificial intelligence"[tiab] OR predict*[tiab] OR estimat*[tiab]) '
        'AND 2023:2026[dp]')
ids = j(f"{base}/esearch.fcgi", {"db": "pubmed", "retmode": "json", "retmax": 100,
                                  "sort": "relevance", "term": term})
idlist = ids.get("esearchresult", {}).get("idlist", [])
out["pubmed_recent"] = {"count": ids.get("esearchresult", {}).get("count"), "records": []}
if idlist:
    time.sleep(0.6)
    summ = j(f"{base}/esummary.fcgi", {"db": "pubmed", "retmode": "json", "id": ",".join(idlist)})
    res = summ.get("result", {})
    for pmid in idlist:
        item = res.get(pmid, {})
        doi = ""
        for aid in item.get("articleids", []):
            if aid.get("idtype") == "doi":
                doi = aid.get("value", "")
        out["pubmed_recent"]["records"].append({
            "pmid": pmid, "title": item.get("title", ""),
            "journal": item.get("fulljournalname") or item.get("source", ""),
            "year": (item.get("pubdate") or "")[:4],
            "authors": [a.get("name", "") for a in item.get("authors", [])][:6],
            "doi": doi,
        })
print("pubmed_recent", out["pubmed_recent"]["count"], flush=True)

with open("lit_search/citations.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)
print("saved")
