"""Preprints, conference proceedings, patents and ICU-specific gap checks."""

import json
import time

import requests

S = requests.Session()
S.headers.update({"User-Agent": "paper-lookup/1.0 (mailto:research@example.org)"})
MAILTO = "research@example.org"


def j(url, params=None, tries=3):
    for attempt in range(tries):
        try:
            r = S.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 503):
                time.sleep(3 * (attempt + 1))
                continue
            return {"_error": f"HTTP {r.status_code}"}
        except Exception as exc:  # noqa: BLE001
            time.sleep(2)
            print("ERR", type(exc).__name__, str(exc)[:100])
    return {}


out = {}

# ---- OpenAlex: preprints and recent conference/review items on CXR + weight
filters = {
    "oa_preprints_cxr_weight": "title_and_abstract.search:(chest radiograph OR chest x-ray) AND (body weight OR weight estimation),type:preprint",
    "oa_icu_weight_imaging": "title_and_abstract.search:(weight estimation OR body weight estimation OR predicted body weight) AND (intensive care OR critically ill OR ICU OR mechanical ventilation)",
    "oa_weight_estimation_radiography_all": "title_and_abstract.search:(radiograph OR radiography OR x-ray) AND (body weight OR weight estimation OR weight prediction)",
}
for tag, filt in filters.items():
    data = j("https://api.openalex.org/works", {"filter": filt, "per_page": 50, "mailto": MAILTO})
    out[tag] = {
        "count": data.get("meta", {}).get("count"),
        "works": [
            {"title": w.get("display_name"), "year": w.get("publication_year"), "type": w.get("type"),
             "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
             "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name")}
            for w in data.get("results", [])
        ],
        "error": data.get("_error"),
    }
    print(tag, out[tag]["count"], flush=True)
    time.sleep(1)

# ---- Crossref: proceedings / conference items
for tag, q in {
    "cr_proceedings_cxr_weight": "body weight prediction chest radiograph deep learning",
    "cr_icu_weight_ai": "body weight estimation intensive care machine learning",
}.items():
    data = j("https://api.crossref.org/works", {
        "query.bibliographic": q, "rows": 40, "mailto": MAILTO,
        "select": "DOI,title,container-title,issued,type",
    })
    out[tag] = {
        "count": data.get("message", {}).get("total-results"),
        "works": [
            {"title": (w.get("title") or [""])[0],
             "year": ((w.get("issued", {}).get("date-parts") or [[None]])[0][0]),
             "doi": w.get("DOI"), "venue": (w.get("container-title") or [""])[0],
             "type": w.get("type")}
            for w in data.get("message", {}).get("items", [])
        ],
        "error": data.get("_error"),
    }
    print(tag, out[tag]["count"], flush=True)
    time.sleep(1)

# ---- Google Patents (best effort)
try:
    r = S.get("https://patents.google.com/xhr/query",
              params={"url": "q=(chest+radiograph+body+weight+estimation)", "exp": ""}, timeout=45)
    out["patents"] = {"status": r.status_code, "len": len(r.text),
                      "head": r.text[:500]}
    print("patents", r.status_code, len(r.text), flush=True)
except Exception as exc:  # noqa: BLE001
    out["patents"] = {"error": f"{type(exc).__name__}: {exc}"}
    print("patents ERR", exc, flush=True)

# ---- Semantic Scholar one more try for the key paper
time.sleep(2)
ss = j("https://api.semanticscholar.org/graph/v1/paper/DOI:10.1007/s12194-023-00697-3",
       {"fields": "title,year,venue,abstract,citationCount,referenceCount,externalIds,tldr"})
out["semantic_scholar_key"] = ss
print("s2", ss.get("_error") or ss.get("title"), flush=True)

with open("lit_search/extra_probe.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)

for tag in ("oa_preprints_cxr_weight", "oa_icu_weight_imaging", "oa_weight_estimation_radiography_all",
            "cr_proceedings_cxr_weight", "cr_icu_weight_ai"):
    print("=" * 100)
    print(tag, "count=", out[tag]["count"])
    for w in out[tag]["works"][:30]:
        print(f"  ({w['year']}) [{w['type']}] {w['title'][:160]}")
        print(f"      {w['venue']} | {w['doi']}")
