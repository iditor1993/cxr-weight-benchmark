"""Targeted follow-up: details, citations and neighbours of the key papers."""

import json
import os
import time

import requests

MAILTO = "research@example.org"
UA = {"User-Agent": f"paper-lookup/1.0 (mailto:{MAILTO})"}
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "lit_search")
os.makedirs(OUT, exist_ok=True)

S = requests.Session()
S.headers.update(UA)


def j(url, params=None):
    for attempt in range(3):
        try:
            r = S.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
        except Exception as exc:  # noqa: BLE001
            print("ERR", url, exc)
        time.sleep(1 + attempt)
    return {}


def openalex_work(doi):
    return j(f"https://api.openalex.org/works/https://doi.org/{doi}")


def crossref_work(doi):
    return j(f"https://api.crossref.org/works/{doi}").get("message", {})


def epmc_by_doi(doi):
    data = j("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
             {"query": f'DOI:"{doi}"', "format": "json", "resultType": "core"})
    res = data.get("resultList", {}).get("result", [])
    return res[0] if res else {}


def pubmed_related(term, retmax=30):
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    ids = j(f"{base}/esearch.fcgi", {"db": "pubmed", "retmode": "json", "retmax": retmax,
                                     "sort": "relevance", "term": term})
    idlist = ids.get("esearchresult", {}).get("idlist", [])
    total = ids.get("esearchresult", {}).get("count")
    records = []
    if idlist:
        time.sleep(0.5)
        summ = j(f"{base}/esummary.fcgi", {"db": "pubmed", "retmode": "json", "id": ",".join(idlist)})
        result = summ.get("result", {})
        for pmid in idlist:
            item = result.get(pmid, {})
            doi = ""
            for aid in item.get("articleids", []):
                if aid.get("idtype") == "doi":
                    doi = aid.get("value", "")
            records.append({"pmid": pmid, "title": item.get("title", ""),
                            "journal": item.get("fulljournalname") or item.get("source", ""),
                            "year": (item.get("pubdate") or "")[:4],
                            "authors": [a.get("name", "") for a in item.get("authors", [])][:8],
                            "doi": doi})
    return total, records


KEY_DOIS = [
    "10.1007/s12194-023-00697-3",           # CXR -> body weight, Radiol Phys Technol 2023
    "10.21203/rs.3.rs-1910857/v1",          # preprint
    "10.1007/s12194-026-01028-y",           # CT scout -> body weight 2026
    "10.1097/rct.0000000000001587",         # CT single-slice body weight 2024
]

out = {"works": {}, "citations": {}, "pm_related": {}}
for doi in KEY_DOIS:
    oa = openalex_work(doi)
    cr = crossref_work(doi)
    ep = epmc_by_doi(doi)
    out["works"][doi] = {
        "openalex": {
            "title": oa.get("display_name"),
            "year": oa.get("publication_year"),
            "venue": ((oa.get("primary_location") or {}).get("source") or {}).get("display_name"),
            "type": oa.get("type"),
            "authors": [a["author"]["display_name"] for a in oa.get("authorships", [])],
            "institutions": sorted({i["display_name"] for a in oa.get("authorships", [])
                                    for i in a.get("institutions", [])}),
            "cited_by_count": oa.get("cited_by_count"),
            "cited_by_api_url": oa.get("cited_by_api_url"),
            "abstract": (oa.get("abstract_inverted_index") is not None),
            "concepts": [c["display_name"] for c in (oa.get("concepts") or [])][:8],
            "ids": oa.get("ids"),
        },
        "crossref": {
            "title": (cr.get("title") or [""])[0],
            "journal": (cr.get("container-title") or [""])[0],
            "year": ((cr.get("issued", {}).get("date-parts") or [[None]])[0][0]),
            "abstract": (cr.get("abstract") or "")[:3000],
            "authors": [" ".join(filter(None, [a.get("given"), a.get("family")]))
                        for a in cr.get("author", [])],
            "references": len(cr.get("reference", []) or []),
            "subject": cr.get("subject", []),
        },
        "europepmc": {
            "title": ep.get("title"),
            "journal": ep.get("journalInfo", {}).get("journal", {}).get("title"),
            "year": ep.get("pubYear"),
            "abstract": (ep.get("abstractText") or "")[:3000],
            "citedByCount": ep.get("citedByCount"),
            "pmid": ep.get("pmid"),
            "authors": ep.get("authorString"),
        },
    }
    time.sleep(0.6)

for doi in KEY_DOIS[:2]:
    oa = openalex_work(doi)
    url = oa.get("cited_by_api_url")
    if url:
        cit = j(url, {"per_page": 50})
        out["citations"][doi] = [
            {"title": w.get("display_name"), "year": w.get("publication_year"),
             "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
             "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name")}
            for w in cit.get("results", [])
        ]

PM_QUERIES = {
    "pm_title_weight_cxr": '("body weight"[ti] OR "weight"[ti]) AND ("chest radiograph*"[ti] OR "chest x-ray*"[ti] OR "chest radiography"[ti] OR "radiograph*"[ti])',
    "pm_cxr_weight_any": '(("chest radiograph*"[tiab] OR "chest x-ray*"[tiab] OR "chest radiography"[tiab] OR "thoracic radiograph*"[tiab]) AND ("body weight"[tiab] OR "body mass"[tiab] OR "weight estimation"[tiab] OR "estimated weight"[tiab] OR "weight prediction"[tiab]))',
    "pm_radiograph_anthropometry_ai": '(radiograph*[tiab] OR "x-ray"[tiab] OR "radiography"[tiab]) AND ("body weight"[tiab] OR "body mass index"[tiab] OR BMI[tiab]) AND ("deep learning"[tiab] OR "convolutional neural network*"[tiab] OR "artificial intelligence"[tiab] OR "machine learning"[tiab])',
    "pm_ct_scout_weight": '("scout image*"[tiab] OR "scout view*"[tiab] OR "topogram*"[tiab] OR "scanogram*"[tiab] OR "computed tomography"[tiab]) AND ("body weight"[tiab] OR "weight estimation"[tiab] OR "weight prediction"[tiab] OR "body mass index"[tiab])',
    "pm_weight_dosing_icu_est": '("weight"[ti] AND ("estimation"[ti] OR "estimated"[ti] OR "prediction"[ti])) AND ("intensive care"[tiab] OR "critically ill"[tiab] OR "emergency"[tiab] OR "ventilation"[tiab] OR "dosing"[tiab])',
}

for tag, q in PM_QUERIES.items():
    total, records = pubmed_related(q)
    out["pm_related"][tag] = {"query": q, "total_hits": total, "records": records}
    print("pm", tag, total, flush=True)
    time.sleep(0.5)

with open(os.path.join(OUT, "targeted.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)
print("saved")
