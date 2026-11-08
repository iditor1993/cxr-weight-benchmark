"""Multi-database literature search for the CR_Weight novelty check.

Queries PubMed, Europe PMC, OpenAlex, Crossref and arXiv with a fixed,
reproducible query set and writes raw JSON plus a readable digest.
"""

import json
import os
import time
import urllib.parse

import requests

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lit_search")
os.makedirs(OUT_DIR, exist_ok=True)

MAILTO = "research@example.org"
UA = {"User-Agent": f"paper-lookup/1.0 (mailto:{MAILTO})"}
SESSION = requests.Session()
SESSION.headers.update(UA)


def get_json(url, params=None, tries=3, sleep=1.0):
    last = None
    for attempt in range(tries):
        try:
            r = SESSION.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(sleep * (attempt + 1))
    return {"_error": last}


def pubmed_search(term, tag, retmax=50):
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    ids = get_json(f"{base}/esearch.fcgi", {
        "db": "pubmed", "retmode": "json", "retmax": retmax,
        "sort": "relevance", "term": term,
    })
    if "_error" in ids:
        return {"query": term, "error": ids["_error"], "records": []}
    idlist = ids.get("esearchresult", {}).get("idlist", [])
    total = ids.get("esearchresult", {}).get("count")
    records = []
    if idlist:
        time.sleep(0.5)
        summ = get_json(f"{base}/esummary.fcgi", {
            "db": "pubmed", "retmode": "json", "id": ",".join(idlist),
        })
        result = summ.get("result", {}) if "_error" not in summ else {}
        for pmid in idlist:
            item = result.get(pmid)
            if not item:
                continue
            doi = ""
            for aid in item.get("articleids", []):
                if aid.get("idtype") == "doi":
                    doi = aid.get("value", "")
            records.append({
                "pmid": pmid,
                "title": item.get("title", ""),
                "journal": item.get("fulljournalname") or item.get("source", ""),
                "year": (item.get("pubdate") or "")[:4],
                "authors": [a.get("name", "") for a in item.get("authors", [])][:6],
                "doi": doi,
                "pubtype": item.get("pubtype", []),
            })
    return {"query": term, "total_hits": total, "records": records}


def europepmc_search(query, tag, pagesize=50):
    data = get_json(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        {"query": query, "format": "json", "pageSize": pagesize, "resultType": "core"},
    )
    if "_error" in data:
        return {"query": query, "error": data["_error"], "records": []}
    records = []
    for item in data.get("resultList", {}).get("result", []):
        records.append({
            "pmid": item.get("pmid", ""),
            "doi": item.get("doi", ""),
            "title": item.get("title", ""),
            "journal": item.get("journalInfo", {}).get("journal", {}).get("title", ""),
            "year": item.get("pubYear", ""),
            "authors": item.get("authorString", ""),
            "abstract": (item.get("abstractText") or "")[:1200],
        })
    return {"query": query, "total_hits": data.get("hitCount"), "records": records}


def openalex_search(query, tag, per_page=50):
    data = get_json("https://api.openalex.org/works", {
        "filter": f"title_and_abstract.search:{query}",
        "per_page": per_page,
        "mailto": MAILTO,
    })
    if "_error" in data:
        return {"query": query, "error": data["_error"], "records": []}
    records = []
    for item in data.get("results", []):
        records.append({
            "doi": (item.get("doi") or "").replace("https://doi.org/", ""),
            "title": item.get("display_name", ""),
            "year": item.get("publication_year"),
            "journal": ((item.get("primary_location") or {}).get("source") or {}).get("display_name", ""),
            "authors": [a["author"]["display_name"] for a in item.get("authorships", [])][:6],
            "type": item.get("type", ""),
        })
    return {"query": query, "total_hits": data.get("meta", {}).get("count"), "records": records}


def crossref_search(query, tag, rows=40):
    data = get_json("https://api.crossref.org/works", {
        "query.bibliographic": query, "rows": rows, "mailto": MAILTO,
        "select": "DOI,title,container-title,issued,author,type,abstract",
    })
    if "_error" in data or "message" not in data:
        return {"query": query, "error": data.get("_error", "no message"), "records": []}
    message = data["message"]
    records = []
    for item in message.get("items", []):
        issued = (item.get("issued", {}).get("date-parts") or [[None]])[0][0]
        records.append({
            "doi": item.get("DOI", ""),
            "title": (item.get("title") or [""])[0],
            "journal": (item.get("container-title") or [""])[0],
            "year": issued,
            "authors": [
                " ".join(filter(None, [a.get("given"), a.get("family")]))
                for a in item.get("author", [])
            ][:6],
            "type": item.get("type", ""),
        })
    return {"query": query, "total_hits": message.get("total-results"), "records": records}


def arxiv_search(query, tag, max_results=40):
    url = "http://export.arxiv.org/api/query"
    r = SESSION.get(url, params={"search_query": query, "max_results": max_results}, timeout=60)
    if r.status_code != 200:
        return {"query": query, "error": f"HTTP {r.status_code}", "records": []}
    import xml.etree.ElementTree as ET

    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(r.text)
    records = []
    for entry in root.findall("a:entry", ns):
        records.append({
            "title": " ".join(entry.findtext("a:title", "", ns).split()),
            "year": (entry.findtext("a:published", "", ns) or "")[:4],
            "authors": [a.findtext("a:name", "", ns) for a in entry.findall("a:author", ns)][:6],
            "url": entry.findtext("a:id", "", ns),
            "summary": " ".join(entry.findtext("a:summary", "", ns).split())[:800],
        })
    return {"query": query, "total_hits": root.findtext("{http://a9.com/-/spec/opensearch/1.1/}totalResults"), "records": records}


PUBMED_QUERIES = {
    "pm_broad": '("chest radiograph*"[tiab] OR "chest x-ray*"[tiab] OR "chest radiography"[tiab] OR "thoracic radiograph*"[tiab] OR "chest radiograms"[tiab] OR CXR[tiab]) AND (weight[tiab] OR "body mass index"[tiab] OR BMI[tiab] OR obesity[tiab] OR "body composition"[tiab] OR "body habitus"[tiab]) AND (estimat*[tiab] OR predict*[tiab] OR infer*[tiab] OR "deep learning"[tiab] OR "machine learning"[tiab] OR "neural network*"[tiab] OR "artificial intelligence"[tiab] OR radiomics[tiab])',
    "pm_weight_estimation": '("body weight estimation"[tiab] OR "estimation of body weight"[tiab] OR "weight estimation"[tiab] OR "estimate body weight"[tiab] OR "estimate the body weight"[tiab] OR "weight prediction"[tiab] OR "predict body weight"[tiab] OR "predicting body weight"[tiab]) AND (radiograph*[tiab] OR "x-ray*"[tiab] OR "chest"[tiab] OR imaging[tiab] OR "deep learning"[tiab] OR "machine learning"[tiab] OR "artificial intelligence"[tiab] OR photograph*[tiab] OR image*[tiab])',
    "pm_cxr_bmi": '("chest radiograph*"[tiab] OR "chest x-ray*"[tiab] OR "chest radiography"[tiab] OR "thoracic radiograph*"[tiab] OR CXR[tiab]) AND ("body mass index"[tiab] OR BMI[tiab] OR "body weight"[tiab] OR "body habitus"[tiab] OR obesity[tiab])',
    "pm_ct_weight": '("body weight"[tiab] OR "body mass index"[tiab] OR BMI[tiab] OR "body composition"[tiab]) AND ("computed tomography"[tiab] OR CT[tiab] OR "radiograph*"[tiab] OR "X-ray"[tiab] OR "dual-energy"[tiab]) AND ("deep learning"[tiab] OR "machine learning"[tiab] OR "neural network*"[tiab] OR "artificial intelligence"[tiab] OR "automated estimation"[tiab] OR "opportunistic"[tiab])',
    "pm_image_anthropometry": '("body weight"[tiab] OR "body mass index"[tiab] OR BMI[tiab] OR "body fat"[tiab] OR "body composition"[tiab]) AND (photograph*[tiab] OR "facial image*"[tiab] OR "face image*"[tiab] OR smartphone[tiab] OR "2D image*"[tiab] OR "3D scan*"[tiab] OR "depth camera"[tiab] OR "digital image*"[tiab]) AND (estimat*[tiab] OR predict*[tiab] OR "deep learning"[tiab] OR "machine learning"[tiab])',
    "pm_critical_care_weight": '("weight estimation"[tiab] OR "estimated body weight"[tiab] OR "body weight"[tiab]) AND ("critically ill"[tiab] OR "intensive care"[tiab] OR "emergency department"[tiab] OR "drug dosing"[tiab] OR "medication dosing"[tiab]) AND (estimat*[tiab] OR predict*[tiab] OR "deep learning"[tiab] OR "machine learning"[tiab])',
    "pm_pediatric_weight": '(child*[tiab] OR pediatric[tiab] OR paediatric[tiab] OR infant*[tiab] OR neonat*[tiab]) AND ("weight estimation"[tiab] OR "body weight"[tiab]) AND (radiograph*[tiab] OR "x-ray"[tiab] OR ultrasound[tiab] OR "medical imaging"[tiab] OR "deep learning"[tiab])',
    "pm_body_composition_cxr": '("chest radiograph*"[tiab] OR "chest x-ray*"[tiab] OR "chest radiography"[tiab]) AND ("body composition"[tiab] OR "skeletal muscle"[tiab] OR "muscle mass"[tiab] OR "adipose tissue"[tiab] OR "fat mass"[tiab] OR "lean mass"[tiab] OR sarcopenia[tiab])',
}

EPMC_QUERIES = {
    "epmc_core": '(TITLE_ABS:"chest radiograph" OR TITLE_ABS:"chest x-ray" OR TITLE_ABS:"chest radiography" OR TITLE_ABS:"thoracic radiograph") AND (TITLE_ABS:"body weight" OR TITLE_ABS:"weight estimation" OR TITLE_ABS:"body mass index" OR TITLE_ABS:BMI) AND (TITLE_ABS:estimation OR TITLE_ABS:prediction OR TITLE_ABS:"deep learning" OR TITLE_ABS:"machine learning")',
    "epmc_weight_imaging": '(TITLE_ABS:"weight estimation" OR TITLE_ABS:"body weight estimation" OR TITLE_ABS:"weight prediction") AND (TITLE_ABS:radiograph OR TITLE_ABS:"x-ray" OR TITLE_ABS:imaging OR TITLE_ABS:photograph OR TITLE_ABS:"deep learning")',
    "epmc_cxr_bmi": '(TITLE_ABS:"chest radiograph" OR TITLE_ABS:"chest x-ray" OR TITLE_ABS:"chest radiography") AND (TITLE_ABS:"body mass index" OR TITLE_ABS:BMI OR TITLE_ABS:obesity OR TITLE_ABS:"body composition")',
}

OPENALEX_QUERIES = {
    "oa_cxr_weight": '"chest radiograph" "body weight"',
    "oa_cxr_bmi": '"chest x-ray" "body mass index" prediction',
    "oa_weight_estimation_imaging": '"weight estimation" radiograph',
    "oa_body_weight_deep_learning_image": '"body weight estimation" deep learning image',
}

CROSSREF_QUERIES = {
    "cr_cxr_weight": "chest radiograph body weight estimation deep learning",
    "cr_weight_estimation": "body weight estimation from medical images deep learning",
    "cr_cxr_bmi": "chest x-ray body mass index prediction",
}

ARXIV_QUERIES = {
    "ax_weight_image": 'all:"body weight estimation" AND all:image',
    "ax_anthropometric": 'all:"body mass index" AND all:"deep learning"',
}


def main():
    out = {"pubmed": {}, "europepmc": {}, "openalex": {}, "crossref": {}, "arxiv": {}}
    for tag, q in PUBMED_QUERIES.items():
        out["pubmed"][tag] = pubmed_search(q, tag)
        print("pubmed", tag, out["pubmed"][tag].get("total_hits"), flush=True)
        time.sleep(0.5)
    for tag, q in EPMC_QUERIES.items():
        out["europepmc"][tag] = europepmc_search(q, tag)
        print("epmc", tag, out["europepmc"][tag].get("total_hits"), flush=True)
    for tag, q in OPENALEX_QUERIES.items():
        out["openalex"][tag] = openalex_search(q, tag)
        print("openalex", tag, out["openalex"][tag].get("total_hits"), flush=True)
    for tag, q in CROSSREF_QUERIES.items():
        out["crossref"][tag] = crossref_search(q, tag)
        print("crossref", tag, out["crossref"][tag].get("total_hits"), flush=True)
    for tag, q in ARXIV_QUERIES.items():
        out["arxiv"][tag] = arxiv_search(q, tag)
        print("arxiv", tag, out["arxiv"][tag].get("total_hits"), flush=True)
        time.sleep(3.5)
    with open(os.path.join(OUT_DIR, "raw_results.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("saved", os.path.join(OUT_DIR, "raw_results.json"))


if __name__ == "__main__":
    main()
