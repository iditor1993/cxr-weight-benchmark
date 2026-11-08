"""Details of the adjacent works (body composition from CXR, CT-based weight)."""

import json
import time

import requests

S = requests.Session()
S.headers.update({"User-Agent": "paper-lookup/1.0 (mailto:research@example.org)"})


def j(url, params=None):
    for attempt in range(3):
        try:
            r = S.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
        except Exception as exc:  # noqa: BLE001
            print("ERR", exc)
        time.sleep(2)
    return {}


def reconstruct(inv):
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))[:2500]


DOIS = [
    "10.64898/2026.01.13.26343990",   # body composition from CXR, medRxiv 2026
    "10.1038/s41598-026-72207-5",     # lower abdominal CT weight
    "10.1109/nssmicrtsd49126.2023.10338236",  # MLP body weight from CT
    "10.3233/xst-230087",             # head CT scout weight 2023
    "10.1016/j.nutos.2024.10.010",    # CXR malnutrition biomarker 2024
    "10.1002/jcsm.13144",             # CXR sarcopenia 2023
]

out = {}
for doi in DOIS:
    w = j(f"https://api.openalex.org/works/https://doi.org/{doi}")
    ep = j("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
           {"query": f'DOI:"{doi}"', "format": "json", "resultType": "core"})
    res = ep.get("resultList", {}).get("result", [])
    out[doi] = {
        "title": w.get("display_name"),
        "year": w.get("publication_year"),
        "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
        "authors": [a["author"]["display_name"] for a in w.get("authorships", [])][:8],
        "abstract": reconstruct(w.get("abstract_inverted_index")) or (res[0].get("abstractText", "")[:2500] if res else ""),
        "cited_by": w.get("cited_by_count"),
    }
    print("=" * 100)
    print(doi, "|", out[doi]["title"])
    print("  ", out[doi]["venue"], out[doi]["year"], "|", ", ".join(out[doi]["authors"]))
    print("  ", out[doi]["abstract"][:1200])
    time.sleep(0.8)

with open("lit_search/adjacent.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)
