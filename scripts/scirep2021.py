import json
import time

import requests

S = requests.Session()
S.headers.update({"User-Agent": "paper-lookup/1.0 (mailto:research@example.org)"})


def reconstruct(inv):
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))[:4000]


for doi in ["10.1038/s41598-021-95170-9", "10.1002/acm2.14080", "10.1016/j.jmir.2022.10.158"]:
    w = S.get(f"https://api.openalex.org/works/https://doi.org/{doi}", timeout=60).json()
    print("=" * 100)
    print(doi)
    print(w.get("display_name"))
    print(((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
          w.get("publication_year"), "| cited:", w.get("cited_by_count"))
    print("authors:", ", ".join(a["author"]["display_name"] for a in w.get("authorships", [])[:8]))
    print(reconstruct(w.get("abstract_inverted_index"))[:2200])
    time.sleep(1)
