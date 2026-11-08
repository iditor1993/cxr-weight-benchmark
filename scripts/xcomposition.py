import json

import requests

S = requests.Session()
S.headers.update({"User-Agent": "paper-lookup/1.0 (mailto:research@example.org)"})
DOI = "10.1093/radadv/umaf035"

cr = S.get(f"https://api.crossref.org/works/{DOI}", timeout=45).json().get("message", {})
print("CROSSREF")
print("title:", (cr.get("title") or [""])[0])
print("journal:", (cr.get("container-title") or [""])[0], cr.get("volume"), cr.get("page"))
print("date:", cr.get("published", {}).get("date-parts"))
print("authors:", [" ".join(filter(None, [a.get("given"), a.get("family")])) for a in cr.get("author", [])])
print("abstract:", (cr.get("abstract") or "")[:3000])

ep = S.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
           params={"query": f'DOI:"{DOI}"', "format": "json", "resultType": "core"}, timeout=45).json()
res = ep.get("resultList", {}).get("result", [])
if res:
    a = res[0]
    print("\nEPMC")
    print("title:", a.get("title"))
    print("journal:", a.get("journalInfo", {}).get("journal", {}).get("title"), a.get("pubYear"))
    print("authors:", a.get("authorString"))
    print("pmid:", a.get("pmid"), "pmcid:", a.get("pmcid"))
    print("abstract:", (a.get("abstractText") or "")[:4000])

w = S.get(f"https://api.openalex.org/works/https://doi.org/{DOI}", timeout=45).json()
print("\nOPENALEX")
print("title:", w.get("display_name"), "| year:", w.get("publication_year"),
      "| cited:", w.get("cited_by_count"), "| oa:", w.get("open_access"))
print("institutions:", sorted({i["display_name"] for a in w.get("authorships", [])
                               for i in a.get("institutions", [])}))
inv = w.get("abstract_inverted_index")
if inv:
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    print("abstract:", " ".join(pos[i] for i in sorted(pos))[:4000])
