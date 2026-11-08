import json

import requests

S = requests.Session()
S.headers.update({"User-Agent": "paper-lookup/1.0 (mailto:research@example.org)"})

PMIDS = {
    "20865881": "Accuracy of weight and height estimation in an intensive care unit (Anaesth Intensive Care 2010)",
    "39973117": "Difference between estimated and actual body weight in an ICU - service evaluation (Nurs Crit Care 2025)",
}

out = {}
for pmid, label in PMIDS.items():
    r = S.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
              params={"query": f"EXT_ID:{pmid}", "format": "json", "resultType": "core"},
              timeout=45).json()
    res = r.get("resultList", {}).get("result", [])
    print("=" * 95)
    if not res:
        print(label, "-> not found")
        continue
    a = res[0]
    rec = {"label": label, "title": a.get("title"),
           "journal": a.get("journalInfo", {}).get("journal", {}).get("title"),
           "year": a.get("pubYear"), "doi": a.get("doi"),
           "authors": a.get("authorString"),
           "abstract": (a.get("abstractText") or "")[:3000]}
    out[pmid] = rec
    print(rec["title"])
    print(rec["journal"], rec["year"], "|", rec["doi"], "|", rec["authors"])
    print(rec["abstract"])

with open("lit_search/icu_weight_abstracts.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)
