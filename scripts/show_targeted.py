import json

with open("lit_search/targeted.json", encoding="utf-8") as fh:
    d = json.load(fh)

for doi, payload in d["works"].items():
    print("=" * 100)
    print("DOI:", doi)
    for source in ("openalex", "crossref", "europepmc"):
        print("-" * 40, source)
        print(json.dumps(payload[source], ensure_ascii=False, indent=1)[:4000])

for doi, cits in d["citations"].items():
    print("=" * 100)
    print("CITATIONS of", doi, "->", len(cits))
    for c in cits:
        print(f"  ({c['year']}) {c['title']} | {c['venue']} | {c['doi']}")

for tag, payload in d["pm_related"].items():
    print("=" * 100)
    print("PUBMED", tag, "total=", payload["total_hits"])
    for rec in payload["records"][:25]:
        print(f"  ({rec['year']}) {rec['title'][:180]}")
        print(f"      {rec['journal']} | {rec['doi']} | {', '.join(rec['authors'][:4])}")
