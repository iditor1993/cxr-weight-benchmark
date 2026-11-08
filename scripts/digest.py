import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(HERE, "lit_search", "raw_results.json")

with open(PATH, encoding="utf-8") as fh:
    data = json.load(fh)

limit = int(sys.argv[1]) if len(sys.argv) > 1 else 20
source_filter = sys.argv[2].split(",") if len(sys.argv) > 2 else None

for source, queries in data.items():
    if source_filter and source not in source_filter:
        continue
    for tag, payload in queries.items():
        print("=" * 100)
        print(f"[{source}] {tag} | query={payload['query'][:160]} | total={payload.get('total_hits')}")
        if payload.get("error"):
            print("  ERROR:", payload["error"])
        for rec in payload.get("records", [])[:limit]:
            year = rec.get("year") or "?"
            print(f"  - ({year}) {rec.get('title','')[:170]}")
            print(f"      {rec.get('journal','')} | doi={rec.get('doi','')} pmid={rec.get('pmid','')}")
        print()
