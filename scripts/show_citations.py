import json

with open("lit_search/citations.json", encoding="utf-8") as fh:
    d = json.load(fh)

for key in ("cites_ichikawa_2023_cxr_weight", "cites_ichikawa_2024_ct_slice_weight"):
    print("=" * 100)
    print(key, "count=", d[key]["count"])
    for w in d[key]["works"]:
        print(f"  ({w['year']}) {w['title'][:170]}")
        print(f"      {w['venue']} | {w['doi']} | {', '.join(w['authors'][:4])}")

for key in ("oa_recent_cxr_weight", "oa_recent_weight_dl"):
    print("=" * 100)
    print(key, "count=", d[key]["count"])
    for w in d[key]["works"][:60]:
        print(f"  ({w['year']}) {w['title'][:170]}")
        print(f"      {w['venue']} | {w['doi']} | {', '.join(w['authors'][:4])}")

print("=" * 100)
print("pubmed_recent count=", d["pubmed_recent"]["count"])
for r in d["pubmed_recent"]["records"][:60]:
    print(f"  ({r['year']}) {r['title'][:170]}")
    print(f"      {r['journal']} | {r['doi']} | {', '.join(r['authors'][:4])}")
