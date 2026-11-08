"""Final gap checks: author trail, title-level radiograph+weight papers, CNKI site search."""

import html
import json
import re
import time

import requests

S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
                  "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"})

out = {}

# --- OpenAlex author trail
au = S.get("https://api.openalex.org/authors", params={"search": "Shota Ichikawa", "mailto": "research@example.org"},
           timeout=45).json()
cands = [a for a in au.get("results", []) if a.get("works_count", 0) > 20][:3]
out["ichikawa_authors"] = []
for a in cands:
    works = S.get("https://api.openalex.org/works", params={
        "filter": f"author.id:{a['id'].split('/')[-1]}", "per_page": 100,
        "mailto": "research@example.org"}, timeout=45).json()
    titles = [{"title": w.get("display_name"), "year": w.get("publication_year"),
               "doi": (w.get("doi") or "").replace("https://doi.org/", "")}
              for w in works.get("results", [])
              if re.search(r"weight|body|anthro|dose|radiograph", w.get("display_name", ""), re.I)]
    out["ichikawa_authors"].append({
        "name": a.get("display_name"), "works_count": a.get("works_count"),
        "institution": (a.get("last_known_institutions") or [{}])[0].get("display_name"),
        "relevant_works": titles})
    time.sleep(1)

# --- PubMed title-level check
BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
term = ('("body weight"[ti] OR "weight"[ti]) AND ("radiograph*"[ti] OR "radiography"[ti] OR "x-ray"[ti] OR "chest"[ti]) '
        'AND 2018:2026[dp]')
r = S.get(f"{BASE}/esearch.fcgi", params={"db": "pubmed", "retmode": "json", "retmax": 100,
                                          "sort": "relevance", "term": term}, timeout=60).json()
idlist = r.get("esearchresult", {}).get("idlist", [])
out["pm_title_level"] = {"count": r.get("esearchresult", {}).get("count"), "records": []}
if idlist:
    time.sleep(0.6)
    summ = S.get(f"{BASE}/esummary.fcgi", params={"db": "pubmed", "retmode": "json",
                                                  "id": ",".join(idlist)}, timeout=60).json()
    for pmid in idlist:
        item = summ.get("result", {}).get(pmid, {})
        doi = ""
        for aid in item.get("articleids", []):
            if aid.get("idtype") == "doi":
                doi = aid.get("value", "")
        out["pm_title_level"]["records"].append({
            "title": item.get("title", ""), "year": (item.get("pubdate") or "")[:4],
            "journal": item.get("fulljournalname") or item.get("source", ""), "doi": doi})

# --- Bing site searches (Chinese databases)
for tag, q in {
    "cnki_site": "site:cnki.net 胸片 体重 预测",
    "wanfang_site": "site:wanfangdata.com.cn 胸片 体重 预测",
    "cnki_weight_ai": "site:cnki.net X线 体重 深度学习",
}.items():
    rr = S.get("https://cn.bing.com/search", params={"q": q, "mkt": "zh-CN"}, timeout=45)
    text = rr.text
    hits = []
    for b in re.findall(r'<li class="b_algo".*?(?=<li class="b_algo"|</ol>)', text, re.S):
        m = re.search(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', b, re.S)
        if m:
            hits.append({"title": html.unescape(re.sub(r"<[^>]+>", "", m.group(2))).strip(),
                         "url": html.unescape(m.group(1))})
    out[tag] = hits[:15]
    print(tag, len(hits), flush=True)
    time.sleep(1)

with open("lit_search/final_gap.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)

print("=" * 100)
for a in out["ichikawa_authors"]:
    print(a["name"], a["works_count"], a["institution"])
    for w in a["relevant_works"]:
        print("   ", w["year"], w["title"], "|", w["doi"])

print("=" * 100)
print("pm_title_level", out["pm_title_level"]["count"])
for rec in out["pm_title_level"]["records"][:40]:
    print(f"  ({rec['year']}) {rec['title'][:160]} | {rec['journal']} | {rec['doi']}")

for tag in ("cnki_site", "wanfang_site", "cnki_weight_ai"):
    print("=" * 100)
    print(tag)
    for h in out[tag]:
        print("  -", h["title"][:140], "|", h["url"][:120])
