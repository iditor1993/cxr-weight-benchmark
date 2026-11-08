import requests

S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0"})

# 1) Sci Rep 2021 article number
r = S.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
          params={"query": 'DOI:"10.1038/s41598-021-95170-9"', "format": "json",
                  "resultType": "core"}, timeout=45).json()
if r.get("resultList", {}).get("result"):
    a = r["resultList"]["result"][0]
    print("SciRep2021:", {k: a.get(k) for k in ["title", "pageInfo", "pubYear", "pmid", "pmcid"]})
    print("journal:", a.get("journalInfo", {}).get("journal", {}).get("title"),
          a.get("journalInfo", {}).get("volume"))
else:
    print("SciRep2021: not found in EPMC")

# 2) medRxiv preprint verify via OpenAlex + landing page
doi = "10.64898/2026.01.13.26343990"
w = S.get(f"https://api.openalex.org/works/https://doi.org/{doi}", timeout=45).json()
print("medRxiv OA:", w.get("display_name"), "|", w.get("publication_date"),
      "|", ((w.get("primary_location") or {}).get("landing_page_url")))
try:
    page = S.get(((w.get("primary_location") or {}).get("landing_page_url") or
                  "https://www.medrxiv.org/content/10.64898/2026.01.13.26343990v1"), timeout=45)
    print("landing:", page.status_code, len(page.text))
except Exception as exc:  # noqa: BLE001
    print("landing ERR", exc)

# 3) Sarcopenia paper year
r2 = S.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
           params={"query": 'DOI:"10.1002/jcsm.13144"', "format": "json",
                   "resultType": "core"}, timeout=45).json()
if r2.get("resultList", {}).get("result"):
    b = r2["resultList"]["result"][0]
    print("sarcopenia:", b.get("pubYear"), b.get("journalInfo", {}).get("journal", {}).get("title"),
          b.get("journalInfo", {}).get("volume"), b.get("pageInfo"))
