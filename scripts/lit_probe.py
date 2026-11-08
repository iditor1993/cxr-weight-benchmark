import json
import sys
import time

import requests

UA = {"User-Agent": "paper-lookup/1.0 (mailto:research@example.org)"}

TARGETS = [
    ("openalex", "https://api.openalex.org/works?per_page=1&search=test"),
    ("semanticscholar", "https://api.semanticscholar.org/graph/v1/paper/search?query=test&limit=1"),
    ("crossref", "https://api.crossref.org/works?query=test&rows=1"),
    ("arxiv", "http://export.arxiv.org/api/query?search_query=all:test&max_results=1"),
    ("pubmed", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&retmax=1&term=test"),
    ("europepmc", "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=test&format=json&pageSize=1"),
]

for name, url in TARGETS:
    try:
        r = requests.get(url, timeout=30, headers=UA)
        print(name, r.status_code, len(r.text))
    except Exception as exc:  # noqa: BLE001
        print(name, "ERR", type(exc).__name__, str(exc)[:160])
    time.sleep(0.5)
