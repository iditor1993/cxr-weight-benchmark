"""Pull abstracts for the key 'recorded/estimated weight is unreliable' papers."""

import json
import time

import requests

S = requests.Session()
S.headers.update({"User-Agent": "paper-lookup/1.0 (mailto:research@example.org)"})

PMIDS = {
    "28589617": "Recording patient bodyweight in hospitals: are we doing well enough? (Intern Med J 2018)",
    "15796724": "How accurate is weight estimation in the emergency department? (EMA 2005)",
    "28469895": "Elimination of ED Medication Errors Due To Estimated Weights (BMJ QIR 2017)",
    "19351637": "Weight assessment in cardiac patients: LMWH prescription (PMJ 2009)",
    "34091516": "tPA dosing on mobile stroke unit: estimated vs measured weights (2021)",
    "28979416": "The weight debate (J Intensive Care Soc 2015)",
}

out = {}
for pmid, label in PMIDS.items():
    r = S.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
              params={"query": f"EXT_ID:{pmid}", "format": "json", "resultType": "core"},
              timeout=45).json()
    res = r.get("resultList", {}).get("result", [])
    if not res:
        print("=" * 95)
        print(label, "-> not found")
        continue
    a = res[0]
    rec = {
        "label": label,
        "title": a.get("title"),
        "journal": a.get("journalInfo", {}).get("journal", {}).get("title"),
        "year": a.get("pubYear"),
        "doi": a.get("doi"),
        "abstract": (a.get("abstractText") or "")[:2500],
    }
    out[pmid] = rec
    print("=" * 95)
    print(rec["title"], "|", rec["journal"], rec["year"], "|", rec["doi"])
    print(rec["abstract"][:1800])
    time.sleep(0.4)

# extra targeted searches
def pm(term, retmax=15):
    r = S.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
              params={"db": "pubmed", "retmode": "json", "retmax": retmax,
                      "sort": "relevance", "term": term}, timeout=60).json()
    ids = r.get("esearchresult", {}).get("idlist", [])
    total = r.get("esearchresult", {}).get("count")
    recs = []
    if ids:
        time.sleep(0.5)
        summ = S.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                     params={"db": "pubmed", "retmode": "json", "id": ",".join(ids)},
                     timeout=60).json()
        for pmid in ids:
            item = summ.get("result", {}).get(pmid, {})
            recs.append({"pmid": pmid, "title": item.get("title", ""),
                         "journal": item.get("fulljournalname") or item.get("source", ""),
                         "year": (item.get("pubdate") or "")[:4]})
    return total, recs

for tag, term in {
    "icu_estimated_vs_measured": '("estimated weight"[tiab] OR "estimated body weight"[tiab] OR "weight estimation"[tiab]) AND ("intensive care"[tiab] OR "critically ill"[tiab])',
    "weight_record_error": '("body weight"[tiab] OR "weight"[ti]) AND ("documentation"[tiab] OR "recording"[tiab] OR "recorded"[tiab]) AND (error[tiab] OR accuracy[tiab] OR inaccuracy[tiab] OR incorrect[tiab])',
    "vancomycin_weight_dosing": 'vancomycin[tiab] AND ("body weight"[tiab] OR "weight-based"[tiab] OR "weight based"[tiab]) AND ("dosing"[tiab] OR "dose"[tiab]) AND (obesity[tiab] OR "critically ill"[tiab] OR "intensive care"[tiab])',
}.items():
    total, recs = pm(term)
    out[tag] = {"total": total, "records": recs}
    print("=" * 95)
    print(tag, "total=", total)
    for r in recs[:15]:
        print(f"  ({r['year']}) {r['title'][:160]} | {r['journal']} | pmid={r['pmid']}")

with open("lit_search/weight_evidence_details.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)
print("\nsaved")
