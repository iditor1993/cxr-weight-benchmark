"""参考文献真实性**实时**再核验：逐条通过 Europe PMC 解析 DOI/PMID 并比对元数据。

核查项
    1. DOI/PMID 能否解析到记录（存在性）；
    2. 标题相似度（≥0.90）、年份一致、期刊名可匹配；
    3. 是否带有撤稿/关注声明（retraction / expression of concern）；
    4. 参考文献表内部重复 DOI。

产出：project/results/tables/reference_verification.json（含逐条结论）
"""

from __future__ import annotations

import json
import re
import ssl
import time
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

from common import TABLE_DIR, save_json, setup_logger

CTX = ssl.create_default_context()
UA = {"User-Agent": "Mozilla/5.0"}
REFS = Path("Insights_into_Imaging/references/references.json")
FLAG_WORDS = ("retract", "withdraw", "expression of concern", "concern has been")


def fetch(url: str, retries: int = 3) -> bytes:
    last = None
    for _ in range(retries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                          timeout=40, context=CTX).read()
        except Exception as e:                      # noqa: BLE001
            last = e
            time.sleep(2)
    raise RuntimeError(str(last))


def lookup(ref: dict) -> dict | None:
    q = f'DOI:"{ref["doi"]}"' if ref.get("doi") else f'EXT_ID:{ref["pmid"]}'
    url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query="
           + urllib.parse.quote(q) + "&format=json&pageSize=1&resultType=core")
    res = json.loads(fetch(url).decode())["resultList"]["result"]
    return res[0] if res else None


def norm(s: str) -> str:
    return re.sub(r"\W+", " ", (s or "").lower()).strip()


def main() -> None:
    log = setup_logger("verify_references_live")
    refs = json.loads(REFS.read_text(encoding="utf-8"))
    rows, problems = [], []
    seen_doi = {}
    for i, r in enumerate(refs, 1):
        rec = None
        err = ""
        try:
            rec = lookup(r)
        except Exception as e:                      # noqa: BLE001
            err = type(e).__name__
        if rec is None:
            problems.append(f"#{i} 无法解析（{r.get('doi') or r.get('pmid')}）{err}")
            rows.append({"n": i, "label": r["label"], "resolved": False, "error": err})
            time.sleep(0.3)
            continue
        title_sim = SequenceMatcher(None, norm(r["title"]), norm(rec.get("title"))).ratio()
        year_ok = str(r["year"]) == str(rec.get("pubYear"))
        j_ref, j_rec = norm(r.get("journal") or ""), norm((rec.get("journalInfo", {})
                                                           .get("journal", {}) or {}).get("title")
                                                          or rec.get("journalTitle") or "")
        journal_ok = bool(j_ref) and (j_ref in j_rec or j_rec in j_ref)
        pubtypes = " ".join(rec.get("pubTypeList", {}).get("pubType", []) or []).lower()
        flagged = [w for w in FLAG_WORDS if w in pubtypes or w in (rec.get("title") or "").lower()]
        doi = (r.get("doi") or "").lower()
        dup = seen_doi.get(doi) if doi else None
        if doi:
            seen_doi[doi] = i
        ok = title_sim >= 0.90 and year_ok and journal_ok and not flagged and not dup
        rows.append({"n": i, "label": r["label"], "resolved": True, "title_similarity": round(title_sim, 3),
                     "year_match": year_ok, "journal_match": journal_ok,
                     "flag_words": flagged, "duplicate_of": dup, "ok": ok})
        if not ok:
            problems.append(f"#{i} {r['label']}: title_sim={title_sim:.2f} year_ok={year_ok} "
                            f"journal_ok={journal_ok} flags={flagged} dup={dup}")
        time.sleep(0.3)

    payload = {"n_total": len(refs), "n_ok": sum(1 for x in rows if x.get("ok")),
               "problems": problems, "rows": rows}
    save_json(TABLE_DIR / "reference_verification.json", payload)
    log.info("参考文献实时核验：%d/%d 通过", payload["n_ok"], payload["n_total"])
    for p in problems:
        log.warning("PROBLEM %s", p)


if __name__ == "__main__":
    main()
