"""稿件机械核查：字数、标点、AI 高频词、引用编号覆盖与孤儿引用。"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

TEXT = Path("Insights_into_Imaging/manuscript/main_text.md")
REFS = Path("Insights_into_Imaging/references/references.json")
FLAGGED = ["delve", "tapestry", "landscape", "pivotal", "crucial", "foster", "showcase", "testament",
           "navigate", "leverage", "realm", "embark", "underscore", "multifaceted", "nuanced",
           "comprehensive", "robust", "intricate", "cornerstone", "paradigm", "synergy", "holistic",
           "streamline", "cutting-edge", "groundbreaking"]


def section(text: str, start: str, end: str | None) -> str:
    i = text.find(start)
    if i < 0:
        return ""
    j = text.find(end) if end else len(text)
    return text[i:j if j > 0 else len(text)]


def main() -> None:
    text = TEXT.read_text(encoding="utf-8")
    refs = json.loads(REFS.read_text(encoding="utf-8"))
    n_refs = len(refs)

    body = section(text, "## Introduction", "## Declarations")
    abstract = section(text, "## Abstract", "## Introduction")
    words = lambda s: len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'’\-\.,%]*", s))
    print(f"正文（Introduction→Discussion）词数: {words(body)}")
    print(f"摘要词数: {words(abstract)}")
    print(f"正文+摘要词数: {words(body) + words(abstract)}")

    print(f"\nem dash 数量: {body.count(chr(8212)) + abstract.count(chr(8212))}（限 ≤3）")
    semis = body.count(";")
    print(f"分号数量: {semis}（限 ≤{2 * words(body) / 1000:.1f}）")
    hits = {t: len(re.findall(rf"\b{re.escape(t)}\b", text, flags=re.I)) for t in FLAGGED}
    hits = {k: v for k, v in hits.items() if v}
    print(f"AI 高频词命中: {hits or '无'}")

    cited = sorted({int(x) for m in re.finditer(r"\[([\d,\-\s]+)\]", body + abstract)
                    for part in m.group(1).split(",")
                    for x in ([part] if "-" not in part else
                              range(int(part.split("-")[0]), int(part.split("-")[1]) + 1))})
    print(f"\n引用编号数: {len(cited)} / 文献表 {n_refs} 篇")
    orphan = [i for i in range(1, n_refs + 1) if i not in cited]
    print(f"孤儿引用（未被正文引用）: {orphan or '无'}")
    over = [i for i in cited if i > n_refs]
    print(f"越界引用: {over or '无'}")


if __name__ == "__main__":
    main()
