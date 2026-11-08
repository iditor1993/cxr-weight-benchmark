"""把参考文献按正文首次出现顺序重新编号（Vancouver 制），同步英文稿、中文稿与文献文件。

步骤
    1. 从英文主稿正文提取全部引用编号，按首次出现顺序得到新编号；
    2. 重写两版正文中的引用（映射 + 升序 + 连续合并为区间）；
    3. 按新顺序重排两版文末的编号文献表；
    4. 输出旧→新对照表与重排后的独立文献表，便于溯源。

产出
    Insights_into_Imaging/references/citation_order_map.csv
    Insights_into_Imaging/references/references_ordered.md
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

EN = Path("Insights_into_Imaging/manuscript/main_text.md")
ZH = Path("Insights_into_Imaging/manuscript/main_text_zh.md")
REF_JSON = Path("Insights_into_Imaging/references/references.json")
OUT_MAP = Path("Insights_into_Imaging/references/citation_order_map.csv")
OUT_LIST = Path("Insights_into_Imaging/references/references_ordered.md")


def expand(group: str) -> list[int]:
    out = []
    for part in group.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part.isdigit():
            out.append(int(part))
    return out


def compress(nums: list[int]) -> str:
    nums = sorted(set(nums))
    parts, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        parts.append(str(nums[i]) if j == i else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return ",".join(parts)


def split_file(text: str) -> tuple[str, list[str]]:
    """返回 (正文, 文献表条目原文列表)。"""
    marker = "\n1. "
    pos = text.rfind(marker)
    body, refs = text[:pos + 1], text[pos + 1:]
    entries = re.findall(r"(?ms)^\d+\.\s(.*?)(?=^\d+\.\s|\Z)", refs)
    return body, [e.strip() for e in entries]


def first_appearance_order(body: str) -> list[int]:
    order: list[int] = []
    for m in re.finditer(r"\[([\d,\-\s]+)\]", body):
        for n in expand(m.group(1)):
            if n not in order:
                order.append(n)
    return order


def renumber_body(body: str, mapping: dict[int, int]) -> str:
    def repl(m: re.Match) -> str:
        return "[" + compress([mapping[n] for n in expand(m.group(1))]) + "]"
    return re.sub(r"\[([\d,\-\s]+)\]", repl, body)


def main() -> None:
    en_text = EN.read_text(encoding="utf-8")
    zh_text = ZH.read_text(encoding="utf-8")
    en_body, en_refs = split_file(en_text)
    zh_body, zh_refs = split_file(zh_text)
    assert len(en_refs) == len(zh_refs) == 60, (len(en_refs), len(zh_refs))

    order = first_appearance_order(en_body)
    n = len(order)
    missing = [i for i in range(1, n + 1) if i not in order]
    print(f"首次出现顺序覆盖 {len(order)} 条；未被引用: {missing}")
    assert not missing, missing

    mapping = {old: new for new, old in enumerate(order, 1)}

    entries_text = "\n".join(f"{i + 1}. {en_refs[old - 1]}" for i, old in enumerate(order))
    EN.write_text(renumber_body(en_body, mapping).rstrip() + "\n\n" + entries_text + "\n",
                  encoding="utf-8")
    ZH.write_text(renumber_body(zh_body, mapping).rstrip() + "\n\n" + entries_text + "\n",
                  encoding="utf-8")

    refs = json.loads(REF_JSON.read_text(encoding="utf-8"))
    rows = []
    for old, new in sorted(mapping.items(), key=lambda kv: kv[1]):
        r = refs[old - 1]
        rows.append({"new_number": new, "old_number": old, "label": r["label"],
                     "first_author": (r["authors"] or "").split(",")[0],
                     "year": r["year"], "title": r["title"], "doi": r["doi"], "pmid": r["pmid"]})
    with OUT_MAP.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    OUT_LIST.write_text("# 参考文献（按正文首次出现顺序编号）\n\n" +
                        "\n".join(f"{r['new_number']}. {r['title']}. *{r['year']}*. doi:{r['doi']}"
                                  for r in rows) + "\n", encoding="utf-8")
    print("已按出现顺序重排；对照表:", OUT_MAP)


if __name__ == "__main__":
    main()
