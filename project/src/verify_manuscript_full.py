"""全文上下文核查：孤儿引用、图表引用、内嵌图片、中英数字一致性、字数与标点约束。

产出：project/results/tables/manuscript_verification.json
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from common import TABLE_DIR, save_json, setup_logger

EN = Path("Insights_into_Imaging/manuscript/main_text.md")
ZH = Path("Insights_into_Imaging/manuscript/main_text_zh.md")
N_REFS = 60


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


def body(text: str, start_marker: str, stop_markers: tuple[str, ...]) -> str:
    i = text.find(start_marker)
    if i < 0:
        return text
    stops = [text.find(m, i) for m in stop_markers if text.find(m, i) > 0]
    return text[i: min(stops)] if stops else text[i:]


def numbers_in(section: str) -> set[float]:
    s = unicodedata.normalize("NFKC", section)
    # 只删除数字之间的千分位逗号；NFKC 会把中文逗号（，）转成英文逗号，不能一律删除
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    vals = set()
    # 注意：Python 的 \w 默认包含中日韩字符，中文紧邻数字时会漏检，故改用 ASCII 边界
    for m in re.finditer(r"(?<![A-Za-z0-9_.])(\d+(?:\.\d+)?)(?![A-Za-z0-9_])", s):
        try:
            vals.add(round(float(m.group(1)), 2))
        except ValueError:
            continue
    return vals


def citations(text: str) -> list[int]:
    sec = body(text, "## Introduction", ("## References", "## 参考文献"))
    return sorted({n for m in re.finditer(r"\[([\d,\-\s]+)\]", sec) for n in expand(m.group(1))})


def main() -> None:
    log = setup_logger("verify_manuscript_full")
    en, zh = EN.read_text(encoding="utf-8"), ZH.read_text(encoding="utf-8")
    report: dict = {}
    problems: list[str] = []

    # A. 引用完整性
    for name, txt in (("en", en), ("zh", zh)):
        cited = citations(txt)
        orphan = [i for i in range(1, N_REFS + 1) if i not in cited]
        out_of_range = [c for c in cited if c > N_REFS]
        report[f"citations_{name}"] = {"n_cited": len(cited), "orphans": orphan,
                                       "out_of_range": out_of_range}
        if orphan:
            problems.append(f"{name}: 孤儿引用 {orphan}")
        if out_of_range:
            problems.append(f"{name}: 越界引用 {out_of_range}")

    # B. 文献表条目数
    en_entries = len(re.findall(r"(?m)^\d+\. ", en[en.find("## References"):]))
    zh_entries = len(re.findall(r"(?m)^\d+\. ", zh[zh.rfind("\n1. "):]))
    report["reference_entries"] = {"en": en_entries, "zh": zh_entries, "expected": N_REFS}
    if en_entries != N_REFS:
        problems.append(f"英文文献表条目数 {en_entries} != {N_REFS}")
    if zh_entries != N_REFS:
        problems.append(f"中文文献表条目数 {zh_entries} != {N_REFS}")

    # C. 图表引用
    for name, txt, fig_pat, tab_pat in (("en", en, r"Fig\. (\d)", r"Table (\d)"),
                                        ("zh", zh, r"图 ?(\d)", r"表 ?(\d)")):
        sec = body(txt, "## Introduction", ("## References", "## 参考文献"))
        figs = {int(m.group(1)) for m in re.finditer(fig_pat, sec)}
        tabs = {int(m.group(1)) for m in re.finditer(tab_pat, sec)}
        report[f"callouts_{name}"] = {"figures": sorted(figs), "tables": sorted(tabs)}
        miss_f = [i for i in range(1, 7) if i not in figs]
        miss_t = [i for i in range(1, 6) if i not in tabs]
        if name == "en" and (miss_f or miss_t):
            problems.append(f"{name}: 未被正文引用的图 {miss_f}，表 {miss_t}")

    # D. 中文版内嵌图片
    imgs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", zh)
    missing = [p for p in imgs if not Path(p).exists()]
    report["zh_embedded_images"] = {"n": len(imgs), "missing": missing}
    if missing:
        problems.append(f"中文版图片路径缺失: {missing}")
    if len(imgs) < 7:
        problems.append(f"中文版内嵌图片数 {len(imgs)} < 7")

    # E. 中英数字一致性（摘要 + 结果）
    table_text = "\n".join(p.read_text(encoding="utf-8")
                           for p in sorted(Path("Insights_into_Imaging/tables").glob("*.md")))
    for label, s_en, s_zh in (
            ("abstract", body(en, "## Abstract", ("## Introduction",)),
             body(zh, "## 摘要", ("## 引言",))),
            ("results", body(en, "## Results", ("## Discussion",)) + "\n" + table_text,
             body(zh, "## 结果", ("## 讨论",)))):
        ne, nz = numbers_in(s_en), numbers_in(s_zh)
        only_en = sorted(ne - nz)
        only_zh = sorted(nz - ne)
        report[f"numbers_{label}"] = {"n_en": len(ne), "n_zh": len(nz),
                                      "only_en": only_en, "only_zh": only_zh}
        if only_en or only_zh:
            problems.append(f"{label}: 数字不一致 only_en={only_en} only_zh={only_zh}")

    # F. 约束
    main_body = body(en, "## Introduction", ("## Declarations",))
    words = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'’\-\.,%]*", main_body))
    abstract_words = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'’\-\.,%]*",
                                    body(en, "## Abstract", ("**Key points**",))))
    report["limits"] = {"main_words": words, "abstract_words": abstract_words,
                        "em_dash": main_body.count("—"), "semicolons": main_body.count(";")}
    if words > 3000:
        problems.append(f"正文 {words} 词 > 3000")
    if abstract_words > 350:
        problems.append(f"摘要 {abstract_words} 词 > 350")
    if main_body.count("—") > 3:
        problems.append("破折号超过 3 个")
    if main_body.count(";") > 2 * words / 1000:
        problems.append("分号密度超标")

    # G. 标题一致性
    expected = ("Formula-based body weight surrogates are unreliable in intensive care: a benchmark "
                "against portable chest radiography deep learning and 1,303 measured weights")
    report["title_en_ok"] = expected in en
    report["title_zh_present"] = "公式类体重替代值" in zh and "1,303" in zh
    if not report["title_en_ok"]:
        problems.append("英文标题与指定标题不一致")
    if not report["title_zh_present"]:
        problems.append("中文标题缺失")

    # H. 参考文献编号是否按正文首次出现顺序（Vancouver 制）
    for name, txt in (("en", en), ("zh", zh)):
        sec = body(txt, "## Introduction", ("## References", "## 参考文献"))
        order: list[int] = []
        for m in re.finditer(r"\[([\d,\-\s]+)\]", sec):
            for n in expand(m.group(1)):
                if n not in order:
                    order.append(n)
        sequential = order == list(range(1, len(order) + 1))
        report[f"citation_order_{name}"] = {"sequential": sequential,
                                            "first_ten": order[:10],
                                            "n": len(order)}
        if not sequential:
            problems.append(f"{name}: 参考文献未按首次出现顺序编号（前 10 位 {order[:10]}）")

    report["problems"] = problems
    save_json(TABLE_DIR / "manuscript_verification.json", report)
    log.info("核查完成：%d 个问题", len(problems))
    for p in problems:
        log.warning("PROBLEM %s", p)
    print(json.dumps({k: v for k, v in report.items() if k != "problems"},
                     ensure_ascii=False, indent=1)[:3000])


if __name__ == "__main__":
    main()
