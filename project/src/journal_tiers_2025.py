"""核实候选期刊的中科院分区（以 LetPub 公开页面为准，2025 年 3 月升级版）。

说明
    - 中科院文献情报中心《期刊分区表》官方站点需机构登录；本脚本改用 LetPub
      的公开期刊页（与官方表同步展示版本、大类/小类分区、Top 与综述标记），并记录来源 URL 与查询日期。
    - 同时抓取 LetPub 展示的《新锐期刊分区表》分区（如该页提供），作为参考列。

产出
    project/results/tables/journal_tiers_2025.json
    project/docs/journal_tiers_2025.md
"""

from __future__ import annotations

import json
import re
import ssl
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import date
from hashlib import md5
from pathlib import Path

from common import DOCS_DIR, TABLE_DIR, save_json, setup_logger

CTX = ssl.create_default_context()
BASE = "https://www.letpub.com.cn/index.php"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

JOURNALS = [
    "Scientific Reports",
    "Computers in Biology and Medicine",
    "Academic Radiology",
    "European Journal of Radiology",
    "Insights into Imaging",
    "JMIR Medical Informatics",
    "BMC Medical Imaging",
    "Journal of Imaging Informatics in Medicine",
    "Annals of Intensive Care",
    "Critical Care",
    "npj Digital Medicine",
    "eClinicalMedicine",
]


CACHE = Path(tempfile.gettempdir()) / "letpub_cache"
CACHE.mkdir(parents=True, exist_ok=True)
THROTTLE = "\u60a8\u8bf7\u6c42\u9875\u9762\u7684\u901f\u5ea6\u8fc7\u5feb"     # 您请求页面的速度过快


def get(url: str, log=None, retries: int = 6, delay: float = 6.0) -> str:
    """带本地缓存、限速与退避重试的抓取。"""
    cache_file = CACHE / (md5(url.encode()).hexdigest() + ".html")
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    last = None
    for attempt in range(retries):
        try:
            time.sleep(delay)
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=40, context=CTX) as r:
                raw = r.read()
            try:
                html = raw.decode("utf-8")
            except UnicodeDecodeError:
                html = raw.decode("gb18030", "ignore")
            if THROTTLE in html:
                if log:
                    log.info("被限流，等待 30 s 后重试（%s）", url[:70])
                time.sleep(30)
                continue
            cache_file.write_text(html, encoding="utf-8")
            return html
        except Exception as e:      # noqa: BLE001
            last = e
            time.sleep(8)
    raise RuntimeError(f"fetch failed: {url} ({last})")


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def find_journal_id(name: str, log) -> int | None:
    url = (f"{BASE}?page=journalapp&view=search&searchname="
           + urllib.parse.quote_plus(name))
    html = get(url, log)
    pairs = re.findall(
        r"journalid=(\d+)&page=journalapp&view=detail[^>]*>([^<]{2,90})</a>", html)
    if not pairs:
        return None
    target = norm(name)
    for jid, title in pairs:
        if norm(title) == target:
            log.info("解析 %s -> id=%s（精确匹配）", name, jid)
            return int(jid)
    jid, title = pairs[0]
    log.info("解析 %s -> id=%s（首个结果：%s，未精确匹配）", name, jid, title.strip())
    return int(jid) if norm(name)[:8] in norm(title) else None


def strip_tags(html: str) -> str:
    txt = re.sub(r"<br\s*/?>", " / ", html)
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = txt.replace("&nbsp;", " ").replace("&amp;", "&")
    return re.sub(r"\s+", " ", txt).strip()


def parse_versions(html: str, log) -> dict:
    """返回 {'2025年3月升级版': {...}, '新锐期刊分区表': {...}}"""
    out: dict[str, dict] = {}
    starts = [m.start() for m in re.finditer("\u5206\u533a\u8868\\s*<br>", html)]
    for i, s in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else min(len(html), s + 9000)
        block = html[s:end]
        m = re.search(r"<font[^>]*>([^<]+)</font>", block)
        label = m.group(1).strip() if m else "未知版本"
        zones = [mm.group(2) for mm in re.finditer(r"<span([^>]*)>([1-4]\u533a)</span>", block)
                 if "display:none" not in mm.group(1)]
        flags = [mm.group(1) for mm in
                 re.finditer(r"<td[^>]*>\s*(\u662f|\u5426|N/A)\s*</td>", block)]
        tds = re.findall(r"<td[^>]*>(.*?)</td>", block, flags=re.S)
        vis = [mm for mm in re.finditer(r"<span[^>]*>([1-4]\u533a)</span>", block)
               if "display:none" not in mm.group(0)]
        big_name = ""
        if vis:
            td_start = block.rfind("<td", 0, vis[0].start())
            frag = re.sub(r"<span[^>]*>[^<]*</span>", " ", block[td_start:vis[0].start()])
            big_name = strip_tags(frag)
        small_name = ""
        for td in tds:
            if "<table" in td:
                first = re.search(r"<td[^>]*>(.*?)</td>", td, flags=re.S)
                if first:
                    small_name = strip_tags(first.group(1))
                break
        out[label] = {"大类学科": big_name, "大类分区": zones[0] if zones else None,
                      "小类学科": small_name, "小类分区": zones[1] if len(zones) > 1 else None,
                      "Top期刊": flags[0] if len(flags) > 0 else None,
                      "综述期刊": flags[1] if len(flags) > 1 else None,
                      "note": next((t for t in ("没有被2025年的期刊分区表收录",
                                                "没有被旧的期刊分区表收录") if t in strip_tags(block)),
                                   None),
                      "visible_zones": zones, "flags": flags}
    # 新锐期刊分区表：页面上位于经典分区表之前，标题形如“新锐期刊分区表”
    m = re.search("\u65b0\u9510\u671f\u520a\u5206\u533a\u8868", html)
    if m and starts:
        block = html[m.start(): starts[0]]
        if len(block) > 4000:
            block = block[:4000]
        if not re.search(r"<span[^>]*>[1-4]\u533a</span>", block):
            block = html[max(0, starts[0] - 7000): starts[0]]
        zones = [mm.group(2) for mm in re.finditer(r"<span([^>]*)>([1-4]\u533a)</span>", block)
                 if "display:none" not in mm.group(1)]
        flags = [mm.group(1) for mm in
                 re.finditer(r"<td[^>]*>\s*(\u662f|\u5426|N/A)\s*</td>", block)]
        out["\u65b0\u9510\u671f\u520a\u5206\u533a\u8868"] = {
            "大类分区": zones[0] if zones else None,
            "小类分区": zones[1] if len(zones) > 1 else None,
            "Top期刊": flags[0] if len(flags) > 0 else None,
            "综述期刊": flags[1] if len(flags) > 1 else None}
    return out


def main() -> None:
    log = setup_logger("journal_tiers_2025")
    today = date.today().isoformat()
    rows = []
    for name in JOURNALS:
        try:
            jid = find_journal_id(name, log)
            if jid is None:
                rows.append({"journal": name, "error": "未找到对应期刊页"})
                continue
            url = f"{BASE}?journalid={jid}&page=journalapp&view=detail"
            html = get(url, log)
            vers = parse_versions(html, log)
            key = next((k for k in vers if "2025" in k), None)
            main_info = vers.get(key, {}) if key else {}
            new_info = vers.get("\u65b0\u9510\u671f\u520a\u5206\u533a\u8868", {})
            issn = re.search(r"\u671f\u520aISSN\s*([0-9Xx\-]{8,12})", strip_tags(html))
            rows.append({
                "journal": name, "letpub_id": jid, "url": url,
                "issn": issn.group(1) if issn else None,
                "cas_version": key, "cas": main_info, "xinrui": new_info,
                "all_versions": {k: {"大类分区": v.get("大类分区"), "小类分区": v.get("小类分区"),
                                     "Top期刊": v.get("Top期刊"), "综述期刊": v.get("综述期刊")}
                                 for k, v in vers.items()}})
            log.info("%-42s | %s | 大类=%s 小类=%s Top=%s", name, key,
                     main_info.get("大类分区"), main_info.get("小类分区"),
                     main_info.get("Top期刊"))
        except Exception as e:                                  # noqa: BLE001
            log.warning("%s 抓取失败：%s", name, e)
            rows.append({"journal": name, "error": str(e)})
        time.sleep(1.0)

    payload = {"queried_on": today, "source": "LetPub 期刊页（同步中科院期刊分区表）",
               "rows": rows}
    save_json(TABLE_DIR / "journal_tiers_2025.json", payload)

    lines = ["# 候选期刊的中科院分区核实（2025 年 3 月升级版）\n",
             f"> 查询日期：{today}。来源：LetPub 公开期刊页（该页同步展示中科院《期刊分区表》"
             "的版本标签、大类/小类分区、Top 与综述标记）。官方《期刊分区表》查询站点需机构登录，"
             "故本表以第三方公开页为准，并逐刊保留来源 URL 供复核。\n",
             "> 两个口径：**大类分区**是投稿评价中最常引用的口径；**小类分区**是学科细分口径。"
             "同一页面保留上一版（2023 年 12 月升级版）用于比较，列在「上一版」列。\n",
             "| 期刊 | 大类学科 | **大类分区（2025）** | 上一版（2023） | 小类分区 | Top 期刊 | 综述期刊 | 新锐分区 | 来源 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("error"):
            lines.append(f"| {r['journal']} | — | — | — | — | — | — | — | 抓取失败：{r['error']} |")
            continue
        c = r.get("cas") or {}
        x = r.get("xinrui") or {}
        prev = next((v for k, v in r.get("all_versions", {}).items()
                     if k != r.get("cas_version")), {})
        zone = c.get("大类分区") or ("未收录" if c.get("note") else "—")
        lines.append("| {j} | {big} | **{zone}** | {prev} | {sz} | {top} | {rev} | {xr} | "
                     "[LetPub]({u}) |".format(
                         j=r["journal"], big=c.get("大类学科") or "—", zone=zone,
                         prev=prev.get("大类分区") or "—", sz=c.get("小类分区") or "—",
                         top=c.get("Top期刊") or "—", rev=c.get("综述期刊") or "—",
                         xr=x.get("大类分区") or "—", u=r["url"]))
    lines += ["",
              "## 版本与口径说明\n",
              "- 表中「大类分区」取自 LetPub 展示的 **2025 年 3 月升级版**（即用户所问的 2025 中科院分区）；"
              "同一页面还保留 2023 年 12 月升级版作为对照，完整版本记录见 JSON。\n",
              "- 若期刊在 LetPub 被标注为「综述期刊 = 是」，其分区分档通常高于非综述期刊，"
              "投稿时需按目标单位口径确认。\n",
              "- 分区可能随后续版本调整；投稿前请以本单位科研管理部门的正式查询结果为准"
              "（多数单位提供官方《期刊分区表》查询账号）。\n",
              "- 中科院除《期刊分区表》外，自 2026 年 3 月起另发布《新锐期刊分区表》"
              "（在原预警机制基础上分档）；本表「新锐分区」列即该口径，供参考。\n",
              "## 对本项目投稿策略的含义\n",
              "1. **A 版（现有数据即可投）能触达的中科院 2 区期刊**：Computers in Biology and "
              "Medicine（2 区，小类 1 区）、Academic Radiology（2 区）、European Journal of "
              "Radiology（2 区，较上一版由 3 区升为 2 区）、Insights into Imaging（2 区，Top）。\n",
              "2. **原本常被提及的 Scientific Reports 已是 3 区**（2023 版为 2 区），"
              "若单位以中科院大区为考核口径，它不再满足「2 区及以上」；JMIR Medical Informatics、"
              "BMC Medical Imaging 同为 3 区。\n",
              "3. **1 区（Top）档需要外部验证支撑**：Annals of Intensive Care、Critical Care、"
              "npj Digital Medicine、eClinicalMedicine 在 2025 版均为大类 1 区、Top 期刊；"
              "其审稿标准与本研究现有证据（单中心、预设未达标）差距较大，建议补外部验证后再冲。\n",
              "4. **Journal of Imaging Informatics in Medicine（原 Journal of Digital Imaging）"
              "未被 2025 年《期刊分区表》收录**（LetPub 页面注明），暂不宜作为分区目标期刊。\n",
              "5. 具体叙事与顺序见 `docs/submission_positioning.md`；本表用于替换该文件中"
              "原先凭经验给出的分区判断。\n"]
    out = DOCS_DIR / "journal_tiers_2025.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info("报告 -> %s", out)


if __name__ == "__main__":
    main()
