"""Bing RSS + Baidu Xueshu probes for Chinese-language coverage."""

import html
import json
import re
import time
import xml.etree.ElementTree as ET

import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"}
S = requests.Session()
S.headers.update(UA)

QUERIES = [
    "胸片 预测 体重 深度学习",
    "胸片 估算 体重 人工智能",
    "胸部X线 体重 预测 模型",
    "X线 胸片 体重 卷积神经网络",
    "chest radiograph body weight estimation deep learning",
    "chest X-ray predict body weight convolutional neural network",
]


def bing_rss(q):
    r = S.get("https://cn.bing.com/search", params={"q": q, "format": "rss", "count": 20}, timeout=30)
    hits = []
    try:
        root = ET.fromstring(r.text)
        for item in root.iter("item"):
            hits.append({
                "title": (item.findtext("title") or "").strip(),
                "url": (item.findtext("link") or "").strip(),
                "snippet": html.unescape((item.findtext("description") or "").strip())[:300],
            })
    except ET.ParseError:
        hits.append({"error": "not xml", "head": r.text[:300]})
    return hits


def baidu_xueshu(q):
    r = S.get("https://xueshu.baidu.com/s", params={"wd": q}, timeout=30)
    text = r.text
    hits = []
    for m in re.finditer(r'<h3 class="t c_font">\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', text, re.S):
        title = re.sub(r"<[^>]+>", "", m.group(2))
        hits.append({"title": html.unescape(title).strip(), "url": m.group(1)})
    if not hits:
        hits.append({"error": f"no parse; status={r.status_code}; len={len(text)}"})
    return hits


out = {"bing_rss": {}, "baidu_xueshu": {}}
for q in QUERIES:
    try:
        out["bing_rss"][q] = bing_rss(q)
    except Exception as exc:  # noqa: BLE001
        out["bing_rss"][q] = [{"error": f"{type(exc).__name__}: {exc}"}]
    print("bing", q, len(out["bing_rss"][q]), flush=True)
    time.sleep(1.2)

for q in QUERIES[:4]:
    try:
        out["baidu_xueshu"][q] = baidu_xueshu(q)
    except Exception as exc:  # noqa: BLE001
        out["baidu_xueshu"][q] = [{"error": f"{type(exc).__name__}: {exc}"}]
    print("xueshu", q, len(out["baidu_xueshu"][q]), flush=True)
    time.sleep(1.2)

with open("lit_search/web_search2.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)

for engine, queries in out.items():
    for q, hits in queries.items():
        print("=" * 90)
        print(engine, "|", q)
        for h in hits[:12]:
            print("  -", h.get("title", ""), "|", h.get("url", ""))
            if h.get("snippet"):
                print("    ", h["snippet"][:220])
