"""Chinese-language literature probes via Bing + Wanfang (best effort)."""

import html
import json
import re
import time
import urllib.parse

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
    "胸片 预测 体重 查新",
]


def bing(q, count=20):
    url = "https://cn.bing.com/search"
    r = S.get(url, params={"q": q, "count": count, "setlang": "zh-Hans"}, timeout=30)
    text = r.text
    hits = []
    for m in re.finditer(r'<li class="b_algo".*?</li>', text, re.S):
        block = m.group(0)
        tm = re.search(r'<h2>.*?<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
        if not tm:
            continue
        link, title = tm.group(1), re.sub(r"<[^>]+>", "", tm.group(2))
        snippet = ""
        sm = re.search(r'<p[^>]*>(.*?)</p>', block, re.S)
        if sm:
            snippet = re.sub(r"<[^>]+>", "", sm.group(1))
        hits.append({"title": html.unescape(title).strip(),
                     "url": html.unescape(link),
                     "snippet": html.unescape(snippet).strip()[:300]})
    return hits


def wanfang(q):
    url = "https://s.wanfangdata.com.cn/paper"
    r = S.get(url, params={"q": q}, timeout=30)
    text = r.text
    hits = []
    for m in re.finditer(r'<a[^>]+href="(https://d\.wanfangdata\.com\.cn/paper/[^"]+)"[^>]*>(.*?)</a>', text, re.S):
        title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        if title:
            hits.append({"title": html.unescape(title), "url": m.group(1)})
    if not hits:
        for m in re.finditer(r'"title":"(.*?)".*?"link":"(.*?)"', text):
            hits.append({"title": html.unescape(m.group(1)), "url": html.unescape(m.group(2))})
    return hits[:25]


out = {"bing": {}, "wanfang": {}}
for q in QUERIES:
    try:
        out["bing"][q] = bing(q)
    except Exception as exc:  # noqa: BLE001
        out["bing"][q] = [{"error": f"{type(exc).__name__}: {exc}"}]
    print("bing", q, len(out["bing"][q]), flush=True)
    time.sleep(1)

for q in QUERIES[:4]:
    try:
        out["wanfang"][q] = wanfang(q)
    except Exception as exc:  # noqa: BLE001
        out["wanfang"][q] = [{"error": f"{type(exc).__name__}: {exc}"}]
    print("wanfang", q, len(out["wanfang"][q]), flush=True)
    time.sleep(1)

with open("lit_search/web_search.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)

for engine, queries in out.items():
    for q, hits in queries.items():
        print("=" * 90)
        print(engine, "|", q)
        for h in hits[:12]:
            print("  -", h.get("title", ""), "|", h.get("url", ""))
            if h.get("snippet"):
                print("    ", h["snippet"][:200])
