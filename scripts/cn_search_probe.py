"""Probe Chinese search engines for the specific query; save raw HTML for inspection."""

import re

import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept-Language": "zh-CN,zh;q=0.9"}
S = requests.Session()
S.headers.update(UA)

TESTS = [
    ("bing_html", "https://www.bing.com/search", {"q": '"胸片" "体重" 预测 深度学习', "mkt": "zh-CN", "setlang": "zh-Hans"}),
    ("sogou", "https://www.sogou.com/web", {"query": "胸片 体重 预测 深度学习"}),
    ("so360", "https://www.so.com/s", {"q": "胸片 体重 预测 深度学习"}),
    ("cnki", "https://kns.cnki.net/kns8s/defaultresult/index", {"kw": "胸片 体重 预测"}),
]

for name, url, params in TESTS:
    try:
        r = S.get(url, params=params, timeout=30)
        text = r.text
        with open(f"lit_search/raw_{name}.html", "w", encoding="utf-8") as fh:
            fh.write(text)
        titles = re.findall(r"<title>(.*?)</title>", text, re.S)[:2]
        print(name, r.status_code, len(text), "|", [t.strip()[:80] for t in titles])
    except Exception as exc:  # noqa: BLE001
        print(name, "ERR", type(exc).__name__, str(exc)[:120])
