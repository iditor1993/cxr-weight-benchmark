import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

urls = [
    "https://www.bing.com/search?q=test",
    "https://cn.bing.com/search?q=test",
    "https://www.baidu.com/s?wd=test",
    "https://kns.cnki.net/kns8s/defaultresult/index",
    "https://www.wanfangdata.com.cn/index.html",
    "https://www.google.com/search?q=test",
    "https://scholar.google.com/scholar?q=test",
    "https://api.crossref.org/works?query=test&rows=1",
    "https://pubmed.ncbi.nlm.nih.gov/?term=test",
]
for u in urls:
    try:
        r = requests.get(u, headers=UA, timeout=25)
        print(u.split("/")[2], r.status_code, len(r.content))
    except Exception as exc:  # noqa: BLE001
        print(u.split("/")[2], "ERR", type(exc).__name__, str(exc)[:100])
