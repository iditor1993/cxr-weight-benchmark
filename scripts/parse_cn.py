import html
import re


def clean(s):
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


for name in ("bing_html", "sogou", "so360"):
    path = f"lit_search/raw_{name}.html"
    try:
        text = open(path, encoding="utf-8").read()
    except FileNotFoundError:
        continue
    print("=" * 100)
    print(name)
    if name == "bing_html":
        blocks = re.findall(r'<li class="b_algo".*?(?=<li class="b_algo"|</ol>)', text, re.S)
        if not blocks:
            blocks = re.findall(r'<h2><a[^>]+href="[^"]+"[^>]*>.*?</h2>', text, re.S)
        for b in blocks[:20]:
            m = re.search(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', b, re.S)
            if not m:
                continue
            print("  -", clean(m.group(2))[:150])
            print("    ", html.unescape(m.group(1))[:200])
            sn = re.search(r'<p[^>]*>(.*?)</p>', b, re.S)
            if sn:
                print("    ", clean(sn.group(1))[:250])
    elif name == "sogou":
        for m in re.finditer(r'<h3[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', text, re.S):
            print("  -", clean(m.group(2))[:150])
            print("    ", html.unescape(m.group(1))[:200])
    else:
        for m in re.finditer(r'<h3[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', text, re.S):
            print("  -", clean(m.group(2))[:150])
            print("    ", html.unescape(m.group(1))[:200])
