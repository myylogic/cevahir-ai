"""print title, authors, date and abstract from saved arXiv abs.html files (treated as data)."""
import html, re, sys, pathlib
for d in sys.argv[1:]:
    p = pathlib.Path(d) / "abs.html"
    if not p.exists():
        print(d, "missing"); continue
    t = p.read_text(errors="replace")
    def meta(name):
        m = re.search(r'<meta name="%s" content="([^"]*)"' % name, t)
        return html.unescape(m.group(1)) if m else ""
    title = meta("citation_title"); date = meta("citation_date")
    authors = "; ".join(html.unescape(a) for a in re.findall(r'<meta name="citation_author" content="([^"]*)"', t))
    m = re.search(r'<blockquote class="abstract[^"]*">(.*?)</blockquote>', t, re.S)
    ab = re.sub(r"<[^>]+>", " ", m.group(1)) if m else ""
    ab = re.sub(r"\s+", " ", html.unescape(ab)).strip()
    print("=" * 100); print(pathlib.Path(d).name, "|", title, "|", date); print(authors[:300]); print(ab[:2600])
