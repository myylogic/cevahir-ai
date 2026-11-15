# -*- coding: utf-8 -*-
"""Zihin, the internet adapter: Google through the Custom Search JSON API, and the text of web pages.

What is pinned here
-------------------
* ``search`` asks the official API (key, cx, q, num at most 5, hl and lr for Turkish) and returns the result URLs
  best first; titles and snippets are kept; links that are not http(s) are left out; no results give [].
* HTML is reduced to its readable text: script, style, navigation, header, footer, aside and form content are
  dropped, paragraphs, headings and list items are one line each, whitespace is collapsed.
* ``extract`` sends the agent's User-Agent with a 20 s timeout and never reads more than 2 MB.
* The declared charset decodes a page (a UTF-8 byte order mark first, then the HTTP header, else a <meta> tag),
  otherwise utf-8 with replacement.
* Non-HTML content (pdf, images) gives "" without its body being read; so does a page that cannot be downloaded.
* Only http(s) addresses are opened: any other scheme is refused before a request is made.
* ``default_source`` is Google only when both GOOGLE_API_KEY and GOOGLE_CSE_ID are set, otherwise Wikipedia.

No socket is opened: the API fetcher and ``urlopen`` are stubs.
"""
from __future__ import annotations

import email.message
import socket
import urllib.error
import urllib.parse

import pytest

from zihin.web import AGENT, MAX_PAGE, GoogleSearch, Wikipedia, default_source, html_text


def no_network(*args, **kwargs):
    raise AssertionError("tests must not reach the network")


@pytest.fixture(autouse=True)
def _no_sockets(monkeypatch):
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)


class FakeResponse:
    """Enough of what ``urllib.request.urlopen`` returns: headers, a bounded read, a context manager."""

    def __init__(self, body: bytes, content_type: str) -> None:
        self.headers = email.message.Message()
        self.headers["Content-Type"] = content_type
        self.body, self.reads = body, []

    def read(self, amount: int = -1) -> bytes:
        self.reads.append(amount)
        return self.body if amount < 0 else self.body[:amount]

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc) -> bool:
        return False


def page_source(body: bytes, content_type: str = "text/html; charset=utf-8"):
    """A GoogleSearch whose ``urlopen`` serves one canned page and records every request."""
    response, requests = FakeResponse(body, content_type), []

    def urlopen(request, timeout):
        requests.append((request.full_url, request.get_header("User-agent"), timeout))
        return response

    return GoogleSearch("test-key", "test-cx", fetch=no_network, urlopen=urlopen), response, requests


SAMPLE = {     # the shape of a Custom Search JSON API answer, trimmed
    "kind": "customsearch#search",
    "searchInformation": {"totalResults": "4"},
    "items": [
        {"kind": "customsearch#result", "title": "Kuantum mekaniği", "link": "https://example.org/wiki/Kuantum",
         "displayLink": "example.org", "snippet": "Kuantum mekaniği atom altı dünyayı inceler."},
        {"kind": "customsearch#result", "title": "Kuantum nedir?", "link": "http://example.com/kuantum",
         "displayLink": "example.com", "snippet": "Kuantum kavramı ..."},
        {"kind": "customsearch#result", "title": "Ders notu", "link": "https://example.net/notlar.pdf",
         "mime": "application/pdf", "fileFormat": "PDF/Adobe Acrobat", "snippet": "Notlar"},
        {"kind": "customsearch#result", "title": "Arşiv", "link": "ftp://example.net/arsiv.html"},
    ],
}


def test_search_asks_the_custom_search_api_and_returns_result_urls():
    asked = []

    def fetch(url):
        asked.append(url)
        return SAMPLE

    source = GoogleSearch("test-key", "test-cx", fetch=fetch, urlopen=no_network)
    links = source.search("kuantum fiziği", limit=10)
    assert links == ["https://example.org/wiki/Kuantum", "http://example.com/kuantum",
                     "https://example.net/notlar.pdf"]
    assert source.found["http://example.com/kuantum"] == ("Kuantum nedir?", "Kuantum kavramı ...")
    assert source.address(links[0]) == links[0]
    base, _, query = asked[0].partition("?")
    assert base == "https://www.googleapis.com/customsearch/v1"
    assert urllib.parse.parse_qs(query) == {"key": ["test-key"], "cx": ["test-cx"], "q": ["kuantum fiziği"],
                                            "num": ["5"], "hl": ["tr"], "lr": ["lang_tr"]}
    source.search("kuantum", limit=2)
    assert urllib.parse.parse_qs(asked[1].partition("?")[2])["num"] == ["2"]
    assert GoogleSearch("test-key", "test-cx", fetch=lambda url: {"kind": "customsearch#search"}).search("x") == []


HTML = """<!doctype html>
<html><head><title>Sayfa başlığı</title><meta charset="utf-8">
<style>p { color: red; }</style><script>var menu = "<p>gizli</p>";</script></head>
<body>
<header><a href="/">Ana sayfa</a> <a href="/giris">Giriş</a></header>
<nav><ul><li><a href="/a">Menü bir</a></li><li><a href="/b">Menü iki</a></li></ul></nav>
<main><article>
<h1>Kuantum   mekaniği</h1>
<p>Kuantum mekaniği, atom altı parçacıkların
   davranışını inceler.</p>
<p>Işık hem dalga <b>hem de</b> parçacık gibi davranır.<br>Bu ikiliğe dalga&nbsp;parçacık ikiliği denir.</p>
<ul><li>Planck sabiti</li><li>Belirsizlik ilkesi</li></ul>
<noscript><p>JavaScript kapalı</p></noscript>
<form><label>E-posta</label><input name="e"><button>Abone ol</button></form>
</article>
<aside>Reklam alanı</aside></main>
<footer><p>© 2026 Örnek</p></footer>
</body></html>"""

TEXT = ("Kuantum mekaniği\n"
        "Kuantum mekaniği, atom altı parçacıkların davranışını inceler.\n"
        "Işık hem dalga hem de parçacık gibi davranır.\n"
        "Bu ikiliğe dalga parçacık ikiliği denir.\n"
        "Planck sabiti\n"
        "Belirsizlik ilkesi")


def test_html_is_reduced_to_its_readable_text():
    assert html_text(HTML) == TEXT


def test_extract_downloads_an_html_page_and_returns_its_text():
    source, response, requests = page_source(HTML.encode("utf-8"))
    assert source.extract("https://example.org/kuantum") == TEXT
    assert requests == [("https://example.org/kuantum", AGENT, 20.0)]
    assert MAX_PAGE == 2 * 1024 * 1024
    assert response.reads and all(0 < amount <= MAX_PAGE for amount in response.reads)


SENTENCE = "Ağaç gölgesinde şarkı söyleyen çocuk."


@pytest.mark.parametrize("content_type, body", [
    ("text/html; charset=windows-1254", ("<p>%s</p>" % SENTENCE).encode("cp1254")),
    ("text/html", ('<meta charset="iso-8859-9"><p>%s</p>' % SENTENCE).encode("iso-8859-9")),
    ("text/html", ('<meta http-equiv="Content-Type" content="text/html; charset=windows-1254"><p>%s</p>'
                   % SENTENCE).encode("cp1254")),
    ("text/html", ("<p>%s</p>" % SENTENCE).encode("utf-8")),
    ("text/html; charset=iso-8859-1", b"\xef\xbb\xbf" + ("<p>%s</p>" % SENTENCE).encode("utf-8")),
])
def test_the_declared_charset_decodes_the_page(content_type, body):
    source, _, _ = page_source(body, content_type)
    assert source.extract("https://example.org/") == SENTENCE


def test_undecodable_bytes_are_replaced():
    source, _, _ = page_source(b"<p>bozuk \xff\xfe bayt</p>", "text/html; charset=no-such-charset")
    assert source.extract("https://example.org/") == "bozuk �� bayt"


@pytest.mark.parametrize("content_type", ["application/pdf", "image/png", "application/octet-stream"])
def test_non_html_content_is_skipped_without_reading_it(content_type):
    source, response, requests = page_source(b"%PDF-1.7 binary", content_type)
    assert source.extract("https://example.net/notlar.pdf") == ""
    assert len(requests) == 1 and response.reads == []


@pytest.mark.parametrize("error", [urllib.error.URLError("unreachable"), TimeoutError("timed out"),
                                   urllib.error.HTTPError("https://example.org/", 403, "Forbidden", None, None)])
def test_a_page_that_cannot_be_downloaded_is_skipped(error):
    def urlopen(request, timeout):
        raise error

    source = GoogleSearch("test-key", "test-cx", fetch=no_network, urlopen=urlopen)
    assert source.extract("https://example.org/") == ""


@pytest.mark.parametrize("url", ["file:///etc/passwd", " file:///etc/passwd", "<URL:file:///etc/passwd>",
                                 "ftp://example.org/a.html", "javascript:alert(1)", "example.org/sayfa"])
def test_only_http_addresses_are_opened(url):
    source, _, requests = page_source(b"<p>gizli</p>")
    with pytest.raises(ValueError):
        source.extract(url)
    assert requests == []


def test_default_source_is_google_only_when_both_variables_are_set(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_CSE_ID", "test-cx")
    google = default_source()
    assert isinstance(google, GoogleSearch) and google.lang == "tr"
    monkeypatch.setenv("GOOGLE_CSE_ID", "  ")
    assert isinstance(default_source(), Wikipedia)
    monkeypatch.delenv("GOOGLE_CSE_ID")
    wiki = default_source("en")
    assert isinstance(wiki, Wikipedia) and wiki.lang == "en"
    with pytest.raises(ValueError):
        GoogleSearch()
    monkeypatch.delenv("GOOGLE_API_KEY")
    monkeypatch.setenv("GOOGLE_CSE_ID", "test-cx")
    assert isinstance(default_source(), Wikipedia)
