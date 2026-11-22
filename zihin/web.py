# -*- coding: utf-8 -*-
"""The internet as a channel of experience (vision 11, 13): an adapter, not a change of the core.

Reading: on the owner's explicit command the session looks a topic up (read only), takes the plain text of what it
finds and lets the mind read it sentence by sentence, like any document: speech of others (provenance "imitated"),
learned by the one rule. Every page read is printed (title and address). The source is Google when GOOGLE_API_KEY
and GOOGLE_CSE_ID are set (the official Custom Search JSON API; result pages are downloaded and reduced to their
text; google.com result pages are never scraped), otherwise Wikipedia (MediaWiki API, plain-text extracts).

Wanting (design section 5): "improve yourself" picks the topics itself, by the one measure every topic the program
chooses has (dusunce.topic_score): telling (few composed contexts hold the word, tau) and not yet well known (high
drive D = use x (1 - c)).

Steering its own learning (design section 8, a first version): there are ways of reading (a short summary, a deep
read). After each reading the mind measures what it gained on that topic per moment spent (new ties to the topic
held with conviction >= 1/2) and writes it as evidence on the tie "way -> relief" of its own internal channel,
with the same evidence rule. The next time it chooses the way with the highest expectation (an untried way first).
So how it learns is itself learned from outcomes and kept in the core.
"""
from __future__ import annotations

import http.client
import json
import os
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Callable, Dict, List, Optional, Protocol, Tuple

import torch

from .evidence import conviction, expectation
from .events import ACTED
from .learn import add_evidence1
from .store import ATOMIC, EXPECT, W_NEG, W_POS

AGENT = "CevahirZihin/0.1 (reader of public encyclopedia pages)"
WAYS = {"ozet": 12, "derin": 60}          # how many sentences a way reads
WAY_CHANNEL = "yol"                       # the mind's own channel for its ways of learning (internal acts)
GOOGLE_API = "https://www.googleapis.com/customsearch/v1"
TIMEOUT = 20.0                            # seconds allowed for one request
MAX_PAGE = 2 * 1024 * 1024                # bytes read of one web page at most
HTML_TYPES = ("text/html", "application/xhtml+xml")


def _get_json(url: str, timeout: float = TIMEOUT) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class Source(Protocol):
    """What research reads from: identifiers from ``search``, plain text and a printable address for each."""

    def search(self, topic: str, limit: int = ...) -> List[str]: ...

    def extract(self, identifier: str) -> str: ...

    def address(self, identifier: str) -> str: ...


class Wikipedia:
    """Search and plain-text extracts from one language edition. ``fetch`` can be replaced (tests, other sources)."""

    def __init__(self, lang: str = "tr", fetch: Callable[[str], dict] = _get_json) -> None:
        self.base = "https://%s.wikipedia.org/w/api.php" % lang
        self.lang, self.fetch = lang, fetch

    def search(self, topic: str, limit: int = 2) -> List[str]:
        q = urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": topic, "srlimit": limit,
                                    "format": "json"})
        data = self.fetch(self.base + "?" + q)
        return [hit["title"] for hit in data.get("query", {}).get("search", [])]

    def extract(self, title: str) -> str:
        q = urllib.parse.urlencode({"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
                                    "titles": title, "format": "json"})
        pages = self.fetch(self.base + "?" + q).get("query", {}).get("pages", {})
        return "\n".join(page.get("extract", "") for page in pages.values())

    def address(self, title: str) -> str:
        return "https://%s.wikipedia.org/wiki/%s" % (self.lang, urllib.parse.quote(title.replace(" ", "_")))


_SKIPPED = frozenset({"script", "style", "noscript", "nav", "header", "footer", "aside", "form",
                      "title", "template", "svg", "button", "select", "iframe", "video", "audio", "object", "canvas"})
_BLOCKS = frozenset({"p", "div", "br", "hr", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "dl", "dt", "dd",
                     "table", "tr", "td", "th", "caption", "blockquote", "pre", "section", "article", "main",
                     "figure", "figcaption", "address", "details", "summary"})
_SPACE = re.compile(r"\s+")
_META_CHARSET = re.compile(rb"""<meta[^>]*?charset\s*=\s*["']?\s*([\w.:-]+)""", re.IGNORECASE)


class _TextParser(HTMLParser):
    """Collects the readable text of a page: block elements end lines, skipped elements give nothing."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self.skipping: List[str] = []         # open skipped elements, innermost last

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _SKIPPED:
            self.skipping.append(tag)
            self.parts.append(" ")
        elif tag in _BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.skipping:              # closes it and whatever was left open inside it
            while self.skipping.pop() != tag:
                pass
            self.parts.append(" ")
        elif tag in _BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skipping:
            self.parts.append(_SPACE.sub(" ", data))


def html_text(html: str) -> str:
    """Readable text of an HTML page: one line per paragraph, heading or list item, whitespace collapsed."""
    parser = _TextParser()
    try:
        parser.feed(html)
        parser.close()
    except AssertionError:                    # markup some Python versions refuse; keep what was read before it
        pass
    lines = (" ".join(line.split()) for line in "".join(parser.parts).split("\n"))
    return "\n".join(line for line in lines if line)


def _decode(raw: bytes, declared: Optional[str]) -> str:
    """Page bytes as text: a UTF-8 byte order mark, else the declared charset (HTTP header, else a <meta> tag),
    otherwise utf-8.
    """
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", errors="replace")
    if not declared:
        found = _META_CHARSET.search(raw[:4096])
        declared = found.group(1).decode("ascii") if found else None
    try:
        return raw.decode(declared or "utf-8", errors="replace")
    except (LookupError, ValueError):         # unknown or unusable charset name
        return raw.decode("utf-8", errors="replace")


def _is_web(url: str) -> bool:
    return url.lower().startswith(("http://", "https://"))


def _google_keys() -> Tuple[str, str]:
    """GOOGLE_API_KEY and GOOGLE_CSE_ID from the environment ("" when unset)."""
    return os.environ.get("GOOGLE_API_KEY", "").strip(), os.environ.get("GOOGLE_CSE_ID", "").strip()


class GoogleSearch:
    """Google web search through the official Custom Search JSON API (google.com pages are never scraped).

    ``search`` returns result URLs, ``extract`` downloads one page and returns its readable text. The key and the
    engine id default to GOOGLE_API_KEY and GOOGLE_CSE_ID and are never printed. ``fetch`` (API answers as JSON)
    and ``urlopen`` (pages, called like ``urllib.request.urlopen``) can be replaced (tests).
    """

    def __init__(self, key: Optional[str] = None, cx: Optional[str] = None, lang: str = "tr",
                 fetch: Callable[[str], dict] = _get_json, urlopen: Callable = urllib.request.urlopen) -> None:
        env_key, env_cx = _google_keys()
        self._key, self._cx = key or env_key, cx or env_cx
        if not (self._key and self._cx):
            raise ValueError("Google search needs GOOGLE_API_KEY and GOOGLE_CSE_ID")
        self.lang, self.fetch, self.urlopen = lang, fetch, urlopen
        self.found: Dict[str, Tuple[str, str]] = {}       # url -> (title, snippet) of every result seen

    def search(self, topic: str, limit: int = 5) -> List[str]:
        """Up to ``limit`` (at most 5) result URLs, best first; in the language ``lang`` (hl and lr)."""
        params = {"key": self._key, "cx": self._cx, "q": topic, "num": max(1, min(int(limit), 5))}
        if self.lang:
            params.update(hl=self.lang, lr="lang_" + self.lang)
        data = self.fetch(GOOGLE_API + "?" + urllib.parse.urlencode(params))
        links: List[str] = []
        for item in data.get("items", []):
            link = str(item.get("link", ""))
            if _is_web(link) and link not in links:
                links.append(link)
                self.found[link] = (str(item.get("title", "")), str(item.get("snippet", "")))
        return links

    def extract(self, url: str) -> str:
        """Readable text of an HTML page; "" for other content (pdf, images) or a page that cannot be downloaded.

        Only http(s) addresses are opened (ValueError otherwise); at most MAX_PAGE bytes are read.
        """
        if not _is_web(url):
            raise ValueError("only http(s) pages are read, not %r" % url)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": AGENT})
            with self.urlopen(request, timeout=TIMEOUT) as response:
                if response.headers.get_content_type() not in HTML_TYPES:
                    return ""
                charset = response.headers.get_content_charset()
                raw = response.read(MAX_PAGE)
        except (OSError, ValueError, http.client.HTTPException):    # unreachable, refused or broken: skipped
            return ""
        return html_text(_decode(raw, charset))

    def address(self, url: str) -> str:
        return url


class Folder:
    """A folder of the owner's own documents as a source (F47: what the mind may turn to by itself without the
    web): every document under the folder that ``__main__.read_document`` reads (.txt, .md, .docx, .pdf, code).
    A topic is found in the documents that hold the word; what is read of a document is its sentences that hold
    the word, in their order (a document is not about one topic as an encyclopedia page is). Nothing leaves the
    machine."""

    def __init__(self, path: str, dil: str = "tr") -> None:
        self.path, self.dil = path, dil
        self._texts: Optional[Dict[str, List[str]]] = None

    def _load(self) -> Dict[str, List[str]]:
        if self._texts is None:
            from .__main__ import read_document, sentences
            self._texts = {}
            names = [self.path] if os.path.isfile(self.path) else sorted(
                os.path.join(root, name) for root, _, files in os.walk(self.path) for name in files)
            for name in names:
                try:
                    text = read_document(name)
                except Exception:                      # not a text, or a reader that is not installed
                    continue
                if text:
                    self._texts[name] = sentences(text, self.dil)
        return self._texts

    def _holding(self, name: str, topic: str) -> List[str]:
        from .__main__ import words
        wanted = set(words(topic, self.dil))
        return [sentence for sentence in self._load()[name] if wanted & set(words(sentence, self.dil))]

    def search(self, topic: str, limit: int = 2) -> List[str]:
        found = [(len(self._holding(name, topic)), name) for name in self._load()]
        found = sorted(((n, name) for n, name in found if n), key=lambda nn: (-nn[0], nn[1]))
        self._topic = topic
        return [name for _, name in found[:limit]]

    def extract(self, identifier: str) -> str:
        return "\n".join(self._holding(identifier, self._topic))

    def address(self, identifier: str) -> str:
        return identifier


class Sources:
    """Several sources asked in order: what the first one that finds the topic holds is what is read (the owner's
    folder before the web)."""

    def __init__(self, sources: List["Source"]) -> None:
        self.sources = list(sources)
        self._of: Dict[str, "Source"] = {}

    def search(self, topic: str, limit: int = 2) -> List[str]:
        for source in self.sources:
            try:
                found = source.search(topic, limit) if isinstance(source, Folder) else source.search(topic)
            except Exception:                          # a service that does not answer: the next source
                continue
            if found:
                self._of = {identifier: source for identifier in found}
                return found
        return []

    def extract(self, identifier: str) -> str:
        return self._of[identifier].extract(identifier)

    def address(self, identifier: str) -> str:
        return self._of[identifier].address(identifier)


def default_source(lang: str = "tr") -> Source:
    """Google (Custom Search JSON API) when GOOGLE_API_KEY and GOOGLE_CSE_ID are both set, otherwise Wikipedia."""
    key, cx = _google_keys()
    return GoogleSearch(key, cx, lang) if key and cx else Wikipedia(lang)


def firm_knowledge(mind, unit: int) -> int:
    """How many lag-1 ties of a unit are held with conviction >= 1/2 (what is firmly known to follow it)."""
    S = mind.S
    n = 0
    for tie in S.out.get((unit, EXPECT, 1), []):
        w = float(((S.w1(W_POS, tie) + S.w1(W_NEG, tie)) * mind.trust).sum())
        n += conviction(w, mind.cfg.k) >= 0.5
    return n


def wanted_topics(mind, channel: str, n: int) -> List[str]:
    """What the mind most wants to know, by its own evidence: of every symbol of the channel it holds, the ``n`` best
    topics by dusunce.topic_score, tau(u) x D(u): telling (few composed contexts hold it) and not yet well known (high
    drive). Nothing is left out by frequency (frequent words are held by many contexts: their tau is small) or by
    what a word looks like; symbols nothing ever followed score 0 and are not offered. Units !unut took back are not
    counted. With F46 (``open_claims``) the items a claim of the mind's own frames was left open for come first,
    the most firmly open first: what it was asked for and did not have, before what it is unsure of."""
    from .dusunce import topic_score
    S = mind.S
    if not mind.ch.has(channel):
        return []
    cid = mind.ch.id(channel)
    n_units = S.n_units
    held = ((S.u_kind[:n_units].to(torch.int64) & ATOMIC) != 0) & (S.u_channel[:n_units] == cid) & S.u_alive[:n_units]
    units = held.nonzero().flatten().tolist()
    first: List[str] = []
    if getattr(mind.cfg, "open_claims", False):            # F46: what a frame of its own needed and did not have
        from .read import open_claim                       # (claims about the item left open, by its own evidence)
        asked = []                                         # comes before what it is merely unsure of
        for u in units:
            symbol = mind.ch.symbol(cid, int(S.u_symbol[u]))
            e = open_claim(mind, symbol)
            if e > 0.5:
                asked.append((e, str(symbol)))
        asked.sort(key=lambda sw: -sw[0])
        first = [w for _, w in asked[:n]]
    scored = []
    for u in units:
        score = topic_score(mind, u)
        if score > 0.0:
            scored.append((score, str(mind.ch.symbol(cid, int(S.u_symbol[u])))))
    scored.sort(key=lambda sw: -sw[0])                     # stable: equal scores keep the order the units were born
    return (first + [w for _, w in scored if w not in first])[:n]


def look_up(session, looking, source: Optional["Source"] = None, say: Callable[[str], None] = print) -> Dict[str, object]:
    """F47: carry out the mind's own decision to look an item up (``looking`` = (its unit, its symbol), act.look):
    read what is found (``research``), then the mind settles the look by reading the claim again where it was left
    open (learn.settle_look). Returns research's report with "met": True, False, or None when nothing could be read
    again."""
    from .learn import settle_look
    filler, symbol = looking
    report = research(session, str(symbol), source=source, say=say)
    report["met"] = settle_look(session.mind, filler)
    return report


class WayChooser:
    """The mind's own evidence about its ways of learning, kept on ties way -> relief of its internal channel."""

    def __init__(self, mind) -> None:
        self.mind = mind
        if not mind.ch.has(WAY_CHANNEL):
            mind.add_channel(WAY_CHANNEL, "knob")

    def _unit(self, way: str) -> int:
        return self.mind._atom(self.mind.ch.id(WAY_CHANNEL), way)

    def _tie(self, way: str) -> int:
        return self.mind.S.tie(self._unit(way), self.mind.feel_unit["relief"], EXPECT, 1)

    def expectation(self, way: str) -> Tuple[float, float]:
        S, tie = self.mind.S, self._tie(way)
        w_pos = float((S.w1(W_POS, tie) * self.mind.trust).sum())
        w_neg = float((S.w1(W_NEG, tie) * self.mind.trust).sum())
        return expectation(w_pos, w_neg, self.mind.cfg.k), w_pos + w_neg

    def choose(self) -> str:
        tried = {way: self.expectation(way) for way in WAYS}
        untried = [way for way, (_, mass) in tried.items() if mass == 0.0]
        if untried:
            return untried[0]
        return max(tried, key=lambda way: tried[way][0])

    def record(self, way: str, gain: float) -> None:
        """gain = new firm ties on the topic per 100 moments; bounded evidence: w+ = g/(g+1), w- = 1/(g+1)."""
        good = gain / (gain + 1.0)
        S = self.mind.S
        S.begin(self.mind.t + 1, self.mind.ch.sizes())     # the outcome is one moment of the mind's own life
        self.mind.t += 1
        tie = self._tie(way)
        add_evidence1(self.mind, W_POS, ACTED, tie, good)
        add_evidence1(self.mind, W_NEG, ACTED, tie, 1.0 - good)
        S.end()


def research(session, topic: str, source: Optional[Source] = None, way: Optional[str] = None,
             say: Callable[[str], None] = print) -> Dict[str, object]:
    """Look a topic up, read it with a way of reading, measure the gain, record it. Returns a report. The topic and
    what is read are split and lower-cased by the session's convention (its mind's dil), as everything it hears."""
    from .events import IMITATED
    from .__main__ import CHANNEL, sentences, words
    source = default_source() if source is None else source
    mind = session.mind
    dil = getattr(session, "dil", getattr(mind, "dil", "tr"))
    adapter = getattr(session, "words", None) or (lambda text: words(text, dil))
    check = getattr(session, "check", None)            # a long reading stops between sentences when asked (Ctrl+C)
    chooser = WayChooser(mind)
    way = way or chooser.choose()
    try:
        titles = source.search(topic)
    except Exception as exc:                           # a search service that refuses (key, quota, closed API)
        if isinstance(source, Wikipedia):
            raise
        say("  arama servisi yanıt vermedi (%s); Vikipedi'ye bakıyorum" % type(exc).__name__)
        source = Wikipedia()
        titles = source.search(topic)
    if not titles:
        return {"topic": topic, "read": [], "note": "bulunamadı"}
    words_topic = adapter(topic)
    key = words_topic[0] if words_topic else topic
    cid = mind.ch.id(CHANNEL)
    before_unit = mind.S.atom_index.get((cid, mind.ch.symbol_index(cid, key, create=False)), -1) \
        if mind.ch.symbol_index(cid, key, create=False) >= 0 else -1
    firm_before = firm_knowledge(mind, before_unit) if before_unit >= 0 else 0
    start, read_titles, lived = mind.t, [], 0
    budget = WAYS[way]
    if getattr(session, "close_turn", None) is not None:
        session.close_turn()                           # an answered turn still open ends before the reading begins
                                                       # (the chat does this for every command; here for any caller)
    if getattr(session, "company", None) is not None:
        session.company(None)                          # F33: a page is no one's in particular, as a document is (read
                                                       # with the partner still present it was kept as lived with them
                                                       # and was not recalled with anyone else: t_page.py)
    for title in titles:
        text = source.extract(title)
        label = getattr(source, "found", {}).get(title, (title,))[0] if hasattr(source, "found") else title
        say("  okunuyor: %s  (%s)" % (label, source.address(title)))
        read_titles.append(title)
        if isinstance(source, Wikipedia):              # Wikipedia section headings (== ... ==), one line each
            text = re.sub(r"(?m)^=+[^=\n]+=+\s*$", " ", text)
        for sentence in sentences(text, dil):
            if lived >= budget:
                break
            if check is not None:
                check()
            session.listen(sentence, IMITATED)
            mind.step([])
            lived += 1
        if lived >= budget:
            break
    unit = mind.S.atom_index.get((cid, mind.ch.symbol_index(cid, key, create=False)), -1) \
        if mind.ch.symbol_index(cid, key, create=False) >= 0 else -1
    firm_after = firm_knowledge(mind, unit) if unit >= 0 else 0
    spent = max(1, mind.t - start)
    gain = 100.0 * max(0, firm_after - firm_before) / spent
    chooser.record(way, gain)
    return {"topic": topic, "way": way, "read": read_titles, "sentences": lived, "moments": spent,
            "firm_before": firm_before, "firm_after": firm_after, "gain_per_100_moments": round(gain, 2),
            "ways": {w: round(chooser.expectation(w)[0], 3) for w in WAYS}}
