# -*- coding: utf-8 -*-
"""The internet as a channel of experience (vision 11, 13): an adapter, not a change of the core.

Reading: on the owner's explicit command the session looks a topic up in Wikipedia (MediaWiki API, read only),
takes the plain text of the article and lets the mind read it sentence by sentence, like any document: speech of
others (provenance "imitated"), learned by the one rule. Every page read is printed (title and address).

Wanting (design section 5): "improve yourself" picks the topics itself: words the mind uses often but holds with
little conviction (high drive D = use x (1 - c)).

Steering its own learning (design section 8, a first version): there are ways of reading (a short summary, a deep
read). After each reading the mind measures what it gained on that topic per moment spent (new ties to the topic
held with conviction >= 1/2) and writes it as evidence on the tie "way -> relief" of its own internal channel,
with the same evidence rule. The next time it chooses the way with the highest expectation (an untried way first).
So how it learns is itself learned from outcomes and kept in the core.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Callable, Dict, List, Optional, Tuple

from .evidence import conviction, expectation
from .events import ACTED
from .learn import add_evidence1
from .read import drive_of
from .store import ATOMIC, EXPECT, W_NEG, W_POS

AGENT = "CevahirZihin/0.1 (reader of public encyclopedia pages)"
WAYS = {"ozet": 12, "derin": 60}          # how many sentences a way reads
WAY_CHANNEL = "yol"                       # the mind's own channel for its ways of learning (internal acts)


def _get_json(url: str, timeout: float = 20.0) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


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


def firm_knowledge(mind, unit: int) -> int:
    """How many lag-1 ties of a unit are held with conviction >= 1/2 (what is firmly known to follow it)."""
    S = mind.S
    n = 0
    for tie in S.out.get((unit, EXPECT, 1), []):
        w = float(((S.t_w[W_POS, tie] + S.t_w[W_NEG, tie]) * mind.trust).sum())
        n += conviction(w, mind.cfg.k) >= 0.5
    return n


def wanted_topics(mind, channel: str, n: int, skip_frequent: int = 100) -> List[str]:
    """Words the mind uses but holds with little conviction: the highest drive, the most frequent words left out
    (they are the function words of the language; frequency decides, no word list)."""
    S = mind.S
    if not mind.ch.has(channel):
        return []
    cid = mind.ch.id(channel)
    atoms = [u for u in range(S.n_units) if int(S.u_kind[u]) & ATOMIC and int(S.u_channel[u]) == cid]
    by_use = sorted(atoms, key=lambda u: -float(S.u_use[u]))
    frequent = set(by_use[:skip_frequent])
    scored = []
    for u in by_use[skip_frequent:skip_frequent + 3000]:
        word = str(mind.ch.symbol(cid, int(S.u_symbol[u])))
        if len(word) >= 4 and word.isalpha():
            scored.append((drive_of(mind, u), word))
    scored.sort(reverse=True)
    return [w for _, w in scored[:n]]


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
        w_pos = float((S.t_w[W_POS, tie] * self.mind.trust).sum())
        w_neg = float((S.t_w[W_NEG, tie] * self.mind.trust).sum())
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


def research(session, topic: str, source: Optional[Wikipedia] = None, way: Optional[str] = None,
             say: Callable[[str], None] = print) -> Dict[str, object]:
    """Look a topic up, read it with a way of reading, measure the gain, record it. Returns a report."""
    from .events import IMITATED
    from .__main__ import CHANNEL, sentences, words
    source = source or Wikipedia()
    mind = session.mind
    chooser = WayChooser(mind)
    way = way or chooser.choose()
    titles = source.search(topic)
    if not titles:
        return {"topic": topic, "read": [], "note": "bulunamadı"}
    words_topic = words(topic)
    key = words_topic[0] if words_topic else topic
    cid = mind.ch.id(CHANNEL)
    before_unit = mind.S.atom_index.get((cid, mind.ch.symbol_index(cid, key, create=False)), -1) \
        if mind.ch.symbol_index(cid, key, create=False) >= 0 else -1
    firm_before = firm_knowledge(mind, before_unit) if before_unit >= 0 else 0
    start, read_titles, lived = mind.t, [], 0
    budget = WAYS[way]
    for title in titles:
        text = source.extract(title)
        say("  okunuyor: %s  (%s)" % (title, source.address(title)))
        read_titles.append(title)
        for sentence in sentences(re.sub(r"=+[^=]+=+", " ", text)):
            if lived >= budget:
                break
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
