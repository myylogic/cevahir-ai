# -*- coding: utf-8 -*-
"""Zihin, the way out to the network (zihin/ag.py) and what reads through it (zihin/web.py). No test here opens a
connection: the transport, the clock and the waiting are replaced.

What is pinned here
-------------------
* A request says who asks; only http(s) addresses are read.
* It waits between two requests to one host, and not between hosts.
* A busy or failing service (429, 5xx) and a broken connection are tried again, after the wait the service names or
  a doubling one, up to the number of tries; a refusal (404) is not tried again; the failure that outlasts the tries
  is raised.
* What was fetched is kept and not fetched twice; a kept page that is too old is fetched again; offline, only what
  is kept is read. An address that carries a key is never kept and never shown whole.
* A page is fetched only when the site's robots.txt allows it; a site without robots.txt allows everything.
* Several pages fetched at the same time come back in the order asked, each standing for itself.
* The encyclopedia and the dictionary are read through it; the report of what can be reached names the reason when
  something cannot.
* Pages given by their addresses are read as documents are, and what they said is known afterwards.
"""
from __future__ import annotations

import io
import urllib.error
from email.message import Message

import pytest
import torch

import zihin.__main__ as chat
from zihin import IMITATED, Mind
from zihin.__main__ import CHANNEL, Session, VARIANTS
from zihin.ag import Net, Offline, Refused, check, shown
from zihin.web import Wikipedia, Wiktionary, read_pages, web_sources


class Clock:
    """Time that passes only when someone waits."""

    def __init__(self) -> None:
        self.t, self.waits = 0.0, []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.waits.append(round(seconds, 3))
        self.t += seconds


class Response:
    def __init__(self, body: bytes, content_type: str = "text/html; charset=utf-8", status: int = 200) -> None:
        self.body, self.status = io.BytesIO(body), status
        self.headers = Message()
        self.headers["Content-Type"] = content_type

    def read(self, amount: int = -1) -> bytes:
        return self.body.read(amount)

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False


def http_error(url: str, code: int, retry_after: str = None) -> urllib.error.HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError(url, code, "error", headers, None)


class Transport:
    """Answers by address: a body, an exception, or a list of them to be given one after another."""

    def __init__(self, answers: dict) -> None:
        self.answers, self.asked, self.agents = answers, [], []

    def __call__(self, request, timeout):
        url = request.full_url
        self.asked.append(url)
        self.agents.append(request.get_header("User-agent"))
        answer = self.answers.get(url, http_error(url, 404))
        if isinstance(answer, list):
            answer = answer.pop(0) if len(answer) > 1 else answer[0]
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, tuple):
            return Response(*answer)
        return Response(answer)


def net_with(answers: dict, **settings):
    clock = Clock()
    transport = Transport(answers)
    net = Net(agent="reader-test/1", urlopen=transport, sleep=clock.sleep, clock=clock.now, now=clock.now, **settings)
    return net, transport, clock


def test_a_request_says_who_asks_and_only_the_web_is_read():
    net, transport, _ = net_with({"https://a.example/x": b"hello"})
    assert net.get("https://a.example/x").body == b"hello"
    assert transport.agents == ["reader-test/1"]
    with pytest.raises(ValueError):
        net.get("file:///etc/passwd")


def test_it_waits_between_requests_to_one_host_only():
    net, _, clock = net_with({"https://a.example/1": b"1", "https://a.example/2": b"2", "https://b.example/1": b"3"},
                             gap=2.0)
    net.get("https://a.example/1")
    net.get("https://b.example/1")
    assert clock.waits == []                               # another host: no wait
    net.get("https://a.example/2")
    assert clock.waits == [2.0]


def test_a_busy_service_is_tried_again_after_the_wait_it_names():
    url = "https://a.example/busy"
    net, transport, clock = net_with({url: [http_error(url, 429, "7"), b"now"]}, gap=0.0)
    assert net.get(url).body == b"now"
    assert transport.asked == [url, url] and clock.waits == [7.0]
    assert net.log[-1]["tries"] == 2 and net.log[-1]["status"] == 200


def test_a_failing_service_is_tried_again_with_a_doubling_wait_and_then_given_up():
    url = "https://a.example/down"
    net, transport, clock = net_with({url: [http_error(url, 503)]}, gap=0.0, retries=3)
    with pytest.raises(urllib.error.HTTPError):
        net.get(url)
    assert len(transport.asked) == 4 and clock.waits == [1.0, 2.0, 4.0]
    broken = "https://a.example/reset"
    net, transport, clock = net_with({broken: [ConnectionResetError("reset"), b"back"]}, gap=0.0)
    assert net.get(broken).body == b"back" and len(transport.asked) == 2


def test_a_refusal_is_not_tried_again():
    net, transport, clock = net_with({}, gap=0.0)
    with pytest.raises(urllib.error.HTTPError):
        net.get("https://a.example/missing")
    assert len(transport.asked) == 1 and clock.waits == []


def test_what_was_fetched_is_kept_and_read_again_offline(tmp_path):
    url = "https://a.example/page"
    net, transport, clock = net_with({url: b"<p>one</p>"}, cache=str(tmp_path), max_age=100.0, gap=0.0)
    assert net.get(url).cached is False and net.get(url).cached is True
    assert transport.asked == [url]                        # not fetched twice
    clock.t += 200.0                                       # too old now: fetched again
    assert net.get(url).cached is False and len(transport.asked) == 2
    offline, silent, _ = net_with({}, cache=str(tmp_path), offline=True)
    clock2 = offline.now
    assert offline.get(url).body == b"<p>one</p>" and silent.asked == []   # kept, however old; nothing leaves
    with pytest.raises(Offline):
        offline.get("https://a.example/never-fetched")
    assert clock2() == 0.0


def test_an_address_with_a_key_is_never_kept_and_never_shown_whole(tmp_path):
    url = "https://api.example/search?key=SECRET123&q=leylek"
    net, transport, _ = net_with({url: (b'{"ok": 1}', "application/json")}, cache=str(tmp_path), gap=0.0)
    assert net.json(url) == {"ok": 1} and net.json(url) == {"ok": 1}
    assert len(transport.asked) == 2                       # asked again: nothing was kept
    assert list(tmp_path.iterdir()) == []
    assert "SECRET123" not in shown(url) and "q=leylek" in shown(url)
    assert all("SECRET123" not in entry["url"] for entry in net.log)


def test_a_page_is_fetched_only_when_robots_allows_it():
    answers = {"https://a.example/robots.txt": (b"User-agent: *\nDisallow: /private/\n", "text/plain"),
               "https://a.example/open": b"<p>open page</p>", "https://a.example/private/x": b"<p>hidden</p>",
               "https://b.example/anything": b"<h1>no rules here</h1>"}
    net, transport, _ = net_with(answers, gap=0.0)
    assert net.page("https://a.example/open") == "open page"
    with pytest.raises(Refused):
        net.page("https://a.example/private/x")
    assert "https://a.example/private/x" not in transport.asked
    assert net.page("https://b.example/anything") == "no rules here"   # no robots.txt (404): everything allowed
    assert transport.asked.count("https://a.example/robots.txt") == 1


def test_several_pages_come_back_in_the_order_asked_each_for_itself():
    answers = {"https://a.example/1": b"<p>first</p>", "https://b.example/2": b"<p>second</p>",
               "https://c.example/3": http_error("https://c.example/3", 404),
               "https://d.example/4": (b"%PDF", "application/pdf")}
    net, _, _ = net_with(answers, gap=0.0)
    got = net.many(["https://b.example/2", "https://c.example/3", "https://a.example/1", "https://d.example/4"])
    assert got[0] == "second" and isinstance(got[1], urllib.error.HTTPError) and got[2] == "first" and got[3] == ""


def test_the_encyclopedia_and_the_dictionary_are_read_through_it():
    base = "https://tr.wikipedia.org/w/api.php?"
    search = base + "action=query&list=search&srsearch=leylek&srlimit=2&format=json"
    extract = base + "action=query&prop=extracts&explaintext=1&redirects=1&titles=Leylek&format=json"
    answers = {search: (b'{"query": {"search": [{"title": "Leylek"}]}}', "application/json"),
               extract: ('{"query": {"pages": {"1": {"extract": "Leylek büyük bir kuştur."}}}}'.encode("utf-8"),
                         "application/json")}
    net, transport, _ = net_with(answers, gap=0.0)
    wiki = Wikipedia("tr", net=net)
    assert wiki.search("leylek") == ["Leylek"] and wiki.extract("Leylek") == "Leylek büyük bir kuştur."
    assert wiki.address("Leylek") == "https://tr.wikipedia.org/wiki/Leylek"
    assert Wiktionary("tr", net=net).base == "https://tr.wiktionary.org/w/api.php"
    chain = web_sources("tr", net)                         # the encyclopedia first, then the dictionary
    assert [type(source).__name__ for source in chain.sources] == ["Wikipedia", "Wiktionary"]
    assert chain.search("leylek") == ["Leylek"] and chain.address("Leylek").startswith("https://tr.wikipedia.org/")


def test_the_report_names_what_can_be_reached_and_why_not(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    wiki = "https://tr.wikipedia.org/w/api.php?action=query&meta=siteinfo&format=json"
    dictionary = "https://tr.wiktionary.org/w/api.php?action=query&meta=siteinfo&format=json"
    net, _, _ = net_with({wiki: (b'{"query": {}}', "application/json"), dictionary: http_error(dictionary, 403)},
                         gap=0.0, retries=0)
    lines = check(net, "tr")
    assert any("Vikipedi (tr)" in line and "ulaşılıyor" in line for line in lines)
    assert any("Vikisözlük (tr)" in line and "ulaşılamıyor" in line and "403" in line for line in lines)
    assert any("Google" in line and "anahtar yok" in line for line in lines)


def test_pages_given_by_their_addresses_are_read_as_documents(tmp_path):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        made = chat._log_path
        chat._log_path = lambda: None
        try:
            cfg = VARIANTS["speaking"]()
        finally:
            chat._log_path = made
        cfg.keep_log, cfg.log_path = False, None
        torch.manual_seed(0)
        mind = Mind(cfg)
        mind.add_channel(CHANNEL, "act")
        s = Session(mind)
        facts = {"gül": "kırmızı bir çiçektir", "çam": "yeşil bir ağaçtır", "çekiç": "ağır bir alettir",
                 "tren": "uzun bir taşıttır", "gemi": "büyük bir taşıttır", "papatya": "beyaz bir çiçektir"}
        for _ in range(3):
            for thing, what in facts.items():
                s.company(None)
                s.listen("%s %s ." % (thing, what), IMITATED)
                s.end_of_turn(IMITATED, tone=".")
                mind.step([])
            for thing, what in facts.items():
                s.watch("%s nedir ?" % thing, "%s %s ." % (thing, what))
        page = "<html><head><title>x</title></head><body><nav>menü</nav><p>zambak mor bir çiçektir .</p></body></html>"
        net, _, _ = net_with({"https://a.example/zambak": page.encode("utf-8"),
                              "https://a.example/yok": http_error("https://a.example/yok", 404)}, gap=0.0)
        said = []
        report = read_pages(s, ["https://a.example/zambak", "https://a.example/yok", "not-an-address"], net, said.append)
        assert report["sentences"] == 1 and len(report["read"]) == 2
        assert any("okunamadı" in line for line in said)
        assert Session(mind.clone()).reply("zambak nedir ?", limit=40) == "zambak mor bir çiçektir"
    finally:
        torch.set_num_threads(threads)
