# -*- coding: utf-8 -*-
"""The way out to the network (ağ): one place through which everything the mind reads from the internet passes.

The mind reads the web as it reads a document (zihin/web.py); this module is only how the bytes are fetched, and it
is made to be a good guest and a dependable one:

* **Only reads.** GET requests; nothing is ever sent but the address asked for.
* **Says who it is** (``AGENT``, or the environment variable ZIHIN_AGENT), as the services' rules ask.
* **Waits between requests to one host** (``gap`` seconds), also when several pages are fetched at the same time.
* **Tries again, and not blindly:** a busy or failing service (429, 5xx) and a broken connection are tried again up
  to ``retries`` times, after the wait the service names (Retry-After) or a doubling one; a refusal (other 4xx) is
  not tried again.
* **Obeys robots.txt** for pages (``page``, ``many``); an API a service publishes for programs is asked as its own
  rules say.
* **Keeps what it fetched** on disk (``cache``; ``max_age`` seconds), so a page is not fetched twice, a session can
  be repeated, and what was read can be read again with no network at all (``offline``). An address that carries a
  key is never kept and never printed whole.
* **Bounded:** ``timeout`` seconds for a request, ``max_bytes`` of a page.
* **Accountable:** every request is in ``log`` (address, status, seconds, tries, whether it came from the cache).

``Net.many`` fetches several pages at the same time (threads: waiting for the network is not the mind's work) and
gives them back in the order asked, so what the mind then reads, in that order, is exactly what it would read had
they been fetched one after another. ``check`` says, in the program's language, what can be reached from here.

    python -m zihin.ag            # what can this machine reach?
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from typing import Callable, Dict, List, NamedTuple, Optional, Union

AGENT = "CevahirZihin/0.1 (reader of public encyclopedia pages)"
SECRET = ("key", "apikey", "api_key", "token", "access_token", "cx")     # query fields never kept or printed
HTML_TYPES = ("text/html", "application/xhtml+xml")


class Offline(OSError):
    """Asked for something that is not kept, while offline."""


class Refused(OSError):
    """The site's robots.txt does not let this page be fetched."""


class Answer(NamedTuple):
    url: str
    status: int
    content_type: str
    charset: Optional[str]
    body: bytes
    cached: bool


def shown(url: str) -> str:
    """An address as it may be printed and kept: the values of fields that carry a key are left out."""
    parts = urllib.parse.urlsplit(url)
    if not parts.query:
        return url
    fields = [(k, "…" if k.lower() in SECRET else v) for k, v in urllib.parse.parse_qsl(parts.query, True)]
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(fields, safe="…")))


def has_secret(url: str) -> bool:
    return any(k.lower() in SECRET for k, _ in urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query, True))


class Net:
    def __init__(self, agent: Optional[str] = None, timeout: float = 20.0, retries: int = 3, gap: float = 1.0,
                 cache: Optional[str] = None, max_age: float = 30 * 86400.0, offline: bool = False,
                 max_bytes: int = 2 * 1024 * 1024, urlopen: Callable = urllib.request.urlopen,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
                 now: Callable[[], float] = time.time) -> None:
        self.agent = agent or os.environ.get("ZIHIN_AGENT", "").strip() or AGENT
        self.timeout, self.retries, self.gap = timeout, retries, gap
        self.cache, self.max_age, self.offline, self.max_bytes = cache, max_age, offline, max_bytes
        self.urlopen, self.sleep, self.clock, self.now = urlopen, sleep, clock, now
        self.log: List[dict] = []
        self._next: Dict[str, float] = {}                 # host -> the moment its next request may start
        self._robots: Dict[str, Optional[urllib.robotparser.RobotFileParser]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ what is kept
    def _paths(self, url: str):
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return os.path.join(self.cache, key + ".meta"), os.path.join(self.cache, key + ".body")

    def _kept(self, url: str) -> Optional[Answer]:
        if not self.cache or has_secret(url):
            return None
        meta_path, body_path = self._paths(url)
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            if not self.offline and self.now() - meta["time"] > self.max_age:
                return None
            with open(body_path, "rb") as f:
                body = f.read()
        except (OSError, ValueError, KeyError):
            return None
        return Answer(url, int(meta["status"]), meta["content_type"], meta.get("charset"), body, True)

    def _keep(self, answer: Answer) -> None:
        if not self.cache or has_secret(answer.url) or answer.status != 200:
            return
        try:
            os.makedirs(self.cache, exist_ok=True)
            meta_path, body_path = self._paths(answer.url)
            with open(body_path + ".tmp", "wb") as f:
                f.write(answer.body)
            os.replace(body_path + ".tmp", body_path)
            with open(meta_path + ".tmp", "w", encoding="utf-8") as f:
                json.dump({"url": answer.url, "time": self.now(), "status": answer.status,
                           "content_type": answer.content_type, "charset": answer.charset}, f)
            os.replace(meta_path + ".tmp", meta_path)
        except OSError:                                    # a cache that cannot be written is no cache
            pass

    # ------------------------------------------------------------------ one request
    def _wait_for(self, host: str) -> None:
        with self._lock:
            now = self.clock()
            start = max(now, self._next.get(host, now))
            self._next[host] = start + self.gap
        if start > now:
            self.sleep(start - now)

    def get(self, url: str, robots: bool = False) -> Answer:
        """The bytes at an http(s) address. Raises ``Offline``, ``Refused`` (robots), ``urllib.error.HTTPError`` (a
        refusal, or a failure that outlasted the tries) or ``OSError`` (unreachable)."""
        if not url.lower().startswith(("http://", "https://")):
            raise ValueError("only http(s) addresses are read, not %r" % url)
        kept = self._kept(url)
        if kept is not None:
            self.log.append({"url": shown(url), "status": kept.status, "seconds": 0.0, "tries": 0, "cached": True})
            return kept
        if self.offline:
            raise Offline("not kept, and offline: %s" % shown(url))
        if robots and not self.allowed(url):
            self.log.append({"url": shown(url), "status": "robots", "seconds": 0.0, "tries": 0, "cached": False})
            raise Refused("robots.txt does not allow %s" % shown(url))
        host = urllib.parse.urlsplit(url).netloc.lower()
        request = urllib.request.Request(url, headers={"User-Agent": self.agent, "Accept-Encoding": "identity"})
        started, tries, wait, last = self.clock(), 0, 1.0, None
        while True:
            self._wait_for(host)
            tries += 1
            try:
                with self.urlopen(request, timeout=self.timeout) as response:
                    headers = response.headers
                    answer = Answer(url, int(getattr(response, "status", 200) or 200), headers.get_content_type(),
                                    headers.get_content_charset(), response.read(self.max_bytes), False)
                self._keep(answer)
                self.log.append({"url": shown(url), "status": answer.status, "seconds": self.clock() - started,
                                 "tries": tries, "cached": False})
                return answer
            except urllib.error.HTTPError as exc:
                last = exc
                again = exc.code == 429 or 500 <= exc.code < 600
                named = exc.headers.get("Retry-After") if exc.headers is not None else None
                if again and named and named.strip().isdigit():
                    wait = min(60.0, float(named.strip()))
                status = exc.code
            except (OSError, http.client.HTTPException) as exc:      # unreachable, reset, timed out, broken answer
                last, again, status = exc, True, type(exc).__name__
            if not again or tries > self.retries:
                self.log.append({"url": shown(url), "status": status, "seconds": self.clock() - started,
                                 "tries": tries, "cached": False})
                raise last
            self.sleep(wait)
            wait = min(60.0, wait * 2.0)

    def json(self, url: str) -> dict:
        answer = self.get(url)
        return json.loads(answer.body.decode(answer.charset or "utf-8", errors="replace"))

    # ------------------------------------------------------------------ pages
    def allowed(self, url: str) -> bool:
        """Whether the site's robots.txt lets this program fetch the page (asked once a site; a site with no
        robots.txt allows everything, one that refuses to show it allows nothing, one that cannot be reached for it
        is not judged by it)."""
        parts = urllib.parse.urlsplit(url)
        site = "%s://%s" % (parts.scheme, parts.netloc)
        with self._lock:
            known = site in self._robots
            rules = self._robots.get(site)
        if not known:
            rules = urllib.robotparser.RobotFileParser()
            try:
                answer = self.get(site + "/robots.txt")
                rules.parse(answer.body.decode(answer.charset or "utf-8", errors="replace").splitlines())
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403):
                    rules.disallow_all = True
                else:
                    rules.allow_all = True
            except OSError:
                rules = None
            with self._lock:
                self._robots[site] = rules
        return True if rules is None else rules.can_fetch(self.agent, url)

    def page(self, url: str) -> str:
        """The readable text of an HTML page (robots.txt obeyed); "" for anything that is not HTML."""
        from .web import _decode, html_text
        answer = self.get(url, robots=True)
        if answer.content_type not in HTML_TYPES:
            return ""
        return html_text(_decode(answer.body, answer.charset))

    def many(self, urls: List[str], workers: int = 4) -> List[Union[str, Exception]]:
        """Several pages at the same time, given back in the order asked: for each its readable text, or the
        exception that kept it from being read. The wait between requests to one host holds across the threads."""
        from concurrent.futures import ThreadPoolExecutor

        def one(url: str):
            try:
                return self.page(url)
            except Exception as exc:                       # each page stands for itself
                return exc
        if len(urls) <= 1 or workers <= 1:
            return [one(url) for url in urls]
        with ThreadPoolExecutor(max_workers=min(workers, len(urls))) as pool:
            return list(pool.map(one, urls))


def why(exc: Exception) -> str:
    """Why something could not be reached, in the program's language."""
    if isinstance(exc, Offline):
        return "çevrimdışı ve önbellekte yok"
    if isinstance(exc, Refused):
        return "sitenin robots.txt dosyası izin vermiyor"
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code == 403:
            return "403: erişim reddedildi (ağ kuralı ya da site izin vermiyor)"
        if exc.code == 429:
            return "429: servis çok istek aldığını söylüyor (sonra yeniden dene)"
        return "HTTP %d" % exc.code
    reason = getattr(exc, "reason", exc)
    text = str(reason) or type(exc).__name__
    if "timed out" in text or isinstance(reason, TimeoutError):
        return "zaman aşımı"
    if "Name or service not known" in text or "getaddrinfo" in text or "nodename" in text:
        return "adres çözülemedi (ağ yok ya da DNS kapalı)"
    return "%s: %s" % (type(exc).__name__, text[:120])


def check(net: Optional[Net] = None, lang: str = "tr") -> List[str]:
    """What can be reached from here: one small request to each service the mind reads from, with the reason when
    one cannot be reached. No key is printed; nothing is read into any mind."""
    net = net or Net(retries=0, timeout=10.0)
    lines = []
    proxies = sorted(urllib.request.getproxies())
    lines.append("  vekil sunucu (proxy): %s" % (", ".join(proxies) if proxies else "yok (doğrudan bağlantı)"))
    lines.append("  kimlik (User-Agent): %s" % net.agent)
    lines.append("  önbellek: %s%s" % (net.cache or "yok", " (çevrimdışı: yalnız önbellek)" if net.offline else ""))
    probes = [("Vikipedi (%s)" % lang, "https://%s.wikipedia.org/w/api.php?action=query&meta=siteinfo&format=json" % lang),
              ("Vikisözlük (%s)" % lang, "https://%s.wiktionary.org/w/api.php?action=query&meta=siteinfo&format=json" % lang)]
    for name, url in probes:
        started = time.monotonic()
        try:
            data = net.json(url)
            ok = "query" in data
            lines.append("  %-18s %s (%.1f sn)" % (name, "ulaşılıyor" if ok else "cevap veriyor ama beklenen biçimde değil",
                                                  time.monotonic() - started))
        except Exception as exc:
            lines.append("  %-18s ulaşılamıyor: %s" % (name, why(exc)))
    key, cx = os.environ.get("GOOGLE_API_KEY", "").strip(), os.environ.get("GOOGLE_CSE_ID", "").strip()
    lines.append("  %-18s %s" % ("Google arama", "anahtarlar tanımlı (GOOGLE_API_KEY, GOOGLE_CSE_ID): !arastir Google'ı "
                                 "kullanır" if key and cx else "anahtar yok: Vikipedi kullanılır (GOOGLE_API_KEY ve "
                                 "GOOGLE_CSE_ID tanımlanırsa Google'ın resmî arama servisi)"))
    return lines


if __name__ == "__main__":
    import sys
    print("\n".join(check(lang=sys.argv[1] if len(sys.argv) > 1 else "tr")))
