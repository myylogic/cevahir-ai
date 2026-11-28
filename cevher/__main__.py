# -*- coding: utf-8 -*-
"""Cevahir'in son biçimiyle uçbirimden konuşmak (cevher/cevahir.py).

    python -m cevher [--load KLASOR] [--ev KLASOR] [--butce MB] [--yeni]

Yazdığınız her satır Cevahir'e söylenir; cevap verir ya da susar ("(sessiz)") ve o turu yaşar. Zihin ev klasöründe
yaşar ($CEVAHIR_HOME, yoksa ~/.cevahir/cevher) ve çıkarken kendiliğinden kaydedilir. İlk açılışta ev boşsa depodaki
açık zihin (cevher/zihin) getirilir; ``--load`` başka bir zihni getirir, ``--yeni`` boş bir zihinle başlar.

    !oku YOL         bir metin dosyasını ya da bir klasördeki .txt/.md/.py dosyalarını okur (uzun dosya parça
                     parça okunur: bellek dosyanın boyuyla büyümez; her parça ayrı bir kayıttır)
    !ogret YOL       bir .tsv dosyasındaki diyalogları izler ("söylenen<TAB>cevap", satır başına bir tane)
    !cevap METIN     son söylediğinize doğru cevabı gösterir (o diyaloğu izlemiş olur)
    !dogru / !yanlis son cevabının nasıl karşılandığı (yanlış cevap doğru söylenmiş gibi yaşanmaz)
    !gecmis          bu oturumun kayıtları;  !unut N  N numaralı kaydı tam olarak unutur
    !uyu             bir gece: harfler bir kez görülen bağlamları bırakır
    !durum           Cevahir'in ne tuttuğu
    !kaydet [KLASOR] şimdi kaydeder (klasör verilmezse eve)
    !cik             kaydedip çıkar
"""
from __future__ import annotations

import argparse
import os
import sys
import time

PIECE = 1 << 18                                              # a file is read so many characters at a time


def pieces(f, most: int = PIECE):
    """A file in pieces of at most about ``most`` characters, each ending at a line's end where it has one. Cevahir
    holds the text it is given while it reads it, so a long file is given a piece at a time."""
    rest = ""
    while True:
        chunk = f.read(most)
        if not chunk:
            break
        chunk = rest + chunk
        cut = chunk.rfind("\n") + 1 or len(chunk)
        yield chunk[:cut]
        rest = chunk[cut:]
    if rest:
        yield rest


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m cevher")
    ap.add_argument("--load", default="", help="getirilecek zihnin klasörü")
    ap.add_argument("--ev", default="", help="zihnin yaşadığı klasör (CEVAHIR_HOME ya da ~/.cevahir/cevher)")
    ap.add_argument("--butce", type=int, default=0, help="bellek bütçesi, MB (yoksa zihnin kendi bütçesi; yeni zihin 512)")
    ap.add_argument("--yeni", action="store_true", help="boş bir zihinle başla")
    a = ap.parse_args()
    for stream in (sys.stdin, sys.stdout):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    t0 = time.time()
    from cevher import Cevahir
    home = a.ev or os.environ.get("CEVAHIR_HOME") or os.path.join(os.path.expanduser("~"), ".cevahir", "cevher")
    shipped = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zihin")
    source = a.load or (home if os.path.exists(os.path.join(home, "manifest.json")) else
                        (shipped if os.path.exists(os.path.join(shipped, "manifest.json")) else ""))
    if a.yeni or not source:
        c, source = Cevahir(budget_mb=a.butce or 512, ledger=True), ""
    else:
        c = Cevahir.load(source, budget_mb=a.butce or None, ledger=True)
    size = c.size()
    print("Cevahir %s (%.1f sn; %d bağ, %d kelime, bütçe %d MB). !cik ile çıkılır." % (
        "yüklendi: " + source if source else "doğdu", time.time() - t0, size["ties"]["ties"], size["words"],
        c.budget_mb), flush=True)
    last = ""
    for raw in sys.stdin:
        line = raw.rstrip("\r\n")
        if not line.strip():
            continue
        if line.startswith("!cik"):
            break
        if line.startswith("!kaydet"):
            where = line[len("!kaydet"):].strip() or home
            m = c.save(where)
            print("  kaydedildi: %s (%d bağ)" % (where, m["size"]["ties"]["ties"]), flush=True)
        elif line.startswith("!dogru") or line.startswith("!yanlis"):
            c.feedback(line.startswith("!dogru"))
            print("  kaydedildi", flush=True)
        elif line.startswith("!cevap "):
            if c.pending is not None:                        # the answer it gave is not lived as said rightly
                c.feedback(False)
            c.watch(last, line[len("!cevap "):].strip())
            print("  izlendi", flush=True)
        elif line.startswith("!gecmis"):
            for i, e in enumerate(c.entries):
                print("  %d %s: %s%s" % (i, e["kind"], e["text"].replace("\n", " "),
                                         " (unutuldu)" if e["forgotten"] else ""), flush=True)
        elif line.startswith("!unut "):
            try:
                print("  %s" % c.forget(int(line.split()[1])), flush=True)
            except (ValueError, IndexError) as err:
                print("  unutulamadı: %s" % err, flush=True)
        elif line.startswith("!uyu"):
            print("  %s" % c.sleep(), flush=True)
        elif line.startswith("!durum"):
            print("  %s" % c.size(), flush=True)
        elif line.startswith("!oku "):
            path = os.path.expanduser(line[len("!oku "):].strip().strip('"'))
            files = [path] if os.path.isfile(path) else sorted(
                os.path.join(r, f) for r, _, fs in os.walk(path) for f in fs if f.endswith((".txt", ".md", ".py")))
            n, parts, t1 = 0, 0, time.time()
            for name in files:
                with open(name, encoding="utf-8", errors="replace") as f:
                    held = None
                    for text in pieces(f):
                        if held is not None:
                            c.read(held, code=name.endswith(".py"))
                        held, n, parts = text, n + len(text), parts + 1
                    if held is not None:                     # the last piece of a file ends its last line
                        c.read(held + "\n", code=name.endswith(".py"))
            print("  okundu: %d dosya, %d karakter, %d parça, %.1f sn" % (len(files), n, parts, time.time() - t1),
                  flush=True)
        elif line.startswith("!ogret "):
            n = 0
            with open(os.path.expanduser(line[len("!ogret "):].strip().strip('"')), encoding="utf-8") as f:
                for row in f:
                    row = row.rstrip("\r\n")
                    if row.startswith("#") or not row.strip():
                        continue
                    if "\t" in row:
                        q, ans = row.split("\t", 1)
                        c.watch(q.strip(), ans.strip())
                    else:
                        c.tell(row.strip())
                    n += 1
            print("  yaşandı: %d satır" % n, flush=True)
        elif line.startswith("!"):
            print("  bilinmeyen komut", flush=True)
        else:
            last = line
            print("  " + (c.converse(line) or "(sessiz)"), flush=True)
    c.save(home)
    print("Cevahir kaydedildi: %s" % home, flush=True)


if __name__ == "__main__":
    main()
