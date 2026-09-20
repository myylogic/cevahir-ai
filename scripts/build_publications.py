"""Build or check publication editions without changing original research records."""
import argparse
import json
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "docs/publications"


def relative(source, destination):
    return Path(os.path.relpath(source, destination)).as_posix()


def edition(paper, catalog):
    source = ROOT / paper["source"]
    parent = PUB / "papers"
    text = source.read_text(encoding="utf-8")
    def rebase(match):
        label, link = match.groups()
        if urlsplit(link).scheme or link.startswith(("#", "/")):
            return match.group(0)
        path, marker, anchor = link.partition("#")
        target = relative((source.parent / path).resolve(), parent)
        return f"[{label}]({target}{marker}{anchor})"
    text = re.sub(r"\[([^\]\n]+)\]\(([^\s)]+)\)", rebase, text)
    text = re.sub(r"\A# ([^\n]+)", r"## Araştırma kaydı: \1", text)
    return f'''# {paper["title"]}

**{catalog["author"]}**

{paper["id"]} · Sürüm {catalog["version"]} · Yayın tarihi {catalog["date"]}

Araştırma raporu / ön baskı. Dış hakem değerlendirmesinden geçmemiştir.

## Özet

{paper["abstract"]}

## Abstract

{paper["abstract_en"]}

## Bulguların yorumu

{paper["finding"]}

{paper["limit"]}

## Yayın ve kanıt bilgisi

Bu makale düzenindeki edisyon, aşağıda özgün araştırma raporunun tam metnini yöntem,
sonuç, negatif kontrol ve kaynak bağlantılarıyla birlikte içerir. Orijinal dosya
[{paper["source"]}]({relative(source, parent)}) olarak korunur. Bağlantı yolları bu
edisyonun konumuna uyarlanmıştır. Rapordaki çalışma tarihi ve geçmiş yayın durumu
ifadeleri tarihsel kayda aittir; bu edisyonun tarihi yukarıdadır.

Bu çalışma [yayın dizisi](../README.md) içindeki ayrı bir metindir. DOI, bütün
metinleri, teknik kitabı ve kodu içeren sürüm arşivini tanımlar; her makaleye ayrı
DOI verildiği anlamına gelmez. Atıfta çalışma kimliği ve sürüm DOI'si birlikte
kullanılmalıdır. Hesaplamalı denetimlerin kapsamı [doğrulama kaydında](../evidence/validation.json)
ve [yayın ilkelerinde](../PUBLISHING.md) açıklanır.

Proje ve araştırma yönü Muhammed Yasin Yılmaz'a aittir. Kod, hesaplamalı deney,
literatür incelemesi ve metin hazırlığında yapay zekâ desteği kullanılmıştır.
Bu katkı açıklaması, DOI kaydı veya otomatik testler dış bilimsel hakemlik sayılmaz.

---

{text.rstrip()}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    catalog = json.loads((PUB / "catalog.json").read_text(encoding="utf-8"))
    for paper in catalog["papers"]:
        path = PUB / "papers" / paper["file"]
        expected = edition(paper, catalog)
        if args.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(expected, encoding="utf-8", newline="\n")
        else:
            assert path.read_text(encoding="utf-8") == expected, str(path)
    print(json.dumps({"publication_editions": len(catalog["papers"]), "mode": "write" if args.write else "check"}))


if __name__ == "__main__":
    main()
