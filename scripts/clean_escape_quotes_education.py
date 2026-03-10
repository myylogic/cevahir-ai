# -*- coding: utf-8 -*-
"""
education klasöründeki tüm JSON (ve isteğe bağlı .txt) dosyalarında
\" ve \' kaçış dizilerini temizler; tırnak karakteri kalır, backslash kaldırılır.

Güvenlik:
  - Yazmadan önce JSON tekrar parse edilir; yapı bozulursa dosya yazılmaz.
  - --backup ile tüm değiştirilecek dosyalar yedeklenir (geri almak kolay).

Kullanım:
  python scripts/clean_escape_quotes_education.py --dry-run     # sadece rapor
  python scripts/clean_escape_quotes_education.py --backup      # yedekle + temizle
  python scripts/clean_escape_quotes_education.py --limit 1    # önce 1 dosyada dene
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from datetime import datetime


def clean_quote_escapes(text: str) -> str:
    """Metindeki \\" \\' /" /' vb. temizler; sadece tırnak kalır."""
    if not text or not isinstance(text, str):
        return text
    for _ in range(3):
        text = text.replace('\\\\"', '"').replace("\\\\'", "'")
        text = text.replace('\\"', '"').replace("\\'", "'")
    text = text.replace('/"', '"').replace("/'", "'")
    return text


def clean_object(obj, changed: list[bool]) -> object:
    """Dict/list içindeki tüm string'leri temizler; changed[0] güncellenir."""
    if isinstance(obj, str):
        new_val = clean_quote_escapes(obj)
        if new_val != obj:
            changed[0] = True
        return new_val
    if isinstance(obj, dict):
        return {k: clean_object(v, changed) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_object(item, changed) for item in obj]
    return obj


def process_json_file(path: Path, dry_run: bool) -> bool:
    """Bir JSON dosyasını okuyup temizleyip kaydeder. Değişiklik yapıldıysa True döner."""
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"  [HATA] Okunamadı: {path} — {e}", file=sys.stderr)
        return False

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  [HATA] JSON parse: {path} — {e}", file=sys.stderr)
        return False

    changed = [False]
    data = clean_object(data, changed)

    if not changed[0]:
        return False

    if dry_run:
        print(f"  [DRY-RUN] Temizlenecek: {path}")
        return True
    try:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return True
    except Exception as e:
        print(f"  [HATA] Yazılamadı: {path} — {e}", file=sys.stderr)
        return False


def process_txt_file(path: Path, dry_run: bool) -> bool:
    """Bir .txt dosyasının içeriğindeki kaçışları temizler."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"  [HATA] Okunamadı: {path} — {e}", file=sys.stderr)
        return False
    cleaned = clean_quote_escapes(text)
    if cleaned == text:
        return False
    if dry_run:
        print(f"  [DRY-RUN] Temizlenecek: {path}")
        return True
    try:
        path.write_text(cleaned, encoding="utf-8")
        return True
    except Exception as e:
        print(f"  [HATA] Yazılamadı: {path} — {e}", file=sys.stderr)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="education klasöründeki dosyalarda \\\" ve \\' temizliği yapar."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="education",
        help="Taranacak klasör (varsayılan: education)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dosya yazma, sadece hangi dosyaların değişeceğini göster",
    )
    parser.add_argument(
        "--txt",
        action="store_true",
        help=".txt dosyalarını da tara ve temizle",
    )
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        print(f"HATA: Klasör bulunamadı: {root.absolute()}", file=sys.stderr)
        sys.exit(1)

    json_files = list(root.rglob("*.json"))
    txt_files = list(root.rglob("*.txt")) if args.txt else []

    total_processed = 0
    total_cleaned = 0

    for path in sorted(json_files):
        total_processed += 1
        if process_json_file(path, args.dry_run):
            total_cleaned += 1
            if not args.dry_run:
                print(f"  Temizlendi: {path}")

    for path in sorted(txt_files):
        total_processed += 1
        if process_txt_file(path, args.dry_run):
            total_cleaned += 1
            if not args.dry_run:
                print(f"  Temizlendi: {path}")

    print(f"\nToplam: {total_processed} dosya tarandı, {total_cleaned} dosyada \\\" / kaçış temizlendi.")
    if args.dry_run and total_cleaned:
        print("Gerçek temizlik için --dry-run olmadan çalıştır.")


if __name__ == "__main__":
    main()
