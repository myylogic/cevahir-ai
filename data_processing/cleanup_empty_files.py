#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boş veya çok küçük TXT dosyalarını temizleme scripti

Bu script, data_processing/output klasöründeki boş veya çok küçük TXT dosyalarını
bulur ve siler. Wikipedia scraping sırasında oluşturulmuş boş dosyaları temizlemek için kullanılır.

HARDCODED AYARLAR:
- Root dir: data_processing/output
- Min size: 1.0 KB (1000 byte)
- Extensions: .txt
- Otomatik silme: Evet (onay istemez)
"""

import os
from pathlib import Path
from typing import List, Tuple

# HARDCODED AYARLAR
ROOT_DIR = Path(__file__).parent / "output"  # data_processing/output
MIN_SIZE_KB = 1.0  # 1 KB'dan küçük dosyalar silinir (sıfır kilobayt dahil)
EXTENSIONS = ('.txt',)  # Sadece .txt dosyaları
AUTO_DELETE = True  # Otomatik silme (onay istemez)


def find_empty_or_small_files() -> List[Tuple[Path, float]]:
    """
    Boş veya çok küçük dosyaları bulur (HARDCODED ayarlar kullanılır).
    
    Returns:
        List[Tuple[Path, float]]: (dosya_yolu, boyut_kb) çiftlerinin listesi
    """
    files_to_delete = []
    
    if not ROOT_DIR.exists():
        print(f"[HATA] Klasör bulunamadı: {ROOT_DIR}")
        return files_to_delete
    
    print(f"[INFO] Taranıyor: {ROOT_DIR.absolute()}")
    print(f"[INFO] Minimum boyut: {MIN_SIZE_KB} KB")
    print(f"[INFO] Uzantılar: {', '.join(EXTENSIONS)}")
    print("-" * 60)
    
    # Recursive olarak tüm dosyaları tara
    for file_path in ROOT_DIR.rglob("*"):
        if not file_path.is_file():
            continue
        
        # Uzantı kontrolü
        if file_path.suffix.lower() not in EXTENSIONS:
            continue
        
        # Dosya boyutunu kontrol et
        try:
            size_bytes = file_path.stat().st_size
            size_kb = size_bytes / 1024.0
            
            # Boş veya çok küçük dosyaları bul
            if size_kb < MIN_SIZE_KB:
                files_to_delete.append((file_path, size_kb))
        except OSError as e:
            print(f"[UYARI] Dosya boyutu kontrol edilemedi: {file_path} - {e}")
            continue
    
    return files_to_delete


def delete_files(files_to_delete: List[Tuple[Path, float]]) -> dict:
    """
    Dosyaları siler (HARDCODED: AUTO_DELETE ayarı kullanılır).
    
    Args:
        files_to_delete: Silinecek dosyaların listesi
    
    Returns:
        dict: İstatistikler
    """
    stats = {
        "total_found": len(files_to_delete),
        "deleted": 0,
        "failed": 0,
        "total_size_kb": 0.0,
        "errors": []
    }
    
    if not files_to_delete:
        print("[INFO] Silinecek dosya bulunamadı.")
        return stats
    
    print(f"[INFO] Toplam {len(files_to_delete)} dosya bulundu.")
    
    for file_path, size_kb in files_to_delete:
        try:
            file_path.unlink()
            stats["deleted"] += 1
            stats["total_size_kb"] += size_kb
        except OSError as e:
            error_msg = f"Silinemedi: {file_path} - {e}"
            print(f"[HATA] {error_msg}")
            stats["failed"] += 1
            stats["errors"].append(error_msg)
    
    return stats


def main():
    """Ana fonksiyon - HARDCODED ayarlar kullanılır"""
    
    print("=" * 60)
    print("BOŞ/KÜÇÜK DOSYA TEMİZLEME SCRIPTİ")
    print("=" * 60)
    print(f"Klasör: {ROOT_DIR.absolute()}")
    print(f"Minimum boyut: {MIN_SIZE_KB} KB")
    print(f"Uzantılar: {', '.join(EXTENSIONS)}")
    print(f"Otomatik silme: {'Aktif' if AUTO_DELETE else 'Kapalı'}")
    print("=" * 60)
    
    # Kök dizini kontrol et
    if not ROOT_DIR.exists():
        print(f"[HATA] Klasör bulunamadı: {ROOT_DIR}")
        return 1
    
    if not ROOT_DIR.is_dir():
        print(f"[HATA] Bu bir klasör değil: {ROOT_DIR}")
        return 1
    
    # Boş veya küçük dosyaları bul
    files_to_delete = find_empty_or_small_files()
    
    if not files_to_delete:
        print("\n[BAŞARILI] Silinecek dosya bulunamadı. Tüm dosyalar yeterince büyük.")
        return 0
    
    # İstatistikler
    total_size_kb = sum(size_kb for _, size_kb in files_to_delete)
    empty_files = sum(1 for _, size_kb in files_to_delete if size_kb == 0)
    
    print(f"\n[BULGULAR]")
    print(f"  - Toplam: {len(files_to_delete)} dosya")
    print(f"  - Boş dosyalar (0 KB): {empty_files}")
    print(f"  - Toplam boyut: {total_size_kb:.2f} KB")
    
    # İlk 20 dosyayı göster
    print(f"\n[ÖRNEKLER] İlk 20 dosya:")
    for i, (file_path, size_kb) in enumerate(files_to_delete[:20], 1):
        rel_path = file_path.relative_to(ROOT_DIR)
        print(f"  {i:2d}. {rel_path} ({size_kb:.2f} KB)")
    if len(files_to_delete) > 20:
        print(f"  ... ve {len(files_to_delete) - 20} dosya daha")
    
    print("\n" + "=" * 60)
    
    # Otomatik silme
    if AUTO_DELETE:
        print(f"[SİLME] {len(files_to_delete)} dosya siliniyor...")
        stats = delete_files(files_to_delete)
        
        # Sonuç raporu
        print("\n" + "=" * 60)
        print("[SONUÇ]")
        print("=" * 60)
        print(f"  - Bulunan: {stats['total_found']} dosya")
        print(f"  - Silinen: {stats['deleted']} dosya")
        print(f"  - Başarısız: {stats['failed']} dosya")
        print(f"  - Toplam boyut: {stats['total_size_kb']:.2f} KB")
        
        if stats['errors']:
            print(f"\n[UYARI] {len(stats['errors'])} hata oluştu:")
            for error in stats['errors'][:5]:
                print(f"  - {error}")
            if len(stats['errors']) > 5:
                print(f"  ... ve {len(stats['errors']) - 5} hata daha")
        
        if stats['failed'] == 0:
            print("\n[BAŞARILI] Tüm dosyalar başarıyla silindi!")
        else:
            print(f"\n[UYARI] {stats['failed']} dosya silinemedi.")
        
        return 0 if stats['failed'] == 0 else 1
    else:
        print("[BİLGİ] AUTO_DELETE=False, dosyalar silinmedi.")
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

