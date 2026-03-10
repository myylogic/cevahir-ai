# -*- coding: utf-8 -*-
"""
Wikipedia API Scraper - Wikipedia Türkçe İçeriklerini Toplama Scripti

Bu script, Wikipedia Türkçe'den içerikleri toplar ve eğitim verisi olarak kullanıma hazır hale getirir.

Kullanım:
    python wikipedia_api_scraper.py --output wikipedia_data --limit 1000
    python wikipedia_api_scraper.py --output wikipedia_data --limit 1000 --format json
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
import json
import time

# Wikipedia API kütüphaneleri
try:
    import wikipediaapi
    WIKIPEDIA_API_AVAILABLE = True
except ImportError:
    WIKIPEDIA_API_AVAILABLE = False
    wikipediaapi = None

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


class WikipediaScraper:
    """Wikipedia Türkçe içeriklerini toplayan sınıf."""
    
    def __init__(self, language: str = "tr", output_format: str = "txt"):
        """
        Args:
            language: Wikipedia dil kodu (varsayılan: "tr" - Türkçe)
            output_format: "txt" veya "json"
        """
        self.language = language
        self.output_format = output_format.lower()
        if self.output_format not in ["txt", "json"]:
            raise ValueError("output_format 'txt' veya 'json' olmalıdır")
        
        # Wikipedia API kontrolü
        if not WIKIPEDIA_API_AVAILABLE:
            raise ImportError(
                "Wikipedia API için 'wikipediaapi' kütüphanesi gerekli.\n"
                "Yükleme: pip install wikipedia-api"
            )
        
        # Wikipedia API başlat
        self.wiki = wikipediaapi.Wikipedia(
            language=self.language,
            user_agent='CevahirProject/1.0 (Educational Purpose)'
        )
    
    def get_page_content(self, page_title: str) -> Optional[Dict[str, any]]:
        """Bir Wikipedia sayfasının içeriğini alır."""
        try:
            page = self.wiki.page(page_title)
            if not page.exists():
                logger.warning(f"Sayfa bulunamadı: {page_title}")
                return None
            
            return {
                "title": page.title,
                "text": page.text,
                "url": page.fullurl,
                "length": len(page.text)
            }
        except Exception as e:
            logger.error(f"Sayfa alınamadı ({page_title}): {e}")
            return None
    
    def get_category_pages(self, category_name: str, max_pages: int = 100) -> List[str]:
        """Bir kategorideki sayfa başlıklarını alır."""
        try:
            category = self.wiki.page(f"Category:{category_name}")
            if not category.exists():
                logger.warning(f"Kategori bulunamadı: {category_name}")
                return []
            
            page_titles = []
            for page_title in category.categorymembers:
                if len(page_titles) >= max_pages:
                    break
                page_titles.append(page_title)
            
            return page_titles
        except Exception as e:
            logger.error(f"Kategori alınamadı ({category_name}): {e}")
            return []
    
    def get_random_pages(self, count: int = 100) -> List[str]:
        """Rastgele sayfa başlıklarını alır."""
        # NOT: wikipediaapi kütüphanesi rastgele sayfa desteği sınırlı
        # Alternatif: Belirli kategorilerden sayfa toplama
        logger.warning("Rastgele sayfa özelliği sınırlı. Kategori bazlı toplama önerilir.")
        return []
    
    def scrape_pages(self, page_titles: List[str], output_dir: Path) -> dict:
        """Birden fazla sayfayı toplar ve kaydeder."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            "success": 0,
            "failed": 0,
            "files": [],
            "total_tokens": 0
        }
        
        for i, page_title in enumerate(page_titles, 1):
            logger.info(f"[{i}/{len(page_titles)}] İşleniyor: {page_title}")
            
            page_data = self.get_page_content(page_title)
            if not page_data:
                results["failed"] += 1
                continue
            
            # Dosya adı (güvenli karakterler)
            safe_title = "".join(c for c in page_title if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_title = safe_title.replace(' ', '_')
            
            if self.output_format == "txt":
                output_file = output_dir / f"{safe_title}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(page_data["text"])
            else:  # json
                output_file = output_dir / f"{safe_title}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "title": page_data["title"],
                        "text": page_data["text"],
                        "url": page_data["url"]
                    }, f, ensure_ascii=False, indent=2)
            
            results["success"] += 1
            results["files"].append(str(output_file))
            results["total_tokens"] += page_data["length"]
            
            logger.info(f"✓ Başarılı: {output_file.name} ({page_data['length']} karakter)")
            
            # Rate limiting (API limitlerini aşmamak için)
            time.sleep(0.5)
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Wikipedia Türkçe içeriklerini toplar"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Çıktı dosyalarının kaydedileceği dizin"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Toplanacak sayfa sayısı (varsayılan: 100)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="txt",
        choices=["txt", "json"],
        help="Çıktı formatı: 'txt' veya 'json' (varsayılan: txt)"
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Belirli bir kategoriden sayfa topla (örn: 'Bilim', 'Tarih')"
    )
    parser.add_argument(
        "--pages",
        type=str,
        nargs="+",
        default=None,
        help="Belirli sayfa başlıklarını topla (örn: 'Python', 'Türkiye')"
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Toplama sonuçlarını JSON formatında kaydet"
    )
    
    args = parser.parse_args()
    
    # Scraper oluştur
    scraper = WikipediaScraper(language="tr", output_format=args.format)
    
    # Sayfa başlıklarını topla
    page_titles = []
    
    if args.pages:
        # Belirli sayfalar
        page_titles = args.pages[:args.limit]
    elif args.category:
        # Kategori bazlı
        logger.info(f"Kategoriden sayfa toplanıyor: {args.category}")
        page_titles = scraper.get_category_pages(args.category, max_pages=args.limit)
    else:
        # Örnek sayfalar (kullanıcı kendi listesini oluşturabilir)
        logger.warning("Sayfa listesi belirtilmedi. Örnek sayfalar kullanılıyor.")
        page_titles = [
            "Python_(programlama_dili)",
            "Türkiye",
            "İstanbul",
            "Ankara",
            "Türkçe",
            "Bilgisayar",
            "Yapay_zeka",
            "Makine_öğrenmesi",
            "Doğal_dil_işleme",
            "Türk_edebiyatı"
        ][:args.limit]
    
    if not page_titles:
        logger.error("Toplanacak sayfa bulunamadı!")
        return 1
    
    logger.info(f"Toplam {len(page_titles)} sayfa toplanacak")
    
    # Sayfaları topla
    output_dir = Path(args.output)
    results = scraper.scrape_pages(page_titles, output_dir)
    
    # Sonuçları yazdır
    logger.info("="*60)
    logger.info(f"Toplam başarılı: {results['success']}")
    logger.info(f"Toplam başarısız: {results['failed']}")
    logger.info(f"Toplam karakter: {results['total_tokens']:,}")
    logger.info("="*60)
    
    # JSON log
    if args.log:
        with open(args.log, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Sonuçlar kaydedildi: {args.log}")
    
    return 0 if results['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

