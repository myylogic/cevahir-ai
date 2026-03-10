# -*- coding: utf-8 -*-
"""
Topic-Based Wikipedia Scraper - Konu Bazlı Derinlemesine Veri Madenciliği

Bu script, belirli konularda derinlemesine Wikipedia içerikleri toplar.
Konu bazlı alt klasörler oluşturur ve organize eder.

Kullanım:
    python topic_based_scraper.py --topic "tarih" --depth deep
    python topic_based_scraper.py --topic "islam_tarihi" --pages 500
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
import json
import time

# Proje kök dizinini sys.path'e ekle
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Topic configuration
from data_processing.topic_config import TOPIC_PAGES, TOPIC_CATEGORIES, get_all_topics

# Wikipedia API
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


# ============================================================================
# NOT: TOPIC_PAGES ve TOPIC_CATEGORIES artık topic_config.py'den import ediliyor
# ============================================================================
# Konu yapılandırmasını genişletmek için data_processing/topic_config.py dosyasını düzenleyin
# ============================================================================



class TopicBasedScraper:
    """Konu bazlı derinlemesine Wikipedia içerikleri toplayan sınıf."""
    
    def __init__(self, language: str = "tr"):
        """Args: language: Wikipedia dil kodu (varsayılan: "tr" - Türkçe)"""
        self.language = language
        
        if not WIKIPEDIA_API_AVAILABLE:
            raise ImportError(
                "Wikipedia API için 'wikipedia-api' kütüphanesi gerekli.\n"
                "Yükleme: pip install wikipedia-api"
            )
        
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
                # Sadece sayfa başlıklarını al (alt kategorileri değil)
                if not page_title.startswith("Category:"):
                    page_titles.append(page_title)
            
            return page_titles
        except Exception as e:
            logger.error(f"Kategori alınamadı ({category_name}): {e}")
            return []
    
    def get_topic_pages(self, topic: str, depth: str = "medium") -> List[str]:
        """Konu bazlı sayfa listesini oluşturur."""
        # Önceden tanımlı sayfalar
        pages = TOPIC_PAGES.get(topic, [])
        
        # Kategori bazlı sayfalar
        categories = TOPIC_CATEGORIES.get(topic, [])
        category_pages = []
        
        if depth == "deep":
            max_pages_per_category = 200
        elif depth == "medium":
            max_pages_per_category = 100
        else:  # shallow
            max_pages_per_category = 50
        
        for category in categories:
            logger.info(f"Kategoriden sayfa toplanıyor: {category}")
            cat_pages = self.get_category_pages(category, max_pages=max_pages_per_category)
            category_pages.extend(cat_pages)
            time.sleep(0.5)  # Rate limiting
        
        # Birleştir ve tekrarları kaldır
        all_pages = list(set(pages + category_pages))
        
        logger.info(f"Toplam {len(all_pages)} sayfa bulundu (önceden tanımlı: {len(pages)}, kategori: {len(category_pages)})")
        
        return all_pages
    
    def scrape_topic(self, topic: str, output_dir: Path, depth: str = "medium", limit: Optional[int] = None) -> dict:
        """Belirli bir konuda derinlemesine içerik toplar."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sayfa listesini oluştur
        page_titles = self.get_topic_pages(topic, depth=depth)
        
        # Limit uygula
        if limit:
            page_titles = page_titles[:limit]
        
        logger.info(f"'{topic}' konusunda {len(page_titles)} sayfa toplanacak")
        
        results = {
            "topic": topic,
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
            
            # TXT formatında kaydet (RAW_TEXT modu için)
            output_file = output_dir / f"{safe_title}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(page_data["text"])
            
            results["success"] += 1
            results["files"].append(str(output_file))
            results["total_tokens"] += page_data["length"]
            
            logger.info(f"✓ Başarılı: {output_file.name} ({page_data['length']} karakter)")
            
            # Rate limiting
            time.sleep(0.5)
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Konu bazlı derinlemesine Wikipedia içerikleri toplar"
    )
    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        choices=list(TOPIC_PAGES.keys()),
        help="Toplanacak konu (örn: tarih, islam_tarihi, dinler_tarihi)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data_processing/output",
        help="Çıktı klasörü (varsayılan: data_processing/output)"
    )
    parser.add_argument(
        "--depth",
        type=str,
        default="medium",
        choices=["shallow", "medium", "deep"],
        help="Derinlik seviyesi: shallow (50 sayfa/kategori), medium (100), deep (200)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Toplam sayfa limiti (None: sınırsız)"
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Toplama sonuçlarını JSON formatında kaydet"
    )
    
    args = parser.parse_args()
    
    # Scraper oluştur
    scraper = TopicBasedScraper(language="tr")
    
    # Çıktı dizini
    output_dir = Path(args.output) / args.topic
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Konu: {args.topic}")
    logger.info(f"Derinlik: {args.depth}")
    logger.info(f"Çıktı: {output_dir}")
    
    # Konu bazlı scraping
    results = scraper.scrape_topic(
        topic=args.topic,
        output_dir=output_dir,
        depth=args.depth,
        limit=args.limit
    )
    
    # Sonuçları yazdır
    logger.info("="*60)
    logger.info(f"Konu: {results['topic']}")
    logger.info(f"Toplam başarılı: {results['success']}")
    logger.info(f"Toplam başarısız: {results['failed']}")
    logger.info(f"Toplam karakter: {results['total_tokens']:,}")
    logger.info(f"Çıktı klasörü: {output_dir}")
    logger.info("="*60)
    
    # JSON log
    if args.log:
        with open(args.log, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Sonuçlar kaydedildi: {args.log}")
    
    return 0 if results['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

