# -*- coding: utf-8 -*-
"""
Batch Scraping Script - Tüm Konular İçin Otomatik Scraping

Bu script, topic_config.py'deki tüm konular için otomatik olarak scraping yapar.
Her konu için 1000 sayfa toplar ve output klasörüne kaydeder.

Kullanım:
    python batch_scrape_all_topics.py
    python batch_scrape_all_topics.py --limit 500  # Her konu için 500 sayfa
    python batch_scrape_all_topics.py --topics tarih fizik kimya  # Sadece belirli konular
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional
import json
import time
from datetime import datetime

# Proje kök dizinini sys.path'e ekle
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Topic configuration
from data_processing.topic_config import get_all_topics

# Topic scraper
from data_processing.topic_based_scraper import TopicBasedScraper

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler('data_processing/scraping_log.txt', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def scrape_all_topics(
    output_base_dir: Path = Path("data_processing/output"),
    limit_per_topic: Optional[int] = 1000,
    depth: str = "deep",
    topics: Optional[List[str]] = None,
    delay_between_topics: float = 2.0
) -> dict:
    """
    Tüm konular için scraping yapar.
    
    Args:
        output_base_dir: Çıktı klasörü
        limit_per_topic: Her konu için maksimum sayfa sayısı
        depth: Derinlik seviyesi (shallow, medium, deep)
        topics: Scrape edilecek konular (None: tüm konular)
        delay_between_topics: Konular arası bekleme süresi (saniye)
    
    Returns:
        Tüm scraping sonuçlarını içeren dict
    """
    # Scraper oluştur
    try:
        scraper = TopicBasedScraper(language="tr")
    except Exception as e:
        logger.error(f"Scraper oluşturulamadı: {e}")
        return {"error": str(e)}
    
    # Konuları belirle
    if topics is None:
        all_topics = get_all_topics()
    else:
        all_topics = [t for t in topics if t in get_all_topics()]
    
    logger.info("="*80)
    logger.info(f"TOPLAM {len(all_topics)} KONU İÇİN SCRAPING BAŞLATILIYOR")
    logger.info(f"Her konu için: {limit_per_topic} sayfa (depth: {depth})")
    logger.info(f"Çıktı klasörü: {output_base_dir}")
    logger.info("="*80)
    
    # Sonuçları sakla
    all_results = {
        "start_time": datetime.now().isoformat(),
        "total_topics": len(all_topics),
        "limit_per_topic": limit_per_topic,
        "depth": depth,
        "topics": {}
    }
    
    # Her konu için scraping
    for i, topic in enumerate(all_topics, 1):
        logger.info("")
        logger.info("="*80)
        logger.info(f"[{i}/{len(all_topics)}] KONU: {topic.upper()}")
        logger.info("="*80)
        
        try:
            # Çıktı dizini
            output_dir = output_base_dir / topic
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Scraping başlat
            start_time = time.time()
            results = scraper.scrape_topic(
                topic=topic,
                output_dir=output_dir,
                depth=depth,
                limit=limit_per_topic
            )
            elapsed_time = time.time() - start_time
            
            # Sonuçları kaydet
            results["elapsed_time"] = elapsed_time
            results["elapsed_time_formatted"] = f"{elapsed_time/60:.2f} dakika"
            all_results["topics"][topic] = results
            
            # Özet
            logger.info("")
            logger.info(f"✓ {topic} tamamlandı!")
            logger.info(f"  Başarılı: {results['success']} sayfa")
            logger.info(f"  Başarısız: {results['failed']} sayfa")
            logger.info(f"  Toplam karakter: {results['total_tokens']:,}")
            logger.info(f"  Süre: {elapsed_time/60:.2f} dakika")
            logger.info(f"  Dosya sayısı: {len(results['files'])}")
            
        except Exception as e:
            logger.error(f"✗ {topic} hatası: {e}")
            all_results["topics"][topic] = {"error": str(e)}
        
        # Konular arası bekleme (rate limiting)
        if i < len(all_topics):
            logger.info(f"Sonraki konu için {delay_between_topics} saniye bekleniyor...")
            time.sleep(delay_between_topics)
    
    # Final özet
    all_results["end_time"] = datetime.now().isoformat()
    
    total_success = sum(r.get("success", 0) for r in all_results["topics"].values() if isinstance(r, dict) and "success" in r)
    total_failed = sum(r.get("failed", 0) for r in all_results["topics"].values() if isinstance(r, dict) and "failed" in r)
    total_tokens = sum(r.get("total_tokens", 0) for r in all_results["topics"].values() if isinstance(r, dict) and "total_tokens" in r)
    total_files = sum(len(r.get("files", [])) for r in all_results["topics"].values() if isinstance(r, dict) and "files" in r)
    
    all_results["summary"] = {
        "total_success": total_success,
        "total_failed": total_failed,
        "total_tokens": total_tokens,
        "total_files": total_files
    }
    
    logger.info("")
    logger.info("="*80)
    logger.info("TÜM SCRAPING TAMAMLANDI!")
    logger.info("="*80)
    logger.info(f"Toplam başarılı sayfa: {total_success:,}")
    logger.info(f"Toplam başarısız sayfa: {total_failed:,}")
    logger.info(f"Toplam karakter: {total_tokens:,}")
    logger.info(f"Toplam dosya: {total_files:,}")
    logger.info(f"Toplam token (tahmini): {total_tokens/4:,}")  # ~4 karakter = 1 token
    logger.info("="*80)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Tüm konular için otomatik Wikipedia scraping"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data_processing/output",
        help="Çıktı klasörü (varsayılan: data_processing/output)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Her konu için maksimum sayfa sayısı (varsayılan: 1000)"
    )
    parser.add_argument(
        "--depth",
        type=str,
        default="deep",
        choices=["shallow", "medium", "deep"],
        help="Derinlik seviyesi (varsayılan: deep)"
    )
    parser.add_argument(
        "--topics",
        type=str,
        nargs="+",
        default=None,
        help="Scrape edilecek konular (boş: tüm konular)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Konular arası bekleme süresi - saniye (varsayılan: 2.0)"
    )
    parser.add_argument(
        "--log",
        type=str,
        default="data_processing/scraping_results.json",
        help="Sonuçları kaydet (JSON formatında)"
    )
    
    args = parser.parse_args()
    
    # Çıktı dizini
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Scraping başlat
    results = scrape_all_topics(
        output_base_dir=output_dir,
        limit_per_topic=args.limit,
        depth=args.depth,
        topics=args.topics,
        delay_between_topics=args.delay
    )
    
    # Sonuçları kaydet
    if args.log:
        with open(args.log, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Sonuçlar kaydedildi: {args.log}")
    
    return 0 if results.get("summary", {}).get("total_failed", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

