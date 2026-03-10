# -*- coding: utf-8 -*-
"""
Subtitle Downloader - Türkçe Altyazı Otomatik İndirme Scripti

Bu script, Opensubtitles API kullanarak Türkçe dizi/film altyazılarını otomatik indirir.

Kullanım:
    # Tek dizi için
    python subtitle_downloader.py --dizi "Kurtlar Vadisi" --sezon 1 --bölüm 1
    
    # Toplu indirme (dizi listesi)
    python subtitle_downloader.py --batch turkce_dizi_listesi.json
    
    # Manuel dizi listesi ile
    python subtitle_downloader.py --dizi-listesi "Kurtlar Vadisi,Ezel,Behzat Ç."

Gereksinimler:
    pip install opensubtitlescom
    # veya
    pip install subliminal

NOT: Opensubtitles.org'a kayıt olmanız ve API key almanız gerekebilir.
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)

# Opensubtitles API denemesi
try:
    from opensubtitlescom import OpenSubtitles
    OPENSUBTITLES_AVAILABLE = True
except ImportError:
    OPENSUBTITLES_AVAILABLE = False
    logger.warning("opensubtitlescom kütüphanesi bulunamadı. 'pip install opensubtitlescom' ile yükleyin.")

# Alternatif: subliminal
try:
    import subliminal
    SUBLIMINAL_AVAILABLE = True
except ImportError:
    SUBLIMINAL_AVAILABLE = False
    logger.warning("subliminal kütüphanesi bulunamadı. 'pip install subliminal' ile yükleyin.")


class SubtitleDownloader:
    """Türkçe altyazı indirme sınıfı"""
    
    # Türkçe dizi listesi (önceden tanımlı)
    TURKCE_DIZILER = [
        "Kurtlar Vadisi",
        "Ezel",
        "Muhteşem Yüzyıl",
        "Diriliş: Ertuğrul",
        "Çukur",
        "Behzat Ç.",
        "Şahsiyet",
        "Fi",
        "Avrupa Yakası",
        "Çocuklar Duymasın",
        "Kara Para Aşk",
        "İçerde",
        "Masum",
        "Börü 2039",
        "50m2",
    ]
    
    def __init__(self, output_dir: str = "data_processing/subtitles", api_key: Optional[str] = None):
        """
        Args:
            output_dir: İndirilen altyazıların kaydedileceği dizin
            api_key: Opensubtitles API key (opsiyonel)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.api_key = api_key or os.getenv("OPENSUBTITLES_API_KEY")
        
        # API client'ı başlat
        self.client = None
        if OPENSUBTITLES_AVAILABLE:
            try:
                self.client = OpenSubtitles(api_key=self.api_key)
                logger.info("Opensubtitles API bağlantısı başarılı")
            except Exception as e:
                logger.warning(f"Opensubtitles API bağlantısı başarısız: {e}")
                logger.warning("Manuel indirme yöntemi kullanılacak")
        else:
            logger.warning("Opensubtitles API kütüphanesi yok, manuel indirme yöntemi kullanılacak")
    
    def normalize_dizi_name(self, dizi_name: str) -> str:
        """Dizi adını normalize et (arama için)"""
        # Türkçe karakterleri normalize et
        replacements = {
            "ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u",
            "Ç": "C", "Ğ": "G", "İ": "I", "Ö": "O", "Ş": "S", "Ü": "U"
        }
        normalized = dizi_name
        for tr, en in replacements.items():
            normalized = normalized.replace(tr, en)
        return normalized
    
    def search_subtitles(self, query: str, language: str = "tur") -> List[Dict]:
        """
        Altyazı ara
        
        Args:
            query: Dizi/film adı
            language: Dil kodu (tur = Türkçe)
        
        Returns:
            Altyazı listesi
        """
        if not self.client:
            logger.error("API client mevcut değil")
            return []
        
        try:
            # Opensubtitles API ile ara
            results = self.client.search(query=query, languages=[language])
            logger.info(f"'{query}' için {len(results)} altyazı bulundu")
            return results
        except Exception as e:
            logger.error(f"Altyazı arama hatası: {e}")
            return []
    
    def download_subtitle(self, subtitle_id: str, output_path: Path) -> bool:
        """
        Altyazı indir
        
        Args:
            subtitle_id: Altyazı ID
            output_path: Kayıt yolu
        
        Returns:
            Başarılı mı?
        """
        if not self.client:
            logger.error("API client mevcut değil")
            return False
        
        try:
            # Altyazı indir
            subtitle_data = self.client.download(subtitle_id)
            
            # Dosyaya kaydet
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(subtitle_data)
            
            logger.info(f"Altyazı indirildi: {output_path.name}")
            return True
        except Exception as e:
            logger.error(f"Altyazı indirme hatası: {e}")
            return False
    
    def download_dizi_subtitles(self, dizi_name: str, sezon: Optional[int] = None, 
                                 bölüm: Optional[int] = None, max_results: int = 10) -> List[Path]:
        """
        Dizi altyazılarını indir
        
        Args:
            dizi_name: Dizi adı
            sezon: Sezon numarası (opsiyonel)
            bölüm: Bölüm numarası (opsiyonel)
            max_results: Maksimum indirme sayısı
        
        Returns:
            İndirilen dosya yolları
        """
        # Arama sorgusu oluştur
        query = dizi_name
        if sezon and bölüm:
            query = f"{dizi_name} S{sezon:02d}E{bölüm:02d}"
        elif sezon:
            query = f"{dizi_name} Season {sezon}"
        
        logger.info(f"'{dizi_name}' için altyazı aranıyor...")
        
        # Altyazı ara
        results = self.search_subtitles(query, language="tur")
        
        if not results:
            logger.warning(f"'{dizi_name}' için Türkçe altyazı bulunamadı")
            return []
        
        # İndirilen dosyalar
        downloaded_files = []
        
        # İlk N sonucu indir
        for i, result in enumerate(results[:max_results]):
            subtitle_id = result.get("id") or result.get("subtitle_id")
            if not subtitle_id:
                continue
            
            # Dosya adı oluştur
            safe_name = self._sanitize_filename(dizi_name)
            if sezon and bölüm:
                filename = f"{safe_name}_S{sezon:02d}E{bölüm:02d}_{i+1}.srt"
            elif sezon:
                filename = f"{safe_name}_S{sezon:02d}_{i+1}.srt"
            else:
                filename = f"{safe_name}_{i+1}.srt"
            
            output_path = self.output_dir / safe_name / filename
            
            # İndir
            if self.download_subtitle(subtitle_id, output_path):
                downloaded_files.append(output_path)
            
            # Rate limiting
            time.sleep(1)
        
        logger.info(f"'{dizi_name}' için {len(downloaded_files)} altyazı indirildi")
        return downloaded_files
    
    def _sanitize_filename(self, filename: str) -> str:
        """Dosya adını güvenli hale getir"""
        # Özel karakterleri temizle
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Boşlukları alt çizgi ile değiştir
        filename = filename.replace(' ', '_')
        return filename
    
    def batch_download(self, dizi_listesi: List[str], max_per_dizi: int = 20) -> Dict[str, List[Path]]:
        """
        Toplu altyazı indirme
        
        Args:
            dizi_listesi: Dizi adları listesi
            max_per_dizi: Dizi başına maksimum indirme
        
        Returns:
            {dizi_adı: [indirilen_dosyalar]}
        """
        results = {}
        
        for dizi_name in dizi_listesi:
            logger.info(f"="*60)
            logger.info(f"Dizi: {dizi_name}")
            logger.info(f"="*60)
            
            # Tüm sezonlar için ara (1-10 sezon)
            downloaded = []
            for sezon in range(1, 11):  # 1-10 sezon
                for bölüm in range(1, 25):  # 1-24 bölüm
                    files = self.download_dizi_subtitles(
                        dizi_name, 
                        sezon=sezon, 
                        bölüm=bölüm,
                        max_results=1  # Her bölüm için 1 altyazı
                    )
                    downloaded.extend(files)
                    
                    if len(downloaded) >= max_per_dizi:
                        break
                    
                    # Rate limiting
                    time.sleep(2)
                
                if len(downloaded) >= max_per_dizi:
                    break
            
            results[dizi_name] = downloaded
            logger.info(f"'{dizi_name}' için toplam {len(downloaded)} altyazı indirildi")
        
        return results


def create_manual_download_guide(dizi_listesi: List[str], output_file: str = "data_processing/MANUAL_SUBTITLE_DOWNLOAD.md"):
    """Manuel indirme rehberi oluştur"""
    # Dizin yoksa oluştur
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    guide = f"""# Türkçe Altyazı Manuel İndirme Rehberi

## 📥 İndirme Kaynakları

### 1. Opensubtitles.org (Önerilen)
**URL:** https://www.opensubtitles.org
**Durum:** ✅ Aktif, ücretsiz kayıt
**Format:** SRT, VTT
**Türkçe Altyazı:** ✅ Mevcut

**Kullanım:**
1. https://www.opensubtitles.org adresine git
2. Kayıt ol (ücretsiz)
3. Dizi/film adını ara
4. Türkçe altyazıları filtrele
5. İndir

### 2. Subscene.com
**URL:** https://subscene.com
**Durum:** ✅ Aktif
**Format:** SRT
**Türkçe Altyazı:** ✅ Mevcut

**Kullanım:**
1. https://subscene.com adresine git
2. Dizi/film adını ara
3. Türkçe altyazıları seç
4. İndir

### 3. YifySubtitles
**URL:** https://yifysubtitles.com
**Durum:** ✅ Aktif
**Format:** SRT
**Türkçe Altyazı:** ⚠️ Sınırlı

---

## 📋 İndirilecek Diziler

"""
    for i, dizi in enumerate(dizi_listesi, 1):
        guide += f"{i}. **{dizi}**\n"
        guide += f"   - Opensubtitles: https://www.opensubtitles.org/en/search/sublanguageid-tur/q-{dizi.replace(' ', '+')}\n"
        guide += f"   - Subscene: https://subscene.com/subtitles/search?q={dizi.replace(' ', '+')}\n"
        guide += "\n"
    
    guide += """
---

## 🔄 İşlem Adımları

1. **Altyazıları İndir**
   - Yukarıdaki kaynaklardan manuel indir
   - `data_processing/subtitles/` klasörüne kaydet
   - Her dizi için ayrı klasör oluştur (örn: `Kurtlar_Vadisi/`)

2. **Altyazıları İşle**
   ```bash
   python data_processing/subtitle_processor.py --input data_processing/subtitles --output data_processing/subtitles_processed --format txt
   ```

3. **Eğitim Verisine Ekle**
   - İşlenmiş altyazıları `education/` klasörüne taşı
   - BPE training ve model eğitimi sırasında otomatik yüklenecek

---

## ⚠️ Önemli Notlar

- **Telif Hakları:** Altyazılar telif hakkı koruması altında olabilir. Sadece eğitim amaçlı kullanın.
- **Kalite Kontrolü:** İndirilen altyazıları kontrol edin, bozuk dosyaları silin.
- **Format:** SRT formatı tercih edilir (en yaygın).
- **Encoding:** UTF-8 encoding kullanın.

---

## 🤖 Otomatik İndirme (Gelecek)

Opensubtitles API ile otomatik indirme scripti geliştirilmektedir.
API key gerektirir ve rate limiting'e dikkat edilmelidir.

"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(guide)
    
    logger.info(f"Manuel indirme rehberi oluşturuldu: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Türkçe altyazı otomatik indirme scripti"
    )
    parser.add_argument(
        "--dizi",
        type=str,
        help="Dizi adı (örn: 'Kurtlar Vadisi')"
    )
    parser.add_argument(
        "--sezon",
        type=int,
        help="Sezon numarası (opsiyonel)"
    )
    parser.add_argument(
        "--bölüm",
        type=int,
        help="Bölüm numarası (opsiyonel)"
    )
    parser.add_argument(
        "--dizi-listesi",
        type=str,
        help="Virgülle ayrılmış dizi listesi (örn: 'Kurtlar Vadisi,Ezel,Behzat Ç.')"
    )
    parser.add_argument(
        "--batch",
        type=str,
        help="JSON dosyası ile toplu indirme (dizi listesi içeren)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data_processing/subtitles",
        help="İndirilen altyazıların kaydedileceği dizin"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="Opensubtitles API key (opsiyonel, OPENSUBTITLES_API_KEY env var'dan da alınabilir)"
    )
    parser.add_argument(
        "--max-per-dizi",
        type=int,
        default=20,
        help="Dizi başına maksimum indirme sayısı"
    )
    parser.add_argument(
        "--manuel-rehber",
        action="store_true",
        help="Manuel indirme rehberi oluştur (API yoksa)"
    )
    
    args = parser.parse_args()
    
    # Manuel rehber oluştur
    if args.manuel_rehber or not OPENSUBTITLES_AVAILABLE:
        logger.info("Manuel indirme rehberi oluşturuluyor...")
        dizi_listesi = SubtitleDownloader.TURKCE_DIZILER
        if args.dizi_listesi:
            dizi_listesi = [d.strip() for d in args.dizi_listesi.split(',')]
        create_manual_download_guide(dizi_listesi)
        logger.info("Manuel indirme rehberi oluşturuldu: data_processing/MANUAL_SUBTITLE_DOWNLOAD.md")
        if not OPENSUBTITLES_AVAILABLE:
            logger.warning("Opensubtitles API kütüphanesi yok, manuel indirme yöntemi kullanılmalı")
            return 0
    
    # Downloader oluştur
    downloader = SubtitleDownloader(output_dir=args.output, api_key=args.api_key)
    
    # Tek dizi indirme
    if args.dizi:
        logger.info(f"'{args.dizi}' için altyazı indiriliyor...")
        files = downloader.download_dizi_subtitles(
            args.dizi,
            sezon=args.sezon,
            bölüm=args.bölüm,
            max_results=args.max_per_dizi
        )
        logger.info(f"Toplam {len(files)} altyazı indirildi")
        return 0
    
    # Dizi listesi ile toplu indirme
    if args.dizi_listesi:
        dizi_listesi = [d.strip() for d in args.dizi_listesi.split(',')]
        logger.info(f"{len(dizi_listesi)} dizi için toplu indirme başlatılıyor...")
        results = downloader.batch_download(dizi_listesi, max_per_dizi=args.max_per_dizi)
        
        total = sum(len(files) for files in results.values())
        logger.info(f"Toplam {total} altyazı indirildi")
        return 0
    
    # JSON dosyası ile toplu indirme
    if args.batch:
        batch_file = Path(args.batch)
        if not batch_file.exists():
            logger.error(f"Batch dosyası bulunamadı: {batch_file}")
            return 1
        
        with open(batch_file, 'r', encoding='utf-8') as f:
            batch_data = json.load(f)
        
        dizi_listesi = batch_data.get("diziler", [])
        if not dizi_listesi:
            logger.error("Batch dosyasında 'diziler' anahtarı bulunamadı")
            return 1
        
        logger.info(f"{len(dizi_listesi)} dizi için toplu indirme başlatılıyor...")
        results = downloader.batch_download(dizi_listesi, max_per_dizi=args.max_per_dizi)
        
        total = sum(len(files) for files in results.values())
        logger.info(f"Toplam {total} altyazı indirildi")
        return 0
    
    # Varsayılan: Tüm önceden tanımlı diziler
    logger.info("Önceden tanımlı tüm diziler için indirme başlatılıyor...")
    results = downloader.batch_download(
        SubtitleDownloader.TURKCE_DIZILER,
        max_per_dizi=args.max_per_dizi
    )
    
    total = sum(len(files) for files in results.values())
    logger.info(f"Toplam {total} altyazı indirildi")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

