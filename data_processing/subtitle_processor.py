# -*- coding: utf-8 -*-
"""
Subtitle Processor - Altyazı Verilerini İşleme ve Genişletme Scripti

Bu script, SRT, VTT ve diğer altyazı formatlarını TXT formatına dönüştürür
ve eğitim verisi olarak kullanıma hazır hale getirir.

Kullanım:
    python subtitle_processor.py --input subtitles_folder --output txt_folder
    python subtitle_processor.py --input subtitles_folder --output txt_folder --format json
"""

import os
import sys
import argparse
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional
import json

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


class SubtitleProcessor:
    """Altyazı dosyalarını işleyen sınıf."""
    
    # SRT format regex
    SRT_PATTERN = re.compile(
        r'(\d+)\s*\n'
        r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s*\n'
        r'(.+?)(?=\n\d+\s*\n|\Z)',
        re.DOTALL | re.MULTILINE
    )
    
    # VTT format regex
    VTT_PATTERN = re.compile(
        r'(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\s*\n'
        r'(.+?)(?=\n\d{2}:\d{2}:\d{2}\.\d{3}|\Z)',
        re.DOTALL | re.MULTILINE
    )
    
    def __init__(self, output_format: str = "txt"):
        """
        Args:
            output_format: "txt" veya "json"
        """
        self.output_format = output_format.lower()
        if self.output_format not in ["txt", "json"]:
            raise ValueError("output_format 'txt' veya 'json' olmalıdır")
    
    def parse_srt(self, content: str) -> List[Dict[str, str]]:
        """SRT formatını parse eder."""
        subtitles = []
        matches = self.SRT_PATTERN.findall(content)
        
        for match in matches:
            index, start_time, end_time, text = match
            # Temizleme
            text = self._clean_text(text)
            if text:
                subtitles.append({
                    "index": int(index),
                    "start": start_time,
                    "end": end_time,
                    "text": text
                })
        
        return subtitles
    
    def parse_vtt(self, content: str) -> List[Dict[str, str]]:
        """VTT formatını parse eder."""
        subtitles = []
        matches = self.VTT_PATTERN.findall(content)
        
        for match in matches:
            start_time, end_time, text = match
            # Temizleme
            text = self._clean_text(text)
            if text:
                subtitles.append({
                    "start": start_time,
                    "end": end_time,
                    "text": text
                })
        
        return subtitles
    
    def _clean_text(self, text: str) -> str:
        """Altyazı metnini temizler."""
        # HTML tag'lerini kaldır
        text = re.sub(r'<[^>]+>', '', text)
        # Çoklu boşlukları temizle
        text = re.sub(r'\s+', ' ', text)
        # Başlangıç/bitiş boşluklarını kaldır
        text = text.strip()
        # Özel karakterleri temizle (isteğe bağlı)
        # text = re.sub(r'[^\w\s.,!?;:()\-\'"]', '', text)
        return text
    
    def process_file(self, subtitle_path: Path) -> Optional[List[Dict[str, str]]]:
        """Tek bir altyazı dosyasını işler."""
        if not subtitle_path.exists():
            logger.error(f"Altyazı dosyası bulunamadı: {subtitle_path}")
            return None
        
        try:
            with open(subtitle_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Format tespiti
            ext = subtitle_path.suffix.lower()
            
            if ext == '.srt':
                subtitles = self.parse_srt(content)
            elif ext == '.vtt':
                subtitles = self.parse_vtt(content)
            else:
                logger.warning(f"Desteklenmeyen format: {ext} ({subtitle_path})")
                return None
            
            logger.info(f"İşlendi: {subtitle_path.name} → {len(subtitles)} altyazı")
            return subtitles
            
        except Exception as e:
            logger.error(f"Altyazı işleme hatası ({subtitle_path}): {e}")
            return None
    
    def save_as_txt(self, subtitles: List[Dict[str, str]], output_path: Path):
        """Altyazıları TXT formatında kaydeder (sadece metinler)."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for sub in subtitles:
                f.write(sub['text'] + '\n')
    
    def save_as_json(self, subtitles: List[Dict[str, str]], output_path: Path):
        """Altyazıları JSON formatında kaydeder (QA formatına uygun)."""
        # Altyazıları diyalog formatına dönüştür
        dialogues = []
        for i in range(len(subtitles) - 1):
            # Ardışık altyazıları soru-cevap olarak düşün
            question = subtitles[i]['text']
            answer = subtitles[i + 1]['text']
            dialogues.append({
                "Soru": question,
                "Cevap": answer
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dialogues, f, ensure_ascii=False, indent=2)
    
    def convert_file(self, subtitle_path: Path, output_path: Path) -> bool:
        """Tek bir altyazı dosyasını dönüştürür."""
        subtitles = self.process_file(subtitle_path)
        if not subtitles:
            return False
        
        # Çıktı dizinini oluştur
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.output_format == "txt":
            self.save_as_txt(subtitles, output_path)
        else:  # json
            self.save_as_json(subtitles, output_path)
        
        return True
    
    def convert_directory(self, input_dir: Path, output_dir: Path, recursive: bool = True) -> dict:
        """
        Bir dizindeki tüm altyazı dosyalarını dönüştürür.
        
        Returns:
            dict: {"success": int, "failed": int, "files": List[str]}
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Altyazı dosyalarını bul
        extensions = ['.srt', '.vtt']
        if recursive:
            subtitle_files = []
            for ext in extensions:
                subtitle_files.extend(input_dir.rglob(f"*{ext}"))
        else:
            subtitle_files = []
            for ext in extensions:
                subtitle_files.extend(input_dir.glob(f"*{ext}"))
        
        logger.info(f"Toplam {len(subtitle_files)} altyazı dosyası bulundu")
        
        results = {
            "success": 0,
            "failed": 0,
            "files": []
        }
        
        for subtitle_path in subtitle_files:
            # Çıktı dosya yolu
            relative_path = subtitle_path.relative_to(input_dir)
            if self.output_format == "txt":
                output_path = output_dir / relative_path.with_suffix('.txt')
            else:  # json
                output_path = output_dir / relative_path.with_suffix('.json')
            
            logger.info(f"Dönüştürülüyor: {subtitle_path.name} → {output_path.name}")
            
            if self.convert_file(subtitle_path, output_path):
                results["success"] += 1
                results["files"].append(str(output_path))
                logger.info(f"✓ Başarılı: {output_path.name}")
            else:
                results["failed"] += 1
                logger.error(f"✗ Başarısız: {subtitle_path.name}")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Altyazı dosyalarını TXT veya JSON formatına dönüştürür"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Altyazı dosyalarının bulunduğu dizin veya dosya yolu"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Çıktı dosyalarının kaydedileceği dizin"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="txt",
        choices=["txt", "json"],
        help="Çıktı formatı: 'txt' veya 'json' (varsayılan: txt)"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Alt dizinleri de tarar"
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Dönüşüm sonuçlarını JSON formatında kaydet"
    )
    
    args = parser.parse_args()
    
    # Processor oluştur
    processor = SubtitleProcessor(output_format=args.format)
    
    # Giriş yolu
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if input_path.is_file():
        # Tek dosya
        logger.info(f"Tek dosya dönüşümü: {input_path.name}")
        if args.format == "txt":
            output_file = output_path / input_path.with_suffix('.txt').name
        else:
            output_file = output_path / input_path.with_suffix('.json').name
        success = processor.convert_file(input_path, output_file)
        results = {
            "success": 1 if success else 0,
            "failed": 0 if success else 1,
            "files": [str(output_file)] if success else []
        }
    else:
        # Dizin
        logger.info(f"Dizin dönüşümü: {input_path} → {output_path}")
        results = processor.convert_directory(input_path, output_path, recursive=args.recursive)
    
    # Sonuçları yazdır
    logger.info("="*60)
    logger.info(f"Toplam başarılı: {results['success']}")
    logger.info(f"Toplam başarısız: {results['failed']}")
    logger.info("="*60)
    
    # JSON log
    if args.log:
        with open(args.log, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Sonuçlar kaydedildi: {args.log}")
    
    return 0 if results['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

