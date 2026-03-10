#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Altyazı işleme ve normalizasyon scripti
Hem altyazı çıkarır hem normalize eder (GÜNCELLENMİŞ VERSİYON)
"""

import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import List, Tuple

# Ana dizine ekle (Eğer bu kütüphane dışarıdan çağrılıyorsa gerekli)
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class SubtitleProcessor:
    """Altyazı dosyalarını işler ve normalize eder"""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # HTML etiketleri için regex
        self.html_pattern = re.compile(r'<[^>]+>', re.IGNORECASE)
        
        # Çoklu noktalama işaretleri için regex
        self.multiple_punct_pattern = re.compile(r'([!?.])\1{2,}', re.UNICODE)
        
        # Çoklu boşluklar için regex
        self.multiple_space_pattern = re.compile(r'\s+', re.UNICODE)
        
        # Çeviri notları için regex
        # <i><b>Çeviri: ...</b></i> tarzındaki notları temizler
        self.translation_pattern = re.compile(r'(?:<i[^>]*>)?(?:<b[^>]*>)?\s*Çeviri:.*?(?:</b[^>]*>)?(?:</i[^>]*>)?', re.IGNORECASE | re.DOTALL)
        
        # Zaman damgaları için regex (00:00:00,000 -> 00:00:00,000)
        self.timestamp_pattern = re.compile(r'\d{2}:\d{2}:\d{2},\d{3}', re.UNICODE)
        
        # Sadece sayı olan satırlar için regex (SRT indeks numaraları)
        # Güçlendirilmiş regex: Sadece sayılar, etrafında boşluklar olabilir.
        self.number_only_pattern = re.compile(r'^\s*\d+\s*$', re.UNICODE)
    
    def read_with_encoding_detection(self, file_path: Path) -> str:
        """Dosyayı encoding auto-detection ile okur"""
        encodings = ['utf-8', 'windows-1254', 'iso-8859-9', 'cp1252', 'latin1']
        
        for encoding in encodings:
            try:
                return file_path.read_text(encoding=encoding, errors='strict')
            except UnicodeDecodeError:
                continue
        
        # Fallback
        return file_path.read_text(encoding='utf-8', errors='ignore')
    
    def clean_subtitle_content(self, content: str) -> str:
        """Altyazı içeriğini temizler ve diyalog verisine çevirir"""
        lines = content.strip().split('\n')
        cleaned_lines = []
        current_dialog = []
        
        for line in lines:
            line = line.strip()
            
            # --- Diyalog Sonu Mantığı ---
            if not line:
                if current_dialog:
                    # Diyalog satırlarını birleştirip listeye ekle
                    dialog = ' '.join(current_dialog).strip()
                    if dialog:
                        cleaned_lines.append(dialog)
                    current_dialog = []
                continue
                
            # --- Temizleme Mantığı ---
            
            # 1. İndeks satırını atla (Sadece sayı olan satırlar - Genellikle SRT/TXT formatında olur)
            if self.number_only_pattern.match(line):
                continue
                
            # 2. Zaman damgası satırını atla (00:00:00,000 --> 00:00:00,000 içerenler)
            if '-->' in line or self.timestamp_pattern.match(line):
                continue
                
            # 3. HTML etiketlerini kaldır (<i>, <b>, <font> vb.)
            line = self.html_pattern.sub('', line)
            
            # 4. Satır başındaki indeks numaralarını kaldır (1, 1., 1 - gibi)
            line = re.sub(r'^\s*\d+\.?\s*-?\s*', '', line)
            
            # 5. Çeviri notlarını temizle (normalize_subtitle_text'te de var, burada da olsun)
            line = self.translation_pattern.sub('', line)

            # --- Diyalog Biriktirme ---
            current_dialog.append(line)
        
        # Son diyalogu ekle
        if current_dialog:
            dialog = ' '.join(current_dialog).strip()
            if dialog:
                cleaned_lines.append(dialog)
        
        # Tüm diyalogları birleştir ve fazla boşlukları temizle
        combined_content = ' '.join(cleaned_lines)
        combined_content = self.multiple_space_pattern.sub(' ', combined_content).strip()
        
        return combined_content
    
    def process_str_format(self, content: str) -> str:
        """STR formatını işler - clean_subtitle_content ile benzer ama STR döngüsü için optimize edilmiş"""
        
        # STR ve SRT formatının yapısı çok benzer olduğu için,
        # sadece döngü yönetiminde küçük farklılıklar vardır.
        # En doğru yöntem, clean_subtitle_content metodunu çağırmaktır.
        # Eğer bu formatlar için özel bir temizlik gereksiz ise, kodu sadeleştirebiliriz.
        # Ancak sizdeki while döngüsü yapısını koruyarak devam edelim:
        
        lines = content.strip().split('\n')
        cleaned_lines = []
        current_dialog = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # --- Diyalog Sonu Mantığı ---
            if not line:
                if current_dialog:
                    dialog = ' '.join(current_dialog).strip()
                    if dialog:
                        cleaned_lines.append(dialog)
                    current_dialog = []
                i += 1
                continue
                
            # --- Temizleme Mantığı (clean_subtitle_content ile aynı) ---
            
            # 1. İndeks satırını atla (Sadece sayı olan satırlar)
            if self.number_only_pattern.match(line):
                i += 1
                continue
                
            # 2. Zaman damgası satırını atla
            if '-->' in line or self.timestamp_pattern.match(line):
                i += 1
                continue
                
            # 3. HTML etiketlerini kaldır
            line = self.html_pattern.sub('', line)
            
            # 4. Satır başındaki indeks numaralarını kaldır
            line = re.sub(r'^\s*\d+\.?\s*-?\s*', '', line)
            
            # 5. Çeviri notlarını temizle
            line = self.translation_pattern.sub('', line)
            
            # --- Diyalog Biriktirme ---
            current_dialog.append(line)
            i += 1
        
        # Son diyalogu ekle
        if current_dialog:
            dialog = ' '.join(current_dialog).strip()
            if dialog:
                cleaned_lines.append(dialog)
        
        # Tüm diyalogları birleştir ve fazla boşlukları temizle
        combined_content = ' '.join(cleaned_lines)
        combined_content = self.multiple_space_pattern.sub(' ', combined_content).strip()
        
        return combined_content
    
    def normalize_subtitle_text(self, text: str) -> str:
        """Altyazı metnini normalize eder (Daha önce birleştirilmiş metin bloğu üzerinde çalışır)"""
        if not text or not isinstance(text, str):
            return ""
        
        # NOT: Satır bazlı temizlik (indeks, zaman damgası) 'clean_subtitle_content' içinde yapıldığı için,
        # burada sadece metnin kendisi üzerinde global temizlik yapılır.
        
        # 1) Unicode normalizasyonu (Örn: "i̇" yerine "i")
        normalized = unicodedata.normalize('NFC', text.strip())
        
        # 2) HTML etiketlerini temizle (Tekrar kontrol)
        normalized = self.html_pattern.sub('', normalized)
        
        # 3) Çeviri notlarını temizle (Tekrar kontrol)
        normalized = self.translation_pattern.sub('', normalized)
        
        # 4) Zaman damgalarını temizle (Tekrar kontrol)
        # Bu aşamada zaman damgası kalmamalı, ama ihtimale karşı:
        normalized = self.timestamp_pattern.sub('', normalized)
        
        # 5) Çoklu noktalama işaretlerini düzelt (!!! -> !)
        normalized = self.multiple_punct_pattern.sub(r'\1', normalized)
        
        # 6) Çoklu boşlukları tek boşluğa çevir
        normalized = self.multiple_space_pattern.sub(' ', normalized)
        
        # 7) Başta ve sonda boşlukları temizle
        normalized = normalized.strip()
        
        # 8) Satır başındaki olası indeks kalıntılarını temizle (Eğer kaldıysa, çok güçlü bir temizlik)
        # normalized = re.sub(r'^\s*\d+\s*', '', normalized) # Bunu yaparsak metnin başındaki rakamlar silinebilir. Bu yüzden bırakıyoruz.
        
        return normalized
    
    def process_subtitle_file(self, input_file: Path) -> Tuple[bool, str, int, int, int]:
        """Tek altyazı dosyasını işler ve normalize eder"""
        try:
            print(f"📄 İşleniyor: {input_file.name}")
            
            # Dosyayı oku
            content = self.read_with_encoding_detection(input_file)
            
            # STR veya SRT formatı için özel işleme
            if input_file.suffix.lower() in ['.str', '.srt']:
                cleaned_content = self.process_str_format(content)
            else:
                # Diğer formatlar için genel temizleme
                cleaned_content = self.clean_subtitle_content(content)
            
            # Normalize et (Temizlenmiş metin bloğu üzerinde son normalizasyon)
            normalized_content = self.normalize_subtitle_text(cleaned_content)
            
            # Sonuçları göster
            print(f"    Orijinal: {len(content)} karakter")
            print(f"    Temizlenmiş: {len(cleaned_content)} karakter")
            print(f"    Normalize edilmiş: {len(normalized_content)} karakter")
            print(f"    Azalma: {len(content) - len(normalized_content)} karakter ({((len(content) - len(normalized_content)) / len(content) * 100):.1f}%)")
            
            return True, normalized_content, len(content), len(cleaned_content), len(normalized_content)
            
        except Exception as e:
            print(f"    ❌ Hata: {input_file.name} - {e}")
            return False, "", 0, 0, 0
    
    def process_directory(self):
        """Tüm altyazı dosyalarını işler ve normalize eder"""
        print(f"🔍 Altyazı dosyaları aranıyor: {self.input_dir}")
        
        # Altyazı dosyalarını bul
        subtitle_files = []
        for ext in ['*.srt', '*.str', '*.txt']:
            subtitle_files.extend(self.input_dir.rglob(ext))
        
        if not subtitle_files:
            print("❌ Altyazı dosyası bulunamadı!")
            return
        
        print(f"📊 {len(subtitle_files)} altyazı dosyası bulundu")
        
        all_dialogs = []
        results = {
            'total_files': len(subtitle_files),
            'successful': 0,
            'failed': 0,
            'total_original_chars': 0,
            'total_cleaned_chars': 0,
            'total_normalized_chars': 0
        }
        
        for subtitle_file in subtitle_files:
            success, normalized_content, orig_chars, cleaned_chars, norm_chars = self.process_subtitle_file(subtitle_file)
            
            if success and normalized_content.strip():
                all_dialogs.append(normalized_content)
                
                # Tekil dosya olarak kaydet
                output_file = self.output_dir / f"{subtitle_file.stem}_processed.txt"
                output_file.write_text(normalized_content, encoding='utf-8')
                print(f"    ✅ Kaydedildi: {output_file.name}")
                
                results['successful'] += 1
                results['total_original_chars'] += orig_chars
                results['total_cleaned_chars'] += cleaned_chars
                results['total_normalized_chars'] += norm_chars
            else:
                results['failed'] += 1
        
        # Tüm diyalogları birleştir
        if all_dialogs:
            combined_file = self.output_dir / "all_dialogs_combined.txt"
            # Diyaloglar arasına iki yeni satır (\n\n) koyarak diyalog bloklarını ayır
            combined_content = '\n\n'.join(all_dialogs)
            combined_file.write_text(combined_content, encoding='utf-8')
            
            print(f"\n📊 Sonuçlar:")
            print(f"    İşlenen dosya: {results['total_files']}")
            print(f"    Başarılı: {results['successful']}")
            print(f"    Başarısız: {results['failed']}")
            print(f"    Toplam orijinal karakter: {results['total_original_chars']:,}")
            print(f"    Toplam temizlenmiş karakter: {results['total_cleaned_chars']:,}")
            print(f"    Toplam normalize karakter: {results['total_normalized_chars']:,}")
            
            # Token tahmini için kelime sayısını kullan
            total_words = len(combined_content.split())
            print(f"    Toplam kelime: {total_words:,}")
            print(f"    Tahmini token: {int(total_words * 1.33):,}")
            print(f"    Birleşik dosya: {combined_file.name}")
            
            # Karakter azalması
            if results['total_original_chars'] > 0:
                reduction_percent = ((results['total_original_chars'] - results['total_normalized_chars']) / results['total_original_chars'] * 100)
                print(f"    Karakter azalması: {reduction_percent:.1f}%")

def main():
    """Ana fonksiyon"""
    print("🎬 Altyazı İşleme ve Normalizasyon")
    print("=" * 60)
    
    # Konfigürasyon
    input_dir = "dataset_subtitle/subtitles"  # Altyazı dosyalarının bulunduğu klasör
    output_dir = "dataset_subtitle/processed"  # İşlenmiş dosyaların kaydedileceği klasör
    
    # Processor oluştur
    processor = SubtitleProcessor(input_dir, output_dir)
    
    # İşle
    processor.process_directory()
    
    print("\n✅ İşlem tamamlandı!")

if __name__ == "__main__":
    main()