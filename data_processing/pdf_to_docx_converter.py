# -*- coding: utf-8 -*-
"""
PDF to DOCX/TXT Converter - Teknik Dokümantasyon İşleme Scripti

Bu script, PDF dosyalarını DOCX veya TXT formatına dönüştürür.
Özellikle teknik dokümantasyonlar, MEB kitapları ve bilimsel içerikler için tasarlandı.

Kullanım:
    python pdf_to_docx_converter.py --input pdf_folder --output docx_folder --format docx
    python pdf_to_docx_converter.py --input pdf_folder --output txt_folder --format txt
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional
import json

# PDF işleme kütüphaneleri
try:
    import PyPDF2
    PDF2_AVAILABLE = True
except ImportError:
    PDF2_AVAILABLE = False
    PyPDF2 = None

try:
    from pdf2docx import Converter
    PDF2DOCX_AVAILABLE = True
except ImportError:
    PDF2DOCX_AVAILABLE = False
    Converter = None

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    pdfplumber = None

# DOCX yazma
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    Document = None

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


class PDFConverter:
    """PDF dosyalarını DOCX veya TXT formatına dönüştüren sınıf."""
    
    def __init__(self, output_format: str = "docx"):
        """
        Args:
            output_format: "docx" veya "txt"
        """
        self.output_format = output_format.lower()
        if self.output_format not in ["docx", "txt"]:
            raise ValueError("output_format 'docx' veya 'txt' olmalıdır")
        
        # Kütüphane kontrolü
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Gerekli kütüphanelerin yüklü olup olmadığını kontrol eder."""
        if self.output_format == "docx":
            if not PDF2DOCX_AVAILABLE and not PDFPLUMBER_AVAILABLE:
                raise ImportError(
                    "DOCX formatı için 'pdf2docx' veya 'pdfplumber' kütüphanesi gerekli.\n"
                    "Yükleme: pip install pdf2docx pdfplumber"
                )
        else:  # txt
            if not PDFPLUMBER_AVAILABLE and not PDF2_AVAILABLE:
                raise ImportError(
                    "TXT formatı için 'pdfplumber' veya 'PyPDF2' kütüphanesi gerekli.\n"
                    "Yükleme: pip install pdfplumber PyPDF2"
                )
    
    def convert_pdf_to_docx(self, pdf_path: Path, docx_path: Path) -> bool:
        """PDF'yi DOCX formatına dönüştürür (pdf2docx kullanarak)."""
        try:
            if PDF2DOCX_AVAILABLE:
                cv = Converter(str(pdf_path))
                cv.convert(str(docx_path))
                cv.close()
                return True
            elif PDFPLUMBER_AVAILABLE and DOCX_AVAILABLE:
                # pdfplumber ile manuel dönüşüm
                doc = Document()
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages, 1):
                        text = page.extract_text()
                        if text:
                            # Paragraf olarak ekle
                            doc.add_paragraph(text)
                            # Sayfa sonu ekle (son sayfa hariç)
                            if page_num < len(pdf.pages):
                                doc.add_page_break()
                doc.save(docx_path)
                return True
            else:
                logger.error(f"DOCX dönüşümü için gerekli kütüphaneler yüklü değil: {pdf_path}")
                return False
        except Exception as e:
            logger.error(f"PDF → DOCX dönüşüm hatası ({pdf_path}): {e}")
            return False
    
    def convert_pdf_to_txt(self, pdf_path: Path, txt_path: Path) -> bool:
        """PDF'yi TXT formatına dönüştürür."""
        try:
            text_content = []
            
            if PDFPLUMBER_AVAILABLE:
                # pdfplumber (daha iyi metin çıkarımı)
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            text_content.append(text)
            elif PDF2_AVAILABLE:
                # PyPDF2 (fallback)
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text = page.extract_text()
                        if text:
                            text_content.append(text)
            else:
                logger.error(f"TXT dönüşümü için gerekli kütüphaneler yüklü değil: {pdf_path}")
                return False
            
            # TXT dosyasına yaz
            if text_content:
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write('\n\n'.join(text_content))
                return True
            else:
                logger.warning(f"PDF'den metin çıkarılamadı: {pdf_path}")
                return False
                
        except Exception as e:
            logger.error(f"PDF → TXT dönüşüm hatası ({pdf_path}): {e}")
            return False
    
    def convert_file(self, pdf_path: Path, output_path: Path) -> bool:
        """Tek bir PDF dosyasını dönüştürür."""
        if not pdf_path.exists():
            logger.error(f"PDF dosyası bulunamadı: {pdf_path}")
            return False
        
        # Çıktı dizinini oluştur
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.output_format == "docx":
            return self.convert_pdf_to_docx(pdf_path, output_path)
        else:  # txt
            return self.convert_pdf_to_txt(pdf_path, output_path)
    
    def convert_directory(self, input_dir: Path, output_dir: Path, recursive: bool = True) -> dict:
        """
        Bir dizindeki tüm PDF dosyalarını dönüştürür.
        
        Returns:
            dict: {"success": int, "failed": int, "files": List[str]}
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # PDF dosyalarını bul
        if recursive:
            pdf_files = list(input_dir.rglob("*.pdf"))
        else:
            pdf_files = list(input_dir.glob("*.pdf"))
        
        logger.info(f"Toplam {len(pdf_files)} PDF dosyası bulundu")
        
        results = {
            "success": 0,
            "failed": 0,
            "files": []
        }
        
        for pdf_path in pdf_files:
            # Çıktı dosya yolu
            relative_path = pdf_path.relative_to(input_dir)
            output_path = output_dir / relative_path.with_suffix(f".{self.output_format}")
            
            logger.info(f"Dönüştürülüyor: {pdf_path.name} → {output_path.name}")
            
            if self.convert_file(pdf_path, output_path):
                results["success"] += 1
                results["files"].append(str(output_path))
                logger.info(f"✓ Başarılı: {output_path.name}")
            else:
                results["failed"] += 1
                logger.error(f"✗ Başarısız: {pdf_path.name}")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="PDF dosyalarını DOCX veya TXT formatına dönüştürür"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="PDF dosyalarının bulunduğu dizin veya dosya yolu"
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
        default="docx",
        choices=["docx", "txt"],
        help="Çıktı formatı: 'docx' veya 'txt' (varsayılan: docx)"
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
    
    # Converter oluştur
    converter = PDFConverter(output_format=args.format)
    
    # Giriş yolu
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if input_path.is_file():
        # Tek dosya
        logger.info(f"Tek dosya dönüşümü: {input_path.name}")
        output_file = output_path / input_path.with_suffix(f".{args.format}").name
        success = converter.convert_file(input_path, output_file)
        results = {
            "success": 1 if success else 0,
            "failed": 0 if success else 1,
            "files": [str(output_file)] if success else []
        }
    else:
        # Dizin
        logger.info(f"Dizin dönüşümü: {input_path} → {output_path}")
        results = converter.convert_directory(input_path, output_path, recursive=args.recursive)
    
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

