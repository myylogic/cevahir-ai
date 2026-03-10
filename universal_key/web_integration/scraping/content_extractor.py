# -*- coding: utf-8 -*-
"""
Content Extractor
=================

Web sayfalarından içerik çıkaran sınıf.
Farklı content tiplerini destekler.
"""

from typing import Dict, Any, List, Optional
import logging
from bs4 import BeautifulSoup
import re
import json

class ContentExtractor:
    """
    Web sayfalarından içerik çıkaran sınıf.
    
    Desteklenen extract türleri:
    - text: Sadece metin
    - structured: Başlık, paragraf vb. yapısal
    - metadata: Meta bilgiler
    - links: Tüm linkler
    - images: Resim URL'leri
    - full: Tüm bilgiler
    """
    
    def __init__(self):
        self.logger = logging.getLogger("ContentExtractor")
        self.is_initialized = False
        
        # Metin temizleme regex'leri
        self.cleanup_patterns = {
            "extra_whitespace": re.compile(r'\s+'),
            "empty_lines": re.compile(r'\n\s*\n'),
            "script_style": re.compile(r'<(script|style)[^>]*>.*?</\1>', re.DOTALL | re.IGNORECASE),
            "html_comments": re.compile(r'<!--.*?-->', re.DOTALL)
        }
    
    async def initialize(self) -> bool:
        """Content Extractor'ı başlat"""
        try:
            self.logger.info("📄 Content Extractor başlatılıyor...")
            self.is_initialized = True
            self.logger.info("✅ Content Extractor başarıyla başlatıldı")
            return True
        except Exception as e:
            self.logger.error(f"Content Extractor başlatma hatası: {e}")
            return False
    
    async def extract(self, html_content: str, extract_type: str, source_url: str = "") -> Dict[str, Any]:
        """HTML içeriğinden bilgi çıkar"""
        if not self.is_initialized:
            return {"error": "Content Extractor başlatılmamış"}
        
        try:
            # HTML parse et
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract type'a göre işle
            if extract_type == "text":
                return await self._extract_text(soup, source_url)
            elif extract_type == "structured":
                return await self._extract_structured(soup, source_url)
            elif extract_type == "metadata":
                return await self._extract_metadata(soup, source_url)
            elif extract_type == "links":
                return await self._extract_links(soup, source_url)
            elif extract_type == "images":
                return await self._extract_images(soup, source_url)
            elif extract_type == "full":
                return await self._extract_full(soup, source_url)
            else:
                return {"error": f"Desteklenmeyen extract türü: {extract_type}"}
                
        except Exception as e:
            self.logger.error(f"Content extraction hatası: {e}")
            return {"error": str(e)}
    
    async def _extract_text(self, soup: BeautifulSoup, source_url: str) -> Dict[str, Any]:
        """Sadece temiz metin çıkar"""
        try:
            # Script ve style taglerini kaldır
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            
            # Metin al
            text = soup.get_text()
            
            # Temizle
            cleaned_text = self._clean_text(text)
            
            return {
                "type": "text",
                "content": cleaned_text,
                "word_count": len(cleaned_text.split()),
                "char_count": len(cleaned_text),
                "source_url": source_url,
                "extraction_type": "text_only"
            }
            
        except Exception as e:
            return {"error": f"Text extraction hatası: {e}"}
    
    async def _extract_structured(self, soup: BeautifulSoup, source_url: str) -> Dict[str, Any]:
        """Yapısal içerik çıkar"""
        try:
            result = {
                "type": "structured",
                "source_url": source_url,
                "extraction_type": "structured"
            }
            
            # Title
            title_tag = soup.find("title")
            result["title"] = title_tag.get_text(strip=True) if title_tag else ""
            
            # Headings
            headings = []
            for level in range(1, 7):  # h1-h6
                for heading in soup.find_all(f"h{level}"):
                    headings.append({
                        "level": level,
                        "text": heading.get_text(strip=True),
                        "tag": f"h{level}"
                    })
            result["headings"] = headings
            
            # Paragraphs
            paragraphs = []
            for p in soup.find_all("p"):
                text = p.get_text(strip=True)
                if len(text) > 20:  # Anlamlı paragraflar
                    paragraphs.append(text)
            result["paragraphs"] = paragraphs
            
            # Lists
            lists = []
            for ul in soup.find_all(["ul", "ol"]):
                list_items = [li.get_text(strip=True) for li in ul.find_all("li")]
                if list_items:
                    lists.append({
                        "type": ul.name,
                        "items": list_items
                    })
            result["lists"] = lists
            
            return result
            
        except Exception as e:
            return {"error": f"Structured extraction hatası: {e}"}
    
    async def _extract_metadata(self, soup: BeautifulSoup, source_url: str) -> Dict[str, Any]:
        """Meta bilgiler çıkar"""
        try:
            metadata = {
                "type": "metadata",
                "source_url": source_url,
                "extraction_type": "metadata_only"
            }
            
            # Meta tags
            meta_tags = {}
            for meta in soup.find_all("meta"):
                name = meta.get("name") or meta.get("property") or meta.get("http-equiv")
                content = meta.get("content")
                if name and content:
                    meta_tags[name] = content
            metadata["meta_tags"] = meta_tags
            
            # Title
            title_tag = soup.find("title")
            metadata["title"] = title_tag.get_text(strip=True) if title_tag else ""
            
            # Language
            html_tag = soup.find("html")
            metadata["language"] = html_tag.get("lang", "") if html_tag else ""
            
            # Canonical URL
            canonical = soup.find("link", rel="canonical")
            metadata["canonical_url"] = canonical.get("href", "") if canonical else ""
            
            return metadata
            
        except Exception as e:
            return {"error": f"Metadata extraction hatası: {e}"}
    
    async def _extract_links(self, soup: BeautifulSoup, source_url: str) -> Dict[str, Any]:
        """Tüm linkleri çıkar"""
        try:
            links = []
            
            for a in soup.find_all("a", href=True):
                href = a.get("href", "")
                text = a.get_text(strip=True)
                
                if href and not href.startswith("#"):  # Fragment linklerini atla
                    links.append({
                        "url": href,
                        "text": text,
                        "title": a.get("title", "")
                    })
            
            return {
                "type": "links",
                "source_url": source_url,
                "extraction_type": "links_only",
                "links": links,
                "total_links": len(links)
            }
            
        except Exception as e:
            return {"error": f"Links extraction hatası: {e}"}
    
    async def _extract_images(self, soup: BeautifulSoup, source_url: str) -> Dict[str, Any]:
        """Resim URL'lerini çıkar"""
        try:
            images = []
            
            for img in soup.find_all("img"):
                src = img.get("src", "")
                alt = img.get("alt", "")
                
                if src:
                    images.append({
                        "url": src,
                        "alt_text": alt,
                        "title": img.get("title", "")
                    })
            
            return {
                "type": "images",
                "source_url": source_url,
                "extraction_type": "images_only",
                "images": images,
                "total_images": len(images)
            }
            
        except Exception as e:
            return {"error": f"Images extraction hatası: {e}"}
    
    async def _extract_full(self, soup: BeautifulSoup, source_url: str) -> Dict[str, Any]:
        """Tüm bilgileri çıkar"""
        try:
            # Tüm extraction türlerini birleştir
            text_data = await self._extract_text(soup, source_url)
            structured_data = await self._extract_structured(soup, source_url)
            metadata = await self._extract_metadata(soup, source_url)
            links_data = await self._extract_links(soup, source_url)
            images_data = await self._extract_images(soup, source_url)
            
            return {
                "type": "full_extraction",
                "source_url": source_url,
                "extraction_type": "complete",
                "text": text_data.get("content", ""),
                "structured": structured_data,
                "metadata": metadata,
                "links": links_data.get("links", []),
                "images": images_data.get("images", []),
                "summary": {
                    "word_count": text_data.get("word_count", 0),
                    "headings_count": len(structured_data.get("headings", [])),
                    "paragraphs_count": len(structured_data.get("paragraphs", [])),
                    "links_count": len(links_data.get("links", [])),
                    "images_count": len(images_data.get("images", []))
                }
            }
            
        except Exception as e:
            return {"error": f"Full extraction hatası: {e}"}
    
    def _clean_text(self, text: str) -> str:
        """Metni temizle"""
        # Extra whitespace'leri temizle
        text = self.cleanup_patterns["extra_whitespace"].sub(" ", text)
        
        # Empty lines temizle
        text = self.cleanup_patterns["empty_lines"].sub("\n", text)
        
        # Başındaki ve sonundaki boşlukları temizle
        text = text.strip()
        
        return text
    
    def get_status(self) -> Dict[str, Any]:
        """Content Extractor durumunu al"""
        return {
            "initialized": self.is_initialized,
            "supported_types": ["text", "structured", "metadata", "links", "images", "full"]
        }
    
    async def shutdown(self) -> bool:
        """Content Extractor'ı kapat"""
        try:
            self.is_initialized = False
            self.logger.info("📄 Content Extractor kapatıldı")
            return True
        except Exception as e:
            self.logger.error(f"Content Extractor kapatma hatası: {e}")
            return False