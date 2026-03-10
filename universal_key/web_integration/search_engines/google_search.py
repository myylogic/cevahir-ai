# -*- coding: utf-8 -*-
"""
Google Search
=============

Google arama motoru entegrasyonu.
Custom Search API ve scraping desteği.
"""

import aiohttp
import asyncio
from typing import Dict, Any, List, Optional
import logging
import json
import time
from urllib.parse import quote_plus
from bs4 import BeautifulSoup

class GoogleSearch:
    """
    Google arama motoru interface'i.
    
    Özellikler:
    - Custom Search API (API key gerekli)
    - HTML scraping fallback
    - Rate limiting
    - Error handling
    """
    
    def __init__(self, api_key: Optional[str] = None, search_engine_id: Optional[str] = None):
        self.logger = logging.getLogger("GoogleSearch")
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_initialized = False
        
        # API credentials
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        
        # URLs
        self.custom_search_url = "https://www.googleapis.com/customsearch/v1"
        self.html_search_url = "https://www.google.com/search"
        
        # Rate limiting
        self.last_request_time = 0
        self.min_delay = 2.0  # Google daha katı
        
    async def initialize(self, config=None) -> bool:
        """Google Search'ü başlat"""
        try:
            self.logger.info("🔍 Google Search başlatılıyor...")
            
            # HTTP session oluştur
            connector = aiohttp.TCPConnector(enable_cleanup_closed=True)
            timeout = aiohttp.ClientTimeout(total=30)
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
            )
            
            self.is_initialized = True
            
            if self.api_key and self.search_engine_id:
                self.logger.info("✅ Google Search (API Mode) başarıyla başlatıldı")
            else:
                self.logger.info("✅ Google Search (Scraping Mode) başarıyla başlatıldı")
                self.logger.warning("⚠️ API credentials yok, scraping modunda çalışıyor")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Google Search başlatma hatası: {e}")
            return False
    
    async def execute(self, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Google search komutunu çalıştır"""
        if command == "search":
            query = parameters.get("query", "")
            max_results = parameters.get("max_results", 10)
            search_type = parameters.get("search_type", "auto")  # "api", "html", "auto"
            
            if not query:
                return {"error": "Query parametresi gerekli"}
            
            # Search type belirleme
            if search_type == "auto":
                if self.api_key and self.search_engine_id:
                    return await self.api_search(query, max_results)
                else:
                    return await self.html_search(query, max_results)
            elif search_type == "api":
                return await self.api_search(query, max_results)
            else:
                return await self.html_search(query, max_results)
        
        else:
            return {"error": f"Desteklenmeyen komut: {command}"}
    
    async def api_search(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Google Custom Search API kullanarak arama"""
        if not self.api_key or not self.search_engine_id:
            return {"error": "Google API credentials eksik"}
        
        if not self.is_initialized or not self.session:
            return {"error": "Google Search başlatılmamış"}
        
        try:
            await self._rate_limit()
            
            # API parametreleri
            params = {
                "key": self.api_key,
                "cx": self.search_engine_id,
                "q": query,
                "num": min(max_results, 10),  # Google API max 10
                "fields": "items(title,link,snippet,displayLink)"
            }
            
            start_time = time.time()
            async with self.session.get(self.custom_search_url, params=params) as response:
                if response.status != 200:
                    return {"error": f"Google API hatası: {response.status}"}
                
                data = await response.json()
                response_time = time.time() - start_time
                
                # Sonuçları işle
                results = self._process_api_results(data.get("items", []))
                
                result = {
                    "query": query,
                    "search_engine": "google_api",
                    "response_time": response_time,
                    "results": results,
                    "total_results": len(results),
                    "timestamp": time.time()
                }
                
                self.logger.info(f"🔍 Google API arama tamamlandı: '{query}' ({len(results)} sonuç)")
                return result
                
        except Exception as e:
            self.logger.error(f"Google API arama hatası: {e}")
            return {"error": str(e)}
    
    async def html_search(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Google HTML scraping ile arama"""
        if not self.is_initialized or not self.session:
            return {"error": "Google Search başlatılmamış"}
        
        try:
            await self._rate_limit()
            
            # HTML search parametreleri
            params = {
                "q": query,
                "num": max_results,
                "hl": "tr",  # Türkçe
                "gl": "tr"   # Türkiye
            }
            
            start_time = time.time()
            async with self.session.get(self.html_search_url, params=params) as response:
                if response.status != 200:
                    return {"error": f"Google HTML arama hatası: {response.status}"}
                
                html_content = await response.text()
                response_time = time.time() - start_time
                
                # HTML'i parse et
                soup = BeautifulSoup(html_content, 'html.parser')
                results = self._parse_html_results(soup, max_results)
                
                result = {
                    "query": query,
                    "search_engine": "google_html",
                    "response_time": response_time,
                    "results": results,
                    "total_results": len(results),
                    "timestamp": time.time()
                }
                
                self.logger.info(f"🔍 Google HTML arama tamamlandı: '{query}' ({len(results)} sonuç)")
                return result
                
        except Exception as e:
            self.logger.error(f"Google HTML arama hatası: {e}")
            return {"error": str(e)}
    
    def _process_api_results(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Google API sonuçlarını işle"""
        results = []
        
        for i, item in enumerate(items):
            results.append({
                "type": "web_result",
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "domain": item.get("displayLink", ""),
                "position": i + 1,
                "relevance_score": 1.0 - (i * 0.05)
            })
        
        return results
    
    def _parse_html_results(self, soup: BeautifulSoup, max_results: int) -> List[Dict[str, Any]]:
        """Google HTML sonuçlarını parse et"""
        results = []
        
        # Google result selectors (güncel selectors gerekebilir)
        result_divs = soup.find_all("div", class_="g")
        
        for i, div in enumerate(result_divs[:max_results]):
            try:
                # Title ve URL
                title_link = div.find("h3")
                if not title_link:
                    continue
                
                parent_link = title_link.find_parent("a")
                if not parent_link:
                    continue
                
                title = title_link.get_text(strip=True)
                url = parent_link.get("href", "")
                
                # Snippet
                snippet_spans = div.find_all("span")
                snippet = ""
                for span in snippet_spans:
                    text = span.get_text(strip=True)
                    if len(text) > 20:  # Uzun text'ler snippet olabilir
                        snippet = text
                        break
                
                # Domain
                cite_element = div.find("cite")
                domain = cite_element.get_text(strip=True) if cite_element else ""
                
                results.append({
                    "type": "web_result",
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "domain": domain,
                    "position": i + 1,
                    "relevance_score": 1.0 - (i * 0.05)
                })
                
            except Exception as e:
                self.logger.warning(f"Google HTML result parse hatası: {e}")
                continue
        
        return results
    
    async def _rate_limit(self):
        """Google için rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_delay:
            sleep_time = self.min_delay - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def get_status(self) -> Dict[str, Any]:
        """Google Search durumunu al"""
        return {
            "initialized": self.is_initialized,
            "session_active": self.session is not None and not self.session.closed,
            "api_available": bool(self.api_key and self.search_engine_id),
            "last_request_time": self.last_request_time,
            "rate_limit_delay": self.min_delay
        }
    
    async def shutdown(self) -> bool:
        """Google Search'ü kapat"""
        try:
            if self.session and not self.session.closed:
                await self.session.close()
            
            self.is_initialized = False
            self.logger.info("🔍 Google Search kapatıldı")
            return True
            
        except Exception as e:
            self.logger.error(f"Google Search kapatma hatası: {e}")
            return False