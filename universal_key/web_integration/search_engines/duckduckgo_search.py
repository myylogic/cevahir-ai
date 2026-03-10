# -*- coding: utf-8 -*-
"""
DuckDuckGo Search
================

DuckDuckGo arama motoru entegrasyonu.
API-based ve scraping-based arama desteği.
"""

import aiohttp
import asyncio
from typing import Dict, Any, List, Optional
import logging
import json
import time
from urllib.parse import quote_plus, urljoin
from bs4 import BeautifulSoup

class DuckDuckGoSearch:
    """
    DuckDuckGo arama motoru interface'i.
    
    Özellikler:
    - Instant Answer API
    - HTML scraping fallback
    - Rate limiting
    - Error handling
    """
    
    def __init__(self):
        self.logger = logging.getLogger("DuckDuckGoSearch")
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_initialized = False
        
        # API endpoints
        self.instant_api_url = "https://api.duckduckgo.com/"
        self.search_url = "https://duckduckgo.com/html/"
        
        # Rate limiting
        self.last_request_time = 0
        self.min_delay = 1.0
        
    async def initialize(self, config=None) -> bool:
        """DuckDuckGo Search'ü başlat"""
        try:
            self.logger.info("🦆 DuckDuckGo Search başlatılıyor...")
            
            # HTTP session oluştur
            connector = aiohttp.TCPConnector(enable_cleanup_closed=True)
            timeout = aiohttp.ClientTimeout(total=30)
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
            )
            
            self.is_initialized = True
            self.logger.info("✅ DuckDuckGo Search başarıyla başlatıldı")
            return True
            
        except Exception as e:
            self.logger.error(f"DuckDuckGo Search başlatma hatası: {e}")
            return False
    
    async def execute(self, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """DuckDuckGo search komutunu çalıştır"""
        if command == "search":
            query = parameters.get("query", "")
            max_results = parameters.get("max_results", 10)
            search_type = parameters.get("search_type", "instant")  # "instant" or "html"
            
            if not query:
                return {"error": "Query parametresi gerekli"}
            
            if search_type == "instant":
                return await self.instant_search(query)
            else:
                return await self.html_search(query, max_results)
        
        else:
            return {"error": f"Desteklenmeyen komut: {command}"}
    
    async def instant_search(self, query: str) -> Dict[str, Any]:
        """DuckDuckGo Instant Answer API kullanarak arama"""
        if not self.is_initialized or not self.session:
            return {"error": "DuckDuckGo Search başlatılmamış"}
        
        try:
            await self._rate_limit()
            
            # API parametreleri
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1"
            }
            
            start_time = time.time()
            async with self.session.get(self.instant_api_url, params=params) as response:
                if response.status != 200:
                    return {"error": f"API hatası: {response.status}"}
                
                data = await response.json()
                response_time = time.time() - start_time
                
                # Sonuçları işle
                processed_results = self._process_instant_results(data, query)
                
                result = {
                    "query": query,
                    "search_engine": "duckduckgo_instant",
                    "response_time": response_time,
                    "results": processed_results,
                    "total_results": len(processed_results),
                    "timestamp": time.time()
                }
                
                self.logger.info(f"🔍 DuckDuckGo Instant arama tamamlandı: '{query}' ({len(processed_results)} sonuç)")
                return result
                
        except Exception as e:
            self.logger.error(f"DuckDuckGo Instant arama hatası: {e}")
            return {"error": str(e)}
    
    async def html_search(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """DuckDuckGo HTML scraping ile arama"""
        if not self.is_initialized or not self.session:
            return {"error": "DuckDuckGo Search başlatılmamış"}
        
        try:
            await self._rate_limit()
            
            # HTML search parametreleri
            params = {
                "q": query,
                "s": "0",  # Start index
                "dc": str(max_results),
                "v": "l",  # Layout
                "o": "json",
                "api": "d.js"
            }
            
            start_time = time.time()
            async with self.session.get(self.search_url, params=params) as response:
                if response.status != 200:
                    return {"error": f"HTML arama hatası: {response.status}"}
                
                html_content = await response.text()
                response_time = time.time() - start_time
                
                # HTML'i parse et
                soup = BeautifulSoup(html_content, 'html.parser')
                results = self._parse_html_results(soup, max_results)
                
                result = {
                    "query": query,
                    "search_engine": "duckduckgo_html",
                    "response_time": response_time,
                    "results": results,
                    "total_results": len(results),
                    "timestamp": time.time()
                }
                
                self.logger.info(f"🔍 DuckDuckGo HTML arama tamamlandı: '{query}' ({len(results)} sonuç)")
                return result
                
        except Exception as e:
            self.logger.error(f"DuckDuckGo HTML arama hatası: {e}")
            return {"error": str(e)}
    
    def _process_instant_results(self, data: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
        """Instant API sonuçlarını işle"""
        results = []
        
        # Abstract (kısa cevap)
        if data.get("Abstract"):
            results.append({
                "type": "abstract",
                "title": data.get("AbstractText", ""),
                "content": data.get("Abstract", ""),
                "source": data.get("AbstractSource", ""),
                "url": data.get("AbstractURL", ""),
                "relevance_score": 0.9
            })
        
        # Definition (tanım)
        if data.get("Definition"):
            results.append({
                "type": "definition",
                "title": f"{query} tanımı",
                "content": data.get("Definition", ""),
                "source": data.get("DefinitionSource", ""),
                "url": data.get("DefinitionURL", ""),
                "relevance_score": 0.95
            })
        
        # Related topics
        if data.get("RelatedTopics"):
            for i, topic in enumerate(data["RelatedTopics"][:5]):  # Max 5 related
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append({
                        "type": "related_topic",
                        "title": topic.get("Text", "")[:100] + "...",
                        "content": topic.get("Text", ""),
                        "url": topic.get("FirstURL", ""),
                        "relevance_score": 0.7 - (i * 0.1)
                    })
        
        # Answer (direkt cevap)
        if data.get("Answer"):
            results.append({
                "type": "answer",
                "title": f"{query} cevabı",
                "content": data.get("Answer", ""),
                "source": data.get("AnswerType", ""),
                "relevance_score": 1.0
            })
        
        return sorted(results, key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    def _parse_html_results(self, soup: BeautifulSoup, max_results: int) -> List[Dict[str, Any]]:
        """HTML arama sonuçlarını parse et"""
        results = []
        
        # DuckDuckGo HTML result selectors
        result_divs = soup.find_all("div", class_="result")
        
        for i, div in enumerate(result_divs[:max_results]):
            try:
                # Title ve URL
                title_link = div.find("a", class_="result__a")
                if not title_link:
                    continue
                
                title = title_link.get_text(strip=True)
                url = title_link.get("href", "")
                
                # Snippet (özet)
                snippet_div = div.find("div", class_="result__snippet")
                snippet = snippet_div.get_text(strip=True) if snippet_div else ""
                
                # Domain
                domain_span = div.find("span", class_="result__url")
                domain = domain_span.get_text(strip=True) if domain_span else ""
                
                results.append({
                    "type": "web_result",
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "domain": domain,
                    "position": i + 1,
                    "relevance_score": 1.0 - (i * 0.05)  # Position-based scoring
                })
                
            except Exception as e:
                self.logger.warning(f"HTML result parse hatası: {e}")
                continue
        
        return results
    
    async def _rate_limit(self):
        """Rate limiting uygula"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_delay:
            sleep_time = self.min_delay - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def get_status(self) -> Dict[str, Any]:
        """DuckDuckGo Search durumunu al"""
        return {
            "initialized": self.is_initialized,
            "session_active": self.session is not None and not self.session.closed,
            "last_request_time": self.last_request_time,
            "endpoints": {
                "instant_api": self.instant_api_url,
                "html_search": self.search_url
            }
        }
    
    async def shutdown(self) -> bool:
        """DuckDuckGo Search'ü kapat"""
        try:
            if self.session and not self.session.closed:
                await self.session.close()
            
            self.is_initialized = False
            self.logger.info("🦆 DuckDuckGo Search kapatıldı")
            return True
            
        except Exception as e:
            self.logger.error(f"DuckDuckGo Search kapatma hatası: {e}")
            return False