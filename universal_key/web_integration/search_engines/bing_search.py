# -*- coding: utf-8 -*-
"""
Bing Search
===========

Bing arama motoru entegrasyonu.
"""

import aiohttp
import asyncio
from typing import Dict, Any, List, Optional
import logging
import time
from bs4 import BeautifulSoup

class BingSearch:
    """Bing arama motoru interface'i"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.logger = logging.getLogger("BingSearch")
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_initialized = False
        self.api_key = api_key
        
        # URLs
        self.api_url = "https://api.bing.microsoft.com/v7.0/search"
        self.html_url = "https://www.bing.com/search"
        
        # Rate limiting
        self.last_request_time = 0
        self.min_delay = 1.5
    
    async def initialize(self, config=None) -> bool:
        """Bing Search'ü başlat"""
        try:
            self.logger.info("🔍 Bing Search başlatılıyor...")
            
            connector = aiohttp.TCPConnector(enable_cleanup_closed=True)
            timeout = aiohttp.ClientTimeout(total=30)
            
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            if self.api_key:
                headers["Ocp-Apim-Subscription-Key"] = self.api_key
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=headers
            )
            
            self.is_initialized = True
            self.logger.info("✅ Bing Search başarıyla başlatıldı")
            return True
            
        except Exception as e:
            self.logger.error(f"Bing Search başlatma hatası: {e}")
            return False
    
    async def execute(self, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Bing search komutunu çalıştır"""
        if command == "search":
            query = parameters.get("query", "")
            max_results = parameters.get("max_results", 10)
            
            if not query:
                return {"error": "Query parametresi gerekli"}
            
            if self.api_key:
                return await self.api_search(query, max_results)
            else:
                return await self.html_search(query, max_results)
        
        else:
            return {"error": f"Desteklenmeyen komut: {command}"}
    
    async def api_search(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Bing API ile arama"""
        if not self.api_key:
            return {"error": "Bing API key eksik"}
        
        try:
            await self._rate_limit()
            
            params = {
                "q": query,
                "count": min(max_results, 50),
                "mkt": "tr-TR",
                "responseFilter": "Webpages"
            }
            
            start_time = time.time()
            async with self.session.get(self.api_url, params=params) as response:
                if response.status != 200:
                    return {"error": f"Bing API hatası: {response.status}"}
                
                data = await response.json()
                response_time = time.time() - start_time
                
                # Sonuçları işle
                results = self._process_api_results(data.get("webPages", {}).get("value", []))
                
                return {
                    "query": query,
                    "search_engine": "bing_api",
                    "response_time": response_time,
                    "results": results,
                    "total_results": len(results),
                    "timestamp": time.time()
                }
                
        except Exception as e:
            self.logger.error(f"Bing API arama hatası: {e}")
            return {"error": str(e)}
    
    async def html_search(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Bing HTML scraping ile arama"""
        try:
            await self._rate_limit()
            
            params = {
                "q": query,
                "count": max_results,
                "setlang": "tr"
            }
            
            start_time = time.time()
            async with self.session.get(self.html_url, params=params) as response:
                if response.status != 200:
                    return {"error": f"Bing HTML arama hatası: {response.status}"}
                
                html_content = await response.text()
                response_time = time.time() - start_time
                
                # HTML parse et
                soup = BeautifulSoup(html_content, 'html.parser')
                results = self._parse_html_results(soup, max_results)
                
                return {
                    "query": query,
                    "search_engine": "bing_html",
                    "response_time": response_time,
                    "results": results,
                    "total_results": len(results),
                    "timestamp": time.time()
                }
                
        except Exception as e:
            self.logger.error(f"Bing HTML arama hatası: {e}")
            return {"error": str(e)}
    
    def _process_api_results(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Bing API sonuçlarını işle"""
        results = []
        
        for i, item in enumerate(items):
            results.append({
                "type": "web_result",
                "title": item.get("name", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", ""),
                "domain": item.get("displayUrl", ""),
                "position": i + 1,
                "relevance_score": 1.0 - (i * 0.05)
            })
        
        return results
    
    def _parse_html_results(self, soup: BeautifulSoup, max_results: int) -> List[Dict[str, Any]]:
        """Bing HTML sonuçlarını parse et"""
        results = []
        
        # Bing result selectors
        result_items = soup.find_all("li", class_="b_algo")
        
        for i, item in enumerate(result_items[:max_results]):
            try:
                # Title ve URL
                title_link = item.find("h2").find("a") if item.find("h2") else None
                if not title_link:
                    continue
                
                title = title_link.get_text(strip=True)
                url = title_link.get("href", "")
                
                # Snippet
                snippet_div = item.find("div", class_="b_caption")
                snippet = snippet_div.get_text(strip=True) if snippet_div else ""
                
                # Domain
                cite_element = item.find("cite")
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
                self.logger.warning(f"Bing HTML result parse hatası: {e}")
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
        """Bing Search durumunu al"""
        return {
            "initialized": self.is_initialized,
            "session_active": self.session is not None and not self.session.closed,
            "api_available": bool(self.api_key),
            "last_request_time": self.last_request_time
        }
    
    async def shutdown(self) -> bool:
        """Bing Search'ü kapat"""
        try:
            if self.session and not self.session.closed:
                await self.session.close()
            
            self.is_initialized = False
            self.logger.info("🔍 Bing Search kapatıldı")
            return True
            
        except Exception as e:
            self.logger.error(f"Bing Search kapatma hatası: {e}")
            return False