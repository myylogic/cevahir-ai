# -*- coding: utf-8 -*-
"""
Academic Search
===============

Akademik arama motorları entegrasyonu.
arXiv, Google Scholar, PubMed vb.
"""

import aiohttp
import asyncio
from typing import Dict, Any, List, Optional
import logging
import time

class AcademicSearch:
    """Akademik arama motorları interface'i"""
    
    def __init__(self):
        self.logger = logging.getLogger("AcademicSearch")
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_initialized = False
        
        # API endpoints
        self.arxiv_api = "http://export.arxiv.org/api/query"
        self.pubmed_api = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        
        # Rate limiting
        self.last_request_times = {}
        self.min_delays = {
            "arxiv": 3.0,
            "pubmed": 0.34,
            "scholar": 5.0
        }
    
    async def initialize(self, config=None) -> bool:
        """Academic Search'ü başlat"""
        try:
            self.logger.info("🎓 Academic Search başlatılıyor...")
            
            connector = aiohttp.TCPConnector(enable_cleanup_closed=True)
            timeout = aiohttp.ClientTimeout(total=45)
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"User-Agent": "CevahirBot/1.0 (Academic Research)"}
            )
            
            self.is_initialized = True
            self.logger.info("✅ Academic Search başarıyla başlatıldı")
            return True
            
        except Exception as e:
            self.logger.error(f"Academic Search başlatma hatası: {e}")
            return False
    
    async def execute(self, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Academic search komutunu çalıştır"""
        if command == "search":
            query = parameters.get("query", "")
            source = parameters.get("source", "arxiv")
            max_results = parameters.get("max_results", 10)
            
            if not query:
                return {"error": "Query parametresi gerekli"}
            
            if source == "arxiv":
                return await self.search_arxiv(query, max_results)
            elif source == "pubmed":
                return await self.search_pubmed(query, max_results)
            else:
                return {"error": f"Desteklenmeyen akademik kaynak: {source}"}
        
        else:
            return {"error": f"Desteklenmeyen komut: {command}"}
    
    async def search_arxiv(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """arXiv API ile arama"""
        try:
            await self._rate_limit("arxiv")
            
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance"
            }
            
            start_time = time.time()
            async with self.session.get(self.arxiv_api, params=params) as response:
                if response.status != 200:
                    return {"error": f"arXiv API hatası: {response.status}"}
                
                xml_content = await response.text()
                response_time = time.time() - start_time
                
                # Basit XML parse (gerçek implementasyon için xml.etree gerekli)
                results = [{"type": "academic_paper", "source": "arXiv", "query": query}]
                
                return {
                    "query": query,
                    "search_engine": "arxiv",
                    "response_time": response_time,
                    "results": results,
                    "total_results": len(results),
                    "timestamp": time.time()
                }
                
        except Exception as e:
            self.logger.error(f"arXiv arama hatası: {e}")
            return {"error": str(e)}
    
    async def search_pubmed(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """PubMed API ile arama"""
        try:
            await self._rate_limit("pubmed")
            
            params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json"
            }
            
            start_time = time.time()
            async with self.session.get(self.pubmed_api, params=params) as response:
                if response.status != 200:
                    return {"error": f"PubMed API hatası: {response.status}"}
                
                data = await response.json()
                response_time = time.time() - start_time
                
                # Sonuçları işle
                results = self._process_pubmed_results(data)
                
                return {
                    "query": query,
                    "search_engine": "pubmed",
                    "response_time": response_time,
                    "results": results,
                    "total_results": len(results),
                    "timestamp": time.time()
                }
                
        except Exception as e:
            self.logger.error(f"PubMed arama hatası: {e}")
            return {"error": str(e)}
    
    def _process_pubmed_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """PubMed sonuçlarını işle"""
        results = []
        
        id_list = data.get("esearchresult", {}).get("idlist", [])
        
        for i, pmid in enumerate(id_list):
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            
            results.append({
                "type": "academic_paper",
                "title": f"PubMed Article {pmid}",
                "url": url,
                "pmid": pmid,
                "source": "PubMed",
                "position": i + 1,
                "relevance_score": 1.0 - (i * 0.05)
            })
        
        return results
    
    async def _rate_limit(self, source: str):
        """Kaynak bazlı rate limiting"""
        current_time = time.time()
        last_time = self.last_request_times.get(source, 0)
        min_delay = self.min_delays.get(source, 1.0)
        
        time_since_last = current_time - last_time
        
        if time_since_last < min_delay:
            sleep_time = min_delay - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_request_times[source] = time.time()
    
    def get_status(self) -> Dict[str, Any]:
        """Academic Search durumunu al"""
        return {
            "initialized": self.is_initialized,
            "session_active": self.session is not None and not self.session.closed,
            "supported_sources": ["arxiv", "pubmed"],
            "last_request_times": dict(self.last_request_times)
        }
    
    async def shutdown(self) -> bool:
        """Academic Search'ü kapat"""
        try:
            if self.session and not self.session.closed:
                await self.session.close()
            
            self.is_initialized = False
            self.logger.info("🎓 Academic Search kapatıldı")
            return True
            
        except Exception as e:
            self.logger.error(f"Academic Search kapatma hatası: {e}")
            return False