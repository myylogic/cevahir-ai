# -*- coding: utf-8 -*-
"""
Web Manager
==========

Web yeteneklerini yöneten ana sınıf.
SOLID prensiplerine uygun olarak tasarlanmıştır.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import logging
import asyncio
import aiohttp
import time
from urllib.parse import urljoin, urlparse
import ssl
import certifi

# Alt modüllerden import
from .scraping.scraping_manager import ScrapingManager
from .search_engines.google_search import GoogleSearch
from .search_engines.duckduckgo_search import DuckDuckGoSearch
from .search_engines.bing_search import BingSearch
from .search_engines.academic_search import AcademicSearch
from .social_media.twitter_interface import TwitterInterface
from .social_media.reddit_interface import RedditInterface
from .social_media.linkedin_interface import LinkedInInterface
from .api_integration.rest_client import RestClient
from .api_integration.graphql_client import GraphQLClient
from .api_integration.websocket_client import WebSocketClient

@dataclass
class WebConfig:
    """Web Manager yapılandırması"""
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    retry_attempts: int = 3
    rate_limit_delay: float = 1.0
    user_agent: str = "CevahirBot/1.0"
    enable_ssl_verification: bool = True
    enable_cookies: bool = True
    enable_javascript: bool = False
    max_response_size: int = 10 * 1024 * 1024  # 10MB
    allowed_domains: List[str] = None
    blocked_domains: List[str] = None
    debug_mode: bool = False

class WebCapability(ABC):
    """Tüm web yetenekleri için base interface"""
    
    @abstractmethod
    async def initialize(self, config: WebConfig) -> bool:
        """Web yeteneğini başlat"""
        pass
    
    @abstractmethod
    async def execute(self, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Web yeteneğini çalıştır"""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Yeteneğin durumunu döndür"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """Yeteneği güvenli şekilde kapat"""
        pass

class WebManager:
    """
    Web yeteneklerini koordine eden ana sınıf.
    
    Sorumluluklar:
    - Tüm web bileşenlerini yönetir
    - Rate limiting ve güvenlik kontrolü
    - Async request coordination
    - Error handling ve retry logic
    - Performance monitoring
    """
    
    def __init__(self, config: Optional[WebConfig] = None):
        self.config = config or WebConfig()
        self.logger = self._setup_logger()
        
        # Web yetenekleri
        self.capabilities: Dict[str, WebCapability] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_initialized = False
        
        # Performance tracking
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "last_request_time": 0.0
        }
        
        # Rate limiting
        self.request_timestamps: List[float] = []
        self.rate_limit_lock = asyncio.Lock()
        
    def _setup_logger(self) -> logging.Logger:
        """Logger kurulumu"""
        logger = logging.getLogger("WebManager")
        logger.setLevel(logging.DEBUG if self.config.debug_mode else logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    async def initialize(self) -> bool:
        """Web Manager'ı ve tüm yetenekleri başlat"""
        try:
            self.logger.info("🌐 Web Manager başlatılıyor...")
            
            # SSL context oluştur
            ssl_context = ssl.create_default_context(cafile=certifi.where()) if self.config.enable_ssl_verification else False
            
            # HTTP session oluştur
            connector = aiohttp.TCPConnector(
                limit=self.config.max_concurrent_requests,
                ssl=ssl_context,
                enable_cleanup_closed=True
            )
            
            timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"User-Agent": self.config.user_agent}
            )
            
            # Web yeteneklerini kaydet ve başlat
            await self._register_capabilities()
            
            self.is_initialized = True
            self.logger.info("✅ Web Manager başarıyla başlatıldı")
            return True
            
        except Exception as e:
            self.logger.error(f"Web Manager başlatma hatası: {e}")
            return False
    
    async def _register_capabilities(self):
        """Tüm web yeteneklerini kaydet"""
        try:
            # Scraping yetenekleri
            self.capabilities["scraping"] = ScrapingManager()
            
            # Arama motorları
            self.capabilities["google_search"] = GoogleSearch()
            self.capabilities["duckduckgo_search"] = DuckDuckGoSearch()
            self.capabilities["bing_search"] = BingSearch()
            self.capabilities["academic_search"] = AcademicSearch()
            
            # Sosyal medya
            self.capabilities["twitter"] = TwitterInterface()
            self.capabilities["reddit"] = RedditInterface()
            self.capabilities["linkedin"] = LinkedInInterface()
            
            # API clients
            self.capabilities["rest_client"] = RestClient()
            self.capabilities["graphql_client"] = GraphQLClient()
            self.capabilities["websocket_client"] = WebSocketClient()
            
            self.logger.info(f"📝 {len(self.capabilities)} web yeteneği kayıtlandı")
            
        except Exception as e:
            self.logger.error(f"Web yetenekleri kayıt hatası: {e}")
            raise
    
    async def search_web(self, query: str, search_engine: str = "duckduckgo", max_results: int = 10) -> Dict[str, Any]:
        """Web'de arama yap"""
        if not self.is_initialized:
            return {"error": "Web Manager başlatılmamış"}
        
        capability_name = f"{search_engine}_search"
        if capability_name not in self.capabilities:
            return {"error": f"Desteklenmeyen arama motoru: {search_engine}"}
        
        try:
            await self._rate_limit_check()
            
            start_time = time.time()
            result = await self.capabilities[capability_name].execute("search", {
                "query": query,
                "max_results": max_results
            })
            
            response_time = time.time() - start_time
            self._update_stats(True, response_time)
            
            self.logger.info(f"🔍 Arama tamamlandı: '{query}' ({response_time:.2f}s)")
            return result
            
        except Exception as e:
            self._update_stats(False, 0)
            self.logger.error(f"Arama hatası: {e}")
            return {"error": str(e)}
    
    async def _rate_limit_check(self):
        """Rate limiting kontrolü"""
        async with self.rate_limit_lock:
            current_time = time.time()
            
            # Eski timestampları temizle (son 1 dakika)
            self.request_timestamps = [
                ts for ts in self.request_timestamps 
                if current_time - ts < 60
            ]
            
            # Rate limit kontrolü (dakikada max 60 istek)
            if len(self.request_timestamps) >= 60:
                sleep_time = 60 - (current_time - self.request_timestamps[0])
                if sleep_time > 0:
                    self.logger.warning(f"Rate limit! {sleep_time:.1f}s bekleniyor...")
                    await asyncio.sleep(sleep_time)
            
            # Timestamp ekle
            self.request_timestamps.append(current_time)
            
            # Minimum delay
            if self.config.rate_limit_delay > 0:
                await asyncio.sleep(self.config.rate_limit_delay)
    
    def _is_domain_allowed(self, url: str) -> bool:
        """Domain izin kontrolü"""
        domain = urlparse(url).netloc.lower()
        
        # Blocked domains kontrolü
        if self.config.blocked_domains:
            if any(blocked in domain for blocked in self.config.blocked_domains):
                return False
        
        # Allowed domains kontrolü
        if self.config.allowed_domains:
            return any(allowed in domain for allowed in self.config.allowed_domains)
        
        # Default: tüm domainlere izin
        return True
    
    def _update_stats(self, success: bool, response_time: float):
        """İstatistikleri güncelle"""
        self.stats["total_requests"] += 1
        
        if success:
            self.stats["successful_requests"] += 1
            # Moving average hesapla
            total_successful = self.stats["successful_requests"]
            current_avg = self.stats["average_response_time"]
            self.stats["average_response_time"] = (
                (current_avg * (total_successful - 1) + response_time) / total_successful
            )
        else:
            self.stats["failed_requests"] += 1
        
        self.stats["last_request_time"] = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """Web Manager istatistiklerini al"""
        if self.stats["total_requests"] > 0:
            success_rate = (self.stats["successful_requests"] / self.stats["total_requests"]) * 100
        else:
            success_rate = 0.0
        
        return {
            **self.stats,
            "success_rate": success_rate,
            "active_capabilities": len(self.capabilities),
            "session_active": self.session is not None and not self.session.closed
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Web Manager durumunu al"""
        return {
            "initialized": self.is_initialized,
            "capabilities_count": len(self.capabilities),
            "session_active": self.session is not None and not self.session.closed,
            "stats": self.get_stats()
        }
    
    async def shutdown(self) -> bool:
        """Web Manager'ı güvenli şekilde kapat"""
        try:
            self.logger.info("🔄 Web Manager kapatılıyor...")
            
            # HTTP session kapat
            if self.session and not self.session.closed:
                await self.session.close()
                self.logger.info("🔐 HTTP session kapatıldı")
            
            self.is_initialized = False
            self.logger.info("🏁 Web Manager kapatıldı")
            return True
            
        except Exception as e:
            self.logger.error(f"Web Manager kapatma hatası: {e}")
            return False

# Factory Pattern
class WebManagerFactory:
    """Web Manager instance'ları oluşturmak için factory"""
    
    @staticmethod
    def create_safe_instance() -> WebManager:
        """Güvenli web manager (sınırlı erişim)"""
        config = WebConfig(
            max_concurrent_requests=5,
            request_timeout=15,
            rate_limit_delay=2.0,
            allowed_domains=[
                "wikipedia.org", "stackoverflow.com", "github.com",
                "arxiv.org", "scholar.google.com", "tdk.gov.tr"
            ],
            enable_javascript=False,
            debug_mode=False
        )
        return WebManager(config)
    
    @staticmethod
    def create_unrestricted_instance() -> WebManager:
        """Sınırsız web manager (tam erişim)"""
        config = WebConfig(
            max_concurrent_requests=20,
            request_timeout=30,
            rate_limit_delay=0.5,
            allowed_domains=None,  # Tüm domainlere izin
            blocked_domains=["malware-site.com", "spam-site.com"],
            enable_javascript=True,
            debug_mode=True
        )
        return WebManager(config)