# -*- coding: utf-8 -*-
"""
Scraping Manager
===============

Web scraping işlemlerini yöneten ana sınıf.
Güvenli, hızlı ve tespit edilemeyen scraping yetenekleri sağlar.
"""

import asyncio
import aiohttp
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import logging
import time
import random
from urllib.parse import urljoin, urlparse, parse_qs
from bs4 import BeautifulSoup
import json
import re

from .content_extractor import ContentExtractor
from .data_validator import DataValidator
from .anti_detection import AntiDetection

@dataclass
class ScrapingConfig:
    """Scraping yapılandırması"""
    max_pages_per_domain: int = 100
    delay_between_requests: float = 1.0
    random_delay_range: tuple = (0.5, 2.0)
    max_retries: int = 3
    timeout: int = 30
    respect_robots_txt: bool = True
    use_proxy_rotation: bool = False
    enable_javascript: bool = False
    max_content_size: int = 5 * 1024 * 1024  # 5MB
    extract_images: bool = False
    extract_links: bool = True
    extract_metadata: bool = True

class ScrapingManager:
    """
    Web scraping işlemlerini yöneten ana sınıf.
    
    Özellikler:
    - Anti-detection teknikleri
    - Rate limiting ve politeness
    - Content extraction ve validation
    - Async/await desteği
    - Error handling ve retry logic
    - Robots.txt uyumluluğu
    """
    
    def __init__(self, config: Optional[ScrapingConfig] = None):
        self.config = config or ScrapingConfig()
        self.logger = self._setup_logger()
        
        # Alt bileşenler
        self.content_extractor = ContentExtractor()
        self.data_validator = DataValidator()
        self.anti_detection = AntiDetection()
        
        # Session ve state
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_initialized = False
        
        # Domain tracking (politeness için)
        self.domain_stats: Dict[str, Dict[str, Any]] = {}
        self.robots_cache: Dict[str, Dict[str, Any]] = {}
        
    def _setup_logger(self) -> logging.Logger:
        """Logger kurulumu"""
        logger = logging.getLogger("ScrapingManager")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    async def initialize(self, web_config=None) -> bool:
        """Scraping Manager'ı başlat"""
        try:
            self.logger.info("🕷️ Scraping Manager başlatılıyor...")
            
            # HTTP session oluştur
            connector = aiohttp.TCPConnector(
                limit=20,
                enable_cleanup_closed=True,
                ssl=False  # Bazı siteler için gerekli olabilir
            )
            
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"User-Agent": "CevahirBot/1.0"}
            )
            
            # Alt bileşenleri başlat
            await self.content_extractor.initialize()
            await self.data_validator.initialize()
            await self.anti_detection.initialize()
            
            self.is_initialized = True
            self.logger.info("✅ Scraping Manager başarıyla başlatıldı")
            return True
            
        except Exception as e:
            self.logger.error(f"Scraping Manager başlatma hatası: {e}")
            return False
    
    async def scrape_url(self, url: str, extract_type: str = "text") -> Dict[str, Any]:
        """Tek URL'yi scrape et"""
        if not self.is_initialized:
            return {"error": "Scraping Manager başlatılmamış"}
        
        try:
            # URL validation
            if not self._is_valid_url(url):
                return {"error": "Geçersiz URL"}
            
            domain = urlparse(url).netloc
            
            # Robots.txt kontrolü
            if self.config.respect_robots_txt:
                if not await self._check_robots_txt(url):
                    return {"error": "Robots.txt tarafından engellendi"}
            
            # Domain rate limiting
            await self._domain_rate_limit(domain)
            
            # Anti-detection hazırlığı
            headers = await self.anti_detection.get_random_headers()
            
            # HTTP isteği
            start_time = time.time()
            async with self.session.get(url, headers=headers) as response:
                
                # Response kontrolü
                if response.status != 200:
                    return {"error": f"HTTP {response.status}: {response.reason}"}
                
                # Content size kontrolü
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > self.config.max_content_size:
                    return {"error": "İçerik çok büyük"}
                
                # Content oku
                content = await response.text()
                response_time = time.time() - start_time
                
                # İçerik çıkarma
                extracted_data = await self.content_extractor.extract(
                    content, extract_type, url
                )
                
                # Veri doğrulama
                validated_data = await self.data_validator.validate(extracted_data)
                
                # Domain istatistiklerini güncelle
                self._update_domain_stats(domain, True, response_time)
                
                result = {
                    "url": url,
                    "status": "success",
                    "response_time": response_time,
                    "content_length": len(content),
                    "extracted_data": validated_data,
                    "timestamp": time.time()
                }
                
                self.logger.info(f"✅ Scraping başarılı: {url} ({response_time:.2f}s)")
                return result
                
        except Exception as e:
            self._update_domain_stats(domain, False, 0)
            self.logger.error(f"Scraping hatası ({url}): {e}")
            return {"error": str(e), "url": url}
    
    def _is_valid_url(self, url: str) -> bool:
        """URL geçerliliğini kontrol et"""
        try:
            parsed = urlparse(url)
            return bool(parsed.scheme and parsed.netloc)
        except:
            return False
    
    async def _check_robots_txt(self, url: str) -> bool:
        """Robots.txt kontrolü"""
        try:
            domain = urlparse(url).netloc
            
            # Cache'den kontrol et
            if domain in self.robots_cache:
                robots_data = self.robots_cache[domain]
                # Cache 1 saat geçerli
                if time.time() - robots_data["timestamp"] < 3600:
                    return robots_data["allowed"]
            
            # Robots.txt'yi al
            robots_url = f"http://{domain}/robots.txt"
            try:
                async with self.session.get(robots_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        robots_content = await response.text()
                        allowed = self._parse_robots_txt(robots_content, url)
                    else:
                        allowed = True  # robots.txt yoksa izin ver
            except:
                allowed = True  # Hata durumunda izin ver
            
            # Cache'e kaydet
            self.robots_cache[domain] = {
                "allowed": allowed,
                "timestamp": time.time()
            }
            
            return allowed
            
        except Exception as e:
            self.logger.warning(f"Robots.txt kontrolü başarısız: {e}")
            return True  # Hata durumunda izin ver
    
    def _parse_robots_txt(self, robots_content: str, url: str) -> bool:
        """Robots.txt içeriğini parse et"""
        # Basit robots.txt parser
        user_agent_section = False
        
        for line in robots_content.split('\n'):
            line = line.strip().lower()
            
            if line.startswith('user-agent:'):
                ua = line.split(':', 1)[1].strip()
                user_agent_section = ua == '*' or 'cevahir' in ua
            
            elif user_agent_section and line.startswith('disallow:'):
                disallowed_path = line.split(':', 1)[1].strip()
                if disallowed_path and urlparse(url).path.startswith(disallowed_path):
                    return False
        
        return True
    
    async def _domain_rate_limit(self, domain: str):
        """Domain bazlı rate limiting"""
        current_time = time.time()
        
        if domain not in self.domain_stats:
            self.domain_stats[domain] = {
                "last_request": 0,
                "request_count": 0,
                "total_requests": 0
            }
        
        stats = self.domain_stats[domain]
        
        # Son istekten bu yana geçen süre
        time_since_last = current_time - stats["last_request"]
        
        # Minimum delay kontrolü
        min_delay = self.config.delay_between_requests
        if time_since_last < min_delay:
            sleep_time = min_delay - time_since_last
            
            # Random delay ekle (daha doğal görünmek için)
            if self.config.random_delay_range:
                random_delay = random.uniform(*self.config.random_delay_range)
                sleep_time += random_delay
            
            await asyncio.sleep(sleep_time)
        
        # İstatistikleri güncelle
        stats["last_request"] = time.time()
        stats["request_count"] += 1
        stats["total_requests"] += 1
    
    def _update_domain_stats(self, domain: str, success: bool, response_time: float):
        """Domain istatistiklerini güncelle"""
        if domain not in self.domain_stats:
            self.domain_stats[domain] = {
                "last_request": 0,
                "request_count": 0,
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "average_response_time": 0.0
            }
        
        stats = self.domain_stats[domain]
        
        if success:
            stats["successful_requests"] += 1
            # Moving average hesapla
            successful_count = stats["successful_requests"]
            current_avg = stats["average_response_time"]
            stats["average_response_time"] = (
                (current_avg * (successful_count - 1) + response_time) / successful_count
            )
        else:
            stats["failed_requests"] += 1
    
    def get_status(self) -> Dict[str, Any]:
        """Scraping Manager durumunu al"""
        total_domains = len(self.domain_stats)
        total_requests = sum(stats.get("total_requests", 0) for stats in self.domain_stats.values())
        successful_requests = sum(stats.get("successful_requests", 0) for stats in self.domain_stats.values())
        
        success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "initialized": self.is_initialized,
            "session_active": self.session is not None and not self.session.closed,
            "total_domains_accessed": total_domains,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "success_rate": success_rate,
            "robots_cache_size": len(self.robots_cache)
        }
    
    async def execute(self, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Scraping komutunu çalıştır"""
        if command == "scrape":
            url = parameters.get("url")
            extract_type = parameters.get("extract_type", "text")
            if not url:
                return {"error": "URL parametresi gerekli"}
            return await self.scrape_url(url, extract_type)
        else:
            return {"error": f"Desteklenmeyen komut: {command}"}
    
    async def shutdown(self) -> bool:
        """Scraping Manager'ı güvenli şekilde kapat"""
        try:
            self.logger.info("🔄 Scraping Manager kapatılıyor...")
            
            # Session kapat
            if self.session and not self.session.closed:
                await self.session.close()
            
            self.is_initialized = False
            self.logger.info("🏁 Scraping Manager kapatıldı")
            return True
            
        except Exception as e:
            self.logger.error(f"Scraping Manager kapatma hatası: {e}")
            return False