# -*- coding: utf-8 -*-
"""
Anti Detection
==============

Bot tespitini önleyen sınıf.
"""

from typing import Dict, Any, List
import logging
import random
import time
import asyncio

class AntiDetection:
    """Bot tespitini önleyen sınıf"""
    
    def __init__(self):
        self.logger = logging.getLogger("AntiDetection")
        self.is_initialized = False
        
        # User agent pool
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0"
        ]
        
        # Common headers
        self.common_headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
    
    async def initialize(self) -> bool:
        """Anti Detection'ı başlat"""
        try:
            self.is_initialized = True
            self.logger.info("🥷 Anti Detection başlatıldı")
            return True
        except Exception as e:
            self.logger.error(f"Anti Detection başlatma hatası: {e}")
            return False
    
    async def get_random_headers(self) -> Dict[str, str]:
        """Rastgele header seti oluştur"""
        headers = dict(self.common_headers)
        
        # Random user agent
        headers["User-Agent"] = random.choice(self.user_agents)
        
        # Random viewport
        viewports = ["1920x1080", "1366x768", "1440x900", "1536x864"]
        headers["Viewport"] = random.choice(viewports)
        
        return headers
    
    async def random_delay(self, min_delay: float = 1.0, max_delay: float = 3.0):
        """Rastgele bekleme süresi"""
        delay = random.uniform(min_delay, max_delay)
        await asyncio.sleep(delay)
    
    def get_status(self) -> Dict[str, Any]:
        """Anti Detection durumunu al"""
        return {
            "initialized": self.is_initialized,
            "user_agents_count": len(self.user_agents),
            "headers_count": len(self.common_headers)
        }
    
    async def shutdown(self) -> bool:
        """Anti Detection'ı kapat"""
        try:
            self.is_initialized = False
            self.logger.info("🥷 Anti Detection kapatıldı")
            return True
        except Exception as e:
            return False