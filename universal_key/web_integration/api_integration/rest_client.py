# -*- coding: utf-8 -*-
"""
REST Client
===========

REST API istekleri için client.
"""

import aiohttp
import asyncio
from typing import Dict, Any, Optional, Union
import logging
import json
import time

class RestClient:
    """REST API istekleri için async client"""
    
    def __init__(self):
        self.logger = logging.getLogger("RestClient")
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_initialized = False
        
        # Authentication storage
        self.auth_configs: Dict[str, Dict[str, Any]] = {}
        
        # Rate limiting
        self.request_timestamps: Dict[str, list] = {}
        self.rate_limits: Dict[str, float] = {}
    
    async def initialize(self, config=None) -> bool:
        """REST Client'ı başlat"""
        try:
            self.logger.info("🔗 REST Client başlatılıyor...")
            
            connector = aiohttp.TCPConnector(limit=50, enable_cleanup_closed=True)
            timeout = aiohttp.ClientTimeout(total=60)
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "CevahirBot/1.0 REST Client"
                }
            )
            
            self.is_initialized = True
            self.logger.info("✅ REST Client başarıyla başlatıldı")
            return True
            
        except Exception as e:
            self.logger.error(f"REST Client başlatma hatası: {e}")
            return False
    
    async def execute(self, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """REST komutunu çalıştır"""
        method = command.upper()
        url = parameters.get("url", "")
        
        if not url:
            return {"error": "URL parametresi gerekli"}
        
        if method == "GET":
            return await self.get(url, parameters.get("params"), parameters.get("headers"))
        elif method == "POST":
            return await self.post(url, parameters.get("data"), parameters.get("headers"))
        else:
            return {"error": f"Desteklenmeyen HTTP method: {method}"}
    
    async def get(self, url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """GET isteği"""
        return await self._make_request("GET", url, params=params, headers=headers)
    
    async def post(self, url: str, data: Optional[Union[Dict[str, Any], str]] = None, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """POST isteği"""
        return await self._make_request("POST", url, json=data if isinstance(data, dict) else None, data=data if isinstance(data, str) else None, headers=headers)
    
    async def _make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """HTTP isteği yap"""
        if not self.is_initialized or not self.session:
            return {"error": "REST Client başlatılmamış"}
        
        try:
            # Rate limiting
            await self._rate_limit_check(url)
            
            # Authentication headers ekle
            headers = kwargs.get("headers", {})
            auth_headers = self._get_auth_headers(url)
            headers.update(auth_headers)
            kwargs["headers"] = headers
            
            # İstek yap
            start_time = time.time()
            async with self.session.request(method, url, **kwargs) as response:
                response_time = time.time() - start_time
                
                # Response işle
                try:
                    if response.content_type == "application/json":
                        response_data = await response.json()
                    else:
                        response_data = await response.text()
                except:
                    response_data = await response.text()
                
                result = {
                    "status": response.status,
                    "success": 200 <= response.status < 300,
                    "data": response_data,
                    "headers": dict(response.headers),
                    "response_time": response_time,
                    "url": url,
                    "method": method,
                    "timestamp": time.time()
                }
                
                if result["success"]:
                    self.logger.info(f"✅ {method} {url} başarılı ({response.status})")
                else:
                    self.logger.warning(f"⚠️ {method} {url} başarısız ({response.status})")
                
                return result
                
        except Exception as e:
            self.logger.error(f"❌ {method} {url} hatası: {e}")
            return {"error": str(e), "url": url, "method": method}
    
    def _get_auth_headers(self, url: str) -> Dict[str, str]:
        """URL için authentication headers al"""
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        
        if domain in self.auth_configs:
            auth_config = self.auth_configs[domain]
            auth_type = auth_config["type"]
            creds = auth_config["credentials"]
            
            if auth_type == "bearer":
                return {"Authorization": f"Bearer {creds.get('token', '')}"}
            elif auth_type == "api_key":
                key_name = creds.get('key_name', 'X-API-Key')
                key_value = creds.get('key_value', '')
                return {key_name: key_value}
        
        return {}
    
    async def _rate_limit_check(self, url: str):
        """URL bazlı rate limiting"""
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        
        current_time = time.time()
        
        if domain not in self.request_timestamps:
            self.request_timestamps[domain] = []
        
        timestamps = self.request_timestamps[domain]
        timestamps[:] = [ts for ts in timestamps if current_time - ts < 60]
        
        rate_limit = self.rate_limits.get(domain, 10)
        
        if len(timestamps) >= rate_limit:
            sleep_time = 60 - (current_time - timestamps[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        timestamps.append(current_time)
    
    def get_status(self) -> Dict[str, Any]:
        """REST Client durumunu al"""
        return {
            "initialized": self.is_initialized,
            "session_active": self.session is not None and not self.session.closed,
            "auth_domains": list(self.auth_configs.keys())
        }
    
    async def shutdown(self) -> bool:
        """REST Client'ı kapat"""
        try:
            if self.session and not self.session.closed:
                await self.session.close()
            
            self.is_initialized = False
            self.logger.info("🔗 REST Client kapatıldı")
            return True
            
        except Exception as e:
            self.logger.error(f"REST Client kapatma hatası: {e}")
            return False
