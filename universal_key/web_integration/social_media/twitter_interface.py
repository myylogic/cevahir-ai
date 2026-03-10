# -*- coding: utf-8 -*-
"""
Twitter Interface
================

Twitter API entegrasyonu.
"""

import aiohttp
from typing import Dict, Any, Optional, List
import logging
import time

class TwitterInterface:
    """
    Twitter API interface'i.
    
    Özellikler:
    - Tweet arama
    - User timeline
    - Tweet gönderme
    - Rate limiting
    """
    
    def __init__(self, bearer_token: Optional[str] = None):
        self.logger = logging.getLogger("TwitterInterface")
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_initialized = False
        
        # API credentials
        self.bearer_token = bearer_token
        
        # API endpoints
        self.base_url = "https://api.twitter.com/2"
        
        # Rate limiting
        self.last_request_time = 0
        self.min_delay = 1.0
    
    async def initialize(self, config=None) -> bool:
        """Twitter Interface'i başlat"""
        try:
            self.logger.info("🐦 Twitter Interface başlatılıyor...")
            
            headers = {"User-Agent": "CevahirBot/1.0 Twitter Client"}
            if self.bearer_token:
                headers["Authorization"] = f"Bearer {self.bearer_token}"
            
            connector = aiohttp.TCPConnector(enable_cleanup_closed=True)
            timeout = aiohttp.ClientTimeout(total=30)
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=headers
            )
            
            self.is_initialized = True
            
            if self.bearer_token:
                self.logger.info("✅ Twitter Interface (API Mode) başarıyla başlatıldı")
            else:
                self.logger.info("✅ Twitter Interface (Read-Only Mode) başarıyla başlatıldı")
                self.logger.warning("⚠️ Bearer token yok, sınırlı erişim")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Twitter Interface başlatma hatası: {e}")
            return False
    
    async def execute(self, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Twitter komutunu çalıştır"""
        if command == "search_tweets":
            query = parameters.get("query", "")
            max_results = parameters.get("max_results", 10)
            
            if not query:
                return {"error": "Query parametresi gerekli"}
            
            return await self.search_tweets(query, max_results)
        
        elif command == "get_user_timeline":
            username = parameters.get("username", "")
            max_results = parameters.get("max_results", 10)
            
            if not username:
                return {"error": "Username parametresi gerekli"}
            
            return await self.get_user_timeline(username, max_results)
        
        else:
            return {"error": f"Desteklenmeyen komut: {command}"}
    
    async def search_tweets(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Tweet arama"""
        if not self.bearer_token:
            return {"error": "Twitter API access token gerekli"}
        
        try:
            await self._rate_limit()
            
            params = {
                "query": query,
                "max_results": min(max_results, 100),
                "tweet.fields": "created_at,author_id,public_metrics,lang"
            }
            
            start_time = time.time()
            async with self.session.get(f"{self.base_url}/tweets/search/recent", params=params) as response:
                response_time = time.time() - start_time
                
                if response.status != 200:
                    return {"error": f"Twitter API hatası: {response.status}"}
                
                data = await response.json()
                
                # Sonuçları işle
                tweets = self._process_tweet_data(data.get("data", []))
                
                result = {
                    "query": query,
                    "response_time": response_time,
                    "tweets": tweets,
                    "total_results": len(tweets),
                    "timestamp": time.time()
                }
                
                self.logger.info(f"🐦 Tweet arama tamamlandı: '{query}' ({len(tweets)} sonuç)")
                return result
                
        except Exception as e:
            self.logger.error(f"Twitter arama hatası: {e}")
            return {"error": str(e)}
    
    async def get_user_timeline(self, username: str, max_results: int = 10) -> Dict[str, Any]:
        """Kullanıcı timeline'ını al"""
        if not self.bearer_token:
            return {"error": "Twitter API access token gerekli"}
        
        try:
            await self._rate_limit()
            
            # Önce user ID al
            user_response = await self._get_user_by_username(username)
            if "error" in user_response:
                return user_response
            
            user_id = user_response["user_id"]
            
            # Timeline al
            params = {
                "max_results": min(max_results, 100),
                "tweet.fields": "created_at,public_metrics,lang"
            }
            
            start_time = time.time()
            async with self.session.get(f"{self.base_url}/users/{user_id}/tweets", params=params) as response:
                response_time = time.time() - start_time
                
                if response.status != 200:
                    return {"error": f"Twitter timeline API hatası: {response.status}"}
                
                data = await response.json()
                
                # Sonuçları işle
                tweets = self._process_tweet_data(data.get("data", []))
                
                result = {
                    "username": username,
                    "user_id": user_id,
                    "response_time": response_time,
                    "tweets": tweets,
                    "total_results": len(tweets),
                    "timestamp": time.time()
                }
                
                self.logger.info(f"🐦 {username} timeline alındı ({len(tweets)} tweet)")
                return result
                
        except Exception as e:
            self.logger.error(f"Twitter timeline hatası: {e}")
            return {"error": str(e)}
    
    async def _get_user_by_username(self, username: str) -> Dict[str, Any]:
        """Username'den user ID al"""
        try:
            async with self.session.get(f"{self.base_url}/users/by/username/{username}") as response:
                if response.status != 200:
                    return {"error": f"User lookup hatası: {response.status}"}
                
                data = await response.json()
                user_data = data.get("data", {})
                
                return {"user_id": user_data.get("id", "")}
                
        except Exception as e:
            return {"error": f"User lookup hatası: {e}"}
    
    def _process_tweet_data(self, tweets_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Tweet verilerini işle"""
        processed_tweets = []
        
        for tweet in tweets_data:
            processed_tweet = {
                "id": tweet.get("id", ""),
                "text": tweet.get("text", ""),
                "created_at": tweet.get("created_at", ""),
                "author_id": tweet.get("author_id", ""),
                "language": tweet.get("lang", ""),
                "url": f"https://twitter.com/i/status/{tweet.get('id', '')}"
            }
            
            # Public metrics
            metrics = tweet.get("public_metrics", {})
            if metrics:
                processed_tweet["metrics"] = {
                    "retweet_count": metrics.get("retweet_count", 0),
                    "like_count": metrics.get("like_count", 0),
                    "reply_count": metrics.get("reply_count", 0),
                    "quote_count": metrics.get("quote_count", 0)
                }
            
            processed_tweets.append(processed_tweet)
        
        return processed_tweets
    
    async def _rate_limit(self):
        """Twitter rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_delay:
            sleep_time = self.min_delay - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def get_status(self) -> Dict[str, Any]:
        """Twitter Interface durumunu al"""
        return {
            "initialized": self.is_initialized,
            "session_active": self.session is not None and not self.session.closed,
            "api_available": bool(self.bearer_token),
            "last_request_time": self.last_request_time
        }
    
    async def shutdown(self) -> bool:
        """Twitter Interface'i kapat"""
        try:
            if self.session and not self.session.closed:
                await self.session.close()
            
            self.is_initialized = False
            self.logger.info("🐦 Twitter Interface kapatıldı")
            return True
            
        except Exception as e:
            self.logger.error(f"Twitter Interface kapatma hatası: {e}")
            return False
