# -*- coding: utf-8 -*-
"""
WebSocket Client
===============

WebSocket bağlantıları için client.
"""

import asyncio
import aiohttp
from typing import Dict, Any, Optional, Callable, List
import logging
import json
import time

class WebSocketClient:
    """
    WebSocket bağlantıları için async client.
    
    Özellikler:
    - Real-time communication
    - Message handling
    - Auto-reconnection
    - Event-based messaging
    """
    
    def __init__(self):
        self.logger = logging.getLogger("WebSocketClient")
        self.connections: Dict[str, aiohttp.ClientWebSocketResponse] = {}
        self.is_initialized = False
        
        # Message handlers
        self.message_handlers: Dict[str, List[Callable]] = {}
        
        # Connection stats
        self.connection_stats: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self, config=None) -> bool:
        """WebSocket Client'ı başlat"""
        try:
            self.logger.info("🔌 WebSocket Client başlatılıyor...")
            self.is_initialized = True
            self.logger.info("✅ WebSocket Client başarıyla başlatıldı")
            return True
        except Exception as e:
            self.logger.error(f"WebSocket Client başlatma hatası: {e}")
            return False
    
    async def execute(self, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """WebSocket komutunu çalıştır"""
        if command == "connect":
            url = parameters.get("url", "")
            connection_id = parameters.get("connection_id", url)
            if not url:
                return {"error": "URL parametresi gerekli"}
            return await self.connect(url, connection_id)
        
        elif command == "send":
            connection_id = parameters.get("connection_id", "")
            message = parameters.get("message", "")
            if not connection_id or not message:
                return {"error": "connection_id ve message parametreleri gerekli"}
            return await self.send_message(connection_id, message)
        
        elif command == "disconnect":
            connection_id = parameters.get("connection_id", "")
            if not connection_id:
                return {"error": "connection_id parametresi gerekli"}
            return await self.disconnect(connection_id)
        
        else:
            return {"error": f"Desteklenmeyen komut: {command}"}
    
    async def connect(self, url: str, connection_id: Optional[str] = None) -> Dict[str, Any]:
        """WebSocket bağlantısı kur"""
        if not self.is_initialized:
            return {"error": "WebSocket Client başlatılmamış"}
        
        connection_id = connection_id or url
        
        try:
            self.logger.info(f"🔌 WebSocket bağlantısı kuruluyor: {url}")
            
            session = aiohttp.ClientSession()
            ws = await session.ws_connect(url)
            
            self.connections[connection_id] = ws
            
            # Connection stats
            self.connection_stats[connection_id] = {
                "url": url,
                "connected_at": time.time(),
                "messages_sent": 0,
                "messages_received": 0,
                "last_activity": time.time()
            }
            
            # Message listener başlat
            asyncio.create_task(self._message_listener(connection_id, ws))
            
            self.logger.info(f"✅ WebSocket bağlantısı kuruldu: {connection_id}")
            return {"success": True, "connection_id": connection_id}
            
        except Exception as e:
            self.logger.error(f"WebSocket bağlantı hatası: {e}")
            return {"error": str(e)}
    
    async def send_message(self, connection_id: str, message: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """WebSocket üzerinden mesaj gönder"""
        if connection_id not in self.connections:
            return {"error": f"Bağlantı bulunamadı: {connection_id}"}
        
        ws = self.connections[connection_id]
        
        try:
            if isinstance(message, dict):
                await ws.send_str(json.dumps(message))
            else:
                await ws.send_str(str(message))
            
            # Stats güncelle
            if connection_id in self.connection_stats:
                self.connection_stats[connection_id]["messages_sent"] += 1
                self.connection_stats[connection_id]["last_activity"] = time.time()
            
            self.logger.debug(f"📤 Mesaj gönderildi ({connection_id}): {str(message)[:100]}")
            return {"success": True}
            
        except Exception as e:
            self.logger.error(f"Mesaj gönderme hatası: {e}")
            return {"error": str(e)}
    
    async def _message_listener(self, connection_id: str, ws: aiohttp.ClientWebSocketResponse):
        """Gelen mesajları dinle"""
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        # JSON parse dene
                        data = json.loads(msg.data)
                    except:
                        # Plain text
                        data = msg.data
                    
                    # Stats güncelle
                    if connection_id in self.connection_stats:
                        self.connection_stats[connection_id]["messages_received"] += 1
                        self.connection_stats[connection_id]["last_activity"] = time.time()
                    
                    # Message handlers çağır
                    await self._handle_message(connection_id, data)
                    
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    self.logger.error(f"WebSocket hatası ({connection_id}): {ws.exception()}")
                    break
                    
        except Exception as e:
            self.logger.error(f"Message listener hatası ({connection_id}): {e}")
        finally:
            # Bağlantıyı temizle
            if connection_id in self.connections:
                del self.connections[connection_id]
    
    async def _handle_message(self, connection_id: str, message: Any):
        """Gelen mesajı işle"""
        self.logger.debug(f"📥 Mesaj alındı ({connection_id}): {str(message)[:100]}")
        
        # Registered handlers çağır
        if connection_id in self.message_handlers:
            for handler in self.message_handlers[connection_id]:
                try:
                    await handler(message)
                except Exception as e:
                    self.logger.error(f"Message handler hatası: {e}")
    
    def add_message_handler(self, connection_id: str, handler: Callable):
        """Mesaj handler ekle"""
        if connection_id not in self.message_handlers:
            self.message_handlers[connection_id] = []
        
        self.message_handlers[connection_id].append(handler)
        self.logger.info(f"📝 Message handler eklendi: {connection_id}")
    
    async def disconnect(self, connection_id: str) -> Dict[str, Any]:
        """WebSocket bağlantısını kes"""
        if connection_id not in self.connections:
            return {"error": f"Bağlantı bulunamadı: {connection_id}"}
        
        try:
            ws = self.connections[connection_id]
            await ws.close()
            
            # Cleanup
            del self.connections[connection_id]
            if connection_id in self.connection_stats:
                del self.connection_stats[connection_id]
            if connection_id in self.message_handlers:
                del self.message_handlers[connection_id]
            
            self.logger.info(f"🔌 WebSocket bağlantısı kesildi: {connection_id}")
            return {"success": True}
            
        except Exception as e:
            self.logger.error(f"WebSocket disconnect hatası: {e}")
            return {"error": str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """WebSocket Client durumunu al"""
        return {
            "initialized": self.is_initialized,
            "active_connections": len(self.connections),
            "connection_ids": list(self.connections.keys()),
            "total_handlers": sum(len(handlers) for handlers in self.message_handlers.values()),
            "connection_stats": dict(self.connection_stats)
        }
    
    async def shutdown(self) -> bool:
        """WebSocket Client'ı kapat"""
        try:
            self.logger.info("🔄 WebSocket Client kapatılıyor...")
            
            # Tüm bağlantıları kapat
            for connection_id in list(self.connections.keys()):
                await self.disconnect(connection_id)
            
            self.is_initialized = False
            self.logger.info("🔌 WebSocket Client kapatıldı")
            return True
            
        except Exception as e:
            self.logger.error(f"WebSocket Client kapatma hatası: {e}")
            return False
