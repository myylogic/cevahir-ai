# -*- coding: utf-8 -*-
"""
Working Memory
==============

Çalışma hafızası sistemi - aktif bilgi işleme.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import logging
import time
import asyncio
from collections import deque

@dataclass
class WorkingMemoryItem:
    """Çalışma hafızası item'ı"""
    id: str
    content: Any
    item_type: str
    priority: float = 0.5
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: float = 300.0  # 5 dakika default TTL

class WorkingMemory:
    """
    Çalışma hafızası sistemi.
    
    Özellikler:
    - Kısa süreli bilgi saklama
    - Priority-based management
    - Automatic cleanup (TTL)
    - Capacity management
    - Fast access patterns
    """
    
    def __init__(self, max_capacity: int = 1000):
        self.logger = logging.getLogger("WorkingMemory")
        self.is_initialized = False
        
        # Storage
        self.items: Dict[str, WorkingMemoryItem] = {}
        self.max_capacity = max_capacity
        
        # Access patterns
        self.access_history: deque = deque(maxlen=1000)
        
        # Priority queue (item_id, priority)
        self.priority_queue: List[tuple] = []
        
        # Background cleanup task
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            "total_items_stored": 0,
            "items_expired": 0,
            "items_evicted": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    async def initialize(self) -> bool:
        """Working Memory'yi başlat"""
        try:
            self.logger.info("⚡ Working Memory başlatılıyor...")
            
            # Background cleanup task başlat
            self.cleanup_task = asyncio.create_task(self._background_cleanup())
            
            self.is_initialized = True
            self.logger.info("✅ Working Memory başarıyla başlatıldı")
            return True
        except Exception as e:
            self.logger.error(f"Working Memory başlatma hatası: {e}")
            return False
    
    async def store(self, item_id: str, content: Any, item_type: str = "general", 
                   priority: float = 0.5, ttl: float = 300.0) -> bool:
        """Working memory'ye item ekle"""
        try:
            # Capacity kontrolü
            if len(self.items) >= self.max_capacity:
                await self._evict_least_important()
            
            # Item oluştur
            item = WorkingMemoryItem(
                id=item_id,
                content=content,
                item_type=item_type,
                priority=priority,
                ttl=ttl
            )
            
            # Sakla
            self.items[item_id] = item
            
            # Priority queue güncelle
            self._update_priority_queue()
            
            # Stats
            self.stats["total_items_stored"] += 1
            
            self.logger.debug(f"💾 Working memory item eklendi: {item_id} (priority: {priority})")
            return True
            
        except Exception as e:
            self.logger.error(f"Working memory store hatası: {e}")
            return False
    
    async def retrieve(self, item_id: str) -> Optional[Any]:
        """Working memory'den item al"""
        try:
            if item_id in self.items:
                item = self.items[item_id]
                
                # TTL kontrolü
                if time.time() - item.created_at > item.ttl:
                    await self._remove_item(item_id, "expired")
                    self.stats["cache_misses"] += 1
                    return None
                
                # Access stats güncelle
                item.last_accessed = time.time()
                item.access_count += 1
                
                # Access history güncelle
                self.access_history.append((item_id, time.time()))
                
                self.stats["cache_hits"] += 1
                self.logger.debug(f"🎯 Working memory hit: {item_id}")
                return item.content
            else:
                self.stats["cache_misses"] += 1
                return None
                
        except Exception as e:
            self.logger.error(f"Working memory retrieve hatası: {e}")
            return None
    
    async def update_priority(self, item_id: str, new_priority: float):
        """Item priority'sini güncelle"""
        if item_id in self.items:
            self.items[item_id].priority = new_priority
            self._update_priority_queue()
            self.logger.debug(f"📊 Priority güncellendi: {item_id} -> {new_priority}")
    
    async def extend_ttl(self, item_id: str, additional_time: float):
        """Item TTL'ini uzat"""
        if item_id in self.items:
            self.items[item_id].ttl += additional_time
            self.logger.debug(f"⏰ TTL uzatıldı: {item_id} (+{additional_time}s)")
    
    async def get_items_by_type(self, item_type: str) -> List[WorkingMemoryItem]:
        """Type'a göre itemları al"""
        return [item for item in self.items.values() if item.item_type == item_type]
    
    async def get_high_priority_items(self, min_priority: float = 0.7) -> List[WorkingMemoryItem]:
        """Yüksek priority itemları al"""
        high_priority_items = [
            item for item in self.items.values() 
            if item.priority >= min_priority
        ]
        
        # Priority'ye göre sırala
        high_priority_items.sort(key=lambda x: x.priority, reverse=True)
        return high_priority_items
    
    def _update_priority_queue(self):
        """Priority queue'yu güncelle"""
        self.priority_queue = [
            (item_id, item.priority) 
            for item_id, item in self.items.items()
        ]
        self.priority_queue.sort(key=lambda x: x[1], reverse=True)
    
    async def _evict_least_important(self):
        """En az önemli item'ı çıkar"""
        if not self.items:
            return
        
        # En düşük priority'li item'ı bul
        least_important_id = min(
            self.items.keys(),
            key=lambda item_id: self.items[item_id].priority
        )
        
        await self._remove_item(least_important_id, "evicted")
    
    async def _remove_item(self, item_id: str, reason: str):
        """Item'ı sil"""
        if item_id in self.items:
            del self.items[item_id]
            
            if reason == "expired":
                self.stats["items_expired"] += 1
            elif reason == "evicted":
                self.stats["items_evicted"] += 1
            
            self.logger.debug(f"🗑️ Working memory item silindi: {item_id} ({reason})")
    
    async def _background_cleanup(self):
        """Background cleanup task"""
        while self.is_initialized:
            try:
                current_time = time.time()
                expired_items = []
                
                # Expired itemları bul
                for item_id, item in self.items.items():
                    if current_time - item.created_at > item.ttl:
                        expired_items.append(item_id)
                
                # Expired itemları sil
                for item_id in expired_items:
                    await self._remove_item(item_id, "expired")
                
                if expired_items:
                    self.logger.info(f"🧹 Background cleanup: {len(expired_items)} item silindi")
                
                # 30 saniye bekle
                await asyncio.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Background cleanup hatası: {e}")
                await asyncio.sleep(60)  # Hata durumunda daha uzun bekle
    
    def get_status(self) -> Dict[str, Any]:
        """Working Memory durumunu al"""
        hit_rate = 0.0
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        if total_requests > 0:
            hit_rate = (self.stats["cache_hits"] / total_requests) * 100
        
        return {
            "initialized": self.is_initialized,
            "current_items": len(self.items),
            "max_capacity": self.max_capacity,
            "usage_percent": (len(self.items) / self.max_capacity) * 100,
            "hit_rate_percent": hit_rate,
            "cleanup_task_active": self.cleanup_task is not None and not self.cleanup_task.done(),
            "stats": dict(self.stats)
        }
    
    async def shutdown(self) -> bool:
        """Working Memory'yi kapat"""
        try:
            # Cleanup task'ı durdur
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass
            
            self.is_initialized = False
            self.logger.info("⚡ Working Memory kapatıldı")
            return True
            
        except Exception as e:
            self.logger.error(f"Working Memory kapatma hatası: {e}")
            return False
