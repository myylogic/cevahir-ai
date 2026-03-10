# -*- coding: utf-8 -*-
"""
Episodic Memory
===============

Episodik hafıza sistemi - deneyimleri ve olayları saklar.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import logging
import time
import json
import hashlib
from datetime import datetime

@dataclass
class Episode:
    """Tek bir episodik hafıza kaydı"""
    id: str
    timestamp: float
    event_type: str
    content: Dict[str, Any]
    context: Dict[str, Any] = field(default_factory=dict)
    emotions: Dict[str, float] = field(default_factory=dict)
    importance: float = 0.5
    tags: List[str] = field(default_factory=list)
    related_episodes: List[str] = field(default_factory=list)

class EpisodicMemory:
    """
    Episodik hafıza sistemi.
    
    Özellikler:
    - Zaman bazlı olay saklama
    - Duygusal etiketleme
    - Önem derecesi hesaplama
    - İlişkisel bağlantılar
    - Otomatik unutma (decay)
    - Arama ve filtreleme
    """
    
    def __init__(self, max_episodes: int = 10000):
        self.logger = logging.getLogger("EpisodicMemory")
        self.is_initialized = False
        
        # Episode storage
        self.episodes: Dict[str, Episode] = {}
        self.max_episodes = max_episodes
        
        # Indexing for fast search
        self.time_index: List[str] = []  # Episode IDs sorted by time
        self.tag_index: Dict[str, List[str]] = {}  # Tag -> Episode IDs
        self.type_index: Dict[str, List[str]] = {}  # Event type -> Episode IDs
        
        # Memory statistics
        self.stats = {
            "total_episodes": 0,
            "episodes_forgotten": 0,
            "last_cleanup": 0,
            "average_importance": 0.0
        }
    
    async def initialize(self) -> bool:
        """Episodic Memory'yi başlat"""
        try:
            self.logger.info("🧠 Episodic Memory başlatılıyor...")
            self.is_initialized = True
            self.logger.info("✅ Episodic Memory başarıyla başlatıldı")
            return True
        except Exception as e:
            self.logger.error(f"Episodic Memory başlatma hatası: {e}")
            return False
    
    async def store_episode(self, event_type: str, content: Dict[str, Any], 
                          context: Optional[Dict[str, Any]] = None,
                          emotions: Optional[Dict[str, float]] = None,
                          tags: Optional[List[str]] = None) -> str:
        """Yeni episod sakla"""
        try:
            # Episode ID oluştur
            episode_id = self._generate_episode_id(event_type, content)
            
            # Importance hesapla
            importance = self._calculate_importance(event_type, content, emotions or {})
            
            # Episode oluştur
            episode = Episode(
                id=episode_id,
                timestamp=time.time(),
                event_type=event_type,
                content=content,
                context=context or {},
                emotions=emotions or {},
                importance=importance,
                tags=tags or [],
                related_episodes=[]
            )
            
            # Sakla
            self.episodes[episode_id] = episode
            
            # Indexleri güncelle
            self._update_indexes(episode)
            
            # İlişkili episodları bul
            related = await self._find_related_episodes(episode)
            episode.related_episodes = related
            
            # Memory limit kontrolü
            await self._manage_memory_limit()
            
            # Stats güncelle
            self._update_stats()
            
            self.logger.info(f"💾 Episode kaydedildi: {event_type} (importance: {importance:.2f})")
            return episode_id
            
        except Exception as e:
            self.logger.error(f"Episode saklama hatası: {e}")
            return ""
    
    async def retrieve_episodes(self, 
                              event_type: Optional[str] = None,
                              tags: Optional[List[str]] = None,
                              time_range: Optional[tuple] = None,
                              min_importance: float = 0.0,
                              max_results: int = 100) -> List[Episode]:
        """Episodları filtrele ve al"""
        try:
            matching_episodes = []
            
            # Tüm episodları kontrol et
            for episode in self.episodes.values():
                # Event type filtresi
                if event_type and episode.event_type != event_type:
                    continue
                
                # Tag filtresi
                if tags and not any(tag in episode.tags for tag in tags):
                    continue
                
                # Time range filtresi
                if time_range:
                    start_time, end_time = time_range
                    if not (start_time <= episode.timestamp <= end_time):
                        continue
                
                # Importance filtresi
                if episode.importance < min_importance:
                    continue
                
                matching_episodes.append(episode)
            
            # Importance ve zamana göre sırala
            matching_episodes.sort(key=lambda e: (e.importance, e.timestamp), reverse=True)
            
            # Limit uygula
            result = matching_episodes[:max_results]
            
            self.logger.info(f"🔍 Episode arama tamamlandı: {len(result)} sonuç")
            return result
            
        except Exception as e:
            self.logger.error(f"Episode retrieve hatası: {e}")
            return []
    
    async def get_recent_episodes(self, hours: int = 24, max_results: int = 50) -> List[Episode]:
        """Son N saatteki episodları al"""
        current_time = time.time()
        start_time = current_time - (hours * 3600)
        
        return await self.retrieve_episodes(
            time_range=(start_time, current_time),
            max_results=max_results
        )
    
    async def get_important_episodes(self, min_importance: float = 0.7, max_results: int = 100) -> List[Episode]:
        """Önemli episodları al"""
        return await self.retrieve_episodes(
            min_importance=min_importance,
            max_results=max_results
        )
    
    def _generate_episode_id(self, event_type: str, content: Dict[str, Any]) -> str:
        """Episode ID oluştur"""
        # Content hash + timestamp
        content_str = json.dumps(content, sort_keys=True)
        content_hash = hashlib.md5(content_str.encode()).hexdigest()[:8]
        timestamp_str = str(int(time.time() * 1000))[-8:]  # Son 8 digit
        
        return f"{event_type}_{timestamp_str}_{content_hash}"
    
    def _calculate_importance(self, event_type: str, content: Dict[str, Any], emotions: Dict[str, float]) -> float:
        """Episode importance hesapla"""
        base_importance = 0.5
        
        # Event type bazlı importance
        type_weights = {
            "learning": 0.8,
            "discovery": 0.9,
            "achievement": 0.9,
            "error": 0.6,
            "interaction": 0.7,
            "decision": 0.8,
            "creative": 0.9,
            "routine": 0.3
        }
        
        importance = type_weights.get(event_type, base_importance)
        
        # Emotion bazlı adjustment
        if emotions:
            emotion_intensity = sum(abs(value) for value in emotions.values())
            importance += min(0.3, emotion_intensity * 0.1)
        
        # Content complexity bazlı adjustment
        content_complexity = len(str(content)) / 1000  # Basit metric
        importance += min(0.2, content_complexity * 0.05)
        
        return min(1.0, max(0.0, importance))
    
    def _update_indexes(self, episode: Episode):
        """Indexleri güncelle"""
        # Time index
        self.time_index.append(episode.id)
        self.time_index.sort(key=lambda eid: self.episodes[eid].timestamp)
        
        # Tag index
        for tag in episode.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = []
            self.tag_index[tag].append(episode.id)
        
        # Type index
        if episode.event_type not in self.type_index:
            self.type_index[episode.event_type] = []
        self.type_index[episode.event_type].append(episode.id)
    
    async def _find_related_episodes(self, episode: Episode) -> List[str]:
        """İlişkili episodları bul"""
        related = []
        
        # Aynı tag'lere sahip episodlar
        for tag in episode.tags:
            if tag in self.tag_index:
                for related_id in self.tag_index[tag]:
                    if related_id != episode.id and related_id not in related:
                        related.append(related_id)
        
        # Yakın zamanda olan episodlar
        time_window = 3600  # 1 saat
        for other_episode in self.episodes.values():
            if (other_episode.id != episode.id and 
                abs(other_episode.timestamp - episode.timestamp) < time_window and
                other_episode.id not in related):
                related.append(other_episode.id)
        
        return related[:10]  # Max 10 related
    
    async def _manage_memory_limit(self):
        """Memory limit kontrolü ve cleanup"""
        if len(self.episodes) <= self.max_episodes:
            return
        
        # En az önemli episodları sil
        episodes_to_remove = len(self.episodes) - self.max_episodes + 100  # 100 extra buffer
        
        # Importance'a göre sırala
        sorted_episodes = sorted(
            self.episodes.values(),
            key=lambda e: (e.importance, e.timestamp)
        )
        
        # En az önemli olanları sil
        for episode in sorted_episodes[:episodes_to_remove]:
            await self._forget_episode(episode.id)
        
        self.logger.info(f"🧹 Memory cleanup: {episodes_to_remove} episode silindi")
    
    async def _forget_episode(self, episode_id: str):
        """Episode'u sil ve indexleri güncelle"""
        if episode_id not in self.episodes:
            return
        
        episode = self.episodes[episode_id]
        
        # Episode'u sil
        del self.episodes[episode_id]
        
        # Indexlerden sil
        if episode_id in self.time_index:
            self.time_index.remove(episode_id)
        
        for tag in episode.tags:
            if tag in self.tag_index and episode_id in self.tag_index[tag]:
                self.tag_index[tag].remove(episode_id)
                if not self.tag_index[tag]:  # Boş liste ise sil
                    del self.tag_index[tag]
        
        if episode.event_type in self.type_index and episode_id in self.type_index[episode.event_type]:
            self.type_index[episode.event_type].remove(episode_id)
            if not self.type_index[episode.event_type]:
                del self.type_index[episode.event_type]
        
        self.stats["episodes_forgotten"] += 1
    
    def _update_stats(self):
        """İstatistikleri güncelle"""
        self.stats["total_episodes"] = len(self.episodes)
        
        if self.episodes:
            total_importance = sum(ep.importance for ep in self.episodes.values())
            self.stats["average_importance"] = total_importance / len(self.episodes)
        else:
            self.stats["average_importance"] = 0.0
    
    def get_status(self) -> Dict[str, Any]:
        """Episodic Memory durumunu al"""
        return {
            "initialized": self.is_initialized,
            "total_episodes": len(self.episodes),
            "max_episodes": self.max_episodes,
            "memory_usage_percent": (len(self.episodes) / self.max_episodes) * 100,
            "unique_tags": len(self.tag_index),
            "unique_event_types": len(self.type_index),
            "stats": dict(self.stats)
        }
    
    async def shutdown(self) -> bool:
        """Episodic Memory'yi kapat"""
        try:
            # Memory'yi kaydet (opsiyonel - file'a yazılabilir)
            self.logger.info(f"💾 {len(self.episodes)} episode ile Episodic Memory kapatılıyor")
            
            self.is_initialized = False
            self.logger.info("🧠 Episodic Memory kapatıldı")
            return True
            
        except Exception as e:
            self.logger.error(f"Episodic Memory kapatma hatası: {e}")
            return False
