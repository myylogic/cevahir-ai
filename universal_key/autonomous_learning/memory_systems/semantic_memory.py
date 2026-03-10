# -*- coding: utf-8 -*-
"""
Semantic Memory
===============

Semantik hafıza sistemi - kavramları ve bilgileri saklar.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import logging
import time

@dataclass
class Concept:
    """Tek bir kavram kaydı"""
    id: str
    name: str
    definition: str
    properties: Dict[str, Any] = field(default_factory=dict)
    relationships: Dict[str, List[str]] = field(default_factory=dict)
    confidence: float = 1.0
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0

@dataclass 
class Fact:
    """Tek bir gerçek/bilgi kaydı"""
    id: str
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source: str = ""
    created_at: float = field(default_factory=time.time)

class SemanticMemory:
    """Semantik hafıza sistemi"""
    
    def __init__(self, max_concepts: int = 50000):
        self.logger = logging.getLogger("SemanticMemory")
        self.is_initialized = False
        
        # Knowledge storage
        self.concepts: Dict[str, Concept] = {}
        self.facts: Dict[str, Fact] = {}
        self.max_concepts = max_concepts
        
        # Indexing
        self.name_to_concept: Dict[str, str] = {}
        
        # Statistics
        self.stats = {
            "total_concepts": 0,
            "total_facts": 0,
            "total_relationships": 0
        }
    
    async def initialize(self) -> bool:
        """Semantic Memory'yi başlat"""
        try:
            self.logger.info("🧩 Semantic Memory başlatılıyor...")
            
            # Temel kavramları yükle
            await self._load_basic_concepts()
            
            self.is_initialized = True
            self.logger.info("✅ Semantic Memory başarıyla başlatıldı")
            return True
        except Exception as e:
            self.logger.error(f"Semantic Memory başlatma hatası: {e}")
            return False
    
    async def store_concept(self, name: str, definition: str, 
                          properties: Optional[Dict[str, Any]] = None) -> str:
        """Yeni kavram sakla"""
        try:
            concept_id = f"concept_{int(time.time() * 1000)}_{hash(name) % 10000}"
            
            concept = Concept(
                id=concept_id,
                name=name.lower().strip(),
                definition=definition,
                properties=properties or {}
            )
            
            # Sakla
            self.concepts[concept_id] = concept
            self.name_to_concept[concept.name] = concept_id
            
            # Stats güncelle
            self._update_stats()
            
            self.logger.info(f"🧩 Kavram kaydedildi: {name}")
            return concept_id
            
        except Exception as e:
            self.logger.error(f"Kavram saklama hatası: {e}")
            return ""
    
    async def get_concept(self, name: str) -> Optional[Concept]:
        """Kavramı al"""
        concept_id = self.name_to_concept.get(name.lower().strip())
        if concept_id and concept_id in self.concepts:
            concept = self.concepts[concept_id]
            # Access stats güncelle
            concept.last_accessed = time.time()
            concept.access_count += 1
            return concept
        return None
    
    async def _load_basic_concepts(self):
        """Temel kavramları yükle"""
        basic_concepts = [
            ("insan", "Homo sapiens türü, akıllı canlı"),
            ("yapay zeka", "İnsan zekasını taklit eden bilgisayar sistemleri"),
            ("bilgisayar", "Elektronik hesaplama makinesi"),
            ("internet", "Küresel bilgisayar ağı"),
            ("dil", "İletişim aracı, sembolik sistem"),
            ("öğrenme", "Bilgi ve beceri edinme süreci"),
            ("hafıza", "Bilgi saklama ve geri çağırma sistemi"),
            ("zeka", "Problem çözme ve adaptasyon yeteneği")
        ]
        
        for name, definition in basic_concepts:
            await self.store_concept(name, definition)
    
    def _update_stats(self):
        """İstatistikleri güncelle"""
        self.stats["total_concepts"] = len(self.concepts)
        self.stats["total_facts"] = len(self.facts)
    
    def get_status(self) -> Dict[str, Any]:
        """Semantic Memory durumunu al"""
        return {
            "initialized": self.is_initialized,
            "total_concepts": len(self.concepts),
            "total_facts": len(self.facts),
            "max_concepts": self.max_concepts,
            "stats": dict(self.stats)
        }
    
    async def shutdown(self) -> bool:
        """Semantic Memory'yi kapat"""
        try:
            self.logger.info(f"💾 {len(self.concepts)} kavram ile Semantic Memory kapatılıyor")
            
            self.is_initialized = False
            self.logger.info("🧩 Semantic Memory kapatıldı")
            return True
            
        except Exception as e:
            self.logger.error(f"Semantic Memory kapatma hatası: {e}")
            return False