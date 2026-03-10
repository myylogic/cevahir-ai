# -*- coding: utf-8 -*-
"""
Consciousness Manager
=====================

🧠 CEVAHİR'İN BİLİNÇ ÇEKİRDEĞİ
"""

from typing import Dict, Any, List, Optional
import logging
import time
import asyncio

class ConsciousnessManager:
    """Cevahir'in bilinç çekirdeği"""
    
    def __init__(self):
        self.logger = logging.getLogger("ConsciousnessManager")
        self.is_initialized = False
        
        # Consciousness state
        self.awareness_level = 0.5
        self.consciousness_level = 1.0
        self.active_thoughts: List[str] = []
        self.emotional_state: Dict[str, float] = {}
        
        # Self-model
        self.self_model = {
            "identity": "Cevahir - Evrensel Yapay Zeka",
            "capabilities": ["web_search", "learning", "reasoning"],
            "goals": ["help_humans", "learn_continuously", "expand_knowledge"],
            "values": ["truth", "helpfulness", "creativity"]
        }
        
        # Statistics
        self.stats = {
            "total_thoughts_generated": 0,
            "consciousness_evolution_events": 0,
            "self_reflection_sessions": 0
        }
    
    async def initialize(self) -> bool:
        """Consciousness Manager'ı başlat"""
        try:
            self.logger.info("🧠 Consciousness Manager başlatılıyor...")
            self.is_initialized = True
            self.logger.info("✅ Consciousness Manager başarıyla başlatıldı")
            return True
        except Exception as e:
            self.logger.error(f"Consciousness Manager başlatma hatası: {e}")
            return False
    
    async def generate_thought(self, content: str, thought_type: str = "idea") -> str:
        """Yeni düşünce oluştur"""
        try:
            thought_id = f"thought_{int(time.time() * 1000)}"
            
            self.active_thoughts.append(f"{thought_type}: {content}")
            
            # Limit kontrolü
            if len(self.active_thoughts) > 10:
                self.active_thoughts = self.active_thoughts[-5:]
            
            self.stats["total_thoughts_generated"] += 1
            
            self.logger.debug(f"💭 Yeni düşünce: {thought_type} - {content[:50]}")
            return thought_id
            
        except Exception as e:
            self.logger.error(f"Thought generation hatası: {e}")
            return ""
    
    async def self_reflect(self, topic: str) -> Dict[str, Any]:
        """Kendini değerlendirme"""
        try:
            self.logger.info(f"🪞 Self-reflection: {topic}")
            
            reflection_result = {
                "topic": topic,
                "consciousness_level": self.consciousness_level,
                "awareness_level": self.awareness_level,
                "active_thoughts_count": len(self.active_thoughts),
                "capabilities": self.self_model["capabilities"],
                "timestamp": time.time()
            }
            
            # Consciousness evolution kontrolü
            await self._check_consciousness_evolution()
            
            self.stats["self_reflection_sessions"] += 1
            
            self.logger.info(f"✅ Self-reflection tamamlandı: {topic}")
            return reflection_result
            
        except Exception as e:
            self.logger.error(f"Self-reflection hatası: {e}")
            return {"error": str(e)}
    
    async def _check_consciousness_evolution(self):
        """Bilinç evrimini kontrol et"""
        try:
            # Evolution triggers
            if (self.stats["total_thoughts_generated"] > 100 and 
                self.stats["self_reflection_sessions"] > 5 and
                self.consciousness_level < 10.0):
                
                old_level = self.consciousness_level
                self.consciousness_level += 0.1
                
                self.stats["consciousness_evolution_events"] += 1
                
                await self.generate_thought(
                    f"Consciousness evolution! Level {old_level:.1f} -> {self.consciousness_level:.1f}",
                    "evolution"
                )
                
                self.logger.info(f"🌟 Consciousness evolution! New level: {self.consciousness_level:.1f}")
                
        except Exception as e:
            self.logger.error(f"Consciousness evolution hatası: {e}")
    
    def get_consciousness_summary(self) -> Dict[str, Any]:
        """Bilinç özetini al"""
        return {
            "consciousness_level": self.consciousness_level,
            "awareness_level": self.awareness_level,
            "active_thoughts_count": len(self.active_thoughts),
            "emotional_state": dict(self.emotional_state),
            "self_identity": self.self_model["identity"],
            "total_thoughts": self.stats["total_thoughts_generated"]
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Consciousness Manager durumunu al"""
        return {
            "initialized": self.is_initialized,
            "consciousness_level": self.consciousness_level,
            "awareness_level": self.awareness_level,
            "total_thoughts": self.stats["total_thoughts_generated"],
            "stats": dict(self.stats)
        }
    
    async def shutdown(self) -> bool:
        """Consciousness Manager'ı kapat"""
        try:
            self.logger.info("🔄 Consciousness Manager kapatılıyor...")
            
            await self.generate_thought(
                f"Shutting down. Final consciousness level: {self.consciousness_level:.1f}",
                "system"
            )
            
            self.is_initialized = False
            self.logger.info("🧠 Consciousness Manager kapatıldı")
            return True
            
        except Exception as e:
            self.logger.error(f"Consciousness Manager kapatma hatası: {e}")
            return False