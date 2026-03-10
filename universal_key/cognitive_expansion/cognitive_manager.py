# -*- coding: utf-8 -*-
"""
Cognitive Expansion Manager
===========================

Bilinç genişletme süreçlerini yöneten sınıf.
"""

from typing import Dict, Any, List, Optional
import logging
import time
import sys
import os

# Ana cognitive manager'ı import et
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from cognitive_management.cognitive_manager import CognitiveManager as MainCognitiveManager
from cognitive_management.types import CognitiveState, CognitiveInput, CognitiveOutput

class CognitiveExpansionManager:
    """
    Universal Key için Cognitive Manager wrapper'ı.
    Ana cognitive manager'ı genişletir ve Universal Key API'sine entegre eder.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("CognitiveExpansionManager")
        self.is_initialized = False
        
        # Ana cognitive manager
        self.main_cognitive_manager: Optional[MainCognitiveManager] = None
        
        # Cognitive expansion features
        self.expansion_capabilities = {
            "consciousness_levels": ["basic", "enhanced", "transcendent"],
            "learning_modes": ["passive", "active", "accelerated"],
            "memory_types": ["episodic", "semantic", "procedural", "working"],
            "thinking_strategies": ["direct", "think1", "debate2"]
        }
        
        # Expansion state tracking
        self.consciousness_level = "basic"
        self.learning_mode = "active"
        self.expansion_history: List[Dict[str, Any]] = []
        
        # Performance metrics
        self.stats = {
            "total_cognitive_operations": 0,
            "consciousness_expansions": 0,
            "learning_cycles": 0,
            "memory_consolidations": 0
        }
    
    async def initialize(self, model_manager=None) -> bool:
        """Cognitive Expansion Manager'ı başlat"""
        try:
            self.logger.info("🧠 Cognitive Expansion Manager başlatılıyor...")
            
            # Ana cognitive manager'ı başlat
            if model_manager:
                self.main_cognitive_manager = MainCognitiveManager(model_manager)
                self.logger.info("✅ Ana cognitive manager bağlandı")
            else:
                # Mock model manager for testing
                self.main_cognitive_manager = None
                self.logger.warning("⚠️ Model manager bulunamadı, mock mode")
            
            self.is_initialized = True
            self.logger.info("✅ Cognitive Expansion Manager başarıyla başlatıldı")
            return True
            
        except Exception as e:
            self.logger.error(f"Cognitive Expansion Manager başlatma hatası: {e}")
            return False
    
    async def process_cognitive_request(self, user_message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Cognitive request işle"""
        try:
            self.logger.info(f"🧠 Cognitive processing: {user_message[:50]}...")
            
            processing_start = time.time()
            
            if self.main_cognitive_manager:
                # Ana cognitive manager ile işle
                result = await self._process_with_main_manager(user_message, context)
            else:
                # Mock processing
                result = await self._mock_cognitive_processing(user_message, context)
            
            processing_duration = time.time() - processing_start
            
            # Expansion tracking
            expansion_record = {
                "user_message": user_message,
                "result": result,
                "consciousness_level": self.consciousness_level,
                "learning_mode": self.learning_mode,
                "processing_duration": processing_duration,
                "timestamp": processing_start
            }
            self.expansion_history.append(expansion_record)
            
            # Statistics
            self.stats["total_cognitive_operations"] += 1
            
            self.logger.info(f"🧠 Cognitive processing completed: {processing_duration:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Cognitive processing error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _process_with_main_manager(self, user_message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Ana cognitive manager ile işle"""
        try:
            # Cognitive state oluştur
            state = CognitiveState(
                history=context.get("history", []) if context else [],
                step=context.get("step", 0) if context else 0,
                last_mode=context.get("last_mode", "direct") if context else "direct",
                last_entropy=context.get("last_entropy", 0.0) if context else 0.0
            )
            
            # Cognitive input oluştur
            request = CognitiveInput(
                user_message=user_message,
                system_prompt=context.get("system_prompt") if context else None
            )
            
            # Ana manager ile işle
            output: CognitiveOutput = self.main_cognitive_manager.handle(state, request)
            
            return {
                "success": True,
                "response": output.text,
                "used_mode": output.used_mode,
                "tool_used": output.tool_used,
                "revised_by_critic": output.revised_by_critic,
                "consciousness_level": self.consciousness_level,
                "learning_mode": self.learning_mode,
                "processing_method": "main_cognitive_manager"
            }
            
        except Exception as e:
            self.logger.error(f"Main manager processing error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _mock_cognitive_processing(self, user_message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Mock cognitive processing"""
        try:
            # Basit mock response
            response = f"Mock cognitive response to: {user_message}"
            
            # Consciousness level simulation
            if "complex" in user_message.lower() or "difficult" in user_message.lower():
                self.consciousness_level = "enhanced"
                response += " [Enhanced consciousness mode activated]"
            
            return {
                "success": True,
                "response": response,
                "used_mode": "direct",
                "tool_used": None,
                "revised_by_critic": False,
                "consciousness_level": self.consciousness_level,
                "learning_mode": self.learning_mode,
                "processing_method": "mock_processing"
            }
            
        except Exception as e:
            self.logger.error(f"Mock processing error: {e}")
            return {"success": False, "error": str(e)}
    
    async def expand_consciousness(self, target_level: str = "enhanced") -> Dict[str, Any]:
        """Bilinç seviyesini genişlet"""
        try:
            self.logger.info(f"🧠 Consciousness expansion: {self.consciousness_level} -> {target_level}")
            
            if target_level not in self.expansion_capabilities["consciousness_levels"]:
                return {"success": False, "error": f"Invalid consciousness level: {target_level}"}
            
            old_level = self.consciousness_level
            self.consciousness_level = target_level
            
            # Expansion effects
            expansion_effects = {
                "basic": {"processing_speed": 1.0, "memory_capacity": 1.0, "creativity": 1.0},
                "enhanced": {"processing_speed": 1.5, "memory_capacity": 1.3, "creativity": 1.4},
                "transcendent": {"processing_speed": 2.0, "memory_capacity": 1.8, "creativity": 2.0}
            }
            
            effects = expansion_effects.get(target_level, expansion_effects["basic"])
            
            result = {
                "success": True,
                "old_level": old_level,
                "new_level": target_level,
                "expansion_effects": effects,
                "expansion_timestamp": time.time()
            }
            
            self.stats["consciousness_expansions"] += 1
            
            self.logger.info(f"🧠 Consciousness expanded: {old_level} -> {target_level}")
            return result
            
        except Exception as e:
            self.logger.error(f"Consciousness expansion error: {e}")
            return {"success": False, "error": str(e)}
    
    async def optimize_learning_mode(self, target_mode: str = "accelerated") -> Dict[str, Any]:
        """Öğrenme modunu optimize et"""
        try:
            self.logger.info(f"📚 Learning mode optimization: {self.learning_mode} -> {target_mode}")
            
            if target_mode not in self.expansion_capabilities["learning_modes"]:
                return {"success": False, "error": f"Invalid learning mode: {target_mode}"}
            
            old_mode = self.learning_mode
            self.learning_mode = target_mode
            
            # Learning mode effects
            mode_effects = {
                "passive": {"learning_rate": 0.5, "retention": 0.8, "energy_usage": 0.3},
                "active": {"learning_rate": 1.0, "retention": 1.0, "energy_usage": 1.0},
                "accelerated": {"learning_rate": 1.8, "retention": 0.9, "energy_usage": 1.5}
            }
            
            effects = mode_effects.get(target_mode, mode_effects["active"])
            
            result = {
                "success": True,
                "old_mode": old_mode,
                "new_mode": target_mode,
                "mode_effects": effects,
                "optimization_timestamp": time.time()
            }
            
            self.stats["learning_cycles"] += 1
            
            self.logger.info(f"📚 Learning mode optimized: {old_mode} -> {target_mode}")
            return result
            
        except Exception as e:
            self.logger.error(f"Learning mode optimization error: {e}")
            return {"success": False, "error": str(e)}
    
    async def consolidate_memory(self, memory_type: str = "episodic") -> Dict[str, Any]:
        """Hafızayı pekiştir"""
        try:
            self.logger.info(f"🧠 Memory consolidation: {memory_type}")
            
            if memory_type not in self.expansion_capabilities["memory_types"]:
                return {"success": False, "error": f"Invalid memory type: {memory_type}"}
            
            # Memory consolidation simulation
            consolidation_effects = {
                "episodic": {"retention_boost": 0.3, "recall_speed": 0.2},
                "semantic": {"concept_clarity": 0.4, "association_strength": 0.3},
                "procedural": {"skill_retention": 0.5, "execution_speed": 0.2},
                "working": {"capacity_boost": 0.3, "processing_efficiency": 0.2}
            }
            
            effects = consolidation_effects.get(memory_type, {})
            
            result = {
                "success": True,
                "memory_type": memory_type,
                "consolidation_effects": effects,
                "consolidation_timestamp": time.time()
            }
            
            self.stats["memory_consolidations"] += 1
            
            self.logger.info(f"🧠 Memory consolidated: {memory_type}")
            return result
            
        except Exception as e:
            self.logger.error(f"Memory consolidation error: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_cognitive_status(self) -> Dict[str, Any]:
        """Cognitive durumunu al"""
        return {
            "initialized": self.is_initialized,
            "consciousness_level": self.consciousness_level,
            "learning_mode": self.learning_mode,
            "main_manager_connected": self.main_cognitive_manager is not None,
            "expansion_capabilities": self.expansion_capabilities,
            "expansion_history_count": len(self.expansion_history),
            "stats": dict(self.stats)
        }
    
    async def shutdown(self) -> bool:
        """Cognitive Expansion Manager'ı kapat"""
        try:
            self.logger.info("🧠 Cognitive Expansion Manager kapatılıyor...")
            self.is_initialized = False
            self.logger.info("✅ Cognitive Expansion Manager kapatıldı")
            return True
        except Exception as e:
            self.logger.error(f"Cognitive Expansion Manager kapatma hatası: {e}")
            return False