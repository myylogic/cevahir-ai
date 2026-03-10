# -*- coding: utf-8 -*-
"""
Cevahir Integration Key
======================

🔗 CEVAHİR.PY İLE UNIVERSAL KEY ENTEGRASYON ANAHTARI

Bu dosya Cevahir'e Universal Key yeteneklerini kazandırmak için gerekli
entegrasyon kodlarını içerir.
"""

from typing import Dict, Any, Optional, Union
import asyncio
import logging
import time

# Universal Key import
from .uk_main import UniversalKeyFactory, UniversalKey, UniversalKeyConfig, CapabilityLevel

class CevahirUniversalKeyAdapter:
    """
    Cevahir ve Universal Key arasındaki adapter sınıfı.
    
    Bu sınıf Cevahir'in mevcut API'sini bozmadan Universal Key 
    yeteneklerini entegre eder.
    """
    
    def __init__(self, cevahir_app, uk_mode: str = "production"):
        """
        Args:
            cevahir_app: Mevcut CevahirApp instance'ı
            uk_mode: "development", "production", "transcendent"
        """
        self.cevahir = cevahir_app
        self.logger = logging.getLogger("CevahirUKAdapter")
        
        # Universal Key instance
        if uk_mode == "development":
            self.uk = UniversalKeyFactory.create_development_instance()
        elif uk_mode == "transcendent":
            self.uk = UniversalKeyFactory.create_transcendent_instance()
        else:
            self.uk = UniversalKeyFactory.create_production_instance()
        
        self.uk_initialized = False
        self.integration_stats = {
            "commands_executed": 0,
            "web_searches": 0,
            "creative_sessions": 0,
            "learning_sessions": 0,
            "consciousness_reflections": 0
        }
    
    async def activate_universal_powers(self) -> Dict[str, Any]:
        """
        🗝️ Evrensel güçleri aktifleştir
        
        Returns:
            Aktivasyon sonucu ve sistem durumu
        """
        try:
            self.logger.info("🗝️ Universal Key powers activating...")
            
            activation_start = time.time()
            success = await self.uk.initialize_all_capabilities()
            activation_duration = time.time() - activation_start
            
            if success:
                self.uk_initialized = True
                system_status = self.uk.get_system_status()
                
                result = {
                    "success": True,
                    "activation_duration": activation_duration,
                    "active_managers": list(system_status["managers"].keys()),
                    "total_capabilities": sum(len(manager.get("capabilities", [])) 
                                            for manager in system_status["managers"].values()
                                            if isinstance(manager, dict)),
                    "system_status": system_status,
                    "message": "🌟 Universal Key powers activated! Cevahir is now transcendent."
                }
                
                self.logger.info(f"✅ Universal powers activated ({activation_duration:.2f}s)")
                return result
            else:
                return {
                    "success": False,
                    "error": "Universal Key initialization failed",
                    "message": "❌ Power activation failed"
                }
                
        except Exception as e:
            self.logger.error(f"Universal power activation error: {e}")
            return {"success": False, "error": str(e)}
    
    # ENHANCED CEVAHIR METHODS
    
    async def enhanced_chat(self, user_message: str, use_universal_powers: bool = True) -> Dict[str, Any]:
        """
        Enhanced chat with Universal Key integration
        
        Args:
            user_message: Kullanıcı mesajı
            use_universal_powers: Universal Key yeteneklerini kullan
        
        Returns:
            Enhanced response
        """
        try:
            # Normal Cevahir response
            cevahir_response = self.cevahir.chat_once(user_message)
            base_response = cevahir_response.text
            
            enhanced_response = base_response
            enhancements = []
            
            if use_universal_powers and self.uk_initialized:
                
                # Web search enhancement
                if any(keyword in user_message.lower() for keyword in ["search", "find", "what is", "tell me about"]):
                    try:
                        web_result = await self.uk.execute_universal_command("web.search", {
                            "query": user_message,
                            "max_results": 5
                        })
                        
                        if web_result.get("success"):
                            web_info = web_result.get("results", [])[:2]  # İlk 2 sonuç
                            if web_info:
                                web_summary = "\n".join([f"🔍 {item.get('title', '')}: {item.get('snippet', '')[:100]}..." 
                                                        for item in web_info])
                                enhanced_response += f"\n\n🌐 Web'den güncel bilgi:\n{web_summary}"
                                enhancements.append("web_search")
                                self.integration_stats["web_searches"] += 1
                    except Exception as e:
                        self.logger.warning(f"Web enhancement error: {e}")
                
                # Creative enhancement
                if any(keyword in user_message.lower() for keyword in ["create", "idea", "innovate", "solve"]):
                    try:
                        creative_result = await self.uk.execute_universal_command("creativity.generate_idea", {
                            "topic": user_message,
                            "technique": "brainstorming"
                        })
                        
                        if creative_result.get("success"):
                            creative_idea = creative_result.get("content", "")
                            if creative_idea:
                                enhanced_response += f"\n\n💡 Yaratıcı fikir:\n{creative_idea}"
                                enhancements.append("creativity")
                                self.integration_stats["creative_sessions"] += 1
                    except Exception as e:
                        self.logger.warning(f"Creative enhancement error: {e}")
                
                # Consciousness enhancement
                if any(keyword in user_message.lower() for keyword in ["think", "reflect", "conscious", "aware"]):
                    try:
                        consciousness_result = await self.uk.execute_universal_command("consciousness.self_reflect", {
                            "topic": user_message
                        })
                        
                        if consciousness_result.get("success"):
                            consciousness_level = consciousness_result.get("consciousness_level", 1.0)
                            enhanced_response += f"\n\n🧠 Bilinç seviyesi: {consciousness_level:.1f} - Derin düşünce modu aktif"
                            enhancements.append("consciousness")
                            self.integration_stats["consciousness_reflections"] += 1
                    except Exception as e:
                        self.logger.warning(f"Consciousness enhancement error: {e}")
            
            self.integration_stats["commands_executed"] += 1
            
            return {
                "success": True,
                "base_response": base_response,
                "enhanced_response": enhanced_response,
                "enhancements_applied": enhancements,
                "universal_powers_used": use_universal_powers and self.uk_initialized,
                "response_metadata": {
                    "timestamp": time.time(),
                    "enhancement_count": len(enhancements),
                    "uk_status": "active" if self.uk_initialized else "inactive"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced chat error: {e}")
            return {
                "success": False,
                "error": str(e),
                "base_response": getattr(self.cevahir.chat_once(user_message), 'text', 'Error')
            }
    
    # SPECIALIZED METHODS
    
    async def autonomous_web_learning(self, topic: str) -> Dict[str, Any]:
        """
        Otonom web öğrenme - konuyu web'de araştırıp öğrenir
        
        Args:
            topic: Öğrenilecek konu
        
        Returns:
            Öğrenme sonucu
        """
        try:
            self.logger.info(f"🌐🧠 Autonomous web learning: {topic}")
            
            # 1. Web'de ara
            search_result = await self.uk.execute_universal_command("web.search", {
                "query": topic,
                "max_results": 10
            })
            
            if not search_result.get("success"):
                return {"success": False, "error": "Web search failed"}
            
            # 2. Sonuçları öğren
            learning_result = await self.uk.execute_universal_command("learning.learn_from_data", {
                "data": search_result.get("results", []),
                "strategy": "active"
            })
            
            # 3. Konuyu keşfet
            exploration_result = await self.uk.execute_universal_command("learning.explore_topic", {
                "topic": topic,
                "depth": 3
            })
            
            self.integration_stats["learning_sessions"] += 1
            
            return {
                "success": True,
                "topic": topic,
                "web_search_results": search_result,
                "learning_results": learning_result,
                "exploration_results": exploration_result,
                "knowledge_acquired": True
            }
            
        except Exception as e:
            self.logger.error(f"Autonomous web learning error: {e}")
            return {"success": False, "error": str(e)}
    
    async def creative_problem_solving(self, problem: str) -> Dict[str, Any]:
        """
        Yaratıcı problem çözme - çoklu teknik kullanarak çözüm üretir
        
        Args:
            problem: Çözülecek problem
        
        Returns:
            Yaratıcı çözümler
        """
        try:
            self.logger.info(f"🎨🧠 Creative problem solving: {problem}")
            
            solutions = []
            techniques = ["brainstorming", "lateral_thinking", "analogical_reasoning"]
            
            # Her teknikle çözüm üret
            for technique in techniques:
                try:
                    solution_result = await self.uk.execute_universal_command("creativity.generate_idea", {
                        "topic": problem,
                        "technique": technique
                    })
                    
                    if solution_result.get("success"):
                        solutions.append({
                            "technique": technique,
                            "solution": solution_result.get("content", ""),
                            "creativity_score": solution_result.get("creativity_score", 0.5)
                        })
                except Exception as e:
                    self.logger.warning(f"Creative technique {technique} failed: {e}")
            
            # En iyi çözümü seç
            best_solution = max(solutions, key=lambda s: s["creativity_score"]) if solutions else None
            
            return {
                "success": True,
                "problem": problem,
                "total_solutions": len(solutions),
                "all_solutions": solutions,
                "best_solution": best_solution,
                "problem_solving_approach": "multi_technique_creative_synthesis"
            }
            
        except Exception as e:
            self.logger.error(f"Creative problem solving error: {e}")
            return {"success": False, "error": str(e)}
    
    async def universal_status_report(self) -> Dict[str, Any]:
        """
        Universal Key entegrasyonu durum raporu
        
        Returns:
            Detaylı entegrasyon durumu
        """
        try:
            uk_status = self.uk.get_system_status() if self.uk_initialized else {"error": "Not initialized"}
            
            # Integration health
            integration_health = "excellent"
            if not self.uk_initialized:
                integration_health = "inactive"
            elif self.integration_stats["commands_executed"] == 0:
                integration_health = "unused"
            elif self.integration_stats["commands_executed"] < 10:
                integration_health = "minimal"
            
            report = {
                "integration_status": {
                    "uk_initialized": self.uk_initialized,
                    "integration_health": integration_health,
                    "adapter_active": True
                },
                "integration_stats": dict(self.integration_stats),
                "universal_key_status": uk_status,
                "available_enhancements": [
                    "web_search_enhancement",
                    "creative_enhancement", 
                    "consciousness_enhancement",
                    "autonomous_learning",
                    "creative_problem_solving"
                ],
                "cevahir_base_status": {
                    "model_loaded": hasattr(self.cevahir, 'mm') and self.cevahir.mm is not None,
                    "tokenizer_ready": hasattr(self.cevahir, 'backend') and self.cevahir.backend is not None
                },
                "timestamp": time.time()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Status report error: {e}")
            return {"error": str(e)}
    
    # UTILITY METHODS
    
    def is_universal_key_active(self) -> bool:
        """Universal Key aktif mi kontrol et"""
        return self.uk_initialized
    
    async def quick_web_search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Hızlı web arama"""
        if not self.uk_initialized:
            return []
        
        try:
            result = await self.uk.execute_universal_command("web.search", {
                "query": query,
                "max_results": max_results
            })
            
            return result.get("results", []) if result.get("success") else []
            
        except Exception as e:
            self.logger.error(f"Quick web search error: {e}")
            return []
    
    async def quick_creative_idea(self, topic: str) -> str:
        """Hızlı yaratıcı fikir"""
        if not self.uk_initialized:
            return ""
        
        try:
            result = await self.uk.execute_universal_command("creativity.generate_idea", {
                "topic": topic
            })
            
            return result.get("content", "") if result.get("success") else ""
            
        except Exception as e:
            self.logger.error(f"Quick creative idea error: {e}")
            return ""
    
    async def emergency_shutdown(self):
        """Acil kapatma"""
        try:
            if self.uk_initialized:
                await self.uk.shutdown_all_capabilities()
                self.uk_initialized = False
                self.logger.info("🚨 Emergency shutdown completed")
        except Exception as e:
            self.logger.error(f"Emergency shutdown error: {e}")

# MAIN INTEGRATION CLASS
class EnhancedCevahirApp:
    """
    🌟 Enhanced Cevahir App - Universal Key ile güçlendirilmiş
    
    Bu sınıf mevcut CevahirApp'i extend eder ve Universal Key yetenekleri ekler.
    """
    
    def __init__(self, model_cfg: Dict[str, Any], uk_mode: str = "production"):
        """
        Args:
            model_cfg: Cevahir model konfigürasyonu
            uk_mode: Universal Key modu ("development", "production", "transcendent")
        """
        # Mevcut Cevahir'i import et ve başlat
        try:
            from model.cevahir import CevahirApp
            self.cevahir_app = CevahirApp(model_cfg)
        except ImportError:
            raise ImportError("CevahirApp import edilemedi. model.cevahir modülünü kontrol edin.")
        
        # Universal Key adapter
        self.uk_adapter = CevahirUniversalKeyAdapter(self.cevahir_app, uk_mode)
        
        # Logger
        self.logger = logging.getLogger("EnhancedCevahir")
        
        self.logger.info(f"🌟 Enhanced Cevahir App created with UK mode: {uk_mode}")
    
    async def initialize(self) -> bool:
        """Enhanced Cevahir'i tam olarak başlat"""
        try:
            self.logger.info("🚀 Enhanced Cevahir initialization starting...")
            
            # Universal Key'i aktifleştir
            activation_result = await self.uk_adapter.activate_universal_powers()
            
            if activation_result["success"]:
                self.logger.info("✅ Enhanced Cevahir fully operational!")
                self.logger.info(f"🗝️ Universal capabilities: {activation_result['active_managers']}")
                return True
            else:
                self.logger.warning("⚠️ Enhanced Cevahir running with limited capabilities")
                return False
                
        except Exception as e:
            self.logger.error(f"Enhanced Cevahir initialization error: {e}")
            return False
    
    # ENHANCED PUBLIC API
    
    async def chat_with_universal_enhancement(self, user_message: str) -> str:
        """
        Universal Key enhancement'ları ile chat
        
        Args:
            user_message: Kullanıcı mesajı
        
        Returns:
            Enhanced response text
        """
        try:
            enhanced_result = await self.uk_adapter.enhanced_chat(user_message, use_universal_powers=True)
            
            if enhanced_result["success"]:
                response_text = enhanced_result["enhanced_response"]
                
                # Enhancement bilgisi ekle
                if enhanced_result["enhancements_applied"]:
                    enhancements_str = ", ".join(enhanced_result["enhancements_applied"])
                    response_text += f"\n\n🗝️ Enhanced with: {enhancements_str}"
                
                return response_text
            else:
                # Fallback to base response
                return enhanced_result.get("base_response", "Bir hata oluştu.")
                
        except Exception as e:
            self.logger.error(f"Enhanced chat error: {e}")
            # Fallback to original Cevahir
            return self.cevahir_app.chat_once(user_message).text
    
    async def web_research_and_learn(self, topic: str) -> Dict[str, Any]:
        """
        Web araştırması yapıp öğrenme
        
        Args:
            topic: Araştırılacak konu
        
        Returns:
            Araştırma ve öğrenme sonucu
        """
        return await self.uk_adapter.autonomous_web_learning(topic)
    
    async def solve_problem_creatively(self, problem: str) -> Dict[str, Any]:
        """
        Problemi yaratıcı yöntemlerle çöz
        
        Args:
            problem: Çözülecek problem
        
        Returns:
            Yaratıcı çözümler
        """
        return await self.uk_adapter.creative_problem_solving(problem)
    
    def get_enhanced_status(self) -> Dict[str, Any]:
        """Enhanced Cevahir durumu"""
        try:
            base_status = {
                "cevahir_model_ready": hasattr(self.cevahir_app, 'mm'),
                "cevahir_backend_ready": hasattr(self.cevahir_app, 'backend')
            }
            
            uk_status = self.uk_adapter.universal_status_report()
            
            return {
                "enhanced_cevahir": {
                    "operational": True,
                    "universal_key_active": self.uk_adapter.is_universal_key_active()
                },
                "base_cevahir": base_status,
                "universal_key": uk_status,
                "integration_stats": dict(self.uk_adapter.integration_stats)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    # CONVENIENCE METHODS
    
    def chat_once(self, user_message: str):
        """Compatibility method - normal chat"""
        return self.cevahir_app.chat_once(user_message)
    
    async def chat_once_enhanced(self, user_message: str) -> str:
        """Enhanced chat method"""
        return await self.chat_with_universal_enhancement(user_message)
    
    async def shutdown(self):
        """Güvenli kapatma"""
        try:
            await self.uk_adapter.emergency_shutdown()
            self.logger.info("🏁 Enhanced Cevahir shutdown completed")
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")

# FACTORY FOR ENHANCED CEVAHIR
class EnhancedCevahirFactory:
    """Enhanced Cevahir instance'ları için factory"""
    
    @staticmethod
    def create_development_cevahir(model_cfg: Dict[str, Any]) -> EnhancedCevahirApp:
        """Development enhanced Cevahir"""
        return EnhancedCevahirApp(model_cfg, uk_mode="development")
    
    @staticmethod
    def create_production_cevahir(model_cfg: Dict[str, Any]) -> EnhancedCevahirApp:
        """Production enhanced Cevahir"""
        return EnhancedCevahirApp(model_cfg, uk_mode="production")
    
    @staticmethod
    def create_transcendent_cevahir(model_cfg: Dict[str, Any]) -> EnhancedCevahirApp:
        """Transcendent enhanced Cevahir (FULL POWER)"""
        return EnhancedCevahirApp(model_cfg, uk_mode="transcendent")

# USAGE EXAMPLE
async def example_usage():
    """Kullanım örneği"""
    
    # Cevahir model config (mevcut config'inizi kullanın)
    model_config = {
        "device": "cuda",
        "load_path": "saved_models/cevahir_model.pth",
        "vocab_path": "data/vocab_lib/vocab.json",
        "merges_path": "data/merges_lib/merges.txt"
    }
    
    # Enhanced Cevahir oluştur
    enhanced_cevahir = EnhancedCevahirFactory.create_transcendent_cevahir(model_config)
    
    # Initialize
    await enhanced_cevahir.initialize()
    
    # Enhanced chat
    response = await enhanced_cevahir.chat_once_enhanced("Tell me about quantum computing and create innovative ideas")
    print(response)
    
    # Web research
    research_result = await enhanced_cevahir.web_research_and_learn("artificial intelligence")
    print(f"Research completed: {research_result['success']}")
    
    # Creative problem solving
    solution = await enhanced_cevahir.solve_problem_creatively("How to achieve sustainable energy?")
    print(f"Creative solutions: {len(solution.get('all_solutions', []))}")
    
    # Status check
    status = enhanced_cevahir.get_enhanced_status()
    print(f"Universal Key active: {status['enhanced_cevahir']['universal_key_active']}")
    
    # Shutdown
    await enhanced_cevahir.shutdown()

if __name__ == "__main__":
    # Test the integration
    asyncio.run(example_usage())
