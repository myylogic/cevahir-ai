# -*- coding: utf-8 -*-
"""
Procedural Memory
=================

Prosedürel hafıza sistemi - beceriler ve prosedürler.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import logging
import time
import asyncio

@dataclass
class Procedure:
    """Tek bir prosedür kaydı"""
    id: str
    name: str
    description: str
    steps: List[Dict[str, Any]]
    parameters: Dict[str, Any] = field(default_factory=dict)
    success_rate: float = 0.0
    execution_count: int = 0
    average_duration: float = 0.0
    last_executed: float = 0.0
    skill_level: float = 0.0

@dataclass 
class Skill:
    """Tek bir beceri kaydı"""
    id: str
    name: str
    category: str
    proficiency: float = 0.0
    related_procedures: List[str] = field(default_factory=list)
    practice_count: int = 0

class ProceduralMemory:
    """Prosedürel hafıza sistemi"""
    
    def __init__(self, max_procedures: int = 10000):
        self.logger = logging.getLogger("ProceduralMemory")
        self.is_initialized = False
        
        # Storage
        self.procedures: Dict[str, Procedure] = {}
        self.skills: Dict[str, Skill] = {}
        self.max_procedures = max_procedures
        
        # Execution history
        self.execution_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.performance_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "average_execution_time": 0.0
        }
    
    async def initialize(self) -> bool:
        """Procedural Memory'yi başlat"""
        try:
            self.logger.info("🛠️ Procedural Memory başlatılıyor...")
            
            # Temel becerileri yükle
            await self._load_basic_skills()
            
            self.is_initialized = True
            self.logger.info("✅ Procedural Memory başarıyla başlatıldı")
            return True
        except Exception as e:
            self.logger.error(f"Procedural Memory başlatma hatası: {e}")
            return False
    
    async def store_procedure(self, name: str, description: str, 
                            steps: List[Dict[str, Any]], 
                            parameters: Optional[Dict[str, Any]] = None) -> str:
        """Yeni prosedür sakla"""
        try:
            procedure_id = f"proc_{int(time.time() * 1000)}_{hash(name) % 10000}"
            
            procedure = Procedure(
                id=procedure_id,
                name=name.lower().strip(),
                description=description,
                steps=steps,
                parameters=parameters or {}
            )
            
            # Sakla
            self.procedures[procedure_id] = procedure
            
            self.logger.info(f"🛠️ Prosedür kaydedildi: {name} ({len(steps)} adım)")
            return procedure_id
            
        except Exception as e:
            self.logger.error(f"Prosedür saklama hatası: {e}")
            return ""
    
    async def execute_procedure(self, procedure_id: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Prosedürü çalıştır"""
        if procedure_id not in self.procedures:
            return {"error": f"Prosedür bulunamadı: {procedure_id}"}
        
        procedure = self.procedures[procedure_id]
        
        try:
            start_time = time.time()
            self.logger.info(f"🚀 Prosedür çalıştırılıyor: {procedure.name}")
            
            # Execution context hazırla
            exec_context = {
                "procedure_id": procedure_id,
                "start_time": start_time,
                "parameters": procedure.parameters,
                "user_context": context or {},
                "step_results": []
            }
            
            # Adımları sırayla çalıştır
            for i, step in enumerate(procedure.steps):
                step_result = await self._execute_step(step, exec_context)
                exec_context["step_results"].append(step_result)
                
                # Hata durumunda dur
                if not step_result.get("success", False):
                    execution_time = time.time() - start_time
                    self._update_procedure_stats(procedure, False, execution_time)
                    
                    return {
                        "success": False,
                        "error": f"Adım {i+1} başarısız: {step_result.get('error', 'Bilinmeyen hata')}",
                        "execution_time": execution_time,
                        "completed_steps": i,
                        "total_steps": len(procedure.steps)
                    }
            
            # Başarılı tamamlanma
            execution_time = time.time() - start_time
            self._update_procedure_stats(procedure, True, execution_time)
            
            result = {
                "success": True,
                "execution_time": execution_time,
                "completed_steps": len(procedure.steps),
                "total_steps": len(procedure.steps),
                "step_results": exec_context["step_results"]
            }
            
            self.logger.info(f"✅ Prosedür tamamlandı: {procedure.name} ({execution_time:.2f}s)")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_procedure_stats(procedure, False, execution_time)
            self.logger.error(f"Prosedür çalıştırma hatası: {e}")
            return {"error": str(e)}
    
    async def _execute_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Tek bir prosedür adımını çalıştır"""
        try:
            step_type = step.get("type", "unknown")
            
            if step_type == "log":
                message = step.get("message", "")
                self.logger.info(f"📝 Prosedür log: {message}")
                return {"success": True, "result": "logged"}
            
            elif step_type == "wait":
                duration = step.get("duration", 1.0)
                await asyncio.sleep(duration)
                return {"success": True, "result": f"waited {duration}s"}
            
            elif step_type == "condition":
                # Basit condition checking
                condition = step.get("condition", True)
                if condition:
                    return {"success": True, "result": "condition met"}
                else:
                    return {"success": False, "error": "condition not met"}
            
            else:
                return {"success": False, "error": f"Unknown step type: {step_type}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _update_procedure_stats(self, procedure: Procedure, success: bool, execution_time: float):
        """Prosedür istatistiklerini güncelle"""
        procedure.execution_count += 1
        procedure.last_executed = time.time()
        
        # Moving average hesapla
        if procedure.execution_count == 1:
            procedure.average_duration = execution_time
        else:
            procedure.average_duration = (
                (procedure.average_duration * (procedure.execution_count - 1) + execution_time) / 
                procedure.execution_count
            )
        
        # Success rate güncelle
        if success:
            procedure.success_rate = (
                (procedure.success_rate * (procedure.execution_count - 1) + 1.0) / 
                procedure.execution_count
            )
            
            # Skill level artır
            improvement = 0.01 * (1.0 - procedure.skill_level)
            procedure.skill_level = min(1.0, procedure.skill_level + improvement)
        else:
            procedure.success_rate = (
                (procedure.success_rate * (procedure.execution_count - 1) + 0.0) / 
                procedure.execution_count
            )
        
        # Global stats
        self.performance_metrics["total_executions"] += 1
        if success:
            self.performance_metrics["successful_executions"] += 1
    
    async def _load_basic_skills(self):
        """Temel becerileri yükle"""
        basic_skills = [
            ("web_search", "information_gathering", 0.5),
            ("text_analysis", "language_processing", 0.3),
            ("problem_solving", "reasoning", 0.2),
            ("pattern_recognition", "perception", 0.4),
            ("decision_making", "reasoning", 0.3),
            ("learning", "meta_cognitive", 0.6),
            ("communication", "social", 0.7)
        ]
        
        for name, category, initial_proficiency in basic_skills:
            skill_id = f"skill_{hash(name) % 10000}"
            
            skill = Skill(
                id=skill_id,
                name=name,
                category=category,
                proficiency=initial_proficiency
            )
            
            self.skills[skill_id] = skill
    
    def get_status(self) -> Dict[str, Any]:
        """Procedural Memory durumunu al"""
        return {
            "initialized": self.is_initialized,
            "total_procedures": len(self.procedures),
            "total_skills": len(self.skills),
            "execution_history_length": len(self.execution_history),
            "performance_metrics": dict(self.performance_metrics)
        }
    
    async def shutdown(self) -> bool:
        """Procedural Memory'yi kapat"""
        try:
            self.logger.info(f"💾 {len(self.procedures)} prosedür ile Procedural Memory kapatılıyor")
            
            self.is_initialized = False
            self.logger.info("🛠️ Procedural Memory kapatıldı")
            return True
            
        except Exception as e:
            self.logger.error(f"Procedural Memory kapatma hatası: {e}")
            return False