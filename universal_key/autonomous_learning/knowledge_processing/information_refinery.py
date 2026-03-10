# -*- coding: utf-8 -*-
"""
Information Refinery
====================

Bilgiyi rafine eden sınıf.
"""

from typing import Dict, Any, List, Optional
import logging
import time
import re

class InformationRefinery:
    """Ham bilgiyi rafine eden sistem"""
    
    def __init__(self):
        self.logger = logging.getLogger("InformationRefinery")
        self.is_initialized = False
        
        # Refinery processes
        self.refinery_stages = [
            "noise_removal",
            "fact_extraction", 
            "credibility_assessment",
            "relevance_scoring",
            "quality_enhancement"
        ]
        
        # Quality metrics
        self.quality_thresholds = {
            "minimum_credibility": 0.6,
            "minimum_relevance": 0.5,
            "minimum_completeness": 0.4
        }
        
        # Processing statistics
        self.stats = {
            "total_refined": 0,
            "high_quality_output": 0,
            "rejected_low_quality": 0,
            "average_quality_improvement": 0.0
        }
    
    async def initialize(self) -> bool:
        """Information Refinery'yi başlat"""
        try:
            self.logger.info("⚗️ Information Refinery başlatılıyor...")
            self.is_initialized = True
            self.logger.info("✅ Information Refinery başarıyla başlatıldı")
            return True
        except Exception as e:
            self.logger.error(f"Information Refinery başlatma hatası: {e}")
            return False
    
    async def refine_information(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Ham bilgiyi rafine et"""
        try:
            self.logger.info("⚗️ Information refining başlatılıyor...")
            
            refining_start = time.time()
            
            # Stage 1: Noise removal
            cleaned_data = await self._remove_noise(raw_data)
            
            # Stage 2: Fact extraction
            extracted_facts = await self._extract_facts(cleaned_data)
            
            # Stage 3: Credibility assessment
            credibility_score = await self._assess_credibility(extracted_facts)
            
            # Stage 4: Relevance scoring
            relevance_score = await self._score_relevance(extracted_facts)
            
            # Stage 5: Quality enhancement
            enhanced_data = await self._enhance_quality(extracted_facts, credibility_score, relevance_score)
            
            refining_duration = time.time() - refining_start
            
            # Overall quality score
            overall_quality = (credibility_score + relevance_score) / 2.0
            
            # Quality check
            meets_standards = (
                credibility_score >= self.quality_thresholds["minimum_credibility"] and
                relevance_score >= self.quality_thresholds["minimum_relevance"]
            )
            
            result = {
                "success": True,
                "original_data": raw_data,
                "refined_data": enhanced_data,
                "quality_metrics": {
                    "credibility_score": credibility_score,
                    "relevance_score": relevance_score,
                    "overall_quality": overall_quality,
                    "meets_quality_standards": meets_standards
                },
                "refining_stages_completed": len(self.refinery_stages),
                "refining_duration": refining_duration,
                "improvement_factor": self._calculate_improvement_factor(raw_data, enhanced_data),
                "timestamp": refining_start
            }
            
            # Update statistics
            self.stats["total_refined"] += 1
            if meets_standards:
                self.stats["high_quality_output"] += 1
            else:
                self.stats["rejected_low_quality"] += 1
            
            # Update average improvement
            improvement = result["improvement_factor"]
            current_avg = self.stats["average_quality_improvement"]
            total_refined = self.stats["total_refined"]
            self.stats["average_quality_improvement"] = ((current_avg * (total_refined - 1)) + improvement) / total_refined
            
            self.logger.info(f"⚗️ Information refined: quality={overall_quality:.2f}, meets_standards={meets_standards}")
            return result
            
        except Exception as e:
            self.logger.error(f"Information refining error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _remove_noise(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Gürültüyü temizle"""
        try:
            cleaned = dict(data)
            
            # Content cleaning
            if "content" in cleaned:
                content = cleaned["content"]
                
                # Remove excessive whitespace
                content = re.sub(r'\s+', ' ', content)
                
                # Remove common noise patterns
                noise_patterns = [
                    r'\b(click here|read more|subscribe now)\b',
                    r'\b(advertisement|sponsored|ad)\b',
                    r'[^\w\s\.\,\!\?\:\;]'  # Remove special chars except basic punctuation
                ]
                
                for pattern in noise_patterns:
                    content = re.sub(pattern, '', content, flags=re.IGNORECASE)
                
                cleaned["content"] = content.strip()
            
            # Remove low-value metadata
            low_value_keys = ["tracking_id", "session_id", "temp_data"]
            for key in low_value_keys:
                cleaned.pop(key, None)
            
            return cleaned
            
        except Exception as e:
            self.logger.error(f"Noise removal error: {e}")
            return data
    
    async def _extract_facts(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Gerçekleri çıkar"""
        try:
            facts = []
            content = data.get("content", "")
            
            if not content:
                return facts
            
            # Simple fact extraction (sentences ending with periods)
            sentences = re.split(r'[.!?]+', content)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10:  # Meaningful sentences
                    fact = {
                        "statement": sentence,
                        "confidence": 0.7,  # Default confidence
                        "source": data.get("source", "unknown"),
                        "extracted_at": time.time()
                    }
                    facts.append(fact)
            
            return facts[:10]  # Limit to 10 facts
            
        except Exception as e:
            self.logger.error(f"Fact extraction error: {e}")
            return []
    
    async def _assess_credibility(self, facts: List[Dict[str, Any]]) -> float:
        """Güvenilirlik değerlendir"""
        try:
            if not facts:
                return 0.0
            
            total_credibility = 0.0
            
            for fact in facts:
                credibility = fact.get("confidence", 0.5)
                
                # Source credibility boost
                source = fact.get("source", "").lower()
                if any(trusted in source for trusted in ["wikipedia", "academic", "official"]):
                    credibility += 0.2
                
                # Content quality indicators
                statement = fact.get("statement", "")
                if len(statement) > 50:  # Detailed statements more credible
                    credibility += 0.1
                
                total_credibility += min(1.0, credibility)
            
            average_credibility = total_credibility / len(facts)
            return min(1.0, average_credibility)
            
        except Exception as e:
            self.logger.error(f"Credibility assessment error: {e}")
            return 0.5
    
    async def _score_relevance(self, facts: List[Dict[str, Any]]) -> float:
        """İlgililik puanla"""
        try:
            if not facts:
                return 0.0
            
            # Simple relevance scoring based on content quality
            total_relevance = 0.0
            
            for fact in facts:
                statement = fact.get("statement", "")
                
                # Length-based relevance
                length_score = min(1.0, len(statement) / 100.0)
                
                # Keyword density
                words = statement.split()
                unique_words = len(set(words))
                keyword_density = unique_words / max(len(words), 1)
                
                relevance = (length_score + keyword_density) / 2.0
                total_relevance += relevance
            
            return total_relevance / len(facts)
            
        except Exception as e:
            self.logger.error(f"Relevance scoring error: {e}")
            return 0.5
    
    async def _enhance_quality(self, facts: List[Dict[str, Any]], credibility: float, relevance: float) -> Dict[str, Any]:
        """Kaliteyi artır"""
        try:
            enhanced = {
                "refined_facts": facts,
                "quality_scores": {
                    "credibility": credibility,
                    "relevance": relevance,
                    "overall": (credibility + relevance) / 2.0
                },
                "fact_count": len(facts),
                "enhancement_applied": True,
                "enhanced_at": time.time()
            }
            
            # Add quality tags
            if credibility > 0.8:
                enhanced["quality_tags"] = enhanced.get("quality_tags", []) + ["high_credibility"]
            if relevance > 0.8:
                enhanced["quality_tags"] = enhanced.get("quality_tags", []) + ["high_relevance"]
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Quality enhancement error: {e}")
            return {"error": str(e)}
    
    def _calculate_improvement_factor(self, original: Dict[str, Any], refined: Dict[str, Any]) -> float:
        """İyileşme faktörünü hesapla"""
        try:
            # Simple improvement calculation
            original_content = original.get("content", "")
            refined_facts = refined.get("refined_facts", [])
            
            if not original_content:
                return 0.0
            
            # Improvement based on structured extraction
            improvement = len(refined_facts) / max(len(original_content.split('.')), 1)
            return min(2.0, improvement)  # Max 2x improvement
            
        except Exception as e:
            return 1.0  # No improvement
    
    def get_status(self) -> Dict[str, Any]:
        """Information Refinery durumunu al"""
        return {
            "initialized": self.is_initialized,
            "refinery_stages": len(self.refinery_stages),
            "quality_thresholds": dict(self.quality_thresholds),
            "stats": dict(self.stats)
        }
    
    async def shutdown(self) -> bool:
        """Information Refinery'yi kapat"""
        try:
            self.logger.info("⚗️ Information Refinery kapatıldı")
            self.is_initialized = False
            return True
        except Exception as e:
            return False