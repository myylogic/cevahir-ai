# -*- coding: utf-8 -*-
"""
Interest Detector
================

İlgi alanlarını tespit eden sınıf.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import time
import re
from collections import defaultdict, Counter

class InterestDetector:
    """İlgi alanlarını tespit eden AI sistemi"""
    
    def __init__(self):
        self.logger = logging.getLogger("InterestDetector")
        self.is_initialized = False
        
        # Interest tracking
        self.interest_scores: Dict[str, float] = defaultdict(float)
        self.topic_interactions: Dict[str, List[float]] = defaultdict(list)
        
        # Interest categories
        self.interest_categories = {
            "technology": ["AI", "machine learning", "programming", "software", "tech"],
            "science": ["physics", "chemistry", "biology", "research", "experiment"],
            "philosophy": ["consciousness", "existence", "meaning", "ethics", "mind"],
            "creativity": ["art", "music", "design", "creative", "innovation"],
            "learning": ["education", "knowledge", "study", "learn", "understand"],
            "social": ["people", "society", "culture", "communication", "relationship"]
        }
        
        # Statistics
        self.stats = {
            "total_content_analyzed": 0,
            "interests_detected": 0,
            "high_interest_topics": 0
        }
    
    async def initialize(self) -> bool:
        """Interest Detector'ı başlat"""
        try:
            self.logger.info("🎯 Interest Detector başlatılıyor...")
            self.is_initialized = True
            self.logger.info("✅ Interest Detector başarıyla başlatıldı")
            return True
        except Exception as e:
            self.logger.error(f"Interest Detector başlatma hatası: {e}")
            return False
    
    async def analyze_content_for_interests(self, content: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """İçeriği analiz ederek ilgi alanlarını tespit et"""
        try:
            self.logger.info("🔍 Content interest analysis başlatılıyor...")
            
            # Content preprocessing
            content_lower = content.lower()
            words = re.findall(r'\b\w+\b', content_lower)
            
            # Category-based interest detection
            category_scores = {}
            
            for category, keywords in self.interest_categories.items():
                score = 0.0
                matches = []
                
                for keyword in keywords:
                    if keyword.lower() in content_lower:
                        score += 1.0
                        matches.append(keyword)
                
                # Normalize score
                if len(words) > 0:
                    normalized_score = score / len(words) * 100
                    category_scores[category] = min(1.0, normalized_score)
                else:
                    category_scores[category] = 0.0
                
                if matches:
                    category_scores[f"{category}_matches"] = matches
            
            # Primary interests (above threshold)
            primary_interests = {
                category: score for category, score in category_scores.items()
                if not category.endswith("_matches") and score >= 0.1
            }
            
            result = {
                "success": True,
                "content_length": len(content),
                "word_count": len(words),
                "category_scores": category_scores,
                "primary_interests": primary_interests,
                "high_interest_categories": [cat for cat, score in primary_interests.items() if score >= 0.5],
                "timestamp": time.time()
            }
            
            # Update tracking
            await self._update_interest_tracking(result)
            
            self.stats["total_content_analyzed"] += 1
            self.stats["interests_detected"] += len(primary_interests)
            
            self.logger.info(f"🎯 Interest analysis: {len(primary_interests)} interests detected")
            return result
            
        except Exception as e:
            self.logger.error(f"Content interest analysis error: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_current_interests(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """Mevcut en yüksek ilgi alanlarını al"""
        try:
            sorted_interests = sorted(
                self.interest_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            return sorted_interests[:top_n]
            
        except Exception as e:
            self.logger.error(f"Get current interests error: {e}")
            return []
    
    async def _update_interest_tracking(self, analysis_result: Dict[str, Any]):
        """Interest tracking'i güncelle"""
        try:
            primary_interests = analysis_result.get("primary_interests", {})
            timestamp = analysis_result.get("timestamp", time.time())
            
            for category, score in primary_interests.items():
                current_score = self.interest_scores[category]
                # Moving average
                self.interest_scores[category] = current_score * 0.7 + score * 0.3
                
                # Add timestamp
                self.topic_interactions[category].append(timestamp)
                
                # Keep recent interactions only
                cutoff_time = timestamp - (30 * 24 * 3600)  # 30 days
                self.topic_interactions[category] = [
                    t for t in self.topic_interactions[category] if t > cutoff_time
                ]
                
        except Exception as e:
            self.logger.error(f"Interest tracking update error: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Interest Detector durumunu al"""
        return {
            "initialized": self.is_initialized,
            "tracked_interests": len(self.interest_scores),
            "interest_categories": len(self.interest_categories),
            "stats": dict(self.stats)
        }
    
    async def shutdown(self) -> bool:
        """Interest Detector'ı kapat"""
        try:
            self.logger.info("🎯 Interest Detector kapatıldı")
            self.is_initialized = False
            return True
        except Exception as e:
            return False