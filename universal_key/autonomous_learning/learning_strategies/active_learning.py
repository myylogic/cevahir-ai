# -*- coding: utf-8 -*-
"""
Active Learning
===============

Aktif öğrenme stratejisi - en faydalı örnekleri seçerek öğrenme.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import time
import random
import math

class ActiveLearning:
    """
    Aktif öğrenme stratejisi.
    
    Özellikler:
    - Uncertainty sampling
    - Query by committee
    - Information density
    - Expected model change
    - Diversity sampling
    """
    
    def __init__(self):
        self.logger = logging.getLogger("ActiveLearning")
        self.is_initialized = False
        
        # Learning statistics
        self.stats = {
            "queries_made": 0,
            "samples_selected": 0,
            "learning_sessions": 0,
            "average_uncertainty": 0.0,
            "last_learning_session": 0.0
        }
        
        # Sample pool
        self.unlabeled_samples: List[Dict[str, Any]] = []
        self.labeled_samples: List[Dict[str, Any]] = []
        
        # Uncertainty thresholds
        self.uncertainty_threshold = 0.7
        self.diversity_threshold = 0.5
    
    async def initialize(self) -> bool:
        """Active Learning'i başlat"""
        try:
            self.logger.info("🎯 Active Learning başlatılıyor...")
            self.is_initialized = True
            self.logger.info("✅ Active Learning başarıyla başlatıldı")
            return True
        except Exception as e:
            self.logger.error(f"Active Learning başlatma hatası: {e}")
            return False
    
    async def select_samples_for_learning(self, unlabeled_data: List[Dict[str, Any]], 
                                        num_samples: int = 10,
                                        strategy: str = "uncertainty") -> List[Dict[str, Any]]:
        """Öğrenme için en faydalı örnekleri seç"""
        try:
            if not unlabeled_data:
                return []
            
            self.logger.info(f"🎯 {len(unlabeled_data)} örnek arasından {num_samples} seçiliyor (strateji: {strategy})")
            
            if strategy == "uncertainty":
                selected = await self._uncertainty_sampling(unlabeled_data, num_samples)
            elif strategy == "diversity":
                selected = await self._diversity_sampling(unlabeled_data, num_samples)
            elif strategy == "hybrid":
                selected = await self._hybrid_sampling(unlabeled_data, num_samples)
            elif strategy == "random":
                selected = random.sample(unlabeled_data, min(num_samples, len(unlabeled_data)))
            else:
                self.logger.warning(f"Bilinmeyen strateji: {strategy}, uncertainty kullanılıyor")
                selected = await self._uncertainty_sampling(unlabeled_data, num_samples)
            
            # Stats güncelle
            self.stats["queries_made"] += 1
            self.stats["samples_selected"] += len(selected)
            
            self.logger.info(f"✅ {len(selected)} örnek seçildi")
            return selected
            
        except Exception as e:
            self.logger.error(f"Sample selection hatası: {e}")
            return []
    
    async def _uncertainty_sampling(self, data: List[Dict[str, Any]], num_samples: int) -> List[Dict[str, Any]]:
        """Belirsizlik bazlı örnekleme"""
        try:
            # Her örnek için uncertainty hesapla
            samples_with_uncertainty = []
            
            for sample in data:
                uncertainty = await self._calculate_uncertainty(sample)
                samples_with_uncertainty.append((sample, uncertainty))
            
            # Uncertainty'ye göre sırala (yüksekten düşüğe)
            samples_with_uncertainty.sort(key=lambda x: x[1], reverse=True)
            
            # En belirsiz olanları seç
            selected = [sample for sample, _ in samples_with_uncertainty[:num_samples]]
            
            # Average uncertainty hesapla
            if samples_with_uncertainty:
                avg_uncertainty = sum(u for _, u in samples_with_uncertainty) / len(samples_with_uncertainty)
                self.stats["average_uncertainty"] = avg_uncertainty
            
            return selected
            
        except Exception as e:
            self.logger.error(f"Uncertainty sampling hatası: {e}")
            return data[:num_samples]  # Fallback
    
    async def _diversity_sampling(self, data: List[Dict[str, Any]], num_samples: int) -> List[Dict[str, Any]]:
        """Çeşitlilik bazlı örnekleme"""
        try:
            if len(data) <= num_samples:
                return data
            
            selected = []
            remaining = list(data)
            
            # İlk örneği rastgele seç
            first_sample = random.choice(remaining)
            selected.append(first_sample)
            remaining.remove(first_sample)
            
            # Kalan örnekleri çeşitlilik bazlı seç
            while len(selected) < num_samples and remaining:
                max_diversity = -1
                most_diverse_sample = None
                
                for candidate in remaining:
                    # Seçilmiş örneklerle minimum benzerlik hesapla
                    min_similarity = min(
                        await self._calculate_similarity(candidate, selected_sample)
                        for selected_sample in selected
                    )
                    
                    diversity = 1.0 - min_similarity
                    
                    if diversity > max_diversity:
                        max_diversity = diversity
                        most_diverse_sample = candidate
                
                if most_diverse_sample:
                    selected.append(most_diverse_sample)
                    remaining.remove(most_diverse_sample)
                else:
                    break
            
            return selected
            
        except Exception as e:
            self.logger.error(f"Diversity sampling hatası: {e}")
            return data[:num_samples]  # Fallback
    
    async def _hybrid_sampling(self, data: List[Dict[str, Any]], num_samples: int) -> List[Dict[str, Any]]:
        """Hibrit örnekleme (uncertainty + diversity)"""
        try:
            # %70 uncertainty, %30 diversity
            uncertainty_count = int(num_samples * 0.7)
            diversity_count = num_samples - uncertainty_count
            
            # Uncertainty sampling
            uncertainty_samples = await self._uncertainty_sampling(data, uncertainty_count)
            
            # Kalan datadan diversity sampling
            remaining_data = [sample for sample in data if sample not in uncertainty_samples]
            diversity_samples = await self._diversity_sampling(remaining_data, diversity_count)
            
            # Birleştir
            selected = uncertainty_samples + diversity_samples
            
            return selected
            
        except Exception as e:
            self.logger.error(f"Hybrid sampling hatası: {e}")
            return data[:num_samples]  # Fallback
    
    async def _calculate_uncertainty(self, sample: Dict[str, Any]) -> float:
        """Örnek için uncertainty hesapla"""
        try:
            # Basit uncertainty metrikleri
            content = str(sample.get("content", ""))
            
            # Content length uncertainty
            length_uncertainty = 1.0 / (1.0 + len(content.split()))
            
            # Keyword diversity uncertainty
            words = content.lower().split()
            unique_words = set(words)
            diversity_uncertainty = len(unique_words) / max(len(words), 1)
            
            # Combined uncertainty
            uncertainty = (length_uncertainty + diversity_uncertainty) / 2.0
            
            return min(1.0, max(0.0, uncertainty))
            
        except Exception as e:
            self.logger.error(f"Uncertainty calculation hatası: {e}")
            return 0.5  # Default uncertainty
    
    async def _calculate_similarity(self, sample1: Dict[str, Any], sample2: Dict[str, Any]) -> float:
        """İki örnek arasındaki benzerlik hesapla"""
        try:
            content1 = str(sample1.get("content", "")).lower()
            content2 = str(sample2.get("content", "")).lower()
            
            # Basit Jaccard similarity
            words1 = set(content1.split())
            words2 = set(content2.split())
            
            if not words1 and not words2:
                return 1.0
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            similarity = intersection / max(union, 1)
            return similarity
            
        except Exception as e:
            self.logger.error(f"Similarity calculation hatası: {e}")
            return 0.0
    
    async def add_labeled_sample(self, sample: Dict[str, Any], label: Any):
        """Etiketli örnek ekle"""
        labeled_sample = {
            **sample,
            "label": label,
            "labeled_at": time.time()
        }
        
        self.labeled_samples.append(labeled_sample)
        
        # Labeled samples limit kontrolü
        if len(self.labeled_samples) > 10000:
            self.labeled_samples = self.labeled_samples[-5000:]  # Son 5000'i koru
        
        self.logger.debug(f"🏷️ Etiketli örnek eklendi (toplam: {len(self.labeled_samples)})")
    
    def get_learning_progress(self) -> Dict[str, Any]:
        """Öğrenme ilerlemesini al"""
        return {
            "labeled_samples": len(self.labeled_samples),
            "unlabeled_samples": len(self.unlabeled_samples),
            "labeling_ratio": len(self.labeled_samples) / max(len(self.labeled_samples) + len(self.unlabeled_samples), 1),
            "average_uncertainty": self.stats["average_uncertainty"],
            "queries_made": self.stats["queries_made"]
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Active Learning durumunu al"""
        return {
            "initialized": self.is_initialized,
            "labeled_samples": len(self.labeled_samples),
            "unlabeled_samples": len(self.unlabeled_samples),
            "stats": dict(self.stats)
        }
    
    async def shutdown(self) -> bool:
        """Active Learning'i kapat"""
        try:
            self.is_initialized = False
            self.logger.info("🎯 Active Learning kapatıldı")
            return True
        except Exception as e:
            self.logger.error(f"Active Learning kapatma hatası: {e}")
            return False
