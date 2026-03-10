# -*- coding: utf-8 -*-
"""
Transfer Learning
=================

Transfer öğrenme stratejisi - önceki bilgileri yeni alanlara aktarma.
"""

from typing import Dict, Any, List, Optional
import logging
import time
import math

class TransferLearning:
    """Transfer öğrenme stratejisi"""
    
    def __init__(self):
        self.logger = logging.getLogger("TransferLearning")
        self.is_initialized = False
        
        # Knowledge domains
        self.domains: Dict[str, Dict[str, Any]] = {}
        
        # Transfer mappings
        self.domain_mappings: Dict[str, Dict[str, float]] = {}
        
        # Transfer history
        self.transfer_history: List[Dict[str, Any]] = []
        
        # Statistics
        self.stats = {
            "total_transfers": 0,
            "successful_transfers": 0,
            "domains_learned": 0,
            "average_transfer_success": 0.0
        }
    
    async def initialize(self) -> bool:
        """Transfer Learning'i başlat"""
        try:
            self.logger.info("🔄 Transfer Learning başlatılıyor...")
            
            # Temel domainleri yükle
            await self._load_basic_domains()
            
            self.is_initialized = True
            self.logger.info("✅ Transfer Learning başarıyla başlatıldı")
            return True
        except Exception as e:
            self.logger.error(f"Transfer Learning başlatma hatası: {e}")
            return False
    
    async def register_domain(self, domain_name: str, domain_knowledge: Dict[str, Any]) -> bool:
        """Yeni domain kaydet"""
        try:
            self.domains[domain_name] = {
                "knowledge": domain_knowledge,
                "features": await self._extract_domain_features(domain_knowledge),
                "created_at": time.time(),
                "transfer_count": 0
            }
            
            self.stats["domains_learned"] += 1
            
            self.logger.info(f"📚 Domain kaydedildi: {domain_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Domain kayıt hatası: {e}")
            return False
    
    async def transfer_knowledge(self, source_domain: str, target_domain: str, 
                               knowledge_item: Dict[str, Any]) -> Dict[str, Any]:
        """Bilgiyi bir domaindan diğerine aktar"""
        try:
            if source_domain not in self.domains:
                return {"error": f"Source domain bulunamadı: {source_domain}"}
            
            if target_domain not in self.domains:
                return {"error": f"Target domain bulunamadı: {target_domain}"}
            
            # Domain similarity hesapla
            similarity = await self._calculate_domain_similarity(source_domain, target_domain)
            
            # Knowledge transfer
            transferred_knowledge = await self._adapt_knowledge(
                knowledge_item, 
                self.domains[source_domain],
                self.domains[target_domain],
                similarity
            )
            
            # Transfer başarısını değerlendir
            transfer_success = similarity * 0.8  # Basit success metric
            
            # Stats güncelle
            self.stats["total_transfers"] += 1
            if transfer_success > 0.7:
                self.stats["successful_transfers"] += 1
            
            result = {
                "success": transfer_success > 0.5,
                "transferred_knowledge": transferred_knowledge,
                "transfer_score": transfer_success,
                "domain_similarity": similarity
            }
            
            self.logger.info(f"🔄 Knowledge transfer: {source_domain} -> {target_domain} (score: {transfer_success:.2f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Knowledge transfer hatası: {e}")
            return {"error": str(e)}
    
    async def _extract_domain_features(self, domain_knowledge: Dict[str, Any]) -> Dict[str, float]:
        """Domain'den özellik çıkar"""
        try:
            features = {}
            
            # Content-based features
            content = str(domain_knowledge.get("content", ""))
            features["content_length"] = len(content.split())
            features["vocabulary_diversity"] = len(set(content.lower().split())) / max(len(content.split()), 1)
            
            # Structure-based features
            features["has_procedures"] = float("procedures" in domain_knowledge)
            features["has_concepts"] = float("concepts" in domain_knowledge)
            features["has_examples"] = float("examples" in domain_knowledge)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction hatası: {e}")
            return {}
    
    async def _calculate_domain_similarity(self, domain1: str, domain2: str) -> float:
        """İki domain arasındaki benzerlik hesapla"""
        try:
            features1 = self.domains[domain1]["features"]
            features2 = self.domains[domain2]["features"]
            
            # Cosine similarity
            common_features = set(features1.keys()).intersection(set(features2.keys()))
            
            if not common_features:
                return 0.0
            
            dot_product = sum(features1[f] * features2[f] for f in common_features)
            norm1 = math.sqrt(sum(v ** 2 for v in features1.values()))
            norm2 = math.sqrt(sum(v ** 2 for v in features2.values()))
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            self.logger.error(f"Domain similarity hesaplama hatası: {e}")
            return 0.0
    
    async def _adapt_knowledge(self, knowledge: Dict[str, Any], 
                             source_domain: Dict[str, Any], 
                             target_domain: Dict[str, Any],
                             similarity: float) -> Dict[str, Any]:
        """Bilgiyi target domain'e adapte et"""
        try:
            adapted = dict(knowledge)
            
            # Similarity bazlı adaptation
            if similarity < 0.5:
                # Low similarity, major adaptation needed
                adapted["adapted"] = True
                adapted["adaptation_level"] = "major"
            elif similarity < 0.8:
                # Medium similarity, minor adaptation
                adapted["adapted"] = True
                adapted["adaptation_level"] = "minor"
            else:
                # High similarity, minimal adaptation
                adapted["adapted"] = False
                adapted["adaptation_level"] = "none"
            
            adapted["transfer_metadata"] = {
                "source_domain": source_domain.get("name", "unknown"),
                "target_domain": target_domain.get("name", "unknown"),
                "similarity": similarity,
                "adapted_at": time.time()
            }
            
            return adapted
            
        except Exception as e:
            self.logger.error(f"Knowledge adaptation hatası: {e}")
            return knowledge
    
    async def _load_basic_domains(self):
        """Temel domainleri yükle"""
        basic_domains = {
            "language": {
                "content": "Dil işleme, gramer, semantik",
                "concepts": ["kelime", "cümle", "anlam"],
                "procedures": ["parse", "translate", "generate"]
            },
            "mathematics": {
                "content": "Sayılar, formüller, hesaplama",
                "concepts": ["sayı", "denklem", "fonksiyon"], 
                "procedures": ["calculate", "solve", "prove"]
            },
            "web": {
                "content": "İnternet, arama, veri toplama",
                "concepts": ["URL", "HTML", "API"],
                "procedures": ["search", "scrape", "parse"]
            }
        }
        
        for domain_name, knowledge in basic_domains.items():
            await self.register_domain(domain_name, knowledge)
    
    def get_status(self) -> Dict[str, Any]:
        """Transfer Learning durumunu al"""
        return {
            "initialized": self.is_initialized,
            "total_domains": len(self.domains),
            "total_transfers": len(self.transfer_history),
            "stats": dict(self.stats)
        }
    
    async def shutdown(self) -> bool:
        """Transfer Learning'i kapat"""
        try:
            self.logger.info(f"💾 {len(self.domains)} domain ile Transfer Learning kapatılıyor")
            
            self.is_initialized = False
            self.logger.info("🔄 Transfer Learning kapatıldı")
            return True
            
        except Exception as e:
            self.logger.error(f"Transfer Learning kapatma hatası: {e}")
            return False