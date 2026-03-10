# -*- coding: utf-8 -*-
"""
Data Validator
==============

Çıkarılan verileri doğrulayan sınıf.
"""

from typing import Dict, Any, List
import logging
import re

class DataValidator:
    """Çıkarılan verileri doğrulayan sınıf"""
    
    def __init__(self):
        self.logger = logging.getLogger("DataValidator")
        self.is_initialized = False
        
        # Spam patterns
        self.spam_patterns = [
            re.compile(r'click here', re.IGNORECASE),
            re.compile(r'buy now', re.IGNORECASE),
            re.compile(r'free download', re.IGNORECASE)
        ]
    
    async def initialize(self) -> bool:
        """Data Validator'ı başlat"""
        try:
            self.is_initialized = True
            self.logger.info("✅ Data Validator başlatıldı")
            return True
        except Exception as e:
            self.logger.error(f"Data Validator başlatma hatası: {e}")
            return False
    
    async def validate(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Çıkarılan veriyi doğrula"""
        if not self.is_initialized:
            return {"error": "Data Validator başlatılmamış"}
        
        try:
            # Temel validation
            if not isinstance(extracted_data, dict):
                return {"error": "Geçersiz veri formatı"}
            
            # Content varsa spam kontrolü
            content = extracted_data.get("content", "")
            if content:
                spam_score = self._calculate_spam_score(content)
                extracted_data["spam_score"] = spam_score
                
                if spam_score > 0.5:
                    self.logger.warning("Yüksek spam puanı tespit edildi")
            
            return extracted_data
            
        except Exception as e:
            self.logger.error(f"Validation hatası: {e}")
            return {"error": str(e)}
    
    def _calculate_spam_score(self, content: str) -> float:
        """Spam puanı hesapla"""
        if not content:
            return 0.0
        
        spam_matches = 0
        for pattern in self.spam_patterns:
            matches = len(pattern.findall(content))
            spam_matches += matches
        
        # Normalize
        content_length = len(content.split())
        spam_score = spam_matches / max(content_length, 1)
        
        return min(1.0, spam_score * 10)
    
    def get_status(self) -> Dict[str, Any]:
        """Data Validator durumunu al"""
        return {
            "initialized": self.is_initialized,
            "spam_patterns_count": len(self.spam_patterns)
        }
    
    async def shutdown(self) -> bool:
        """Data Validator'ı kapat"""
        try:
            self.is_initialized = False
            self.logger.info("🛡️ Data Validator kapatıldı")
            return True
        except Exception as e:
            return False