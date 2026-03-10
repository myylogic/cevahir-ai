# -*- coding: utf-8 -*-
"""
Voice Interface
==============

Ses arayüzü.
"""

import logging
from typing import Dict, Any

class VoiceInterface:
    """Ses arayüzü"""
    
    def __init__(self):
        self.logger = logging.getLogger("VoiceInterface")
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        try:
            self.is_initialized = True
            self.logger.info("✅ Voice Interface başlatıldı")
            return True
        except:
            return False
    
    def get_status(self) -> Dict[str, Any]:
        return {"initialized": self.is_initialized}
    
    async def shutdown(self) -> bool:
        try:
            self.is_initialized = False
            return True
        except:
            return False
