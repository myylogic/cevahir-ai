# -*- coding: utf-8 -*-
"""
Text Interface
==============

Metin arayüzü.
"""

import logging
from typing import Dict, Any

class TextInterface:
    """Metin arayüzü"""
    
    def __init__(self):
        self.logger = logging.getLogger("TextInterface")
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        try:
            self.is_initialized = True
            self.logger.info("✅ Text Interface başlatıldı")
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
