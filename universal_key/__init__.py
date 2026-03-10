# -*- coding: utf-8 -*-
"""
Universal Key - Cevahir'in Evrensel Yetenek Sistemi
==================================================

Bu paket Cevahir'e evrensel yetenekler kazandıran modüler sistem içerir.
SOLID prensiplerine uygun olarak tasarlanmıştır.

Modules:
    - uk_main: Ana orchestrator
    - web_integration: Web yetenekleri
    - autonomous_learning: Otonom öğrenme
    - physical_world_interface: Fiziksel dünya arayüzü
    - quantum_capabilities: Kuantum yetenekleri
    - cognitive_expansion: Bilinç genişletme
    - temporal_manipulation: Zaman manipülasyonu
    - security_warfare: Güvenlik ve savaş
    - creative_synthesis: Yaratıcı sentez
    - communication_hub: İletişim merkezi
    - consciousness_core: Bilinç çekirdeği
"""

__version__ = "1.0.0"
__author__ = "Cevahir Development Team"

from .uk_main import UniversalKey, UniversalKeyConfig, UniversalKeyFactory

__all__ = [
    "UniversalKey",
    "UniversalKeyConfig", 
    "UniversalKeyFactory"
]
