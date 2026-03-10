# -*- coding: utf-8 -*-
"""
Multi Channel Module
===================

Çoklu kanal iletişimi için modül.
"""

from .voice_interface import VoiceInterface
from .text_interface import TextInterface
from .visual_interface import VisualInterface

__all__ = [
    "VoiceInterface",
    "TextInterface",
    "VisualInterface"
]
