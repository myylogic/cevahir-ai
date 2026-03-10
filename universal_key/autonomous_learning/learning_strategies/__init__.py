# -*- coding: utf-8 -*-
"""
Learning Strategies Module
==========================

Öğrenme stratejileri için modül.
"""

from .active_learning import ActiveLearning
from .reinforcement_learning import ReinforcementLearning
from .transfer_learning import TransferLearning

__all__ = [
    "ActiveLearning",
    "ReinforcementLearning",
    "TransferLearning"
]
