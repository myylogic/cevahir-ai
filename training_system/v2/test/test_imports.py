# -*- coding: utf-8 -*-
"""
Test Training System V2 Imports

Bu test dosyası tüm import'ların doğru çalıştığını doğrular.
"""

import pytest
import sys
import os

# Proje kök dizinini sys.path'e ekle
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_training_service_import():
    """TrainingService import testi"""
    from training_system.v2.core.training_service import TrainingService
    assert TrainingService is not None


def test_bpe_validator_import():
    """BPEValidator import testi"""
    from training_system.v2.core.bpe_validator import BPEValidator
    assert BPEValidator is not None


def test_criterion_manager_import():
    """CriterionManager import testi"""
    from training_system.v2.core.criterion_manager import CriterionManager
    assert CriterionManager is not None


def test_data_preparator_import():
    """DataPreparator import testi"""
    from training_system.v2.core.data_preparator import DataPreparator
    assert DataPreparator is not None


def test_config_manager_import():
    """ConfigManager import testi"""
    from training_system.v2.core.config_manager import ConfigManager
    assert ConfigManager is not None


def test_v2_training_manager_import():
    """V2 TrainingManager import testi (doğru path)"""
    from training_management.v2.core.training_manager import TrainingManager
    assert TrainingManager is not None


def test_data_loader_wrapper_import():
    """DataLoaderWrapper import testi"""
    from training_system.v2.utils.data_loader_wrapper import create_dataloaders
    assert create_dataloaders is not None


def test_all_core_modules_import():
    """Tüm core modüllerin import testi"""
    from training_system.v2.core import (
        TrainingService,
        BPEValidator,
        CriterionManager,
        DataPreparator,
        ConfigManager
    )
    assert all([
        TrainingService is not None,
        BPEValidator is not None,
        CriterionManager is not None,
        DataPreparator is not None,
        ConfigManager is not None
    ])

