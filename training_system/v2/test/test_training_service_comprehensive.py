# -*- coding: utf-8 -*-
"""
Training Service V2 - Kapsamlı Test Dosyası

Bu test dosyası TrainingService'in tüm özelliklerini test eder:
- Initialization
- BPE validation
- Model initialization
- Data preparation
- Criterion creation (EOS weight 0.1)
- V2 TrainingManager integration
- Error handling
- Edge cases
"""

import pytest
import os
import sys
import torch
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
from typing import Dict, Any

# Proje kök dizinini sys.path'e ekle
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# Fixtures
@pytest.fixture
def temp_dir():
    """Geçici dizin oluştur"""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def sample_config(temp_dir):
    """Örnek config"""
    vocab_path = os.path.join(temp_dir, "vocab.json")
    merges_path = os.path.join(temp_dir, "merges.txt")
    data_dir = os.path.join(temp_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Basit BPE dosyaları oluştur
    with open(vocab_path, 'w') as f:
        f.write('{"<PAD>": {"id": 0}, "<UNK>": {"id": 1}, "<BOS>": {"id": 2}, "<EOS>": {"id": 3}}')
    with open(merges_path, 'w') as f:
        f.write("#version: 0.2\n")
    
    return {
        "data_dir": data_dir,
        "vocab_path": vocab_path,
        "merges_path": merges_path,
        "batch_size": 4,
        "epochs": 2,
        "max_seq_length": 128,
        "device": "cpu",
        "use_gpu": False,
        "enable_data_cache": False,  # Test için cache'i kapat
        "pad_token_id": 0,
        "checkpoint_dir": os.path.join(temp_dir, "checkpoints"),
        "tensorboard_log_dir": os.path.join(temp_dir, "runs"),
        "enable_tensorboard": False,
        "track_memory": False,
        "track_performance": False,
    }


@pytest.fixture
def mock_tokenizer_core():
    """Mock TokenizerCore"""
    mock = MagicMock()
    mock.get_vocab.return_value = {
        "<PAD>": {"id": 0},
        "<UNK>": {"id": 1},
        "<BOS>": {"id": 2},
        "<EOS>": {"id": 3},
    }
    mock.get_vocab_size.return_value = 4
    mock.get_pad_token_id.return_value = 0
    return mock


@pytest.fixture
def mock_model_manager():
    """Mock ModelManager"""
    mock = MagicMock()
    mock.model = torch.nn.Linear(10, 4)  # Basit model
    mock.optimizer = torch.optim.Adam(mock.model.parameters(), lr=0.001)
    mock.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(mock.optimizer)
    mock.criterion = None  # TrainingService tarafından set edilecek
    return mock


class TestTrainingServiceInitialization:
    """TrainingService initialization testleri"""
    
    def test_init_with_valid_config(self, sample_config, temp_dir):
        """Valid config ile initialization testi"""
        with patch('training_system.v2.core.training_service.TokenizerCore') as mock_tc_class, \
             patch('training_system.v2.core.training_service.ModelManager') as mock_mm_class, \
             patch('training_system.v2.core.training_service.DataCache') as mock_cache_class:
            
            # Mock setup
            mock_tc = MagicMock()
            mock_tc.get_vocab.return_value = {"<PAD>": {"id": 0}, "<UNK>": {"id": 1}, "<BOS>": {"id": 2}, "<EOS>": {"id": 3}}
            mock_tc.get_vocab_size.return_value = 4
            mock_tc.get_pad_token_id.return_value = 0
            mock_tc_class.return_value = mock_tc
            
            mock_mm = MagicMock()
            mock_model = torch.nn.Linear(10, 4)
            mock_mm.model = mock_model
            mock_mm.optimizer = torch.optim.Adam(mock_model.parameters(), lr=0.001)
            mock_mm.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(mock_mm.optimizer)
            mock_mm.initialize.return_value = None
            mock_mm_class.return_value = mock_mm
            
            mock_cache = MagicMock()
            mock_cache_class.return_value = mock_cache
            
            from training_system.v2.core.training_service import TrainingService
            
            service = TrainingService(sample_config)
            
            # Assertions
            assert service.config == sample_config
            assert service.tokenizer_core == mock_tc
            assert service.model_manager == mock_mm
            assert service.data_cache == mock_cache
            assert service.criterion is not None
            assert hasattr(service.criterion, 'weight')  # EOS weight olmalı
    
    def test_init_bpe_validation_fails(self, sample_config, temp_dir):
        """BPE dosyaları yoksa hata vermeli"""
        # Vocab dosyasını sil
        os.remove(sample_config["vocab_path"])
        
        from training_system.v2.core.training_service import TrainingService
        
        with pytest.raises(RuntimeError, match="BPE dosyaları bulunamadı"):
            TrainingService(sample_config)
    
    def test_init_data_dir_missing(self, sample_config):
        """data_dir yoksa hata vermeli"""
        sample_config["data_dir"] = "/nonexistent/path"
        
        from training_system.v2.core.training_service import TrainingService
        
        # data_dir kontrolü model initialization'dan önce yapılır
        # Eğer mock kullanılmıyorsa, model initialization'da da hata oluşabilir
        with pytest.raises(RuntimeError, match="(valid 'data_dir'|Model.*başlatılamadı)"):
            TrainingService(sample_config)
    
    def test_init_device_setup_cpu(self, sample_config, temp_dir):
        """CPU device setup testi"""
        sample_config["device"] = "cpu"
        sample_config["use_gpu"] = False
        
        with patch('training_system.v2.core.training_service.TokenizerCore') as mock_tc_class, \
             patch('training_system.v2.core.training_service.ModelManager') as mock_mm_class, \
             patch('training_system.v2.core.training_service.DataCache') as mock_cache_class:
            
            mock_tc = MagicMock()
            mock_tc.get_vocab.return_value = {"<PAD>": {"id": 0}, "<UNK>": {"id": 1}, "<BOS>": {"id": 2}, "<EOS>": {"id": 3}}
            mock_tc.get_vocab_size.return_value = 4
            mock_tc.get_pad_token_id.return_value = 0
            mock_tc_class.return_value = mock_tc
            
            mock_mm = MagicMock()
            mock_model = torch.nn.Linear(10, 4)
            mock_mm.model = mock_model
            mock_mm.optimizer = torch.optim.Adam(mock_model.parameters(), lr=0.001)
            mock_mm.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(mock_mm.optimizer)
            mock_mm.initialize.return_value = None
            mock_mm_class.return_value = mock_mm
            
            mock_cache = MagicMock()
            mock_cache_class.return_value = mock_cache
            
            from training_system.v2.core.training_service import TrainingService
            
            service = TrainingService(sample_config)
            assert service.device == "cpu"


class TestCriterionManager:
    """CriterionManager entegrasyonu testleri"""
    
    def test_eos_weight_applied(self, sample_config, temp_dir):
        """EOS weight 0.1 uygulanmalı"""
        with patch('training_system.v2.core.training_service.TokenizerCore') as mock_tc_class, \
             patch('training_system.v2.core.training_service.ModelManager') as mock_mm_class, \
             patch('training_system.v2.core.training_service.DataCache') as mock_cache_class:
            
            mock_tc = MagicMock()
            mock_tc.get_vocab.return_value = {"<PAD>": {"id": 0}, "<UNK>": {"id": 1}, "<BOS>": {"id": 2}, "<EOS>": {"id": 3}}
            mock_tc.get_vocab_size.return_value = 4
            mock_tc.get_pad_token_id.return_value = 0
            mock_tc_class.return_value = mock_tc
            
            mock_mm = MagicMock()
            mock_model = torch.nn.Linear(10, 4)
            mock_mm.model = mock_model
            mock_mm.optimizer = torch.optim.Adam(mock_model.parameters(), lr=0.001)
            mock_mm.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(mock_mm.optimizer)
            mock_mm.initialize.return_value = None
            mock_mm_class.return_value = mock_mm
            
            mock_cache = MagicMock()
            mock_cache_class.return_value = mock_cache
            
            from training_system.v2.core.training_service import TrainingService
            
            service = TrainingService(sample_config)
            
            # EOS weight kontrolü
            assert service.criterion.weight is not None
            weights = service.criterion.weight
            assert weights[3].item() == pytest.approx(0.1, abs=1e-6)  # EOS id=3, weight=0.1
            assert weights[0].item() == pytest.approx(1.0, abs=1e-6)  # Diğerleri 1.0


class TestDataPreparation:
    """Data preparation testleri"""
    
    def test_data_preparator_integration(self, sample_config, temp_dir):
        """DataPreparator entegrasyonu testi"""
        with patch('training_system.v2.core.training_service.TokenizerCore') as mock_tc_class, \
             patch('training_system.v2.core.training_service.ModelManager') as mock_mm_class, \
             patch('training_system.v2.core.training_service.DataCache') as mock_cache_class:
            
            mock_tc = MagicMock()
            mock_tc.get_vocab.return_value = {"<PAD>": {"id": 0}, "<UNK>": {"id": 1}, "<BOS>": {"id": 2}, "<EOS>": {"id": 3}}
            mock_tc.get_vocab_size.return_value = 4
            mock_tc.get_pad_token_id.return_value = 0
            mock_tc_class.return_value = mock_tc
            
            mock_mm = MagicMock()
            mock_model = torch.nn.Linear(10, 4)
            mock_mm.model = mock_model
            mock_mm.optimizer = torch.optim.Adam(mock_model.parameters(), lr=0.001)
            mock_mm.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(mock_mm.optimizer)
            mock_mm.initialize.return_value = None
            mock_mm_class.return_value = mock_mm
            
            mock_cache = MagicMock()
            mock_cache_class.return_value = mock_cache
            
            from training_system.v2.core.training_service import TrainingService
            
            service = TrainingService(sample_config)
            
            # DataPreparator instance kontrolü
            assert service.data_preparator is not None
            assert hasattr(service.data_preparator, 'prepare_from_cache')


class TestTrainingServiceIntegration:
    """TrainingService entegrasyon testleri"""
    
    def test_v2_training_manager_integration(self, sample_config, temp_dir):
        """V2 TrainingManager entegrasyonu testi"""
        with patch('training_system.v2.core.training_service.TokenizerCore') as mock_tc_class, \
             patch('training_system.v2.core.training_service.ModelManager') as mock_mm_class, \
             patch('training_system.v2.core.training_service.DataCache') as mock_cache_class, \
             patch('training_system.v2.core.training_service.create_dataloaders') as mock_create_dl, \
             patch('training_system.v2.core.training_service.V2TrainingManager') as mock_tm_class:
            
            # Mock setup
            mock_tc = MagicMock()
            mock_tc.get_vocab.return_value = {"<PAD>": {"id": 0}, "<UNK>": {"id": 1}, "<BOS>": {"id": 2}, "<EOS>": {"id": 3}}
            mock_tc.get_vocab_size.return_value = 4
            mock_tc.get_pad_token_id.return_value = 0
            mock_tc_class.return_value = mock_tc
            
            mock_mm = MagicMock()
            mock_model = torch.nn.Linear(10, 4)
            mock_mm.model = mock_model
            mock_mm.optimizer = torch.optim.Adam(mock_model.parameters(), lr=0.001)
            mock_mm.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(mock_mm.optimizer)
            mock_mm.initialize.return_value = None
            mock_mm_class.return_value = mock_mm
            
            mock_cache = MagicMock()
            mock_cache_class.return_value = mock_cache
            
            # Mock data
            mock_train_data = [(torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4]))] * 10
            mock_val_data = [(torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4]))] * 5
            mock_cache.get_or_process.return_value = (mock_train_data, False)
            
            # Mock DataLoader
            mock_train_loader = MagicMock()
            mock_val_loader = MagicMock()
            mock_create_dl.return_value = (mock_train_loader, mock_val_loader)
            
            # Mock TrainingManager - training_service modülündeki V2TrainingManager'ı patch'le
            mock_tm = MagicMock()
            mock_tm.train.return_value = (0.5, 0.6)  # (train_loss, val_loss)
            mock_tm_class.return_value = mock_tm
            
            from training_system.v2.core.training_service import TrainingService
            
            service = TrainingService(sample_config)
            
            # Mock data_preparator
            service.data_preparator.prepare_from_cache = MagicMock(
                return_value=(mock_train_data, mock_val_data, 4)
            )
            
            # Train çağrısı
            train_loss, val_loss = service.train()
            
            # Assertions
            assert train_loss == 0.5
            assert val_loss == 0.6
            mock_tm_class.assert_called_once()
            mock_tm.train.assert_called_once()


class TestErrorHandling:
    """Error handling testleri"""
    
    def test_model_initialization_failure(self, sample_config, temp_dir):
        """Model initialization başarısızlığı testi"""
        with patch('training_system.v2.core.training_service.TokenizerCore') as mock_tc_class, \
             patch('training_system.v2.core.training_service.ModelManager') as mock_mm_class, \
             patch('training_system.v2.core.training_service.DataCache') as mock_cache_class:
            
            mock_tc = MagicMock()
            mock_tc.get_vocab.return_value = {"<PAD>": {"id": 0}, "<UNK>": {"id": 1}, "<BOS>": {"id": 2}, "<EOS>": {"id": 3}}
            mock_tc.get_vocab_size.return_value = 4
            mock_tc.get_pad_token_id.return_value = 0
            mock_tc_class.return_value = mock_tc
            
            mock_mm = MagicMock()
            mock_mm.model = None  # Model None!
            mock_mm.initialize.return_value = None
            mock_mm_class.return_value = mock_mm
            
            mock_cache = MagicMock()
            mock_cache_class.return_value = mock_cache
            
            from training_system.v2.core.training_service import TrainingService
            
            # Model None olduğu için __init__ içinde hata atılmalı
            with pytest.raises(RuntimeError, match="model oluşturamadı"):
                TrainingService(sample_config)
    
    def test_optimizer_none_handling(self, sample_config, temp_dir):
        """Optimizer None kontrolü testi"""
        with patch('training_system.v2.core.training_service.TokenizerCore') as mock_tc_class, \
             patch('training_system.v2.core.training_service.ModelManager') as mock_mm_class, \
             patch('training_system.v2.core.training_service.DataCache') as mock_cache_class, \
             patch('training_system.v2.core.training_service.create_dataloaders') as mock_create_dl:
            
            mock_tc = MagicMock()
            mock_tc.get_vocab.return_value = {"<PAD>": {"id": 0}, "<UNK>": {"id": 1}, "<BOS>": {"id": 2}, "<EOS>": {"id": 3}}
            mock_tc.get_vocab_size.return_value = 4
            mock_tc.get_pad_token_id.return_value = 0
            mock_tc_class.return_value = mock_tc
            
            mock_mm = MagicMock()
            mock_model = torch.nn.Linear(10, 4)
            mock_mm.model = mock_model
            mock_mm.optimizer = None  # Optimizer None!
            mock_mm.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(torch.optim.Adam(mock_model.parameters(), lr=0.001))
            mock_mm.initialize.return_value = None
            mock_mm_class.return_value = mock_mm
            
            mock_cache = MagicMock()
            mock_cache_class.return_value = mock_cache
            
            mock_train_loader = MagicMock()
            mock_val_loader = MagicMock()
            mock_create_dl.return_value = (mock_train_loader, mock_val_loader)
            
            from training_system.v2.core.training_service import TrainingService
            
            service = TrainingService(sample_config)
            
            # Mock data_preparator
            mock_train_data = [(torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4]))] * 10
            mock_val_data = [(torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4]))] * 5
            service.data_preparator.prepare_from_cache = MagicMock(
                return_value=(mock_train_data, mock_val_data, 4)
            )
            
            # train() çağrısı optimizer None olduğu için hata vermeli
            with pytest.raises(RuntimeError, match="optimizer None"):
                service.train()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

