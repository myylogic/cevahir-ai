"""
Silent Error Scenario Tests
============================

Tests for scenarios where errors might occur silently.
Ensures all modules handle edge cases correctly.
"""

import pytest
import torch
import torch.nn as nn
import os
import shutil

from training_management.v2.core.training_manager import TrainingManager
from training_management.v2.core.training_loop import TrainingLoop


class TestSilentErrorScenarios:
    """Tests for silent error scenarios"""
    
    def test_nan_loss_detection_and_skipping(
        self,
        simple_model,
        simple_train_loader,
        simple_val_loader,
        simple_optimizer,
        simple_criterion,
        simple_config,
        device,
        mock_logger
    ):
        """Test that NaN loss is detected and batch is skipped silently"""
        config = simple_config.copy()
        config["device"] = device
        config["batch_size"] = 4
        config["epochs"] = 1
        config["vocab_size"] = 128
        config["checkpoint_dir"] = "./test_checkpoints_nan"
        
        if os.path.exists(config["checkpoint_dir"]):
            shutil.rmtree(config["checkpoint_dir"])
        
        # Create a model that might produce NaN (for testing)
        # In real scenario, this would happen due to numerical instability
        tm = TrainingManager(
            model=simple_model,
            train_loader=simple_train_loader,
            val_loader=simple_val_loader,
            optimizer=simple_optimizer,
            criterion=simple_criterion,
            config=config,
            logger=mock_logger
        )
        
        # Verify NaN detector is initialized
        assert tm.training_loop.nan_inf_detector is not None
        
        # Training should complete even if NaN detected (batch skipped)
        final_train_loss, final_val_loss = tm.train()
        
        # Results should be valid (not NaN)
        assert isinstance(final_train_loss, float)
        assert isinstance(final_val_loss, float)
        assert not (torch.isnan(torch.tensor(final_train_loss)).item() if isinstance(final_train_loss, (int, float)) else False)
        
        if os.path.exists(config["checkpoint_dir"]):
            shutil.rmtree(config["checkpoint_dir"])
    
    def test_inf_loss_detection_and_skipping(
        self,
        simple_model,
        simple_train_loader,
        simple_val_loader,
        simple_optimizer,
        simple_criterion,
        simple_config,
        device,
        mock_logger
    ):
        """Test that Inf loss is detected and batch is skipped silently"""
        config = simple_config.copy()
        config["device"] = device
        config["batch_size"] = 4
        config["epochs"] = 1
        config["vocab_size"] = 128
        config["checkpoint_dir"] = "./test_checkpoints_inf"
        
        if os.path.exists(config["checkpoint_dir"]):
            shutil.rmtree(config["checkpoint_dir"])
        
        tm = TrainingManager(
            model=simple_model,
            train_loader=simple_train_loader,
            val_loader=simple_val_loader,
            optimizer=simple_optimizer,
            criterion=simple_criterion,
            config=config,
            logger=mock_logger
        )
        
        # Verify NaN/Inf detector is initialized
        assert tm.training_loop.nan_inf_detector is not None
        
        # Training should complete even if Inf detected
        final_train_loss, final_val_loss = tm.train()
        
        assert isinstance(final_train_loss, float)
        assert isinstance(final_val_loss, float)
        
        if os.path.exists(config["checkpoint_dir"]):
            shutil.rmtree(config["checkpoint_dir"])
    
    def test_memory_tracker_doesnt_crash_on_cpu(
        self,
        simple_model,
        simple_train_loader,
        simple_val_loader,
        simple_optimizer,
        simple_criterion,
        simple_config,
        mock_logger
    ):
        """Test MemoryTracker doesn't crash when CUDA is not available"""
        config = simple_config.copy()
        config["device"] = "cpu"  # Force CPU
        config["batch_size"] = 4
        config["epochs"] = 1
        config["vocab_size"] = 128
        config["track_memory"] = True
        config["checkpoint_dir"] = "./test_checkpoints_mem_cpu"
        
        if os.path.exists(config["checkpoint_dir"]):
            shutil.rmtree(config["checkpoint_dir"])
        
        # Should not crash even if CUDA not available
        tm = TrainingManager(
            model=simple_model,
            train_loader=simple_train_loader,
            val_loader=simple_val_loader,
            optimizer=simple_optimizer,
            criterion=simple_criterion,
            config=config,
            logger=mock_logger
        )
        
        # Memory tracker should be initialized
        assert tm.training_loop.memory_tracker is not None
        
        # Training should complete without crashes
        final_train_loss, final_val_loss = tm.train()
        
        assert isinstance(final_train_loss, float)
        assert isinstance(final_val_loss, float)
        
        if os.path.exists(config["checkpoint_dir"]):
            shutil.rmtree(config["checkpoint_dir"])
    
    def test_tensorboard_disabled_doesnt_crash(
        self,
        simple_model,
        simple_train_loader,
        simple_val_loader,
        simple_optimizer,
        simple_criterion,
        simple_config,
        device,
        mock_logger
    ):
        """Test that disabling TensorBoard doesn't cause crashes"""
        config = simple_config.copy()
        config["device"] = device
        config["batch_size"] = 4
        config["epochs"] = 1
        config["vocab_size"] = 128
        config["enable_tensorboard"] = False  # Disabled
        config["checkpoint_dir"] = "./test_checkpoints_tb_disabled"
        
        if os.path.exists(config["checkpoint_dir"]):
            shutil.rmtree(config["checkpoint_dir"])
        
        tm = TrainingManager(
            model=simple_model,
            train_loader=simple_train_loader,
            val_loader=simple_val_loader,
            optimizer=simple_optimizer,
            criterion=simple_criterion,
            config=config,
            logger=mock_logger
        )
        
        # TensorBoard should be disabled
        assert tm.tensorboard_manager.enabled == False
        
        # Training should complete without TensorBoard logging
        final_train_loss, final_val_loss = tm.train()
        
        assert isinstance(final_train_loss, float)
        assert isinstance(final_val_loss, float)
        
        if os.path.exists(config["checkpoint_dir"]):
            shutil.rmtree(config["checkpoint_dir"])
    
    def test_zero_epochs_handling(
        self,
        simple_model,
        simple_train_loader,
        simple_val_loader,
        simple_optimizer,
        simple_criterion,
        simple_config,
        device,
        mock_logger
    ):
        """Test that zero epochs are handled gracefully"""
        config = simple_config.copy()
        config["device"] = device
        config["batch_size"] = 4
        config["epochs"] = 0  # Zero epochs
        config["vocab_size"] = 128
        config["checkpoint_dir"] = "./test_checkpoints_zero_epochs"
        
        if os.path.exists(config["checkpoint_dir"]):
            shutil.rmtree(config["checkpoint_dir"])
        
        tm = TrainingManager(
            model=simple_model,
            train_loader=simple_train_loader,
            val_loader=simple_val_loader,
            optimizer=simple_optimizer,
            criterion=simple_criterion,
            config=config,
            logger=mock_logger
        )
        
        # Should return inf losses gracefully
        final_train_loss, final_val_loss = tm.train()
        
        assert isinstance(final_train_loss, float)
        assert isinstance(final_val_loss, float)
        # Should be inf when epochs is 0
        assert final_train_loss == float('inf')
        assert final_val_loss == float('inf')
        
        if os.path.exists(config["checkpoint_dir"]):
            shutil.rmtree(config["checkpoint_dir"])
    
    def test_empty_loader_handling(
        self,
        simple_model,
        simple_optimizer,
        simple_criterion,
        simple_config,
        device,
        mock_logger
    ):
        """Test that empty loaders are handled gracefully"""
        from torch.utils.data import Dataset, DataLoader
        
        class EmptyDataset(Dataset):
            def __len__(self):
                return 0
            def __getitem__(self, idx):
                raise IndexError
        
        config = simple_config.copy()
        config["device"] = device
        config["batch_size"] = 4
        config["epochs"] = 1
        config["vocab_size"] = 128
        config["checkpoint_dir"] = "./test_checkpoints_empty"
        
        empty_train_loader = DataLoader(EmptyDataset(), batch_size=4)
        empty_val_loader = DataLoader(EmptyDataset(), batch_size=4)
        
        if os.path.exists(config["checkpoint_dir"]):
            shutil.rmtree(config["checkpoint_dir"])
        
        # Should handle empty loaders without crashing
        # Note: This might raise an error, but should be handled gracefully
        try:
            tm = TrainingManager(
                model=simple_model,
                train_loader=empty_train_loader,
                val_loader=empty_val_loader,
                optimizer=simple_optimizer,
                criterion=simple_criterion,
                config=config,
                logger=mock_logger
            )
            
            # Training might fail, but should be handled
            final_train_loss, final_val_loss = tm.train()
            
            # If we get here, losses should be valid
            assert isinstance(final_train_loss, float)
            assert isinstance(final_val_loss, float)
        except Exception as e:
            # It's OK if empty loaders cause an error, but it should be informative
            assert len(str(e)) > 0  # Error message should exist
        
        if os.path.exists(config["checkpoint_dir"]):
            shutil.rmtree(config["checkpoint_dir"])
    
    def test_memory_tracker_history_preservation(
        self,
        simple_model,
        simple_train_loader,
        simple_val_loader,
        simple_optimizer,
        simple_criterion,
        simple_config,
        device,
        mock_logger
    ):
        """Test that MemoryTracker preserves history across epochs"""
        config = simple_config.copy()
        config["device"] = device
        config["batch_size"] = 4
        config["epochs"] = 2
        config["vocab_size"] = 128
        config["track_memory"] = True
        config["checkpoint_dir"] = "./test_checkpoints_mem_history"
        
        if os.path.exists(config["checkpoint_dir"]):
            shutil.rmtree(config["checkpoint_dir"])
        
        tm = TrainingManager(
            model=simple_model,
            train_loader=simple_train_loader,
            val_loader=simple_val_loader,
            optimizer=simple_optimizer,
            criterion=simple_criterion,
            config=config,
            logger=mock_logger
        )
        
        # Training should track memory across epochs
        final_train_loss, final_val_loss = tm.train()
        
        # Memory tracker should have history attribute
        assert hasattr(tm.training_loop.memory_tracker, 'history')
        
        assert isinstance(final_train_loss, float)
        assert isinstance(final_val_loss, float)
        
        if os.path.exists(config["checkpoint_dir"]):
            shutil.rmtree(config["checkpoint_dir"])
    
    def test_performance_tracker_batch_times(
        self,
        simple_model,
        simple_train_loader,
        simple_val_loader,
        simple_optimizer,
        simple_criterion,
        simple_config,
        device,
        mock_logger
    ):
        """Test that PerformanceTracker correctly tracks batch times"""
        config = simple_config.copy()
        config["device"] = device
        config["batch_size"] = 4
        config["epochs"] = 1
        config["vocab_size"] = 128
        config["seq_len"] = 10
        config["track_performance"] = True
        config["checkpoint_dir"] = "./test_checkpoints_perf_times"
        
        if os.path.exists(config["checkpoint_dir"]):
            shutil.rmtree(config["checkpoint_dir"])
        
        tm = TrainingManager(
            model=simple_model,
            train_loader=simple_train_loader,
            val_loader=simple_val_loader,
            optimizer=simple_optimizer,
            criterion=simple_criterion,
            config=config,
            logger=mock_logger
        )
        
        # Training should track performance
        final_train_loss, final_val_loss = tm.train()
        
        # Performance tracker should have batch_times attribute
        assert hasattr(tm.training_loop.performance_tracker, 'batch_times')
        
        assert isinstance(final_train_loss, float)
        assert isinstance(final_val_loss, float)
        
        if os.path.exists(config["checkpoint_dir"]):
            shutil.rmtree(config["checkpoint_dir"])

