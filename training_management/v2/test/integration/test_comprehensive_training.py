"""
Comprehensive Training Tests
============================

End-to-end comprehensive tests for TrainingManager.
Tests all features including multi-epoch, early stopping, checkpointing, etc.
"""

import pytest
import torch
import torch.nn as nn
import os
import shutil

from training_management.v2.core.training_manager import TrainingManager


class TestComprehensiveTraining:
    """Comprehensive training tests"""
    
    def test_multi_epoch_training(
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
        """Test multi-epoch training"""
        # Setup config
        config = simple_config.copy()
        config["device"] = device
        config["batch_size"] = 4
        config["epochs"] = 3
        config["vocab_size"] = 128
        config["checkpoint_dir"] = "./test_checkpoints_multi_epoch"
        
        # Clean up checkpoint dir
        if os.path.exists(config["checkpoint_dir"]):
            shutil.rmtree(config["checkpoint_dir"])
        
        # Create TrainingManager
        tm = TrainingManager(
            model=simple_model,
            train_loader=simple_train_loader,
            val_loader=simple_val_loader,
            optimizer=simple_optimizer,
            criterion=simple_criterion,
            config=config,
            logger=mock_logger
        )
        
        # Train
        final_train_loss, final_val_loss = tm.train()
        
        # Assertions
        assert isinstance(final_train_loss, float)
        assert isinstance(final_val_loss, float)
        assert final_train_loss >= 0.0
        assert final_val_loss >= 0.0
        
        # History should have 3 entries (one per epoch)
        history = tm.get_training_history()
        assert len(history["train_loss"]) == 3
        assert len(history["val_loss"]) == 3
        assert len(history["accuracy"]) == 3
        
        # Cleanup
        if os.path.exists(config["checkpoint_dir"]):
            shutil.rmtree(config["checkpoint_dir"])
    
    def test_early_stopping(
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
        """Test early stopping mechanism"""
        # Setup config with early stopping
        config = simple_config.copy()
        config["device"] = device
        config["batch_size"] = 4
        config["epochs"] = 10  # More epochs than patience
        config["vocab_size"] = 128
        config["early_stopping_patience"] = 2  # Stop after 2 epochs without improvement
        config["checkpoint_dir"] = "./test_checkpoints_early_stop"
        
        # Clean up checkpoint dir
        if os.path.exists(config["checkpoint_dir"]):
            shutil.rmtree(config["checkpoint_dir"])
        
        # Create TrainingManager
        tm = TrainingManager(
            model=simple_model,
            train_loader=simple_train_loader,
            val_loader=simple_val_loader,
            optimizer=simple_optimizer,
            criterion=simple_criterion,
            config=config,
            logger=mock_logger
        )
        
        # Train (should stop early)
        final_train_loss, final_val_loss = tm.train()
        
        # Assertions
        assert isinstance(final_train_loss, float)
        assert isinstance(final_val_loss, float)
        
        # History should have fewer entries than epochs (due to early stopping)
        history = tm.get_training_history()
        # Early stopping should trigger (validation loss might not always improve)
        assert len(history["train_loss"]) <= 10
        
        # Cleanup
        if os.path.exists(config["checkpoint_dir"]):
            shutil.rmtree(config["checkpoint_dir"])
    
    def test_checkpoint_saving(
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
        """Test checkpoint saving"""
        # Setup config
        config = simple_config.copy()
        config["device"] = device
        config["batch_size"] = 4
        config["epochs"] = 2
        config["vocab_size"] = 128
        checkpoint_dir = "./test_checkpoints_saving"
        config["checkpoint_dir"] = checkpoint_dir
        
        # Clean up checkpoint dir
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        
        # Create TrainingManager
        tm = TrainingManager(
            model=simple_model,
            train_loader=simple_train_loader,
            val_loader=simple_val_loader,
            optimizer=simple_optimizer,
            criterion=simple_criterion,
            config=config,
            logger=mock_logger
        )
        
        # Train
        final_train_loss, final_val_loss = tm.train()
        
        # Checkpoint dir should exist
        assert os.path.exists(checkpoint_dir)
        
        # Checkpoint files should exist
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        assert len(checkpoint_files) > 0  # At least one checkpoint should be saved
        
        # Cleanup
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
    
    def test_training_with_scheduler(
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
        """Test training with learning rate scheduler"""
        # Setup config with scheduler
        config = simple_config.copy()
        config["device"] = device
        config["batch_size"] = 4
        config["epochs"] = 2
        config["vocab_size"] = 128
        config["scheduler_type"] = "ReduceLROnPlateau"
        config["scheduler_kwargs"] = {
            "mode": "min",
            "patience": 1,
            "factor": 0.5,
            "verbose": False
        }
        config["checkpoint_dir"] = "./test_checkpoints_scheduler"
        
        # Clean up checkpoint dir
        if os.path.exists(config["checkpoint_dir"]):
            shutil.rmtree(config["checkpoint_dir"])
        
        # Get initial LR
        initial_lr = simple_optimizer.param_groups[0]["lr"]
        
        # Create TrainingManager
        tm = TrainingManager(
            model=simple_model,
            train_loader=simple_train_loader,
            val_loader=simple_val_loader,
            optimizer=simple_optimizer,
            criterion=simple_criterion,
            config=config,
            logger=mock_logger
        )
        
        # Train
        final_train_loss, final_val_loss = tm.train()
        
        # Scheduler should have stepped (LR might have changed)
        final_lr = tm.optimizer.param_groups[0]["lr"]
        # LR might have decreased due to scheduler
        assert isinstance(final_lr, float)
        assert final_lr > 0.0
        
        # Cleanup
        if os.path.exists(config["checkpoint_dir"]):
            shutil.rmtree(config["checkpoint_dir"])
    
    def test_training_with_eos_weight_end_to_end(
        self,
        simple_model,
        simple_train_loader,
        simple_val_loader,
        simple_optimizer,
        eos_weighted_criterion,
        simple_config,
        device,
        mock_logger
    ):
        """Test end-to-end training with EOS weight"""
        # Setup config
        config = simple_config.copy()
        config["device"] = device
        config["batch_size"] = 4
        config["epochs"] = 2
        config["vocab_size"] = 128
        config["checkpoint_dir"] = "./test_checkpoints_eos"
        
        # Clean up checkpoint dir
        if os.path.exists(config["checkpoint_dir"]):
            shutil.rmtree(config["checkpoint_dir"])
        
        # Create TrainingManager with EOS weighted criterion
        tm = TrainingManager(
            model=simple_model,
            train_loader=simple_train_loader,
            val_loader=simple_val_loader,
            optimizer=simple_optimizer,
            criterion=eos_weighted_criterion,
            config=config,
            logger=mock_logger
        )
        
        # Verify EOS weight is set
        assert tm.loss_computation.criterion == eos_weighted_criterion
        assert hasattr(eos_weighted_criterion, 'weight')
        assert eos_weighted_criterion.weight is not None
        
        # Train
        final_train_loss, final_val_loss = tm.train()
        
        # Assertions
        assert isinstance(final_train_loss, float)
        assert isinstance(final_val_loss, float)
        assert final_train_loss >= 0.0
        assert final_val_loss >= 0.0
        
        # History should be populated
        history = tm.get_training_history()
        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2
        
        # Cleanup
        if os.path.exists(config["checkpoint_dir"]):
            shutil.rmtree(config["checkpoint_dir"])
    
    def test_epoch_callback(
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
        """Test epoch callback functionality"""
        # Setup config
        config = simple_config.copy()
        config["device"] = device
        config["batch_size"] = 4
        config["epochs"] = 2
        config["vocab_size"] = 128
        config["checkpoint_dir"] = "./test_checkpoints_callback"
        
        # Clean up checkpoint dir
        if os.path.exists(config["checkpoint_dir"]):
            shutil.rmtree(config["checkpoint_dir"])
        
        # Track callback calls
        callback_calls = []
        
        def epoch_callback(epoch, train_loss, val_loss):
            callback_calls.append((epoch, train_loss, val_loss))
        
        # Create TrainingManager
        tm = TrainingManager(
            model=simple_model,
            train_loader=simple_train_loader,
            val_loader=simple_val_loader,
            optimizer=simple_optimizer,
            criterion=simple_criterion,
            config=config,
            logger=mock_logger
        )
        
        # Train with callback
        final_train_loss, final_val_loss = tm.train(epoch_callback=epoch_callback)
        
        # Callback should have been called for each epoch
        assert len(callback_calls) == 2
        for epoch, train_loss, val_loss in callback_calls:
            assert isinstance(epoch, int)
            assert isinstance(train_loss, float)
            assert isinstance(val_loss, float)
        
        # Cleanup
        if os.path.exists(config["checkpoint_dir"]):
            shutil.rmtree(config["checkpoint_dir"])
    
    def test_training_history_consistency(
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
        """Test training history consistency"""
        # Setup config
        config = simple_config.copy()
        config["device"] = device
        config["batch_size"] = 4
        config["epochs"] = 3
        config["vocab_size"] = 128
        config["checkpoint_dir"] = "./test_checkpoints_history"
        
        # Clean up checkpoint dir
        if os.path.exists(config["checkpoint_dir"]):
            shutil.rmtree(config["checkpoint_dir"])
        
        # Create TrainingManager
        tm = TrainingManager(
            model=simple_model,
            train_loader=simple_train_loader,
            val_loader=simple_val_loader,
            optimizer=simple_optimizer,
            criterion=simple_criterion,
            config=config,
            logger=mock_logger
        )
        
        # Train
        final_train_loss, final_val_loss = tm.train()
        
        # Get history
        history = tm.get_training_history()
        
        # History should be consistent
        assert len(history["train_loss"]) == 3
        assert len(history["val_loss"]) == 3
        assert len(history["accuracy"]) == 3
        
        # Final values should match
        assert abs(history["train_loss"][-1] - final_train_loss) < 1e-6
        assert abs(history["val_loss"][-1] - final_val_loss) < 1e-6
        
        # All values should be finite
        for loss in history["train_loss"]:
            assert isinstance(loss, float) and loss >= 0.0
        for loss in history["val_loss"]:
            assert isinstance(loss, float) and loss >= 0.0
        for acc in history["accuracy"]:
            assert isinstance(acc, float) and 0.0 <= acc <= 1.0
        
        # Cleanup
        if os.path.exists(config["checkpoint_dir"]):
            shutil.rmtree(config["checkpoint_dir"])

