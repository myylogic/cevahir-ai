"""
Test TrainingManager Full Integration
======================================

End-to-end integration tests for TrainingManager.
Tests the complete training workflow.
"""

import pytest
import torch
import torch.nn as nn

from training_management.v2.core.training_manager import TrainingManager


class TestTrainingManagerFullIntegration:
    """End-to-end integration tests for TrainingManager"""
    
    def test_training_manager_train_method_implemented(
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
        """Test that train() method is now implemented"""
        # Setup config
        config = simple_config.copy()
        config["device"] = device
        config["batch_size"] = 4
        config["epochs"] = 1  # Single epoch for quick test
        config["vocab_size"] = 128
        config["checkpoint_dir"] = "./test_checkpoints"
        
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
        
        # Train method should execute successfully
        final_train_loss, final_val_loss = tm.train()
        
        # Assertions
        assert isinstance(final_train_loss, float)
        assert isinstance(final_val_loss, float)
        assert final_train_loss >= 0.0
        assert final_val_loss >= 0.0
        
        # Training history should be populated
        history = tm.get_training_history()
        assert "train_loss" in history
        assert "val_loss" in history
        assert "accuracy" in history
    
    def test_training_manager_can_execute_training_loop_directly(
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
        """Test that TrainingManager can execute training loop through training_loop"""
        # Setup config
        config = simple_config.copy()
        config["device"] = device
        config["batch_size"] = 4
        
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
        
        # Execute training epoch through training_loop
        avg_loss, avg_acc, avg_grad_norm = tm.training_loop.train_epoch(tm.train_loader)
        
        # Assertions
        assert isinstance(avg_loss, float)
        assert isinstance(avg_acc, float)
        assert isinstance(avg_grad_norm, float)
        assert avg_loss >= 0.0
        assert 0.0 <= avg_acc <= 1.0
    
    def test_training_manager_can_execute_validation_loop_directly(
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
        """Test that TrainingManager can execute validation loop through training_loop"""
        # Setup config
        config = simple_config.copy()
        config["device"] = device
        config["batch_size"] = 4
        
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
        
        # Execute validation epoch through training_loop
        avg_loss, avg_acc = tm.training_loop.validate_epoch(tm.val_loader)
        
        # Assertions
        assert isinstance(avg_loss, float)
        assert isinstance(avg_acc, float)
        assert avg_loss >= 0.0
        assert 0.0 <= avg_acc <= 1.0
    
    def test_training_manager_eos_weight_propagation(
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
        """Test that EOS weight is correctly propagated to LossComputation"""
        # Setup config
        config = simple_config.copy()
        config["device"] = device
        config["batch_size"] = 4
        
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
        
        # CRITICAL: LossComputation should use the EOS weighted criterion
        assert tm.loss_computation.criterion == eos_weighted_criterion
        
        # Verify EOS weight is present in criterion
        assert hasattr(eos_weighted_criterion, 'weight')
        assert eos_weighted_criterion.weight is not None
        
        # Execute training epoch to verify EOS weight is used
        avg_loss, avg_acc, avg_grad_norm = tm.training_loop.train_epoch(tm.train_loader)
        
        # Assertions
        assert isinstance(avg_loss, float)
        assert avg_loss >= 0.0
        # EOS weight should affect loss (integration test)

