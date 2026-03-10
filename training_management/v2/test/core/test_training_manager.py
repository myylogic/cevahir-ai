"""
Test TrainingManager (Facade)
==============================

Comprehensive unit and integration tests for TrainingManager module.
"""

import pytest
import torch
import torch.nn as nn

from training_management.v2.core.training_manager import TrainingManager


class TestTrainingManager:
    """Test TrainingManager class (Facade)"""
    
    def test_initialization(
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
        """Test TrainingManager initialization"""
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
        
        # Assertions: Components initialized
        assert tm.model == simple_model
        assert tm.train_loader == simple_train_loader
        assert tm.val_loader == simple_val_loader
        assert tm.optimizer == simple_optimizer
        assert tm.config == config
        
        # Assertions: Sub-components initialized
        assert tm.batch_processor is not None
        assert tm.loss_computation is not None
        assert tm.gradient_manager is not None
        assert tm.training_loop is not None
    
    def test_initialization_device_setup(
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
        """Test TrainingManager device setup"""
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
        
        # Assertions: Device setup
        assert tm.device == device
        # Model should be on correct device
        assert next(tm.model.parameters()).device == device
    
    def test_initialization_with_eos_weight(
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
        """Test TrainingManager initialization with EOS weighted criterion"""
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
        
        # Assertions: Components initialized
        assert tm.loss_computation.criterion == eos_weighted_criterion
        
        # CRITICAL: LossComputation should use the criterion with EOS weight
        # (This is already tested in LossComputation tests, this is integration test)
    
    def test_initialization_config_parameters(
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
        """Test TrainingManager initialization with various config parameters"""
        # Setup config with various parameters
        config = simple_config.copy()
        config["device"] = device
        config["batch_size"] = 4
        config["max_grad_norm"] = 0.5
        config["grad_accum_steps"] = 2
        config["use_amp"] = False
        config["pad_token_id"] = 0
        
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
        
        # Assertions: Config parameters passed to sub-components
        assert tm.gradient_manager.max_grad_norm == 0.5
        assert tm.training_loop.grad_accum_steps == 2
        assert tm.training_loop.pad_token_id == 0
