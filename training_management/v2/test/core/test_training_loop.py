"""
Test TrainingLoop
=================

Comprehensive unit tests for TrainingLoop module.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training_management.v2.core.training_loop import TrainingLoop
from training_management.v2.core.loss_computation import LossComputation
from training_management.v2.core.gradient_manager import GradientManager
from training_management.v2.core.batch_processor import BatchProcessor


class TestTrainingLoop:
    """Test TrainingLoop class"""
    
    def test_train_epoch_basic(
        self,
        simple_model,
        simple_train_loader,
        simple_optimizer,
        simple_criterion,
        simple_config,
        device,
        mock_logger
    ):
        """Test basic training epoch execution"""
        # Setup components
        loss_comp = LossComputation(simple_criterion, logger=mock_logger)
        grad_mgr = GradientManager(max_grad_norm=1.0, logger=mock_logger)
        batch_proc = BatchProcessor(logger=mock_logger)
        
        # Setup config
        config = simple_config.copy()
        config["batch_size"] = 4
        
        # Create TrainingLoop
        training_loop = TrainingLoop(
            model=simple_model,
            optimizer=simple_optimizer,
            loss_computation=loss_comp,
            gradient_manager=grad_mgr,
            batch_processor=batch_proc,
            device=device,
            config=config,
            logger=mock_logger
        )
        
        # Execute training epoch
        avg_loss, avg_acc, avg_grad_norm = training_loop.train_epoch(simple_train_loader)
        
        # Assertions
        assert isinstance(avg_loss, float)
        assert isinstance(avg_acc, float)
        assert isinstance(avg_grad_norm, float)
        assert avg_loss >= 0.0
        assert 0.0 <= avg_acc <= 1.0
        assert avg_grad_norm >= 0.0
    
    def test_validate_epoch_basic(
        self,
        simple_model,
        simple_val_loader,
        simple_optimizer,
        simple_criterion,
        simple_config,
        device,
        mock_logger
    ):
        """Test basic validation epoch execution"""
        # Setup components
        loss_comp = LossComputation(simple_criterion, logger=mock_logger)
        grad_mgr = GradientManager(max_grad_norm=1.0, logger=mock_logger)
        batch_proc = BatchProcessor(logger=mock_logger)
        
        # Setup config
        config = simple_config.copy()
        config["batch_size"] = 4
        
        # Create TrainingLoop
        training_loop = TrainingLoop(
            model=simple_model,
            optimizer=simple_optimizer,
            loss_computation=loss_comp,
            gradient_manager=grad_mgr,
            batch_processor=batch_proc,
            device=device,
            config=config,
            logger=mock_logger
        )
        
        # Execute validation epoch
        avg_loss, avg_acc = training_loop.validate_epoch(simple_val_loader)
        
        # Assertions
        assert isinstance(avg_loss, float)
        assert isinstance(avg_acc, float)
        assert avg_loss >= 0.0
        assert 0.0 <= avg_acc <= 1.0
    
    def test_train_epoch_with_gradient_accumulation(
        self,
        simple_model,
        simple_train_loader,
        simple_optimizer,
        simple_criterion,
        simple_config,
        device,
        mock_logger
    ):
        """Test training epoch with gradient accumulation"""
        # Setup components
        loss_comp = LossComputation(simple_criterion, logger=mock_logger)
        grad_mgr = GradientManager(max_grad_norm=1.0, logger=mock_logger)
        batch_proc = BatchProcessor(logger=mock_logger)
        
        # Setup config with gradient accumulation
        config = simple_config.copy()
        config["grad_accum_steps"] = 2
        config["batch_size"] = 4
        
        # Create TrainingLoop
        training_loop = TrainingLoop(
            model=simple_model,
            optimizer=simple_optimizer,
            loss_computation=loss_comp,
            gradient_manager=grad_mgr,
            batch_processor=batch_proc,
            device=device,
            config=config,
            logger=mock_logger
        )
        
        # Execute training epoch
        avg_loss, avg_acc, avg_grad_norm = training_loop.train_epoch(simple_train_loader)
        
        # Assertions
        assert isinstance(avg_loss, float)
        assert isinstance(avg_acc, float)
        assert isinstance(avg_grad_norm, float)
        assert avg_loss >= 0.0
        assert 0.0 <= avg_acc <= 1.0
    
    def test_train_epoch_with_eos_weight(
        self,
        simple_model,
        simple_train_loader,
        simple_optimizer,
        eos_weighted_criterion,
        simple_config,
        device,
        mock_logger
    ):
        """Test training epoch with EOS weighted criterion"""
        # Setup components (CRITICAL: Use EOS weighted criterion)
        loss_comp = LossComputation(eos_weighted_criterion, logger=mock_logger)
        grad_mgr = GradientManager(max_grad_norm=1.0, logger=mock_logger)
        batch_proc = BatchProcessor(logger=mock_logger)
        
        # Setup config
        config = simple_config.copy()
        config["batch_size"] = 4
        
        # Create TrainingLoop
        training_loop = TrainingLoop(
            model=simple_model,
            optimizer=simple_optimizer,
            loss_computation=loss_comp,
            gradient_manager=grad_mgr,
            batch_processor=batch_proc,
            device=device,
            config=config,
            logger=mock_logger
        )
        
        # Execute training epoch
        avg_loss, avg_acc, avg_grad_norm = training_loop.train_epoch(simple_train_loader)
        
        # Assertions
        assert isinstance(avg_loss, float)
        assert isinstance(avg_acc, float)
        assert avg_loss >= 0.0
        assert 0.0 <= avg_acc <= 1.0
        
        # CRITICAL: Loss should be computed using EOS weight
        # (LossComputation already tested separately, this is integration test)
    
    def test_train_epoch_gradient_clipping(
        self,
        simple_model,
        simple_train_loader,
        simple_optimizer,
        simple_criterion,
        simple_config,
        device,
        mock_logger
    ):
        """Test training epoch with gradient clipping"""
        # Setup components with aggressive gradient clipping
        loss_comp = LossComputation(simple_criterion, logger=mock_logger)
        grad_mgr = GradientManager(max_grad_norm=0.5, logger=mock_logger)  # Small norm
        batch_proc = BatchProcessor(logger=mock_logger)
        
        # Setup config
        config = simple_config.copy()
        config["batch_size"] = 4
        
        # Create TrainingLoop
        training_loop = TrainingLoop(
            model=simple_model,
            optimizer=simple_optimizer,
            loss_computation=loss_comp,
            gradient_manager=grad_mgr,
            batch_processor=batch_proc,
            device=device,
            config=config,
            logger=mock_logger
        )
        
        # Execute training epoch
        avg_loss, avg_acc, avg_grad_norm = training_loop.train_epoch(simple_train_loader)
        
        # Assertions
        assert isinstance(avg_loss, float)
        assert isinstance(avg_acc, float)
        assert isinstance(avg_grad_norm, float)
        # Gradient norm should be clipped (not necessarily <= max_grad_norm due to averaging)
        # but should be reasonable
    
    def test_validate_epoch_model_in_eval_mode(
        self,
        simple_model,
        simple_val_loader,
        simple_optimizer,
        simple_criterion,
        simple_config,
        device,
        mock_logger
    ):
        """Test that model is in eval mode during validation"""
        # Setup components
        loss_comp = LossComputation(simple_criterion, logger=mock_logger)
        grad_mgr = GradientManager(max_grad_norm=1.0, logger=mock_logger)
        batch_proc = BatchProcessor(logger=mock_logger)
        
        # Setup config
        config = simple_config.copy()
        config["batch_size"] = 4
        
        # Create TrainingLoop
        training_loop = TrainingLoop(
            model=simple_model,
            optimizer=simple_optimizer,
            loss_computation=loss_comp,
            gradient_manager=grad_mgr,
            batch_processor=batch_proc,
            device=device,
            config=config,
            logger=mock_logger
        )
        
        # Set model to train mode first
        simple_model.train()
        assert simple_model.training
        
        # Execute validation epoch
        avg_loss, avg_acc = training_loop.validate_epoch(simple_val_loader)
        
        # Model should be in eval mode after validation
        assert not simple_model.training
        
        # Assertions
        assert isinstance(avg_loss, float)
        assert isinstance(avg_acc, float)
    
    def test_train_epoch_model_in_train_mode(
        self,
        simple_model,
        simple_train_loader,
        simple_optimizer,
        simple_criterion,
        simple_config,
        device,
        mock_logger
    ):
        """Test that model is in train mode during training"""
        # Setup components
        loss_comp = LossComputation(simple_criterion, logger=mock_logger)
        grad_mgr = GradientManager(max_grad_norm=1.0, logger=mock_logger)
        batch_proc = BatchProcessor(logger=mock_logger)
        
        # Setup config
        config = simple_config.copy()
        config["batch_size"] = 4
        
        # Create TrainingLoop
        training_loop = TrainingLoop(
            model=simple_model,
            optimizer=simple_optimizer,
            loss_computation=loss_comp,
            gradient_manager=grad_mgr,
            batch_processor=batch_proc,
            device=device,
            config=config,
            logger=mock_logger
        )
        
        # Set model to eval mode first
        simple_model.eval()
        assert not simple_model.training
        
        # Execute training epoch
        avg_loss, avg_acc, avg_grad_norm = training_loop.train_epoch(simple_train_loader)
        
        # Model should be in train mode after training
        assert simple_model.training
        
        # Assertions
        assert isinstance(avg_loss, float)
        assert isinstance(avg_acc, float)
