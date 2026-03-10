# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: batch_processor.py
Modül: training_management/v2/core
Görev: Batch Processor - Batch parsing ve processing utilities. Farklı batch
       formatlarını (tuple, dict, tensor) parse eder, batch shape ve type
       validasyonu yapar, edge case'leri (boş batch, malformed batch) yönetir.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (batch parsing ve validation),
                     Open/Closed (strategy pattern ile genişletilebilir),
                     Dependency Inversion (abstraksiyonlara bağımlı)
- Design Patterns: Strategy Pattern (farklı batch formatları için)
- Endüstri Standartları: Batch processing best practices

KULLANIM:
- Batch parsing için
- Batch validation için
- Farklı batch formatlarını işlemek için

BAĞIMLILIKLAR:
- torch: Tensor işlemleri

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from typing import Tuple, Optional, Dict, Any, Union, List
import torch


class BatchProcessor:
    """
    Batch parsing and processing utility.
    
    Responsibilities:
    - Parse different batch formats (tuple, dict, tensor)
    - Validate batch shapes and types
    - Handle edge cases (empty batches, malformed batches)
    
    SOLID: Single Responsibility Principle
    """
    
    def __init__(self, logger: Optional[Any] = None):
        """
        Initialize BatchProcessor.
        
        Args:
            logger: Optional logger instance for error logging
        """
        self.logger = logger
    
    def parse_batch(
        self, 
        batch: Union[torch.Tensor, Tuple, Dict, List],
        allow_missing_target: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parse batch into (inputs, targets) tuple.
        
        Supported formats:
        - Tensor -> (inputs, inputs) [LM default]
        - (inputs, targets) tuple/list
        - dict: {'input_ids'|'inputs'|'x', 'labels'|'targets'|'y'}
        - list of (inp, tgt): torch.stack ile birleştirilir
        
        Args:
            batch: Batch data in various formats
            allow_missing_target: If True, targets can be None
            
        Returns:
            Tuple of (inputs, targets) tensors
            
        Raises:
            ValueError: If batch format cannot be parsed
            TypeError: If batch types are invalid
        """
        inputs: Optional[torch.Tensor] = None
        targets: Optional[torch.Tensor] = None
        
        # Format 1: Tensor (LM default)
        if isinstance(batch, torch.Tensor):
            inputs = batch
            if not allow_missing_target:
                targets = batch
            return inputs, targets
        
        # Format 2: Tuple or List of two tensors
        if isinstance(batch, (tuple, list)):
            if len(batch) == 2 and all(isinstance(b, torch.Tensor) for b in batch):
                return batch[0], batch[1]
            
            # Format 3: List of tuples (list of (inp, tgt) pairs)
            if len(batch) > 0 and all(
                isinstance(b, (tuple, list)) and len(b) == 2 
                for b in batch
            ):
                inps = [b[0] for b in batch]
                tgts = [b[1] for b in batch]
                if all(isinstance(t, torch.Tensor) for t in inps) and \
                   all(isinstance(t, torch.Tensor) for t in tgts):
                    return torch.stack(inps, dim=0), torch.stack(tgts, dim=0)
            
            # Format 4: Single tensor in list
            if len(batch) == 1 and isinstance(batch[0], torch.Tensor):
                inputs = batch[0]
                if not allow_missing_target:
                    targets = inputs
                return inputs, targets
        
        # Format 5: Dictionary
        if isinstance(batch, dict):
            # Try input keys
            for k in ("input_ids", "inputs", "x"):
                if k in batch and isinstance(batch[k], torch.Tensor):
                    inputs = batch[k]
                    break
            
            # Try target keys
            for k in ("labels", "targets", "y"):
                if k in batch and isinstance(batch[k], torch.Tensor):
                    targets = batch[k]
                    break
            
            if inputs is not None:
                if targets is None and not allow_missing_target:
                    targets = inputs
                return inputs, targets
        
        # If we get here, batch format is unsupported
        error_msg = f"Batch formatı çözümlenemedi: type={type(batch)}"
        if self.logger:
            self.logger.log_error(error_msg)
        raise ValueError(error_msg)
    
    def validate_batch(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor],
        expected_input_dim: int = 2,
        expected_target_dim: Optional[int] = None,
    ) -> bool:
        """
        Validate batch shapes and types.
        
        Args:
            inputs: Input tensor
            targets: Target tensor (optional)
            expected_input_dim: Expected input tensor dimension
            expected_target_dim: Expected target tensor dimension
            
        Returns:
            True if valid, raises ValueError otherwise
            
        Raises:
            ValueError: If batch is invalid
            TypeError: If types are invalid
        """
        if not isinstance(inputs, torch.Tensor):
            raise TypeError(f"Inputs must be Tensor, got {type(inputs)}")
        
        if inputs.dim() < expected_input_dim:
            raise ValueError(
                f"Inputs dimension must be >= {expected_input_dim}, "
                f"got {inputs.dim()}"
            )
        
        if targets is not None:
            if not isinstance(targets, torch.Tensor):
                raise TypeError(f"Targets must be Tensor, got {type(targets)}")
            
            if expected_target_dim is not None:
                if targets.dim() != expected_target_dim:
                    raise ValueError(
                        f"Targets dimension must be {expected_target_dim}, "
                        f"got {targets.dim()}"
                    )
            
            # Check batch size match
            if inputs.shape[0] != targets.shape[0]:
                raise ValueError(
                    f"Batch size mismatch: inputs={inputs.shape[0]}, "
                    f"targets={targets.shape[0]}"
                )
        
        return True

