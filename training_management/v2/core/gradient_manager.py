# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: gradient_manager.py
Modül: training_management/v2/core
Görev: Gradient Manager - Gradient calculation, clipping ve accumulation işlemleri.
       Gradient norm hesaplama, gradient clipping, gradient accumulation ve
       gradient sorunlarını (NaN, Inf, explosion) tespit etme işlemlerini yapar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (gradient yönetimi),
                     Dependency Inversion (model abstraksiyonuna bağımlı)
- Design Patterns: Manager Pattern (gradient yönetimi)
- Endüstri Standartları: Gradient clipping ve accumulation best practices

KULLANIM:
- Gradient norm hesaplama için
- Gradient clipping için
- Gradient accumulation için
- Gradient sorunlarını tespit etmek için

BAĞIMLILIKLAR:
- torch: Gradient işlemleri

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from typing import Optional, Any
import torch


class GradientManager:
    """
    Gradient management utility.
    
    Responsibilities:
    - Calculate gradient norms
    - Clip gradients
    - Handle gradient accumulation
    - Detect gradient issues (NaN, Inf, explosion)
    
    SOLID: Single Responsibility Principle
    """
    
    def __init__(
        self,
        max_grad_norm: float = 1.0,
        logger: Optional[Any] = None
    ):
        """
        Initialize GradientManager.
        
        Args:
            max_grad_norm: Maximum gradient norm for clipping
            logger: Optional logger instance
        """
        self.max_grad_norm = max_grad_norm
        self.logger = logger
    
    def calculate_gradient_norm(
        self,
        model: torch.nn.Module
    ) -> float:
        """
        Calculate total gradient norm (L2 norm).
        
        Args:
            model: Model with gradients computed
            
        Returns:
            Total gradient norm as float
        """
        total_norm = 0.0
        num_params = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                try:
                    param_norm = param.grad.norm(2).item()
                    
                    # Check for NaN/Inf
                    if not (torch.isfinite(torch.tensor(param_norm)) and param_norm > 0):
                        if self.logger:
                            self.logger.log_warning(
                                f"Parameter '{name}' gradient norm invalid: "
                                f"{param_norm} (inf/nan/zero)"
                            )
                        continue
                    
                    total_norm += param_norm ** 2
                    num_params += 1
                    
                except Exception as e:
                    if self.logger:
                        self.logger.log_error(
                            f"Error calculating gradient norm for '{name}': {e}"
                        )
                    continue
        
        total_norm = total_norm ** 0.5
        avg_norm = total_norm / num_params if num_params > 0 else 0.0
        
        # Final validation
        if not (torch.isfinite(torch.tensor(avg_norm)) and avg_norm > 0):
            if self.logger:
                self.logger.log_warning(
                    f"Average gradient norm invalid: {avg_norm}, returning 0.0"
                )
            return 0.0
        
        return float(avg_norm)
    
    def clip_gradients(
        self,
        model: torch.nn.Module
    ) -> Optional[float]:
        """
        Clip gradients to max_grad_norm.
        
        Args:
            model: Model with gradients computed
            
        Returns:
            Total gradient norm before clipping (or None if no gradients)
        """
        # Calculate current norm
        total_norm = 0.0
        has_gradients = False
        
        for param in model.parameters():
            if param.grad is not None:
                has_gradients = True
                param_norm = param.grad.norm(2).item()
                total_norm += param_norm ** 2
        
        if not has_gradients:
            return None
        
        total_norm = total_norm ** 0.5
        # NaN/Inf ise genelde mimari veya bir batch'te sayısal patlama (log(0), overflow)
        if not (total_norm == total_norm and abs(total_norm) != float("inf")):
            if self.logger:
                self.logger.log_warning(
                    "Gradient norm NaN/Inf — bu batch atlanıyor (muhtemel: mimari sayısal kararsızlık veya bozuk batch)"
                )
            return None
        total_norm = float(total_norm)
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        
        # Clip if norm exceeds threshold
        if clip_coef < 1.0:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.mul_(clip_coef)
        
        return total_norm
    
    def detect_gradient_issues(
        self,
        model: torch.nn.Module
    ) -> dict:
        """
        Detect gradient issues (NaN, Inf, explosion).
        
        Args:
            model: Model with gradients computed
            
        Returns:
            Dictionary with issue detection results:
            {
                "has_nan": bool,
                "has_inf": bool,
                "has_explosion": bool,
                "max_grad_value": float,
                "total_norm": float
            }
        """
        has_nan = False
        has_inf = False
        max_grad_value = 0.0
        total_norm_sq = 0.0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                
                # Check for NaN
                if not has_nan and torch.isnan(grad).any():
                    has_nan = True
                    if self.logger:
                        self.logger.log_error(f"NaN detected in '{name}' gradients")
                
                # Check for Inf
                if not has_inf and torch.isinf(grad).any():
                    has_inf = True
                    if self.logger:
                        self.logger.log_error(f"Inf detected in '{name}' gradients")
                
                # Track max value and norm
                max_val = grad.abs().max().item()
                max_grad_value = max(max_grad_value, max_val)
                total_norm_sq += grad.norm(2).item() ** 2
        
        total_norm = total_norm_sq ** 0.5
        has_explosion = total_norm > self.max_grad_norm * 10.0  # 10x threshold
        
        return {
            "has_nan": has_nan,
            "has_inf": has_inf,
            "has_explosion": has_explosion,
            "max_grad_value": max_grad_value,
            "total_norm": total_norm
        }

