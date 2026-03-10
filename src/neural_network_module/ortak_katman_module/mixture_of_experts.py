# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: mixture_of_experts.py
Modül: src/neural_network_module/ortak_katman_module
Görev: Mixture of Experts (MoE) - Büyük modeller için sparse activation sağlar.
       N adet expert (FFN), router (her token için en uygun expert'leri seçer),
       top-k routing (her token için k expert seçilir) ve load balancing
       (expert'lerin dengeli kullanımı) işlemlerini yapar. GPT-4, Gemini, GShard
       standardı. Referans: "Outrageously Large Neural Networks" (2017),
       "GShard: Scaling Giant Models" (2020).

MİMARİ:
- SOLID Prensipleri: Single Responsibility (MoE işlemleri),
                     Open/Closed (genişletilebilir),
                     Dependency Inversion (nn.Module abstraction'ına bağımlı)
- Design Patterns: Expert Pattern (mixture of experts)
- Endüstri Standartları: GPT-4, Gemini, GShard MoE standardı

KULLANIM:
- MoE layer oluşturmak için
- Sparse activation için
- Büyük model kapasitesi için

BAĞIMLILIKLAR:
- torch.nn: Module base class
- torch.nn.functional: Activation fonksiyonları

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Tuple
import math


class Router(nn.Module):
    """
    [OK] V4: Expert Router
    Endüstri standardı: GPT-4, Gemini, GShard
    
    Her token için en uygun expert'leri seçer.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_experts: int,
        top_k: int = 1,
        log_level: int = logging.INFO,
    ):
        """
        Args:
            embed_dim: Embedding dimension
            num_experts: Expert sayısı
            top_k: Her token için seçilecek expert sayısı (genellikle 1 veya 2)
            log_level: Logging level
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router: [embed_dim] -> [num_experts]
        self.router = nn.Linear(embed_dim, num_experts, bias=False)
        
        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Xavier initialization
        nn.init.xavier_uniform_(self.router.weight)
        
        self.logger.info(
            f"[V4] Router initialized: embed_dim={embed_dim}, num_experts={num_experts}, top_k={top_k}"
        )
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Router forward pass.
        
        Args:
            x: [B, T, embed_dim] - Input tokens
        
        Returns:
            expert_weights: [B, T, top_k] - Expert weights (softmax)
            expert_indices: [B, T, top_k] - Expert indices
            load_balancing_loss: Scalar - Load balancing loss
        """
        B, T, D = x.shape
        
        # Router logits: [B, T, num_experts]
        router_logits = self.router(x)  # [B, T, num_experts]
        
        # Top-k expert selection
        expert_weights, expert_indices = torch.topk(
            router_logits, 
            k=self.top_k, 
            dim=-1
        )  # [B, T, top_k]
        
        # Softmax over top-k experts
        expert_weights = F.softmax(expert_weights, dim=-1)  # [B, T, top_k]
        
        # Load balancing loss (expert diversity için)
        # Her expert'in kullanım oranını dengeler
        router_probs = F.softmax(router_logits, dim=-1)  # [B, T, num_experts]
        expert_usage = router_probs.mean(dim=[0, 1])  # [num_experts] - Her expert'in ortalama kullanımı
        load_balancing_loss = self.num_experts * (expert_usage ** 2).sum()  # Diversity loss
        
        return expert_weights, expert_indices, load_balancing_loss


class MixtureOfExperts(nn.Module):
    """
    [OK] V4: Mixture of Experts (MoE)
    Endüstri standardı: GPT-4, Gemini, GShard
    
    N adet expert (FFN) ve router ile sparse activation.
    """
    
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        activation: str = "swiglu",
        log_level: int = logging.INFO,
    ):
        """
        Args:
            embed_dim: Embedding dimension
            ffn_dim: FFN dimension (her expert için)
            num_experts: Expert sayısı (GPT-4: 8, Gemini: 16)
            top_k: Her token için seçilecek expert sayısı (GPT-4: 2)
            dropout: Dropout rate
            activation: Activation function ("gelu", "relu", "swiglu")
            log_level: Logging level
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router
        self.router = Router(embed_dim, num_experts, top_k, log_level)
        
        # Experts: N adet FFN
        from .feed_forward_network import FeedForwardNetwork
        self.experts = nn.ModuleList([
            FeedForwardNetwork(
                embed_dim=embed_dim,
                ffn_dim=ffn_dim,
                dropout=dropout,
                activation=activation,
                log_level=log_level,
            ) for _ in range(num_experts)
        ])
        
        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.logger.info(
            f"[V4] MixtureOfExperts initialized: embed_dim={embed_dim}, ffn_dim={ffn_dim}, "
            f"num_experts={num_experts}, top_k={top_k}, activation={activation}"
        )
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MoE forward pass.
        
        Args:
            x: [B, T, embed_dim] - Input tokens
        
        Returns:
            output: [B, T, embed_dim] - MoE output
            load_balancing_loss: Scalar - Load balancing loss
        """
        B, T, D = x.shape
        
        # Router: Expert selection
        expert_weights, expert_indices, load_balancing_loss = self.router(x)
        # expert_weights: [B, T, top_k]
        # expert_indices: [B, T, top_k]
        
        # Initialize output
        output = torch.zeros_like(x)  # [B, T, embed_dim]
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_mask = (expert_indices == expert_idx)  # [B, T, top_k]
            
            if expert_mask.any():
                # Get expert output
                expert_output = self.experts[expert_idx](x)  # [B, T, embed_dim]
                
                # Weighted sum over top_k experts
                for k in range(self.top_k):
                    mask = expert_mask[:, :, k]  # [B, T]
                    weight = expert_weights[:, :, k]  # [B, T]
                    
                    # Add weighted expert output
                    output += expert_output * mask.unsqueeze(-1) * weight.unsqueeze(-1)
        
        return output, load_balancing_loss

