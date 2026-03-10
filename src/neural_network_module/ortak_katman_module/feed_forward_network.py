# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: feed_forward_network.py
Modül: src/neural_network_module/ortak_katman_module
Görev: Feed Forward Network - Transformer standardı: 2-layer MLP (expand →
       contract). Endüstri standardı: GPT-2/3/4, BERT, T5. SwiGLU activation
       desteği (GPT-4, PaLM standardı). Activation parametresi ile genişletilebilir.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (FFN işlemleri),
                     Open/Closed (Activation parametresi ile genişletilebilir),
                     Dependency Inversion (nn.Module abstraction'ına bağımlı)
- Design Patterns: Network Pattern (feed-forward network)
- Endüstri Standartları: GPT-2/3/4, BERT, T5 FFN standardı

KULLANIM:
- Feed-forward network oluşturmak için
- MLP katmanı için
- SwiGLU activation için

BAĞIMLILIKLAR:
- torch.nn: Linear modülü
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
import logging


class FeedForwardNetwork(nn.Module):
    """
    Transformer standardı: 2-layer MLP (expand → contract)
    Endüstri standardı: GPT-2/3/4, BERT, T5
    
    Yapı:
    Input: [B, T, embed_dim]
      ↓ Linear (expand)
    [B, T, ffn_dim] (genellikle 4x embed_dim)
      ↓ Activation (GELU)
    [B, T, ffn_dim]
      ↓ Dropout
    [B, T, ffn_dim]
      ↓ Linear (contract)
    [B, T, embed_dim]
    """
    
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        log_level: int = logging.INFO,
    ):
        """
        Args:
            embed_dim: Embedding dimension (input/output)
            ffn_dim: Feed-forward network dimension (expand size)
            dropout: Dropout rate
            activation: Activation function ("gelu", "relu", veya "swiglu")
                [OK] V4: "swiglu" eklendi (GPT-4, PaLM standardı)
            log_level: Logging level
        """
        super().__init__()
        
        # [OK] ENDÜSTRİ STANDARDI: Parametre validasyonu (GPT-2/3/4, BERT, T5)
        if not isinstance(embed_dim, int) or embed_dim <= 0:
            raise ValueError(f"embed_dim ({embed_dim}) pozitif bir tamsayı olmalıdır.")
        if not isinstance(ffn_dim, int) or ffn_dim <= 0:
            raise ValueError(f"ffn_dim ({ffn_dim}) pozitif bir tamsayı olmalıdır.")
        if not isinstance(dropout, (int, float)) or not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout ({dropout}) 0.0 ile 1.0 arasında olmalıdır.")
        
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout
        self.activation_name = activation
        
        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Katmanlar
        # [OK] V4: SwiGLU desteği (GPT-4, PaLM standardı)
        if activation.lower() == "swiglu":
            # SwiGLU için iki gate kullanılır: gate_proj ve up_proj
            # SwiGLU(x) = Swish(gate_proj(x)) ⊙ up_proj(x)
            # Swish(x) = x * sigmoid(x)
            # ffn_dim 2/3'ü gate için, 1/3'ü up için kullanılır (PaLM standardı)
            # Ama genellikle her ikisi de ffn_dim boyutunda yapılır
            self.gate_proj = nn.Linear(embed_dim, ffn_dim)  # Gate projection
            self.up_proj = nn.Linear(embed_dim, ffn_dim)    # Up projection
            self.fc1 = None  # SwiGLU için kullanılmaz
            self.fc2 = nn.Linear(ffn_dim, embed_dim)  # Contract
            self.activation = None  # SwiGLU kendi activation'ını kullanır
            self.activation_name = "swiglu"
        else:
            # Standart activation'lar için
            self.fc1 = nn.Linear(embed_dim, ffn_dim)  # Expand
            self.fc2 = nn.Linear(ffn_dim, embed_dim)  # Contract
            self.gate_proj = None
            self.up_proj = None
            
            # Activation
            if activation.lower() == "gelu":
                self.activation = nn.GELU()
            elif activation.lower() == "relu":
                self.activation = nn.ReLU()
            else:
                raise ValueError(f"Unsupported activation: {activation}. Use 'gelu', 'relu', or 'swiglu'.")
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Endüstri standardı: Xavier initialization (GPT-2/3/4, BERT, T5)
        if self.activation_name == "swiglu":
            # SwiGLU için gate_proj ve up_proj initialize et
            nn.init.xavier_uniform_(self.gate_proj.weight)
            nn.init.xavier_uniform_(self.up_proj.weight)
            if self.gate_proj.bias is not None:
                nn.init.zeros_(self.gate_proj.bias)
            if self.up_proj.bias is not None:
                nn.init.zeros_(self.up_proj.bias)
        else:
            # Standart activation'lar için
            nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        if self.fc1 is not None and self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)
        
        self.logger.info(
            f"FeedForwardNetwork initialized: embed_dim={embed_dim}, "
            f"ffn_dim={ffn_dim}, dropout={dropout}, activation={activation}"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [B, T, embed_dim]
        
        Returns:
            x: [B, T, embed_dim]
        """
        # [OK] V4: SwiGLU activation (GPT-4, PaLM standardı)
        if self.activation_name == "swiglu":
            # SwiGLU(x) = Swish(gate_proj(x)) ⊙ up_proj(x)
            # Swish(x) = x * sigmoid(x)
            gate = self.gate_proj(x)  # [B, T, embed_dim] → [B, T, ffn_dim]
            up = self.up_proj(x)       # [B, T, embed_dim] → [B, T, ffn_dim]
            
            # Swish activation: gate * sigmoid(gate)
            swish_gate = gate * torch.sigmoid(gate)
            
            # Element-wise multiplication (gating)
            x = swish_gate * up  # [B, T, ffn_dim]
            
            # Dropout
            x = self.dropout(x)
            
            # Contract
            x = self.fc2(x)  # [B, T, ffn_dim] → [B, T, embed_dim]
        else:
            # Standart activation'lar (GELU, ReLU)
            # 1) Expand
            x = self.fc1(x)  # [B, T, embed_dim] → [B, T, ffn_dim]
            
            # 2) Activation
            x = self.activation(x)  # GELU veya ReLU
            
            # 3) Dropout
            x = self.dropout(x)
            
            # 4) Contract
            x = self.fc2(x)  # [B, T, ffn_dim] → [B, T, embed_dim]
        
        return x
    
    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, ffn_dim={self.ffn_dim}, "
            f"dropout={self.dropout_rate}, activation={self.activation_name}"
        )

