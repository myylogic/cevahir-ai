# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: advanced_checkpointing.py
Modül: src/neural_network_module/ortak_katman_module
Görev: Advanced Checkpointing - Gradient Checkpointing'in gelişmiş versiyonu.
       Selective checkpointing (sadece belirli layer'ları checkpoint'le),
       layer-wise checkpointing (her layer için ayrı checkpoint stratejisi) ve
       memory-efficient training için optimize edilmiş. Transformer
       standardı.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (checkpointing işlemleri),
                     Open/Closed (genişletilebilir),
                     Dependency Inversion (nn.Module abstraction'ına bağımlı)
- Design Patterns: Checkpointing Pattern (gelişmiş checkpointing)
- Endüstri Standartları: Transformer checkpointing standardı

KULLANIM:
- Advanced checkpointing için
- Selective checkpointing için
- Layer-wise checkpointing için

BAĞIMLILIKLAR:
- torch.utils.checkpoint: Checkpoint fonksiyonları

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
from typing import Optional, List, Callable, Any
from torch.utils.checkpoint import checkpoint
import logging


class AdvancedCheckpointing:
    """
    [OK] V4: Advanced Checkpointing
    Endüstri standardı: Transformer
    
    Selective ve layer-wise checkpointing stratejileri.
    
    Stratejiler:
    1. Selective: Sadece belirli layer'ları checkpoint'le
    2. Layer-wise: Her layer için ayrı strateji
    3. Adaptive: Tarihsel ad; ilk/son ve çift indeksli katmanları seçen sabit kural
    """
    
    def __init__(
        self,
        strategy: str = "selective",
        checkpoint_layers: Optional[List[int]] = None,
        checkpoint_every_n: int = 2,
        log_level: int = logging.INFO,
    ):
        """
        Args:
            strategy: Checkpointing stratejisi ("selective", "layer_wise", "adaptive")
            checkpoint_layers: Checkpoint'lenecek layer index'leri (selective için)
            checkpoint_every_n: Her N layer'da bir checkpoint (layer_wise için)
            log_level: Logging level
        """
        if not isinstance(strategy, str):
            raise ValueError("checkpoint strategy must be a string")
        self.strategy = strategy.strip().lower()
        if self.strategy not in ["selective", "layer_wise", "adaptive"]:
            raise ValueError(f"Desteklenmeyen checkpointing stratejisi: {strategy}. "
                             f"Geçerli seçenekler: 'selective', 'layer_wise', 'adaptive'.")
        if isinstance(checkpoint_every_n, bool) or not isinstance(checkpoint_every_n, int) or checkpoint_every_n <= 0:
            raise ValueError("checkpoint_every_n must be a positive integer")
        if checkpoint_layers is not None and any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in checkpoint_layers):
            raise ValueError("checkpoint_layers must contain non-negative integers")
        self.checkpoint_layers = list(checkpoint_layers) if checkpoint_layers is not None else []
        self.checkpoint_every_n = checkpoint_every_n
        
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
            f"[V4] Advanced Checkpointing initialized: strategy={strategy}, "
            f"checkpoint_layers={checkpoint_layers}, checkpoint_every_n={checkpoint_every_n}"
        )
    
    def should_checkpoint(
        self,
        layer_idx: int,
        total_layers: int,
        training: bool = True,
    ) -> bool:
        """
        Bu layer'ı checkpoint'lemeli mi?
        
        Args:
            layer_idx: Layer index (0-based)
            total_layers: Toplam layer sayısı
            training: Training modunda mı?
        
        Returns:
            True ise checkpoint'le, False ise normal forward
        """
        if not training:
            # Inference'da checkpoint yok
            return False
        
        if self.strategy == "selective":
            # Sadece belirtilen layer'ları checkpoint'le
            return layer_idx in self.checkpoint_layers
        
        elif self.strategy == "layer_wise":
            # Her N layer'da bir checkpoint
            return layer_idx % self.checkpoint_every_n == 0
        
        elif self.strategy == "adaptive":
            # İlk ve son layer'ları checkpoint'le, ortadakileri seçici
            if layer_idx == 0 or layer_idx == total_layers - 1:
                return True
            # Ortadaki layer'lar için her 2'de bir
            return layer_idx % 2 == 0
        
        else:
            # Default: Her layer'ı checkpoint'le (full checkpointing)
            return True
    
    def checkpoint_forward(
        self,
        func: Callable,
        *args,
        use_reentrant: bool = False,
        **kwargs,
    ) -> Any:
        """
        Checkpoint ile forward pass.
        
        Args:
            func: Forward function
            *args: Function arguments
            use_reentrant: Reentrant checkpointing kullan
            **kwargs: Function keyword arguments
        
        Returns:
            Function output
        """
        return checkpoint(func, *args, use_reentrant=use_reentrant, **kwargs)


def create_checkpointing_strategy(
    strategy: str = "selective",
    num_layers: int = 12,
    **kwargs,
) -> AdvancedCheckpointing:
    """
    Checkpointing stratejisi oluştur.
    
    Args:
        strategy: Strateji tipi
        num_layers: Toplam layer sayısı
        **kwargs: Ek parametreler
    
    Returns:
        AdvancedCheckpointing instance
    """
    if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers <= 0:
        raise ValueError("num_layers must be a positive integer")
    policy = AdvancedCheckpointing(strategy=strategy, **kwargs)
    if policy.strategy == "selective" and "checkpoint_layers" not in kwargs:
        policy.checkpoint_layers = sorted({0, num_layers - 1, *range(1, num_layers - 1, 2)})
    if any(i >= num_layers for i in policy.checkpoint_layers):
        raise ValueError("checkpoint layer index exceeds num_layers")
    return policy
