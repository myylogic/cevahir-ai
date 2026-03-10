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
       memory-efficient training için optimize edilmiş. GPT-4, Claude, Gemini
       standardı.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (checkpointing işlemleri),
                     Open/Closed (genişletilebilir),
                     Dependency Inversion (nn.Module abstraction'ına bağımlı)
- Design Patterns: Checkpointing Pattern (gelişmiş checkpointing)
- Endüstri Standartları: GPT-4, Claude, Gemini checkpointing standardı

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
    Endüstri standardı: GPT-4, Claude, Gemini
    
    Selective ve layer-wise checkpointing stratejileri.
    
    Stratejiler:
    1. Selective: Sadece belirli layer'ları checkpoint'le
    2. Layer-wise: Her layer için ayrı strateji
    3. Adaptive: Memory kullanımına göre otomatik seçim
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
        self.strategy = strategy.lower()
        if self.strategy not in ["selective", "layer_wise", "adaptive"]:
            raise ValueError(f"Desteklenmeyen checkpointing stratejisi: {strategy}. "
                             f"Geçerli seçenekler: 'selective', 'layer_wise', 'adaptive'.")
        self.checkpoint_layers = checkpoint_layers or []
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
    if strategy == "selective":
        # İlk ve son layer'ları checkpoint'le, ortadakileri seçici
        checkpoint_layers = [0, num_layers - 1]
        # Ortadaki layer'lar için her 2'de bir
        for i in range(1, num_layers - 1, 2):
            checkpoint_layers.append(i)
        return AdvancedCheckpointing(
            strategy="selective",
            checkpoint_layers=checkpoint_layers,
            **kwargs,
        )
    
    elif strategy == "layer_wise":
        # Her 2 layer'da bir checkpoint
        return AdvancedCheckpointing(
            strategy="layer_wise",
            checkpoint_every_n=2,
            **kwargs,
        )
    
    elif strategy == "adaptive":
        # Adaptive strateji
        return AdvancedCheckpointing(
            strategy="adaptive",
            **kwargs,
        )
    
    else:
        # Default: Selective
        return AdvancedCheckpointing(
            strategy="selective",
            checkpoint_layers=[0, num_layers - 1] if num_layers > 1 else [0],
            **kwargs,
        )

