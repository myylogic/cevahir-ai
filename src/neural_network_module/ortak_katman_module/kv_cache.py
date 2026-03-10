# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: kv_cache.py
Modül: src/neural_network_module/ortak_katman_module
Görev: KV Cache (Key-Value Cache) - Autoregressive generation sırasında key ve
       value'ları cache'ler. Her yeni token için sadece yeni token'ın key/value'sunu
       hesaplar, önceki token'ların key/value'larını tekrar hesaplamadan cache'den
       kullanır. GPT-4, Claude, Gemini standardı. Memory-efficient ve hızlı
       inference sağlar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (KV cache işlemleri),
                     Open/Closed (genişletilebilir),
                     Dependency Inversion (nn.Module abstraction'ına bağımlı)
- Design Patterns: Cache Pattern (key-value cache)
- Endüstri Standartları: GPT-4, Claude, Gemini KV cache standardı

KULLANIM:
- KV cache oluşturmak için
- Autoregressive generation için
- Inference optimizasyonu için

BAĞIMLILIKLAR:
- torch.nn: Module base class

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
import logging
from typing import Optional, Tuple, Dict, Any


class KVCache:
    """
    [OK] V4: KV Cache (Key-Value Cache)
    Endüstri standardı: GPT-4, Claude, Gemini
    
    Autoregressive generation için key ve value'ları cache'ler.
    Her layer için ayrı cache tutulur.
    
    Cache yapısı:
    - key_cache: [B, num_heads, cache_len, head_dim]
    - value_cache: [B, num_heads, cache_len, head_dim]
    
    Kullanım:
    1. İlk forward: Tüm sequence için key/value hesapla ve cache'le
    2. Sonraki forward'lar: Sadece yeni token için key/value hesapla, cache'e ekle
    3. Attention: Cached key/value + new key/value concatenate et
    """
    
    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        max_cache_len: int = 2048,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        log_level: int = logging.INFO,
    ):
        """
        Args:
            batch_size: Batch size
            num_heads: Number of attention heads
            head_dim: Dimension per head
            max_cache_len: Maximum cache length
            device: Device for cache tensors
            dtype: Data type for cache tensors
            log_level: Logging level
        """
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_cache_len = max_cache_len
        self.device = device or torch.device("cpu")
        self.dtype = dtype or torch.float32
        
        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Cache storage: [B, H, cache_len, D]
        self.key_cache: Optional[torch.Tensor] = None
        self.value_cache: Optional[torch.Tensor] = None
        self.cache_len: int = 0
        self._initialized: bool = False  # [OK] V4: Initialization flag
        
        self.logger.info(
            f"[V4] KVCache initialized: batch_size={batch_size}, num_heads={num_heads}, "
            f"head_dim={head_dim}, max_cache_len={max_cache_len}"
        )
    
    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cache'i güncelle ve concatenated key/value döndür.
        
        Args:
            key: [B, H, new_len, D] - Yeni key tensor
            value: [B, H, new_len, D] - Yeni value tensor
            cache_position: [new_len] - Cache'deki pozisyonlar (None ise append)
        
        Returns:
            key_concat: [B, H, total_len, D] - Cached + new key
            value_concat: [B, H, total_len, D] - Cached + new value
        """
        B, H, new_len, D = key.shape
        
        # Batch size'ı ilk forward'da belirle (dinamik batch size desteği)
        if not self._initialized:
            self.batch_size = B
            self._initialized = True
        
        # İlk forward: Cache yok, direkt key/value'yu cache'le
        if self.key_cache is None:
            # Cache'i oluştur (max_cache_len boyutunda)
            self.key_cache = torch.zeros(
                self.batch_size, H, self.max_cache_len, D,
                device=self.device, dtype=self.dtype
            )
            self.value_cache = torch.zeros(
                self.batch_size, H, self.max_cache_len, D,
                device=self.device, dtype=self.dtype
            )
            self.cache_len = 0
        
        # Yeni key/value'yu cache'e ekle
        if cache_position is not None:
            # Belirli pozisyonlara yaz (incremental generation)
            max_pos = int(cache_position.max().item()) + 1
            if max_pos > self.cache_len:
                self.cache_len = max_pos
            
            # Belirli pozisyonlara yaz
            self.key_cache[:, :, cache_position] = key.to(device=self.device, dtype=self.dtype)
            self.value_cache[:, :, cache_position] = value.to(device=self.device, dtype=self.dtype)
        else:
            # Append mode: Cache'in sonuna ekle
            new_cache_len = self.cache_len + new_len
            if new_cache_len > self.max_cache_len:
                raise RuntimeError(
                    f"Cache length ({new_cache_len}) exceeds max_cache_len ({self.max_cache_len}). "
                    f"Consider increasing max_cache_len or using sliding window attention."
                )
            
            # Append
            self.key_cache[:, :, self.cache_len:new_cache_len] = key.to(device=self.device, dtype=self.dtype)
            self.value_cache[:, :, self.cache_len:new_cache_len] = value.to(device=self.device, dtype=self.dtype)
            self.cache_len = new_cache_len
        
        self.logger.debug(f"[V4] KVCache: Updated, cache_len={self.cache_len}")
        return self.key_cache[:, :, :self.cache_len], self.value_cache[:, :, :self.cache_len]
        
    
    def _grow_cache(self, new_len: int) -> None:
        """Cache'i yeni uzunluğa genişlet."""
        old_len = self.key_cache.size(2) if self.key_cache is not None else 0
        if new_len <= old_len:
            return
        
        # Yeni cache oluştur
        new_key_cache = torch.zeros(
            self.batch_size,
            self.num_heads,
            new_len,
            self.head_dim,
            device=self.device,
            dtype=self.dtype,
        )
        new_value_cache = torch.zeros(
            self.batch_size,
            self.num_heads,
            new_len,
            self.head_dim,
            device=self.device,
            dtype=self.dtype,
        )
        
        # Eski cache'i kopyala
        if self.key_cache is not None:
            new_key_cache[:, :, :old_len] = self.key_cache
            new_value_cache[:, :, :old_len] = self.value_cache
        
        self.key_cache = new_key_cache
        self.value_cache = new_value_cache
        self.logger.debug(f"[V4] KVCache: Grown from {old_len} to {new_len}")
    
    def get(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Cache'i döndür."""
        if self.key_cache is None:
            return None, None
        return self.key_cache[:, :, :self.cache_len], self.value_cache[:, :, :self.cache_len]
    
    def reset(self) -> None:
        """Cache'i sıfırla (clear ile aynı)."""
        self.clear()
    
    def clear(self) -> None:
        """Cache'i temizle."""
        self.key_cache = None
        self.value_cache = None
        self.cache_len = 0
        self._initialized = False
        self.logger.debug("[V4] KVCache: Cleared")
    
    def __len__(self) -> int:
        """Cache uzunluğunu döndür."""
        return self.cache_len

