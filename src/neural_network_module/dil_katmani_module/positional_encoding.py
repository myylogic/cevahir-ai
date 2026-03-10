# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: positional_encoding.py
Modül: src/neural_network_module/dil_katmani_module
Görev: Positional Encoding - Pozisyonel kodlama modülü. "sinusoidal" (Transformer
       orijinal) ve "learned" (öğrenilebilir) modları destekler. register_buffer
       ile güvenli, pickle ve device/dtype uyumlu. seq_len büyüdükçe otomatik
       genişleme (ensure_length) sağlar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (pozisyonel kodlama)
- Design Patterns: Encoding Pattern (pozisyonel kodlama)
- Endüstri Standartları: Transformer positional encoding standardı

KULLANIM:
- Pozisyonel kodlama eklemek için
- Sinusoidal veya learned positional encoding için
- Sequence uzunluğu değişikliklerini yönetmek için

BAĞIMLILIKLAR:
- torch.nn: Module base class
- math: Matematik işlemleri

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations

import math
import logging
from typing import Optional

import torch
import torch.nn as nn


__all__ = ["PositionalEncoding"]


class PositionalEncoding(nn.Module):
    """
    Pozisyonel kodlama katmanı.

    Giriş/Çıkış:
        forward(x): x [B, T, D] → aynı boyutta tensor döner (x + PE).

    Parametreler:
        embed_dim (int): Gömme boyutu (D).
        max_len (int): Başlangıçtaki maksimum sekans uzunluğu (gerektiğinde otomatik büyür).
        dropout (float): PE eklendikten sonra uygulanacak dropout (0-1).
        mode (str): "sinusoidal" veya "learned".
        log_level (int): Logger seviyesi.

    Notlar:
        - "sinusoidal" modunda PE bir buffer olarak saklanır (eğitilmez).
        - "learned" modunda PE nn.Embedding ile öğrenilir ve uzunluk büyürse güvenle genişletilir.
        - Cihaz ve dtype, giriş tensörüne uyarlanır.
    """

    def __init__(
        self,
        embed_dim: int,
        max_len: int = 2048,
        dropout: float = 0.1,
        mode: str = "sinusoidal",
        num_heads: Optional[int] = None,  # ✅ YENİ: RoPE için head_dim hesaplama
        log_level: int = logging.INFO,
    ) -> None:
        super().__init__()

        # ---- Doğrulamalar
        if not isinstance(embed_dim, int) or embed_dim <= 0:
            raise ValueError(f"embed_dim > 0 olmalı; gelen={embed_dim}")
        if not isinstance(max_len, int) or max_len <= 0:
            raise ValueError(f"max_len > 0 olmalı; gelen={max_len}")
        dropout = float(dropout)
        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout 0-1 arasında olmalı; gelen={dropout}")
        mode = str(mode).lower()
        # ✅ V3: RoPE (Rotary Position Embedding) desteği (endüstri standardı: GPT-3+, Claude, Gemini)
        if mode not in {"sinusoidal", "learned", "rope"}:
            raise ValueError("mode 'sinusoidal', 'learned' veya 'rope' olmalıdır.")

        # ---- Logger (tekil handler)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            _h = logging.StreamHandler()
            _f = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            _h.setFormatter(_f)
            self.logger.addHandler(_h)

        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)
        self.mode = mode

        # Pozisyon indekslerini buffer olarak tut (int64); gerektiğinde büyütülecek
        pos_idx = torch.arange(max_len, dtype=torch.long).unsqueeze(0)  # [1, L]
        self.register_buffer("pos_idx", pos_idx, persistent=False)

        if self.mode == "sinusoidal":
            # Sinüsoidal PE'yi hesaplayıp buffer olarak sakla (float32)
            pe = self._build_sinusoidal_pe(max_len, embed_dim)
            self.register_buffer("pe", pe, persistent=False)  # [1, L, D]
            # learned/rope modunda kullanılmadığı için placeholder
            self.pe_embed = None  # type: ignore
            self.rope_freqs = None  # type: ignore
        elif self.mode == "learned":
            # Öğrenilebilir PE
            self.pe = None  # type: ignore
            self.pe_embed = nn.Embedding(max_len, embed_dim)
            self._init_learned(self.pe_embed)
            self.rope_freqs = None  # type: ignore
        else:  # rope
            # ✅ V3: RoPE (Rotary Position Embedding) - endüstri standardı
            # RoPE direkt PE eklemez, rotary matrix'leri hesaplar
            # Attention içinde kullanılacak
            self.pe = None  # type: ignore
            self.pe_embed = None  # type: ignore
            # ✅ DÜZELTME: RoPE için head_dim kullanılmalı (embed_dim değil)
            # head_dim = embed_dim // num_heads
            if num_heads is not None and num_heads > 0:
                self.rope_dim = embed_dim // num_heads  # head_dim
                rope_freqs = self._build_rope_freqs(max_len, self.rope_dim)
                self.register_buffer("rope_freqs", rope_freqs, persistent=False)  # [max_len, rope_dim // 2]
                self.logger.info(f"[V3] RoPE initialized: max_len={max_len}, rope_dim={self.rope_dim} (head_dim, num_heads={num_heads})")
            else:
                # Fallback: num_heads verilmemişse embed_dim kullan (uyumluluk için)
                self.rope_dim = embed_dim
                rope_freqs = self._build_rope_freqs(max_len, self.rope_dim)
                self.register_buffer("rope_freqs", rope_freqs, persistent=False)  # [max_len, rope_dim // 2]
                self.logger.warning(f"[V3] RoPE initialized without num_heads: max_len={max_len}, rope_dim={self.rope_dim} (embed_dim). "
                                  f"Attention'da head_dim kullanılıyorsa uyumsuzluk olabilir!")

        self.max_len = max_len  # mevcut kapasite

    # ------------------------ Yardımcılar ------------------------ #
    @staticmethod
    def _build_sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
        """[1, max_len, d_model] sinüsoidal PE oluşturur (float32)."""
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [L,1]
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even kanallar
        if d_model % 2 == 1:
            # tek boyutta cos için son kanal yok → div_term[:-1]
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, L, D]

    @staticmethod
    def _init_learned(emb: nn.Embedding) -> None:
        """Öğrenilebilir embedding başlatma (kaiming-normal'e yakın bir ölçek)."""
        nn.init.normal_(emb.weight, mean=0.0, std=0.02)
    
    @staticmethod
    def _build_rope_freqs(max_len: int, dim: int, base: float = 10000.0) -> torch.Tensor:
        """
        ✅ V3: RoPE (Rotary Position Embedding) frequencies hesaplama
        Endüstri standardı: GPT-3+, Claude, Gemini
        
        Args:
            max_len: Maximum sequence length
            dim: Embedding dimension (head_dim genellikle)
            base: Base frequency (default: 10000.0, GPT standardı)
        
        Returns:
            freqs: [max_len, dim // 2] - Rotary frequencies
        """
        # RoPE frequencies: θ_i = 10000^(-2i/d) for i in [0, d//2-1]
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        # Position indices: [0, 1, 2, ..., max_len-1]
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]
        # Outer product: [max_len, 1] x [1, dim//2] -> [max_len, dim//2]
        freqs = positions * inv_freq.unsqueeze(0)  # [max_len, dim//2]
        return freqs

    def _grow_to(self, new_len: int) -> None:
        """
        İç kapasiteyi (max_len) new_len'e büyütür.
        Sinusoidal modda buffer yeniden hesaplanır;
        learned modda nn.Embedding genişletilir (ağırlıklar korunur).
        """
        if new_len <= self.max_len:
            return

        # pos_idx büyüt
        new_pos_idx = torch.arange(new_len, dtype=torch.long, device=self.pos_idx.device).unsqueeze(0)
        self.pos_idx = new_pos_idx  # type: ignore[attr-defined]

        if self.mode == "sinusoidal":
            # Mevcut device'a uygun şekilde yeniden üret
            new_pe = self._build_sinusoidal_pe(new_len, self.embed_dim).to(self.pos_idx.device)
            self.pe = new_pe  # type: ignore[assignment]
        elif self.mode == "learned":
            # nn.Embedding'i genişlet
            old_embed = self.pe_embed  # type: ignore[assignment]
            assert isinstance(old_embed, nn.Embedding)
            new_embed = nn.Embedding(new_len, self.embed_dim).to(old_embed.weight.device, dtype=old_embed.weight.dtype)
            with torch.no_grad():
                new_embed.weight[: self.max_len].copy_(old_embed.weight)
                # yeni satırları başlat
                nn.init.normal_(new_embed.weight[self.max_len :], mean=0.0, std=0.02)
            self.pe_embed = new_embed
        else:  # rope
            # ✅ V3: RoPE frequencies'i genişlet
            new_rope_freqs = self._build_rope_freqs(new_len, self.rope_dim).to(self.pos_idx.device)
            self.rope_freqs = new_rope_freqs  # type: ignore[assignment]

        self.max_len = new_len
        self.logger.info(f"PositionalEncoding kapasitesi büyütüldü: max_len={self.max_len}")

    # --------------------------- Forward --------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]  →  return: x + PE (aynı dtype/device).
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input torch.Tensor olmalı; gelen={type(x)}")
        if x.ndim != 3:
            raise ValueError(f"[B,T,D] bekleniyor; gelen şekil={tuple(x.shape)}")
        B, T, D = x.shape
        
        # ✅ JIT tracing/scripting sırasında shape kontrollerini atla (TracerWarning önleme)
        try:
            is_tracing = torch._C._get_tracing_state() is not None
        except (AttributeError, RuntimeError):
            is_tracing = False
        
        if not (torch.jit.is_scripting() or is_tracing):
            if int(D) != self.embed_dim:
                raise ValueError(f"embed_dim uyuşmazlığı: beklenen={self.embed_dim}, gelen={int(D)}")

        # Kapasite yetmiyorsa büyüt
        if not (torch.jit.is_scripting() or is_tracing):
            if int(T) > self.max_len:
                self._grow_to(int(T))

        # Device/dtype eşle
        if self.mode == "sinusoidal":
            pe_slice = self.pe[:, :T, :].to(device=x.device, dtype=x.dtype)  # type: ignore[index]
            out = x + pe_slice
        elif self.mode == "learned":
            # learned
            # pos indekslerini doğru cihaza taşı
            pos_idx = self.pos_idx[:, :T].to(x.device)  # [1, T]
            pe_slice = self.pe_embed(pos_idx).to(dtype=x.dtype)  # [1, T, D]  # type: ignore[operator]
            out = x + pe_slice
        else:  # rope
            # ✅ V3: RoPE (Rotary Position Embedding)
            # RoPE direkt PE eklemez, rotary matrix'leri döndürür
            # Attention içinde kullanılacak (apply_rotary_pos_emb)
            # Burada sadece pass-through yapıyoruz (RoPE attention'da uygulanır)
            # Alternatif: RoPE'yi burada da uygulayabiliriz ama genellikle attention'da yapılır
            out = x  # RoPE attention mekanizması içinde uygulanacak
        
        out = self.dropout(out)
        return out
    
    def apply_rotary_pos_emb(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        ✅ V3: RoPE (Rotary Position Embedding) uygulama
        Endüstri standardı: GPT-3+, Claude, Gemini
        
        Args:
            x: [B, T, D] veya [B, H, T, D] - Input tensor (query/key/value)
            positions: [T] veya [B, T] - Position indices (None ise [0, 1, ..., T-1])
        
        Returns:
            x_rotated: [B, T, D] veya [B, H, T, D] - RoPE uygulanmış tensor
        """
        if self.mode != "rope":
            raise ValueError(f"apply_rotary_pos_emb sadece 'rope' modunda kullanılabilir, mevcut mod: {self.mode}")
        
        original_shape = x.shape
        is_4d = x.ndim == 4
        if is_4d:
            # [B, H, T, D] -> [B*H, T, D]
            B, H, T, D = x.shape
            x = x.reshape(B * H, T, D)
        else:
            # [B, T, D]
            B, T, D = x.shape
            H = 1
        
        # Position indices
        if positions is None:
            positions = torch.arange(T, device=x.device, dtype=torch.long)  # [T]
        elif positions.ndim == 2:
            # [B, T] -> [T] (ilk batch'i al, tüm batch'ler için aynı positions)
            positions = positions[0]
        
        # RoPE frequencies: [max_len, D//2]
        # Sadece ihtiyacımız olan positions'ları al
        # ✅ JIT tracing/scripting sırasında max_pos hesaplamasını optimize et (TracerWarning önleme)
        try:
            is_tracing = torch._C._get_tracing_state() is not None
        except (AttributeError, RuntimeError):
            is_tracing = False
        
        if torch.jit.is_scripting() or is_tracing:
            # JIT tracing için: max_len kullan (dinamik büyütme yok)
            max_pos = self.max_len
        else:
            # Normal mod: max_pos hesapla
            max_pos = int(positions.max().item()) + 1
            if max_pos > self.max_len:
                self._grow_to(max_pos)
        
        # RoPE frequencies: [T, D//2]
        rope_freqs = self.rope_freqs[positions].to(device=x.device, dtype=x.dtype)  # [T, D//2]
        
        # ✅ ENDÜSTRİ STANDARDI: RoPE rotation (GPT-3+, Claude, Gemini)
        # Split x into pairs: [B*H, T, D] -> [B*H, T, D//2, 2]
        # Her çift (x[2i], x[2i+1]) bir complex number temsil eder
        x_reshaped = x.reshape(B * H, T, D // 2, 2)  # [B*H, T, D//2, 2]
        x1, x2 = x_reshaped[..., 0], x_reshaped[..., 1]  # [B*H, T, D//2] each
        
        # RoPE rotation: [cos(θ), -sin(θ); sin(θ), cos(θ)] * [x1; x2]
        # rope_freqs: [T, D//2] -> [1, T, D//2] -> [B*H, T, D//2]
        cos_freqs = torch.cos(rope_freqs).unsqueeze(0).expand(B * H, T, D // 2)  # [B*H, T, D//2]
        sin_freqs = torch.sin(rope_freqs).unsqueeze(0).expand(B * H, T, D // 2)  # [B*H, T, D//2]
        
        # Rotation matrix multiplication:
        # [x1_rotated]   [cos(θ)  -sin(θ)] [x1]
        # [x2_rotated] = [sin(θ)   cos(θ)] [x2]
        x1_rotated = x1 * cos_freqs - x2 * sin_freqs  # [B*H, T, D//2]
        x2_rotated = x1 * sin_freqs + x2 * cos_freqs  # [B*H, T, D//2]
        
        # Concatenate back: [B*H, T, D//2, 2] -> [B*H, T, D]
        x_rotated = torch.stack([x1_rotated, x2_rotated], dim=-1)  # [B*H, T, D//2, 2]
        x_rotated = x_rotated.reshape(B * H, T, D)  # [B*H, T, D]
        
        # Reshape back to original
        if is_4d:
            # [B*H, T, D] -> [B, H, T, D]
            x_rotated = x_rotated.reshape(B, H, T, D)
        
        return x_rotated

    # ------------------------- Temsil ------------------------- #
    def extra_repr(self) -> str:
        rope_info = f", rope_dim={self.rope_dim}" if self.mode == "rope" else ""
        return (
            f"embed_dim={self.embed_dim}, max_len={self.max_len}, mode='{self.mode}'{rope_info}, "
            f"dropout={self.dropout.p:.2f}"
        )
