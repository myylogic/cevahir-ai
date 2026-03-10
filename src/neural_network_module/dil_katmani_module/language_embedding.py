# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: language_embedding.py
Modül: src/neural_network_module/dil_katmani_module
Görev: Language Embedding - Kelime id'lerini sürekli vektörlere dönüştürür.
       Güvenli ağırlık başlatma (padding satırı korunur), isteğe bağlı
       sqrt(d_model) ölçekleme, opsiyonel LayerNorm + Dropout, padding_idx,
       weight-tie, freeze/unfreeze yardımcıları ve vocab büyütme/küçültme
       işlemlerini yapar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (embedding işlemleri)
- Design Patterns: Embedding Pattern (kelime embedding)
- Endüstri Standartları: Transformer embedding standardı

KULLANIM:
- Kelime id'lerini embedding'e çevirmek için
- Embedding ağırlıklarını yönetmek için
- Vocab büyütme/küçültme için

BAĞIMLILIKLAR:
- torch.nn: Embedding modülü

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

import math
import logging
from typing import Optional

import torch
import torch.nn as nn


class LanguageEmbedding(nn.Module):
    """
    Kelime id'lerini sürekli vektörlere dönüştürür.
    - Güvenli ağırlık başlatma (padding satırı korunur)
    - İsteğe bağlı sqrt(d_model) ölçekleme (Transformer geleneği)
    - Opsiyonel LayerNorm + Dropout
    - padding_idx, weight-tie, freeze/unfreeze yardımcıları
    - Vocab büyütme/küçültme (ağırlıkları koruyarak)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        init_method: str = "xavier",
        log_level: int = logging.INFO,
        *,
        padding_idx: Optional[int] = None,
        scale_by_sqrt: bool = True,
        dropout: float = 0.1,
        norm_type: Optional[str] = None,  # "layer_norm" | None
        norm_eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        # --- Logger ---
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            _h = logging.StreamHandler()
            _f = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            _h.setFormatter(_f)
            self.logger.addHandler(_h)

        # --- Parametreler ---
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            raise ValueError(f"vocab_size > 0 olmalı; gelen={vocab_size}")
        if not isinstance(embed_dim, int) or embed_dim <= 0:
            raise ValueError(f"embed_dim > 0 olmalı; gelen={embed_dim}")
        if dropout < 0.0 or dropout > 1.0:
            raise ValueError(f"dropout [0,1] aralığında olmalı; gelen={dropout}")
        if padding_idx is not None and (padding_idx < 0 or padding_idx >= vocab_size):
            raise ValueError(f"padding_idx [0,{vocab_size-1}] aralığında olmalı; gelen={padding_idx}")

        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.init_method = str(init_method).lower()
        self.padding_idx = padding_idx
        self.scale_by_sqrt = bool(scale_by_sqrt)

        # --- Katmanlar ---
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embed_dim,
            padding_idx=self.padding_idx,
            device=device,
            dtype=dtype,
        )

        self._initialize_weights(self.init_method)

        # [FIX] Opsiyonel normalizasyon (token-wise)
        # By default, add LayerNorm to stabilize output distribution
        # This prevents dominant token collapse regardless of embedding init
        if norm_type is None:
            # [FIX] DEFAULT: Add LayerNorm unless explicitly disabled
            self.norm = nn.LayerNorm(self.embed_dim, eps=norm_eps, elementwise_affine=True)
            self.logger.info(f"[FIX] LayerNorm added by default to embedding output (prevent collapse)")
        elif norm_type.lower() in {"layer_norm", "layernorm", "ln"}:
            self.norm = nn.LayerNorm(self.embed_dim, eps=norm_eps, elementwise_affine=True)
        else:
            raise ValueError(f"Desteklenmeyen norm_type: {norm_type}")

        self.dropout = nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity()

        self.logger.info(
            f"LanguageEmbedding initialized | "
            f"vocab_size={self.vocab_size}, embed_dim={self.embed_dim}, "
            f"init={self.init_method}, padding_idx={self.padding_idx}, "
            f"scale_by_sqrt={self.scale_by_sqrt}, norm={type(self.norm).__name__ if self.norm else None}, "
            f"dropout={dropout}"
        )

    # -------------------- Public API -------------------- #
    
    @property
    def num_embeddings(self) -> int:
        """
        PyTorch nn.Embedding standardı: num_embeddings property.
        vocab_size ile aynı değeri döndürür.
        """
        return self.vocab_size

    @torch.no_grad()
    def resize_embedding(self, new_vocab_size: int, init_method: Optional[str] = None) -> None:
        """
        Embedding tablosunu yeni vocab boyutuna göre yeniden kurar.
        Mevcut ağırlıkları korur, yeni satırları başlatır.
        Küçültme de desteklenir.

        Not: padding_idx korunur ve aralık dışına düşerse None'a alınır.
        """
        if not isinstance(new_vocab_size, int) or new_vocab_size <= 0:
            raise ValueError(f"new_vocab_size > 0 olmalı; gelen={new_vocab_size}")
        if new_vocab_size == self.vocab_size:
            self.logger.info("resize_embedding: Boyut değişmedi, işlem yapılmadı.")
            return

        old_weight = self.embedding.weight.data
        old_vocab = self.vocab_size
        new_emb = nn.Embedding(
            num_embeddings=new_vocab_size,
            embedding_dim=self.embed_dim,
            padding_idx=self.padding_idx if (self.padding_idx is not None and self.padding_idx < new_vocab_size) else None,
            device=old_weight.device,
            dtype=old_weight.dtype,
        )

        # yeni tabloyu başlat
        self._initialize_weights(init_method or self.init_method, weight=new_emb.weight)

        # ortak kısmı kopyala
        n_copy = min(old_vocab, new_vocab_size)
        new_emb.weight[:n_copy].copy_(old_weight[:n_copy])

        # shrink ile padding_idx aralık dışına çıktıysa sıfırla
        if self.padding_idx is not None and self.padding_idx >= new_vocab_size:
            self.logger.warning(
                f"padding_idx ({self.padding_idx}) yeni vocab aralığının dışında; padding devre dışı bırakılıyor."
            )
            new_emb.padding_idx = None
            self.padding_idx = None

        self.embedding = new_emb
        self.vocab_size = new_vocab_size
        self.logger.warning(f"Embedding yeniden boyutlandırıldı: {old_vocab} → {new_vocab_size}")

    @torch.no_grad()
    def load_pretrained(self, weight: torch.Tensor, freeze: bool = False) -> None:
        """
        Önceden eğitilmiş embedding matrisi yükler.
        - weight.shape == (vocab_size, embed_dim) olmalı
        - Boyut uyuşmazlığında ValueError
        """
        if not isinstance(weight, torch.Tensor):
            raise TypeError("weight bir torch.Tensor olmalı.")
        if weight.ndim != 2 or weight.shape != (self.vocab_size, self.embed_dim):
            raise ValueError(
                f"Ağırlık boyutu uyumsuz: beklenen={(self.vocab_size, self.embed_dim)}, gelen={tuple(weight.shape)}"
            )
        self.embedding.weight.data.copy_(weight)
        if self.padding_idx is not None:
            self.embedding.weight.data[self.padding_idx].zero_()
        for p in self.embedding.parameters():
            p.requires_grad = (not freeze)
        self.logger.info(f"Ön-eğitimli embedding yüklendi. freeze={freeze}")

    @torch.no_grad()
    def freeze_embeddings(self, freeze: bool = True) -> None:
        """Embedding’i dondur/çöz."""
        for p in self.embedding.parameters():
            p.requires_grad = (not freeze)
        self.logger.info(f"Embeddings {'frozen' if freeze else 'unfrozen'}.")

    def tie_weights_to(self, linear: nn.Linear) -> None:
        """
        Çıkış katmanıyla ağırlık paylaşımı (weight tying).
        linear.out_features == vocab_size ve linear.in_features == embed_dim olmalı.
        """
        if not isinstance(linear, nn.Linear):
            raise TypeError("linear bir nn.Linear olmalı.")
        if linear.out_features != self.vocab_size or linear.in_features != self.embed_dim:
            raise ValueError(
                f"Boyut uyumsuz: linear({linear.in_features}->{linear.out_features}), "
                f"embedding({self.embed_dim}), vocab={self.vocab_size}"
            )
        linear.weight = self.embedding.weight  # weight tying
        self.logger.info("Embedding weights tied to the given Linear layer.")

    # -------------------- Forward -------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: LongTensor, şekil: [B, T] veya [..., T]
        Returns:
            out: FloatTensor, şekil: [B, T, D] (giriş şekline uygun son eksen D)
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input torch.Tensor olmalı; gelen={type(x)}")
        if x.dtype != torch.long:
            raise TypeError(f"Input dtype torch.long olmalı; gelen={x.dtype}")
        if x.ndim < 1:
            raise ValueError(f"Geçersiz giriş boyutu: {x.shape}")

        out = self.embedding(x)
        if self.scale_by_sqrt:
            out = out * math.sqrt(self.embed_dim)

        if self.norm is not None:
            # LayerNorm son eksen (D) üzerinde çalışır
            out = self.norm(out)

        out = self.dropout(out)
        self._log_tensor_stats(out, "After LanguageEmbedding")
        return out

    # -------------------- Helpers -------------------- #

    @torch.no_grad()
    def _initialize_weights(self, method: str, *, weight: Optional[torch.Tensor] = None) -> None:
        """
        Embedding ağırlıklarını başlatır. padding satırı 0'lanır ve başlatmadan sonra korunur.
        [FIX] Special token importance: BOS, EOS scaled up to prevent suppression
        [FIX] Unit normalization: All tokens initialized with uniform importance
        """
        method = str(method).lower()
        w = weight if weight is not None else self.embedding.weight

        pad_idx = self.padding_idx
        pad_backup = None
        if pad_idx is not None and 0 <= pad_idx < w.shape[0]:
            pad_backup = torch.zeros_like(w[pad_idx])

        if method in {"xavier", "xavier_uniform"}:
            nn.init.xavier_uniform_(w)
        elif method in {"xavier_normal"}:
            nn.init.xavier_normal_(w)
        elif method in {"kaiming", "kaiming_uniform"}:
            nn.init.kaiming_uniform_(w, nonlinearity="relu")
        elif method in {"kaiming_normal"}:
            nn.init.kaiming_normal_(w, nonlinearity="relu")
        elif method == "normal":
            nn.init.normal_(w, mean=0.0, std=0.02)
        elif method == "uniform":
            nn.init.uniform_(w, a=-0.1, b=0.1)
        else:
            raise ValueError(
                f"Invalid init_method: {method}. "
                f"Seçenekler: ['xavier', 'xavier_normal', 'kaiming', 'kaiming_normal', 'normal', 'uniform']"
            )

        # [FIX] NORMALIZE embedding magnitudes to uniform importance
        # This ensures no token is suppressed at initialization
        with torch.no_grad():
            # Unit normalization per token
            norms = torch.norm(w, p=2, dim=1, keepdim=True)
            norms = torch.clamp(norms, min=1e-8)
            w.div_(norms)  # w = w / ||w||
            
            # Scale to reasonable range (target std ≈ 0.15)
            w.mul_(0.15)
        
        # padding satırını sıfırla/koru
        if pad_idx is not None and 0 <= pad_idx < w.shape[0]:
            w[pad_idx].zero_() if pad_backup is None else w[pad_idx].copy_(pad_backup)
        
        self.logger.info(
            f"[FIX] Embedding weights unit-normalized: "
            f"All tokens initialized with uniform importance (prevent special token suppression)"
        )

    def _log_tensor_stats(self, tensor: torch.Tensor, stage: str) -> None:
        if not self.logger.isEnabledFor(logging.DEBUG):
            return
        try:
            self.logger.debug(
                f"{stage} | shape={tuple(tensor.shape)} "
                f"min={tensor.min().item():.4f} max={tensor.max().item():.4f} "
                f"mean={tensor.mean().item():.4f} std={tensor.std().item():.4f}"
            )
        except Exception as e:
            self.logger.debug(f"Stat log error at {stage}: {e}")
