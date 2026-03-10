# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: cross_attention.py
Modül: src/neural_network_module/ortak_katman_module/attention_manager_module
Görev: Cross-Attention - Katmanlar arası dikkat mekanizması. Farklı normalizasyon
       türlerini destekler, attention scaling ve scaling strategy parametreleri
       ile genişletilebilir. Debug modu ile hata ayıklama desteği sağlar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (cross-attention işlemleri)
- Design Patterns: Attention Pattern (cross-attention)
- Endüstri Standartları: Transformer cross-attention standardı

KULLANIM:
- Cross-attention oluşturmak için
- Katmanlar arası dikkat mekanizması için
- Attention scaling için

BAĞIMLILIKLAR:
- torch.nn: Attention modülleri
- TrainingLogger: Logging işlemleri

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
from training_management.training_logger import TrainingLogger

logger = TrainingLogger()


class CrossAttention(nn.Module):
    """
    Katmanlar arası dikkat mekanizması.
    """

    _logger_initialized = False

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.3,
        attention_scaling=True,
        normalization_type="layer_norm",
        scaling_strategy="sqrt",
        debug=False,
    ):
        super(CrossAttention, self).__init__()

        # ---- Parametre doğrulama ----
        if not isinstance(embed_dim, int) or embed_dim <= 0:
            raise ValueError(f"embed_dim ({embed_dim}) pozitif bir tamsayı olmalıdır.")
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(f"num_heads ({num_heads}) pozitif bir tamsayı olmalıdır.")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) num_heads ({num_heads}) ile tam bölünmelidir."
            )
        if not (isinstance(dropout, float) and 0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout ({dropout}) 0..1 aralığında bir float olmalıdır.")
        if normalization_type not in ["layer_norm", "batch_norm", "instance_norm", "group_norm"]:
            raise ValueError(
                f"Geçersiz normalization_type: {normalization_type}. "
                f"Seçenekler: 'layer_norm','batch_norm','instance_norm','group_norm'."
            )
        if scaling_strategy not in ["sqrt", "linear", "none"]:
            raise ValueError(
                f"Geçersiz scaling_strategy: {scaling_strategy}. "
                f"Seçenekler: 'sqrt','linear','none'."
            )

        # ---- Özellikler ----
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attention_scaling = attention_scaling
        self.normalization_type = normalization_type
        self.scaling_strategy = scaling_strategy
        self.debug = debug

        # ---- MHA + çıkış projeksiyonu ----
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.final_dropout = nn.Dropout(dropout)

        # ---- Normalizasyon modülü ----
        if normalization_type == "layer_norm":
            self.norm = nn.LayerNorm(embed_dim)
        elif normalization_type == "batch_norm":
            self.norm = nn.BatchNorm1d(embed_dim, eps=1e-5, momentum=0.9, track_running_stats=True)
        elif normalization_type == "instance_norm":
            self.norm = nn.InstanceNorm1d(embed_dim, eps=1e-5, affine=True, track_running_stats=True)
        else:  # group_norm
            # Grupları başlık sayısına eşitlemek pratik bir seçimdir
            self.norm = nn.GroupNorm(num_groups=self.num_heads, num_channels=embed_dim, eps=1e-5)

        # ---- Logger ----
        self.logger = logger
        if self.debug:
            self.logger.debug(
                f"[CrossAttention:init] embed_dim={embed_dim}, num_heads={num_heads}, "
                f"dropout={dropout}, norm={normalization_type}, scaling={scaling_strategy}, "
                f"attn_scaling={attention_scaling}"
            )

        # Dahili yapı doğrulama
        self.validate_internal_config()

    # ----------------------- Yardımcılar ----------------------- #
    def validate_internal_config(self):
        if self.head_dim <= 0:
            raise ValueError(
                f"Head dim <= 0. embed_dim={self.embed_dim}, num_heads={self.num_heads}"
            )

    def _apply_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        LayerNorm: [B, L, E] -> LN([..., E]) doğrudan
        Batch/Instance/GroupNorm: (N,C,L) bekler; (B,L,E) -> (B,E,L) -> norm -> (B,L,E)
        """
        if self.normalization_type == "layer_norm":
            return self.norm(x)

        # Diğer normlar için dönüşüm
        if x.dim() != 3:
            raise ValueError(f"Normalization için 3D tensör beklenir, alındı: {x.shape}")
        B, L, E = x.shape
        if E != self.embed_dim:
            raise ValueError(f"Normalization: embed_dim uyuşmuyor: {E} != {self.embed_dim}")

        xt = x.transpose(1, 2)  # (B, E, L)
        yt = self.norm(xt)      # (B, E, L)
        y = yt.transpose(1, 2)  # (B, L, E)
        return y

    def _to_bool_key_padding_mask(self, key_padding_mask: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        MultiheadAttention için key_padding_mask: şekil (B, S), dtype=bool, True=maskla (yoksay).
        Giren mask 2/3/4D olabilir; bool değilse >0 olanları masked kabul ederiz.
        """
        mask = key_padding_mask

        # 4D/3D tekil eksenleri sıkıştır
        while mask.dim() > 2:
            mask = mask.squeeze(1)

        if mask.dim() != 2:
            raise ValueError(f"key_padding_mask 2D olmalı, alındı: {mask.shape}")

        if mask.size(1) != seq_len:
            raise ValueError(
                f"key_padding_mask seq_len uyuşmuyor: {mask.size(1)} != {seq_len}"
            )

        if mask.dtype == torch.bool:
            return mask
        # sayısal ise: 0=keep, >0=mask varsayımı
        return (mask > 0).to(torch.bool)

    def _prepare_attn_mask(
        self,
        attention_mask: torch.Tensor,
        batch_size: int,
        tgt_len: int,
        src_len: int,
        device: torch.device,
    ):
        """
        MultiheadAttention.attn_mask beklentileri:
          - bool maske: True -> engelle
          - float maske: additive; engellenen yerlere büyük negatif (örn. -inf)
        Şekil:
          - 2D: (tgt_len, src_len)
          - 3D: (batch_size * num_heads, tgt_len, src_len)
        Giriş (B, L, S) ise başlık bazında (B*num_heads, L, S) şekline dönüştürülür.
        """
        m = attention_mask.to(device)

        # Eğer (B, L, S) geldiyse -> (B*num_heads, L, S)
        if m.dim() == 3 and m.size(0) == batch_size and m.size(1) == tgt_len and m.size(2) == src_len:
            # bool ise direkt çoğalt; float ise de aynı
            m = m.unsqueeze(1).expand(batch_size, self.num_heads, tgt_len, src_len)
            m = m.reshape(batch_size * self.num_heads, tgt_len, src_len)

        elif m.dim() == 2:
            if m.size(0) != tgt_len or m.size(1) != src_len:
                raise ValueError(
                    f"attention_mask (tgt_len, src_len)=({tgt_len},{src_len}) beklenir, alındı {tuple(m.shape)}"
                )
        else:
            raise ValueError(
                f"attention_mask 2D (L,S) ya da 3D (B,L,S) olmalı, alındı: {tuple(m.shape)}"
            )

        # Tip/semantik
        if m.dtype == torch.bool:
            return m  # True -> engelle
        else:
            # Sayısal mask: eğer {0,1} gibi ise 1=keep varsayıp 0 olan yerlere -inf koy
            # Eğer zaten -inf/neg büyük sayılar içeriyorsa direkt bırak.
            if torch.isfinite(m).all() and m.min() >= 0.0 and m.max() <= 1.0:
                # 1=keep -> 0=engelle => engellenen yerlere -inf
                keep = (m > 0.5)
                add_mask = torch.zeros_like(m, dtype=m.dtype, device=m.device)
                add_mask[~keep] = float("-inf")
                return add_mask
            return m  # additive maske olduğunu varsay

    def calculate_scaling_factor(self):
        """
        'sqrt' -> sqrt(head_dim), 'linear' -> head_dim, 'none' -> 1.0
        """
        dev = next(self.parameters()).device
        if self.scaling_strategy == "sqrt":
            return torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32, device=dev))
        elif self.scaling_strategy == "linear":
            return torch.tensor(float(self.head_dim), dtype=torch.float32, device=dev)
        elif self.scaling_strategy == "none":
            return torch.tensor(1.0, dtype=torch.float32, device=dev)
        else:
            raise ValueError(f"Unsupported scaling strategy: {self.scaling_strategy}")

    def validate_inputs(self, query, key, value):
        # Boyut/cihaz/tür kontrolleri
        for name, t in [("query", query), ("key", key), ("value", value)]:
            if not isinstance(t, torch.Tensor):
                raise ValueError(f"[INPUT ERROR] {name} bir torch.Tensor olmalı, alındı: {type(t)}")
            if t.dim() != 3:
                raise ValueError(f"[INPUT ERROR] {name} 3D olmalı (B,L,E). Alındı: {t.shape}")
            if t.size(-1) != self.embed_dim:
                raise ValueError(f"[INPUT ERROR] {name} embed_dim uyuşmuyor: {t.size(-1)} != {self.embed_dim}")

        if query.size(0) != key.size(0) or key.size(0) != value.size(0):
            raise ValueError("[INPUT ERROR] Batch boyutları uyuşmuyor.")
        if key.size(1) != value.size(1):
            raise ValueError("[INPUT ERROR] Key ve Value seq_len uyuşmuyor.")
        if not (query.device == key.device == value.device):
            raise ValueError("[INPUT ERROR] Tüm tensörler aynı cihazda olmalı.")
        if not (query.dtype == key.dtype == value.dtype):
            raise ValueError("[INPUT ERROR] Tüm tensörler aynı dtype olmalı.")

        if self.debug:
            self.logger.debug(
                f"[CrossAttention:validate] shapes: Q{tuple(query.shape)} K{tuple(key.shape)} V{tuple(value.shape)} "
                f"device={query.device} dtype={query.dtype}"
            )

    def check_tensor_values(self, *tensors):
        for idx, t in enumerate(tensors):
            if torch.isnan(t).any():
                raise ValueError(f"[INPUT ERROR] Tensor {idx} içinde NaN bulundu.")
            if torch.isinf(t).any():
                raise ValueError(f"[INPUT ERROR] Tensor {idx} içinde Inf bulundu.")
            if self.debug:
                self.logger.debug(
                    f"[CrossAttention:check] T{idx} -> min={t.min().item():.6f} "
                    f"max={t.max().item():.6f} mean={t.mean().item():.6f} std={t.std().item():.6f}"
                )

    # ----------------------- İleri Yayılım ----------------------- #
    def forward(self, query, key, value, key_padding_mask=None, attention_mask=None):
        """
        Args:
            query: (B, Lq, E)
            key:   (B, Sk, E)
            value: (B, Sk, E)
            key_padding_mask: (B, Sk) bool (True=maskla) veya yayınlanabilir varyantlar
            attention_mask: (Lq, Sk) bool/float veya (B, Lq, Sk) -> (B*H, Lq, Sk)
        Returns:
            attn_out: (B, Lq, E)
            attn_weights: (B, Lq, Sk)  (average_attn_weights=True varsayılan)
        """
        # 1) giriş doğrulama
        self.validate_inputs(query, key, value)
        B, Lq, _ = query.shape
        Sk = key.shape[1]
        dev = query.device

        # 2) maskeleri hazırla
        kpm = None
        attn_m = None
        if key_padding_mask is not None:
            kpm = self._to_bool_key_padding_mask(key_padding_mask, Sk).to(dev)
        if attention_mask is not None:
            attn_m = self._prepare_attn_mask(attention_mask, B, Lq, Sk, dev)

        # 3) MHA
        try:
            attn_output, attn_weights = self.multihead_attn(
                query, key, value, key_padding_mask=kpm, attn_mask=attn_m
            )
        except Exception as e:
            raise RuntimeError(f"[ATTENTION MECHANISM ERROR] Dikkat mekanizmasında hata: {e}")

        if self.debug:
            self.logger.debug(
                f"[CrossAttention:MHA] out={tuple(attn_output.shape)} weights={tuple(attn_weights.shape)}"
            )

        # 4) Projeksiyon + dropout
        try:
            attn_output = self.final_dropout(self.output_proj(attn_output))
        except Exception as e:
            raise RuntimeError(f"[PROJECTION ERROR] Projeksiyon/dropout sırasında hata: {e}")

        # 5) Residual + normalization (doğru eksenlerle)
        try:
            residual_output = attn_output + query
            normalized_output = self._apply_norm(residual_output)
        except Exception as e:
            raise RuntimeError(f"[NORMALIZATION ERROR] Residual/normalizasyon sırasında hata: {e}")

        # 6) Opsiyonel ölçekleme (standart: sqrt(head_dim))
        if self.attention_scaling:
            sf = self.calculate_scaling_factor()
            normalized_output = normalized_output / sf
            if self.debug:
                self.logger.debug(f"[CrossAttention:scale] factor={sf.item():.6f}")

        # 7) Sayısal temizlik
        normalized_output = torch.nan_to_num(normalized_output, nan=0.0, posinf=1e9, neginf=-1e9)

        if self.debug:
            self.check_tensor_values(normalized_output)
            self.logger.debug(f"[CrossAttention:out] shape={tuple(normalized_output.shape)}")

        return normalized_output, attn_weights

    # (Eski imzayı koruyarak) infernallerde kullandığımız sadeleştirilmiş maske işleyici:
    def process_attention_masks(self, key_padding_mask, attention_mask, seq_len):
        # Geriye dönük uyumluluk için bırakıldı; asıl mantık forward içinde yeni fonksiyonlara taşındı.
        if key_padding_mask is not None:
            # Sıkıştır
            while key_padding_mask.dim() > 2:
                key_padding_mask = key_padding_mask.squeeze(1)
            if key_padding_mask.dim() != 2:
                raise ValueError("key_padding_mask must be 2D after squeezing.")
        if attention_mask is not None and attention_mask.dim() == 2:
            # (L,S) kalıbında bırak
            attention_mask = attention_mask
        return key_padding_mask, attention_mask

    def extra_repr(self):
        return (
            f"embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads}, "
            f"dropout={getattr(self.final_dropout, 'p', 'N/A')}, "
            f"attention_scaling={self.attention_scaling}, "
            f"normalization_type={self.normalization_type}, "
            f"scaling_strategy={self.scaling_strategy}, "
            f"debug={self.debug}"
        )
