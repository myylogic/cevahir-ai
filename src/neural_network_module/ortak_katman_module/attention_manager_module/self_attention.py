# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: self_attention.py
Modül: src/neural_network_module/ortak_katman_module/attention_manager_module
Görev: Self-Attention - Sekans içi çok başlıklı dikkat (self-attention). Farklı
       normalizasyon türlerini destekler, sağlam maske işleme ve hata ayıklama
       içerir. Normalization type, num_groups ve eps parametreleri ile
       genişletilebilir.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (self-attention işlemleri)
- Design Patterns: Attention Pattern (self-attention)
- Endüstri Standartları: Transformer self-attention standardı

KULLANIM:
- Self-attention oluşturmak için
- Sekans içi dikkat mekanizması için
- Normalizasyon seçenekleri için

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
import torch.nn.functional as F
from training_management.training_logger import TrainingLogger

logger = TrainingLogger()


class SelfAttention(nn.Module):
    """
    Sekans içi çok başlıklı dikkat (self-attention).
    Farklı normalizasyon türlerini destekler, sağlam maske işleme ve hata ayıklama içerir.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.3,
        normalization_type="layer_norm",
        num_groups=None,
        eps=1e-5,
        debug=False,
    ):
        super(SelfAttention, self).__init__()

        # --- Parametre doğrulama ---
        if not isinstance(embed_dim, int) or embed_dim <= 0:
            raise ValueError(f"Gömme boyutu (embed_dim={embed_dim}) pozitif bir tamsayı olmalı.")
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(f"Çok başlık sayısı (num_heads={num_heads}) pozitif bir tamsayı olmalı.")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Gömme boyutu ({embed_dim}) çok başlık sayısına ({num_heads}) tam bölünmelidir."
            )
        if not (isinstance(dropout, float) and 0.0 <= dropout <= 1.0):
            raise ValueError(f"Dropout ({dropout}) 0..1 aralığında float olmalıdır.")
        if normalization_type not in ["layer_norm", "batch_norm", "group_norm", "instance_norm"]:
            raise ValueError(
                f"Geçersiz normalizasyon tipi: {normalization_type}. "
                f"Desteklenenler: ['layer_norm','batch_norm','group_norm','instance_norm']"
            )
        if not isinstance(eps, (float, int)) or eps <= 0:
            raise ValueError(f"eps pozitif olmalıdır (verilen: {eps}).")

        # --- Özellikler ---
        self.logger = logger
        self.logger.debug("SelfAttention __init__ çağrıldı.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.normalization_type = normalization_type
        self.eps = float(eps)
        self.debug = debug

        # GroupNorm için grup sayısı
        if normalization_type == "group_norm":
            if num_groups is None:
                num_groups = self.num_heads  # başlık sayısı iyi bir varsayılan
            if not isinstance(num_groups, int) or num_groups <= 0:
                raise ValueError(f"num_groups pozitif bir tamsayı olmalı (verilen: {num_groups}).")
            if embed_dim % num_groups != 0:
                # en yakın bölünebilir grup sayısına düşür
                g = min(num_groups, embed_dim)
                while g > 1 and (embed_dim % g != 0):
                    g -= 1
                if g == 1 and embed_dim % g != 0:
                    raise ValueError(
                        f"embed_dim={embed_dim} için geçerli num_groups bulunamadı (aday: {num_groups})."
                    )
                num_groups = g
        self.num_groups = num_groups

        # --- Projeksiyon katmanları ---
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # --- Dropout ---
        self.attn_dropout = nn.Dropout(dropout)
        self.final_dropout = nn.Dropout(dropout)

        # --- Normalizasyon modülü ---
        if normalization_type == "layer_norm":
            self.norm = nn.LayerNorm(embed_dim, eps=self.eps)
        elif normalization_type == "batch_norm":
            self.norm = nn.BatchNorm1d(embed_dim, eps=self.eps, momentum=0.9, track_running_stats=True)
        elif normalization_type == "instance_norm":
            self.norm = nn.InstanceNorm1d(embed_dim, eps=self.eps, affine=True, track_running_stats=True)
        else:  # group_norm
            self.norm = nn.GroupNorm(num_groups=self.num_groups, num_channels=embed_dim, eps=self.eps)

        if self.debug:
            print(
                f"SelfAttention başlatıldı: E={embed_dim}, H={num_heads}, D={self.head_dim}, "
                f"dropout={dropout}, norm={normalization_type}, groups={self.num_groups}, eps={self.eps}"
            )

    # ------------------------ Yardımcılar ------------------------ #
    def _apply_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        LayerNorm: [B, L, E] -> direkt.
        Batch/Instance/GroupNorm: (N,C,L) beklediği için (B,L,E)->(B,E,L)->norm->(B,L,E)
        """
        if self.normalization_type == "layer_norm":
            return self.norm(x)

        if x.dim() != 3:
            raise ValueError(f"Normalization için 3D tensör beklenir, alındı: {x.shape}")
        B, L, E = x.shape
        if E != self.embed_dim:
            raise ValueError(f"Normalization: embed_dim uyuşmazlığı: {E} != {self.embed_dim}")

        xt = x.transpose(1, 2)  # (B,E,L)
        yt = self.norm(xt)      # (B,E,L)
        y = yt.transpose(1, 2)  # (B,L,E)
        return y

    def _prepare_attention_mask(self, mask: torch.Tensor, batch_size: int, tgt_len: int, src_len: int, device):
        """
        Softmax öncesi skorlar için additive maske üretir.
        Desteklenen girdiler:
          - bool maske: True=ENGELLE (additive -inf)
          - {0,1} maske: 0=ENGELLE
          - additive float maske: direkt eklenir
        Boyutlar:
          - (tgt_len, src_len)
          - (batch_size, tgt_len, src_len)
          - (batch_size, 1/num_heads, tgt_len, src_len)
        Çıkış: (B, H veya 1, T, S) float mask.
        """
        m = mask.to(device)
        if m.dim() == 2:
            if m.size(0) != tgt_len or m.size(1) != src_len:
                raise ValueError(f"attention mask boyutu (L,S)=({tgt_len},{src_len}) beklenir, alındı {tuple(m.shape)}")
            m = m.unsqueeze(0).unsqueeze(0)  # (1,1,L,S)
        elif m.dim() == 3:
            if m.size(0) != batch_size or m.size(1) != tgt_len or m.size(2) != src_len:
                raise ValueError(f"(B,L,S)=({batch_size},{tgt_len},{src_len}) beklenir, alındı {tuple(m.shape)}")
            m = m.unsqueeze(1)  # (B,1,L,S)
        elif m.dim() == 4:
            if not (m.size(0) == batch_size and m.size(2) == tgt_len and m.size(3) == src_len):
                raise ValueError(f"(B,*,L,S) beklenir, alındı {tuple(m.shape)}")
        else:
            raise ValueError(f"attention mask 2D/3D/4D olmalı, alındı: {tuple(m.shape)}")

        if m.dtype == torch.bool:
            add = torch.zeros_like(m, dtype=torch.float32)
            add[m] = float("-inf")
            return add
        else:
            if torch.isfinite(m).all() and m.min() >= 0.0 and m.max() <= 1.0:
                keep = (m > 0.5)
                add = torch.zeros_like(m, dtype=torch.float32)
                add[~keep] = float("-inf")
                return add
            return m.to(torch.float32)

    def _check_tensor(self, name, t: torch.Tensor):
        if torch.isnan(t).any():
            raise ValueError(f"[ERROR] {name} içinde NaN bulundu.")
        if torch.isinf(t).any():
            raise ValueError(f"[ERROR] {name} içinde Inf bulundu.")
        if self.debug:
            self.logger.debug(
                f"[SelfAttn:check] {name} -> min={t.min().item():.6f} max={t.max().item():.6f} "
                f"mean={t.mean().item():.6f} std={t.std().item():.6f}"
            )

    # ---------------- Scaled Dot-Product Attention ---------------- #
    def scaled_dot_product_attention(self, query, key, value, add_mask=None):
        """
        Args:
            query,key,value: (B, H, T, D)
            add_mask: additive maske, (B, 1/H, T, S) ile yayınlanabilir (float; -inf engeller)
        Returns:
            output, attn_weights
        """
        d_k = query.size(-1)
        scale = max(d_k ** 0.5, 1e-6)

        # (B,H,T,D) x (B,H,D,S) -> (B,H,T,S)
        scores = torch.matmul(query, key.transpose(-2, -1)) / scale

        if add_mask is not None:
            scores = scores + add_mask  # -inf eklenen yerler engellenir

        # sayısal kararlılık
        max_scores, _ = scores.max(dim=-1, keepdim=True)
        scores = scores - max_scores
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1e9, neginf=-1e9)

        attn_weights = F.softmax(scores, dim=-1)
        if self.training:
            attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, value)
        return output, attn_weights

    # ---------------------------- Forward ---------------------------- #
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, L, E)
            mask: (L,S) veya (B,L,S) ya da (B,*,L,S). Self-attention için S=L olur.
                  Bool/0-1/additive desteklenir.
        Returns:
            (B, L, E)
        """
        # 1) doğrulama
        self.validate_inputs(x, mask)

        B, L, E = x.shape
        device = x.device

        if self.debug:
            # Debug print'leri kaldırıldı - gereksiz tensor bilgisi
            pass

        x = torch.nan_to_num(x, nan=0.0, posinf=1e9, neginf=-1e9)

        # 2) Projeksiyonlar -> (B,H,L,D)
        q = self.query_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        self._check_tensor("Q", q)
        self._check_tensor("K", k)
        self._check_tensor("V", v)

        # 3) Maske hazırlama
        add_mask = None
        if mask is not None:
            add_mask = self._prepare_attention_mask(mask, B, L, L, device)
            if add_mask.dim() == 4 and add_mask.size(1) == 1:
                add_mask = add_mask.expand(B, self.num_heads, L, L)

        # 4) SDPA
        attn_out, attn_weights = self.scaled_dot_product_attention(q, k, v, add_mask=add_mask)
        self._check_tensor("attn_out", attn_out)

        # 5) Başlıkları birleştir -> (B,L,E)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, E)

        # 6) Çıkış projeksiyonu + dropout + residual + norm
        y = self.out_proj(attn_out)
        y = self.final_dropout(y)
        out = self._apply_norm(y + x)

        out = torch.nan_to_num(out, nan=0.0, posinf=1e9, neginf=-1e9)
        self._check_tensor("output", out)

        if self.debug:
            # Debug print kaldırıldı - gereksiz tensor bilgisi
            pass

        return out

    # ----------------- Mevcut yardımcıların güncellenmiş hâlleri ----------------- #
    def validate_inputs(self, x, mask=None):
        if not isinstance(x, torch.Tensor):
            raise RuntimeError("x bir torch.Tensor olmalıdır.")
        if x.dim() != 3:
            raise RuntimeError(f"x 3D olmalıdır (B,L,E); alındı: {tuple(x.shape)}")
        if x.size(-1) != self.embed_dim:
            raise RuntimeError(f"E uyuşmazlığı: {x.size(-1)} != {self.embed_dim}")

        if mask is not None:
            if mask.dim() not in (2, 3, 4):
                raise RuntimeError(f"Maske 2D/3D/4D olmalıdır; alındı: {mask.dim()}")
            if mask.size(-1) not in (x.size(1),):  # S genelde L
                # 2D (L,S) durumunda zaten _prepare_attention_mask içinde kontrol var
                pass

        if not torch.is_floating_point(x):
            raise RuntimeError(f"x tipi {x.dtype}. Beklenen: float türleri.")

        if self.debug:
            print(
                f"validate_inputs OK: x={tuple(x.shape)}, "
                f"mask={(tuple(mask.shape) if mask is not None else None)}"
            )

    def combine_heads(self, attn_output, batch_size, seq_len):
        # KORUNDU (bazı dış çağrılar bu imzayı kullanıyor olabilir)
        if not isinstance(attn_output, torch.Tensor) or attn_output.dim() != 4:
            raise RuntimeError("attn_output (B,H,L,D) biçiminde olmalıdır.")
        num_heads, head_dim = attn_output.size(1), attn_output.size(-1)
        if self.embed_dim != num_heads * head_dim:
            raise RuntimeError(
                f"E uyuşmazlığı: {self.embed_dim} != {num_heads}*{head_dim}"
            )
        return (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )

    # Geriye dönük uyumluluk adına, dıştan çağrılmıyorsa kullanılmayacaktır.
    def process_mask(self, mask, seq_len):
        # Self-attention'da S=L; bu yardımcı artık _prepare_attention_mask ile kapsanıyor.
        if not isinstance(mask, torch.Tensor):
            raise RuntimeError("Maske torch.Tensor olmalıdır.")
        if mask.dim() == 2 and mask.size(1) != seq_len:
            raise RuntimeError(
                f"2D maske için S={mask.size(1)} seq_len={seq_len} ile uyuşmuyor."
            )
        return mask.to(torch.float32)

    def extra_repr(self):
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"dropout={self.attn_dropout.p}, debug={self.debug}, "
            f"normalization_type={self.normalization_type}, "
            f"num_groups={self.num_groups}, eps={self.eps}"
        )
