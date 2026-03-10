# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: attention_normalizer.py
Modül: src/neural_network_module/ortak_katman_module/attention_manager_module/attention_utils_module
Görev: Attention Normalizer - Dikkat mekanizmaları için tek arayüzlü normalizasyon
       modülü. Giriş: [B, T, D] veya [B, H, T, Hd] → Çıkış: girişle aynı şekil.
       Farklı normalizasyon türlerini (layer_norm, batch_norm, group_norm, instance_norm)
       destekler.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (attention normalizasyonu)
- Design Patterns: Normalizer Pattern (normalizasyon)
- Endüstri Standartları: Normalization best practices

KULLANIM:
- Attention normalizasyonu için
- Farklı normalizasyon türleri için
- Embedding ve sequence dimension normalizasyonu için

BAĞIMLILIKLAR:
- torch.nn: Normalization modülleri
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


class AttentionNormalizer(nn.Module):
    """
    Dikkat mekanizmaları için tek arayüzlü normalizasyon modülü.
    Giriş: [B, T, D] veya [B, H, T, Hd]  →  Çıkış: girişle aynı şekil.
    """

    def __init__(
        self,
        normalization_type: str = "layer_norm",
        embed_dim: int | None = None,
        seq_len: int | None = None,
        eps: float | None = None,
        verbose: bool = False,
        momentum: float | None = None,
    ):
        super().__init__()
        self.logger = logger
        self.logger.debug("AttentionNormalizer __init__ çağrıldı.")

        # tip doğrulamaları
        if not isinstance(normalization_type, str):
            raise TypeError("`normalization_type` str olmalıdır.")
        if not isinstance(verbose, bool):
            raise TypeError("`verbose` bool olmalıdır.")
        if eps is not None and not isinstance(eps, (float, int)):
            raise TypeError("`eps` float/int olmalıdır.")
        if momentum is not None and not isinstance(momentum, (float, int)):
            raise TypeError("`momentum` float/int olmalıdır.")

        self.normalization_type = normalization_type.lower()
        self.verbose = verbose
        self.momentum = None if momentum is None else float(momentum)

        # varsayılan eps
        default_eps = {
            "layer_norm": 1e-5,
            "batch_norm": 1e-5,
            "group_norm": 1e-5,
            "instance_norm": 1e-5,
        }
        self.eps = float(eps) if eps is not None else default_eps.get(self.normalization_type, 1e-5)

        self.supported_normalizations = ["layer_norm", "batch_norm", "group_norm", "instance_norm"]
        if self.normalization_type not in self.supported_normalizations:
            raise ValueError(
                f"Geçersiz normalizasyon tipi: '{self.normalization_type}'. "
                f"Desteklenen türler: {', '.join(self.supported_normalizations)}."
            )

        # parametre gereksinimleri
        # Not: BN1d -> num_features = D (embed_dim). seq_len sadece permütasyon için kullanılır.
        if self.normalization_type in ["layer_norm", "group_norm", "instance_norm", "batch_norm"]:
            if embed_dim is None or embed_dim <= 0:
                raise ValueError(f"{self.normalization_type} için 'embed_dim' pozitif olmalıdır.")
        self.embed_dim_hint = embed_dim
        self.seq_len_hint = seq_len

        # normalizer modülünü kur
        self.normalizer = self._initialize_normalizer(embed_dim=self.embed_dim_hint, seq_len=self.seq_len_hint)

        if self.verbose:
            self.logger.info(
                f"[AttentionNormalizer] init: type={self.normalization_type}, eps={self.eps}, "
                f"embed_dim={self.embed_dim_hint}, seq_len={self.seq_len_hint}, momentum={self.momentum}"
            )

    # -------- public forward -------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D] veya [B, H, T, Hd]
        """
        import time
        t0 = time.time()
        self._validate_input(x)

        orig_shape = x.shape
        B = orig_shape[0]

        # 4D -> 3D dönüşümü (gerektiğinde)
        if x.ndim == 4:
            # [B,H,T,Hd] -> [B,T,D]
            B, H, T, Hd = x.shape
            D = H * Hd
            if self.normalization_type == "batch_norm":
                # BN1d: [N, C, L] bekler → [B,D,T]
                x3 = x.reshape(B, H * Hd, T)  # [B, D, T]
                x3 = x3  # zaten [B,D,T]
            elif self.normalization_type in ["group_norm", "instance_norm"]:
                # GN/IN de [N, C, *] bekler → [B,D,T]
                x3 = x.reshape(B, H * Hd, T)  # [B, D, T]
            else:
                # LayerNorm: [B,T,D]
                x3 = x.reshape(B, T, H * Hd)
        else:
            # 3D giriş
            if self.normalization_type == "batch_norm":
                # BN1d için [B, D, T] formatına geçeceğiz
                B, T, D = x.shape
                x3 = x.transpose(1, 2)  # [B, D, T]
            elif self.normalization_type in ["group_norm", "instance_norm"]:
                # GN/IN: [B, D, T]
                B, T, D = x.shape
                x3 = x.transpose(1, 2)  # [B, D, T]
            else:
                # LayerNorm: [B, T, D]
                x3 = x

        # normalizasyon uygula
        y3 = self._apply_normalization(x3)

        # 3D -> orijinal şekle dönüş
        if orig_shape != y3.shape:
            if len(orig_shape) == 4:
                # orijinale dön: [B,H,T,Hd]
                B, H, T, Hd = orig_shape
                if self.normalization_type in ["batch_norm", "group_norm", "instance_norm"]:
                    # y3 şu an [B, D, T] (D = H*Hd)
                    y = y3.reshape(B, H * Hd, T).reshape(B, H, Hd, T).permute(0, 1, 3, 2)  # [B,H,T,Hd]
                else:
                    # layer_norm: y3 [B, T, D]
                    y = y3.reshape(B, T, H * Hd).reshape(B, T, H, Hd).permute(0, 2, 1, 3)  # [B,H,T,Hd]
            else:
                # orijinal 3D: [B,T,D]
                if self.normalization_type in ["batch_norm", "group_norm", "instance_norm"]:
                    y = y3.transpose(1, 2)  # [B, T, D]
                else:
                    y = y3
        else:
            y = y3

        # çıkış doğrulama
        if y.shape != orig_shape:
            self.logger.error(
                f"[AttentionNormalizer] Çıkış şekli orijinali ile uyuşmuyor. "
                f"orijinal={orig_shape}, çıktı={y.shape}"
            )
            raise ValueError("Normalizasyon sonrası şekil uyuşmazlığı.")

        if not torch.isfinite(y).all():
            raise RuntimeError("Normalizasyon sonrası NaN/Inf tespit edildi.")

        if self.verbose:
            dt = time.time() - t0
            self.logger.debug(
                f"[AttentionNormalizer] forward tamamlandı. shape={tuple(y.shape)}, mean={y.mean().item():.6f}, "
                f"std={y.std().item():.6f}, time={dt:.4f}s"
            )
        return y

    # -------- init helpers -------- #
    def _initialize_normalizer(self, embed_dim: int | None = None, seq_len: int | None = None) -> nn.Module:
        if self.verbose:
            self.logger.debug(
                f"[AttentionNormalizer] _initialize_normalizer: type={self.normalization_type}, "
                f"embed_dim={embed_dim}, seq_len={seq_len}, eps={self.eps}, momentum={self.momentum}"
            )

        if self.normalization_type == "layer_norm":
            return self._initialize_layer_norm(embed_dim)
        if self.normalization_type == "batch_norm":
            return self._initialize_batch_norm(embed_dim, seq_len)
        if self.normalization_type == "group_norm":
            return self._initialize_group_norm(embed_dim)
        if self.normalization_type == "instance_norm":
            return self._initialize_instance_norm(embed_dim)

        raise RuntimeError(f"Desteklenmeyen normalizasyon: {self.normalization_type}")

    def _initialize_layer_norm(self, embed_dim: int, seq_len: int | None = None) -> nn.LayerNorm:
        self._validate_positive_param("embed_dim", embed_dim)
        norm = nn.LayerNorm(normalized_shape=embed_dim, eps=self.eps)
        if self.verbose:
            self.logger.debug(f"LayerNorm init: embed_dim={embed_dim}, eps={self.eps}")
        return norm

    def _initialize_batch_norm(self, embed_dim: int, seq_len: int | None) -> nn.BatchNorm1d:
        # BN1d -> num_features=D
        self._validate_positive_param("embed_dim", embed_dim)
        eps_value = self.eps
        momentum_value = 0.1 if self.momentum is None else float(self.momentum)
        norm = nn.BatchNorm1d(num_features=embed_dim, eps=eps_value, momentum=momentum_value, track_running_stats=True)
        if self.verbose:
            self.logger.debug(
                f"BatchNorm1d init: num_features={embed_dim}, eps={eps_value}, momentum={momentum_value}, "
                f"seq_len_hint={seq_len}"
            )
        return norm

    def _initialize_group_norm(self, embed_dim: int, seq_len: int | None = None) -> nn.GroupNorm:
        self._validate_positive_param("embed_dim", embed_dim)
        num_groups = self._calculate_num_groups(embed_dim)
        norm = nn.GroupNorm(num_groups=num_groups, num_channels=embed_dim, eps=self.eps)
        if self.verbose:
            self.logger.debug(f"GroupNorm init: num_groups={num_groups}, num_channels={embed_dim}, eps={self.eps}")
        return norm

    def _initialize_instance_norm(self, embed_dim: int, seq_len: int | None = None) -> nn.InstanceNorm1d:
        self._validate_positive_param("embed_dim", embed_dim)
        eps_value = self.eps
        norm = nn.InstanceNorm1d(num_features=embed_dim, eps=eps_value, affine=True, track_running_stats=True)
        if self.verbose:
            self.logger.debug(f"InstanceNorm1d init: num_features={embed_dim}, eps={eps_value}, affine=True")
        return norm

    def _calculate_num_groups(self, embed_dim: int) -> int:
        # pratik: (D // 16) sınırlı, en az 1, en çok 32
        ng = max(1, min(embed_dim // 16, 32))
        if embed_dim % ng != 0:
            if self.verbose:
                self.logger.warning(
                    f"embed_dim ({embed_dim}) % num_groups ({ng}) != 0; num_groups=1 seçiliyor."
                )
            ng = 1
        return ng

    # -------- core apply -------- #
    def _apply_normalization(self, x: torch.Tensor) -> torch.Tensor:
        """
        x beklenen eksen düzeninde olmalı:
          - LN: [B, T, D]
          - BN/GN/IN: [B, D, T]
        """
        self._ensure_finite(x)

        if self.normalization_type == "layer_norm":
            # LN: [B, T, D] üzerinde çalışır (normalized_shape=D)
            if x.ndim != 3 or x.shape[-1] != self.normalizer.normalized_shape[0]:
                # güvenliğe al: [B,T,D] beklenmezse dönüştürme yapma, hata ver
                raise ValueError("LayerNorm için giriş [B, T, D] ve D eşleşmeli.")
            y = self.normalizer(x)
        elif self.normalization_type in ["batch_norm", "group_norm", "instance_norm"]:
            # BN/GN/IN: [B, D, T]
            if x.ndim != 3 or x.shape[1] != self._get_num_features():
                raise ValueError(
                    f"{self.normalization_type} için giriş [B, D, T] olmalı ve D=num_features ({self._get_num_features()})."
                )
            y = self.normalizer(x)
        else:
            raise RuntimeError(f"Desteklenmeyen normalizasyon: {self.normalization_type}")

        self._ensure_finite(y)
        return y

    # -------- utilities -------- #
    def _get_num_features(self) -> int:
        # BN/IN: num_features, GN: num_channels; LN: normalized_shape
        if hasattr(self.normalizer, "num_features") and self.normalizer.num_features is not None:
            return int(self.normalizer.num_features)
        if hasattr(self.normalizer, "num_channels") and self.normalizer.num_channels is not None:
            return int(self.normalizer.num_channels)
        if hasattr(self.normalizer, "normalized_shape") and self.normalizer.normalized_shape is not None:
            return int(self.normalizer.normalized_shape[0])
        # fallback
        return self.embed_dim_hint if self.embed_dim_hint is not None else -1

    def _validate_positive_param(self, name: str, value: int | float, allow_zero: bool = False):
        if value is None:
            raise ValueError(f"{name} belirtilmelidir.")
        if allow_zero:
            if value < 0:
                raise ValueError(f"{name} negatif olamaz.")
        else:
            if value <= 0:
                raise ValueError(f"{name} pozitif olmalıdır.")

    def _ensure_finite(self, x: torch.Tensor):
        if not torch.isfinite(x).all():
            raise ValueError("Tensör NaN/Inf içeriyor.")

    def _validate_input(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Giriş bir torch.Tensor olmalıdır; got {type(x)}.")
        if x.ndim not in (3, 4):
            raise ValueError(f"Giriş 3D veya 4D olmalıdır; got {x.ndim}D.")
        if any(d <= 0 for d in x.shape):
            raise ValueError(f"Geçersiz şekil: {tuple(x.shape)}.")
        self._ensure_finite(x)

    def extra_repr(self) -> str:
        parts = [
            f"normalization_type={self.normalization_type}",
            f"eps={self.eps}",
            f"verbose={self.verbose}",
        ]
        if hasattr(self.normalizer, "normalized_shape"):
            parts.append(f"embed_dim={self.normalizer.normalized_shape}")
        if hasattr(self.normalizer, "num_features"):
            parts.append(f"num_features={getattr(self.normalizer, 'num_features', None)}")
        if hasattr(self.normalizer, "num_channels"):
            parts.append(f"num_channels={getattr(self.normalizer, 'num_channels', None)}")
        return ", ".join(parts)
