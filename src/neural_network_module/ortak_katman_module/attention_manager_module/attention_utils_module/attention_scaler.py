# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: attention_scaler.py
Modül: src/neural_network_module/ortak_katman_module/attention_manager_module/attention_utils_module
Görev: Attention Scaler - Dikkat çıktılarının ölçeklenmesi ve (opsiyonel) kırpılması /
       yeniden normalize edilmesi. Giriş: 2D [T, D], 3D [B, T, D] veya 4D [B, H, T, Hd]
       → Çıkış: girişle aynı şekil. Scale factor, clip range ve num_heads parametreleri
       ile genişletilebilir.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (attention scaling)
- Design Patterns: Scaler Pattern (ölçekleme)
- Endüstri Standartları: Attention scaling best practices

KULLANIM:
- Attention çıktılarını ölçeklemek için
- Attention clipping için
- Attention renormalization için

BAĞIMLILIKLAR:
- torch.nn: Module base class
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


class AttentionScaler(nn.Module):
    """
    Dikkat çıktılarının ölçeklenmesi ve (opsiyonel) kırpılması / yeniden normalize edilmesi.
    Giriş: 2D [T, D], 3D [B, T, D] veya 4D [B, H, T, Hd] → Çıkış: girişle aynı şekil.
    """

    def __init__(self, scale_factor=1.0, clip_range=None, verbose=False, num_heads=None):
        super().__init__()
        self.logger = logger
        self.logger.debug("AttentionScaler __init__ çağrıldı.")

        # scale_factor
        if not isinstance(scale_factor, (int, float)) or scale_factor <= 0:
            raise ValueError(f"'scale_factor' pozitif bir sayı olmalıdır. Bulunan: {scale_factor}")
        self.scale_factor = float(scale_factor)

        # clip_range
        if clip_range is not None:
            if not (isinstance(clip_range, tuple) and len(clip_range) == 2):
                raise ValueError(f"'clip_range' (min, max) tuple olmalıdır. Bulunan: {clip_range}")
            min_val, max_val = clip_range
            if not (isinstance(min_val, (int, float)) and isinstance(max_val, (int, float))):
                raise ValueError(f"'clip_range' sayısal olmalıdır. Bulunan: {clip_range}")
            if min_val >= max_val:
                raise ValueError(f"'clip_range' min ({min_val}), max ({max_val}) değerinden küçük olmalıdır.")
        self.clip_range = clip_range

        # verbose
        if not isinstance(verbose, bool):
            raise TypeError(f"'verbose' bool olmalıdır. Bulunan: {verbose}")
        self.verbose = verbose

        # num_heads (isteğe bağlı)
        if num_heads is not None and (not isinstance(num_heads, int) or num_heads <= 0):
            raise ValueError(f"'num_heads' pozitif bir tam sayı olmalıdır. Bulunan: {num_heads}")
        self.num_heads = num_heads

        if self.verbose:
            self.logger.info(
                f"[AttentionScaler] init → scale_factor={self.scale_factor}, clip_range={self.clip_range}, "
                f"num_heads={self.num_heads}"
            )

    # -------------------- Public -------------------- #
    def forward(self, attention_scores: torch.Tensor) -> torch.Tensor:
        """
        attention_scores:
          - 2D: [T, D]
          - 3D: [B, T, D]
          - 4D: [B, H, T, Hd]
        """
        import time
        t0 = time.time()

        # 1) giriş doğrulama
        self._validate_tensor(attention_scores)
        original_ndim = attention_scores.ndim
        orig_shape = attention_scores.shape

        if self.verbose:
            self.logger.debug(f"[AttentionScaler] input shape={tuple(orig_shape)}, dtype={attention_scores.dtype}")

        # 2) 2D → 3D (gerektiğinde)
        x = attention_scores
        if original_ndim == 2:
            x = x.unsqueeze(0)  # [1, T, D]
            if self.verbose:
                self.logger.debug(f"[AttentionScaler] 2D → 3D: shape={tuple(x.shape)}")

        # 3) 3D → 4D (yalnızca num_heads verilmişse ve mümkünse)
        converted_3d_to_4d = False
        if x.ndim == 3 and self.num_heads is not None:
            B, T, D = x.shape
            if D % self.num_heads != 0:
                # başlık bazlı dönüşüm mümkün değil; 3D olarak devam
                self.logger.warning(
                    f"[AttentionScaler] D={D} num_heads={self.num_heads} ile bölünemiyor. 3D üzerinde devam edilecek."
                )
            else:
                x = self._convert_3d_to_4d(x)  # [B, H, T, Hd]
                converted_3d_to_4d = True
                if self.verbose:
                    self.logger.debug(f"[AttentionScaler] 3D → 4D: shape={tuple(x.shape)}")

        # 4) dtype güvenliği (float değilse float'a al)
        if not torch.is_floating_point(x):
            x = x.float()
            if self.verbose:
                self.logger.debug("[AttentionScaler] giriş float değil; float'a çevrildi.")

        # 5) ölçekleme
        try:
            y = x * self.scale_factor
            if self.verbose:
                self._log_stats(y, prefix="scaled")
        except Exception as e:
            self.logger.error(f"[AttentionScaler] Ölçekleme sırasında hata: {e}", exc_info=True)
            raise

        # 6) kırpma (opsiyonel)
        if self.clip_range is not None:
            min_val, max_val = self.clip_range
            try:
                y = torch.clamp(y, min=min_val, max=max_val)
                if self.verbose:
                    self.logger.debug(f"[AttentionScaler] clipping applied: [{min_val}, {max_val}]")
                    self._log_stats(y, prefix="clipped")
            except Exception as e:
                self.logger.error(f"[AttentionScaler] Kırpma sırasında hata: {e}", exc_info=True)
                raise

        # 7) yeniden normalize (opsiyonel, dışarıdan attribute set edilebilir)
        try:
            if hasattr(self, "re_normalize") and self.re_normalize:
                # son eksen üzerinde toplam = 1
                attn_sum = y.sum(dim=-1, keepdim=True)
                y = y / (attn_sum + 1e-8)
                if self.verbose:
                    self.logger.debug("[AttentionScaler] re-normalize uygulandı (sum==1).")
        except Exception as e:
            self.logger.error(f"[AttentionScaler] Re-normalization sırasında hata: {e}", exc_info=True)
            raise

        # 8) geriye dönüş: 4D → 3D (gerekirse), ardından 3D → 2D (gerekirse)
        if converted_3d_to_4d and original_ndim in (2, 3):
            y = self._convert_4d_to_3d(y)  # [B, T, D]
            if self.verbose:
                self.logger.debug(f"[AttentionScaler] 4D → 3D: shape={tuple(y.shape)}")

        if original_ndim == 2:
            y = y.squeeze(0)  # [T, D]
            if self.verbose:
                self.logger.debug(f"[AttentionScaler] 3D → 2D: shape={tuple(y.shape)}")

        # 9) son doğrulama
        self._validate_tensor(y, expect_ndim=original_ndim, expect_shape=orig_shape)
        if self.verbose:
            dt = time.time() - t0
            self.logger.info(f"[AttentionScaler] forward tamamlandı: shape={tuple(y.shape)}, time={dt:.6f}s")

        return y

    # -------------------- Helpers -------------------- #
    def _convert_3d_to_4d(self, attention_scores: torch.Tensor) -> torch.Tensor:
        """
        [B, T, D] → [B, H, T, Hd]  (H = num_heads, Hd = D/H)
        """
        if not isinstance(attention_scores, torch.Tensor) or attention_scores.ndim != 3:
            raise ValueError(f"3D tensör bekleniyor; got shape={tuple(attention_scores.shape)}")

        if self.num_heads is None or self.num_heads <= 0:
            raise ValueError(f"'num_heads' pozitif bir tamsayı olmalıdır. Bulunan: {self.num_heads}")

        B, T, D = attention_scores.size()
        if D % self.num_heads != 0:
            raise ValueError(f"embed_dim ({D}), num_heads ({self.num_heads}) ile tam bölünmelidir.")

        Hd = D // self.num_heads
        try:
            x = attention_scores.view(B, T, self.num_heads, Hd).permute(0, 2, 1, 3).contiguous()
        except RuntimeError as re:
            raise RuntimeError(
                f"3D→4D dönüşüm hatası: {re}. shape={tuple(attention_scores.shape)}, H={self.num_heads}, Hd={Hd}"
            ) from re

        exp_shape = (B, self.num_heads, T, Hd)
        if x.shape != exp_shape:
            raise ValueError(f"3D→4D çıktı beklenenden farklı: {tuple(x.shape)} vs {exp_shape}")
        return x

    def _convert_4d_to_3d(self, attention_scores: torch.Tensor) -> torch.Tensor:
        """
        [B, H, T, Hd] → [B, T, D]  (D = H*Hd)
        """
        if not isinstance(attention_scores, torch.Tensor) or attention_scores.ndim != 4:
            raise ValueError(f"4D tensör bekleniyor; got shape={tuple(attention_scores.shape)}")

        B, H, T, Hd = attention_scores.size()
        D = H * Hd
        try:
            x = attention_scores.permute(0, 2, 1, 3).contiguous().view(B, T, D)
        except RuntimeError as re:
            raise RuntimeError(
                f"4D→3D dönüşüm hatası: {re}. shape={tuple(attention_scores.shape)}"
            ) from re

        if x.shape != (B, T, D):
            raise ValueError(f"4D→3D çıktı beklenenden farklı: {tuple(x.shape)} vs {(B, T, D)}")
        return x

    def _validate_tensor(self, attention_scores: torch.Tensor, expect_ndim: int | None = None, expect_shape=None):
        """
        Geçerleme: tip, boyut, pozitif şekiller. 2D/3D/4D desteklenir.
        """
        if not isinstance(attention_scores, torch.Tensor):
            raise TypeError(f"Giriş bir torch.Tensor olmalıdır; got {type(attention_scores)}")

        if attention_scores.ndim not in (2, 3, 4):
            raise ValueError(f"Tensör 2D/3D/4D olmalıdır; got {attention_scores.ndim}D, shape={tuple(attention_scores.shape)}")

        if any(d <= 0 for d in attention_scores.shape):
            raise ValueError(f"Tensör şekli geçersiz: {tuple(attention_scores.shape)}")

        if expect_ndim is not None and attention_scores.ndim != expect_ndim:
            raise ValueError(f"Beklenen boyut: {expect_ndim}D, alınan: {attention_scores.ndim}D")

        if expect_shape is not None and tuple(attention_scores.shape) != tuple(expect_shape):
            raise ValueError(f"Beklenen şekil: {tuple(expect_shape)}, alınan: {tuple(attention_scores.shape)}")

        if not torch.is_floating_point(attention_scores):
            # çoğu attention çıktısı float olmalı; güvenlik için uyarı ver
            self.logger.warning("[AttentionScaler] Tensör float değil; işlem sırasında float'a çevrilecektir.")

    def _log_stats(self, t: torch.Tensor, prefix: str = "tensor"):
        try:
            self.logger.debug(
                f"[AttentionScaler] {prefix} stats → shape={tuple(t.shape)}, "
                f"min={t.min().item():.6f}, max={t.max().item():.6f}, mean={t.mean().item():.6f}, std={t.std().item():.6f}"
            )
        except Exception:
            pass

    def validate_tensor(self, attention_scores: torch.Tensor):
        """
        (Kamu) doğrulama — geriye dönük uyumluluk için bırakıldı.
        2D/3D/4D desteklenir.
        """
        self._validate_tensor(attention_scores)

    def extra_repr(self) -> str:
        scale_factor_info = f"scale_factor={self.scale_factor:.3f}"
        clip_range_info = "clip_range=None" if self.clip_range is None else f"clip_range=({self.clip_range[0]:.3f}, {self.clip_range[1]:.3f})"
        verbose_info = f"verbose={'Enabled' if self.verbose else 'Disabled'}"
        num_heads_info = f"num_heads={self.num_heads}" if self.num_heads is not None else "num_heads=None"
        return ", ".join([scale_factor_info, clip_range_info, verbose_info, num_heads_info])
