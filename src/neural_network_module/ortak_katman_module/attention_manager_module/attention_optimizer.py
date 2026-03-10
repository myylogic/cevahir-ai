# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: attention_optimizer.py
Modül: src/neural_network_module/ortak_katman_module/attention_manager_module
Görev: Attention Optimizer - Dikkat mekanizmaları için optimizasyon sınıfı.
       Scaling method (SOFTMAX, SIGMOID, ZSCORE, SQRT), epsilon, verbose ve
       default scaling method parametreleri ile genişletilebilir. Attention
       normalizasyonu ve clipping işlemlerini yapar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (attention optimizasyonu)
- Design Patterns: Optimizer Pattern (attention optimizasyonu)
- Endüstri Standartları: Attention optimization best practices

KULLANIM:
- Attention optimizasyonu için
- Attention normalizasyonu için
- Attention clipping için

BAĞIMLILIKLAR:
- torch: Tensor işlemleri
- TrainingLogger: Logging işlemleri

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

import time
import torch
import torch.nn as nn
from enum import Enum
from training_management.training_logger import TrainingLogger

logger = TrainingLogger()


class ScalingMethod(Enum):
    SOFTMAX = "softmax"
    SIGMOID = "sigmoid"
    ZSCORE  = "zscore"
    SQRT    = "sqrt"   # ek: normalize_attention içinde güvenli sqrt dönüşümü desteklenir


class AttentionOptimizer:
    """
    Dikkat mekanizmaları için optimizasyon sınıfı.
    """

    def __init__(
        self,
        epsilon=1e-9,
        verbose=False,
        default_scaling_method=ScalingMethod.SOFTMAX,
        default_clipping_value=None,
    ):
        # logger
        self.logger = logger

        # epsilon
        if not isinstance(epsilon, (float, int)) or epsilon <= 0:
            raise ValueError("[ERROR] Epsilon değeri sıfırdan büyük bir float veya int olmalıdır.")
        self.epsilon = float(epsilon)

        # verbose
        if not isinstance(verbose, bool):
            raise TypeError("[ERROR] Verbose parametresi bir boolean olmalıdır.")
        self.verbose = verbose

        # scaling method
        if isinstance(default_scaling_method, str):
            try:
                default_scaling_method = ScalingMethod(default_scaling_method.lower())
            except ValueError:
                raise ValueError(
                    f"[ERROR] Bilinmeyen scaling method: '{default_scaling_method}'. "
                    f"Geçerli seçenekler: {[e.value for e in ScalingMethod]}"
                )
        elif not isinstance(default_scaling_method, ScalingMethod):
            raise TypeError(
                "[ERROR] default_scaling_method bir ScalingMethod Enum değeri veya geçerli bir string olmalıdır."
            )
        self.default_scaling_method = default_scaling_method

        # clipping
        if default_clipping_value is not None:
            if not isinstance(default_clipping_value, (float, int)) or default_clipping_value < 0:
                raise ValueError(
                    "[ERROR] default_clipping_value sıfırdan büyük bir float/int olmalıdır veya None bırakılmalıdır."
                )
        self.default_clipping_value = float(default_clipping_value) if default_clipping_value is not None else 1.0

        if self.verbose:
            print("[INFO] AttentionOptimizer başlatıldı:")
            print(f"  - Epsilon: {self.epsilon}")
            print(f"  - Verbose modu: {self.verbose}")
            print(f"  - Varsayılan Ölçeklendirme Yöntemi: {self.default_scaling_method.value}")
            print(f"  - Varsayılan Clipping Değeri: {self.default_clipping_value}")

    def __repr__(self):
        return (
            f"AttentionOptimizer(epsilon={self.epsilon}, verbose={self.verbose}, "
            f"default_scaling_method={self.default_scaling_method.value}, "
            f"default_clipping_value={self.default_clipping_value})"
        )

    # ------------------------ Utils / Logging ------------------------ #
    def log_tensor_info(self, tensor, name="Tensor", verbose_level=1):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"[ERROR] {name} bir PyTorch tensörü olmalıdır. Bulunan tür: {type(tensor)}")

        try:
            info = {
                "Shape": tuple(tensor.shape),
                "Dim": tensor.dim(),
                "NumElements": tensor.numel(),
                "Min": tensor.min().item() if tensor.numel() > 0 else None,
                "Max": tensor.max().item() if tensor.numel() > 0 else None,
                "Mean": tensor.mean().item() if tensor.numel() > 0 else None,
                "Std": tensor.std().item() if tensor.numel() > 0 else None,
            }
            if info["NumElements"] and info["NumElements"] > 1e6:
                info["Warning"] = f"Tensör çok büyük ({info['NumElements']} eleman)."

            if verbose_level >= 2:
                info["FirstFewElements"] = tensor.flatten()[:10].tolist()
                info["HasNaN"] = torch.isnan(tensor).any().item()
                info["HasInf"] = torch.isinf(tensor).any().item()

            if verbose_level >= 3:
                try:
                    info["Histogram"] = torch.histc(tensor.float(), bins=10).tolist()
                except Exception:
                    pass

            if self.verbose:
                print(f"[INFO] {name} Bilgileri:")
                for k, v in info.items():
                    print(f"  {k}: {v}")

            self.logger.debug(f"[{name}] {info}")

        except Exception as e:
            if self.verbose:
                print(f"[ERROR] {name} tensör bilgileri loglanırken hata oluştu: {e}")

        finally:
            if torch.isnan(tensor).any():
                self.logger.warning(f"[{name}] Tensörde NaN değerler mevcut.")
            if torch.isinf(tensor).any():
                self.logger.warning(f"[{name}] Tensörde sonsuz değerler mevcut.")

    # ------------------------ Core Ops ------------------------ #
    def normalize_attention(self, attention_scores, method=None):
        """
        attention_scores: genelde [B, H, S, S] (veya 3D [B, S, S]) float tensör
        method: 'softmax' | 'sigmoid' | 'zscore' | 'sqrt'
        """
        if not isinstance(attention_scores, torch.Tensor):
            raise TypeError("[ERROR] 'attention_scores' bir PyTorch tensörü olmalıdır.")
        if not torch.is_floating_point(attention_scores):
            raise TypeError("[ERROR] 'attention_scores' float dtype olmalıdır.")
        if attention_scores.dim() not in (3, 4):
            raise ValueError(f"[ERROR] Geçersiz tensör boyutu: {attention_scores.dim()} (beklenen 3D/4D).")

        # temizle (güvenlik)
        if torch.isnan(attention_scores).any() or torch.isinf(attention_scores).any():
            if self.verbose:
                print("[WARNING] Attention scores içinde NaN/Inf bulundu ve temizleniyor.")
            attention_scores = torch.nan_to_num(attention_scores, nan=0.0, posinf=1e9, neginf=-1e9)

        # yöntem seçimi
        if method is None:
            method = self.default_scaling_method
        elif isinstance(method, str):
            try:
                method = ScalingMethod(method.lower())
            except ValueError:
                raise ValueError(
                    f"[ERROR] Geçersiz normalizasyon yöntemi: '{method}'. "
                    f"Geçerli yöntemler: {[e.value for e in ScalingMethod]}"
                )
        elif not isinstance(method, ScalingMethod):
            raise TypeError("[ERROR] 'method', ScalingMethod Enum veya geçerli bir string olmalıdır.")

        try:
            if method == ScalingMethod.SOFTMAX:
                normalized_scores = torch.softmax(attention_scores, dim=-1)
            elif method == ScalingMethod.SIGMOID:
                normalized_scores = torch.sigmoid(attention_scores)
            elif method == ScalingMethod.ZSCORE:
                mean = attention_scores.mean(dim=-1, keepdim=True)
                std = attention_scores.std(dim=-1, keepdim=True)
                std = torch.where(std == 0, torch.full_like(std, self.epsilon), std)
                normalized_scores = (attention_scores - mean) / (std + self.epsilon)
            elif method == ScalingMethod.SQRT:
                # Güvenli sqrt dönüşümü (işaret korunur)
                normalized_scores = torch.sign(attention_scores) * torch.sqrt(torch.abs(attention_scores) + self.epsilon)
            else:
                raise ValueError(
                    f"[ERROR] Desteklenmeyen normalizasyon yöntemi: '{method}'. "
                    f"Geçerli yöntemler: {[e.value for e in ScalingMethod]}"
                )
        except Exception as e:
            raise RuntimeError(f"[ERROR] Normalizasyon işlemi sırasında hata oluştu: {e}")

        if self.verbose:
            if torch.isnan(normalized_scores).any():
                print("[ERROR] Normalizasyon sonrası NaN değerler bulundu.")
            if torch.isinf(normalized_scores).any():
                print("[ERROR] Normalizasyon sonrası sonsuz değerler bulundu.")
            if torch.isfinite(normalized_scores).all():
                print("[INFO] Normalizasyon başarılı.")

        self.log_tensor_info(normalized_scores, f"Normalized Attention ({method.value})")
        return normalized_scores

    def mask_attention(self, attention_scores, attention_mask=None, mask_type="default"):
        MaskingHelper.validate_attention_inputs(attention_scores, attention_mask, mask_type)
        if mask_type == "default":
            if self.verbose:
                print("[INFO] Varsayılan maske uygulanıyor...")
            attention_scores = MaskingHelper.apply_default_mask(attention_scores, attention_mask, verbose=self.verbose)
        elif mask_type == "causal":
            if self.verbose:
                print("[INFO] Causal maske uygulanıyor...")
            attention_scores = MaskingHelper.apply_causal_mask(attention_scores, verbose=self.verbose)

        attention_scores = MaskingHelper.clean_tensor(attention_scores, verbose=self.verbose)
        if self.verbose:
            self.log_tensor_info(attention_scores, f"Masked Attention ({mask_type})")
        return attention_scores

    def scale_attention(self, attention_scores, scaling_factor=None, adaptive=False, embed_dim=None, num_heads=None):
        if not isinstance(attention_scores, torch.Tensor):
            raise TypeError("[ERROR] 'attention_scores' bir PyTorch tensörü olmalıdır.")
        if not torch.is_floating_point(attention_scores):
            raise TypeError("[ERROR] 'attention_scores' float dtype olmalıdır.")

        if adaptive:
            if embed_dim is None or num_heads is None:
                raise ValueError("[ERROR] Adaptive scaling için 'embed_dim' ve 'num_heads' sağlanmalıdır.")
            try:
                scaling_factor = torch.sqrt(torch.tensor(embed_dim / num_heads, dtype=torch.float32))
                if self.verbose:
                    print(f"[INFO] Adaptive scaling kullanılıyor. Hesaplanan scaling_factor: {scaling_factor:.4f}")
            except Exception as e:
                raise RuntimeError(f"[ERROR] Adaptive scaling factor hesaplanırken hata oluştu: {e}")
        elif scaling_factor is None or scaling_factor <= 0:
            raise ValueError("[ERROR] Scaling factor pozitif bir değer olmalıdır.")

        try:
            attention_scores = attention_scores / float(scaling_factor)
            if self.verbose:
                print(f"[INFO] Attention scores, scaling_factor={float(scaling_factor):.4f} ile ölçeklendirildi.")
        except Exception as e:
            raise RuntimeError(f"[ERROR] Attention scores ölçeklendirilirken hata oluştu: {e}")

        if torch.isnan(attention_scores).any() or torch.isinf(attention_scores).any():
            if self.verbose:
                print("[WARNING] Ölçeklendirme sonrası NaN/Inf bulundu. Temizleniyor...")
            attention_scores = torch.nan_to_num(attention_scores, nan=0.0, posinf=1e9, neginf=-1e9)

        if self.verbose:
            self.log_tensor_info(attention_scores, "Scaled Attention")
        return attention_scores

    def clip_attention(self, attention_scores, clip_value=None):
        if not isinstance(attention_scores, torch.Tensor):
            raise TypeError("[ERROR] 'attention_scores' bir PyTorch tensörü olmalıdır.")
        if not torch.is_floating_point(attention_scores):
            raise TypeError("[ERROR] 'attention_scores' float dtype olmalıdır.")

        if clip_value is None:
            clip_value = self.default_clipping_value
            if self.verbose:
                print(f"[INFO] clip_value belirtilmedi. Varsayılan değer ({clip_value}) kullanılacak.")
        if not isinstance(clip_value, (float, int)) or clip_value <= 0:
            raise ValueError("[ERROR] 'clip_value' pozitif bir sayı olmalıdır.")

        try:
            attention_scores = torch.clamp(attention_scores, min=-float(clip_value), max=float(clip_value))
            if self.verbose:
                print(f"[INFO] Dikkat tensörleri clip_value={clip_value} ile sınırlandırıldı.")
        except Exception as e:
            raise RuntimeError(f"[ERROR] Clipping işlemi sırasında hata oluştu: {e}")

        if torch.isnan(attention_scores).any() or torch.isinf(attention_scores).any():
            if self.verbose:
                print("[WARNING] Clipping sonrası NaN/Inf bulundu. Temizleniyor...")
            attention_scores = torch.nan_to_num(attention_scores, nan=0.0, posinf=clip_value, neginf=-clip_value)

        if self.verbose:
            self.log_tensor_info(attention_scores, "Clipped Attention")
        return attention_scores

    def optimize(
        self,
        attention_scores,
        attention_mask=None,
        scaling_factor=None,
        clip_value=None,
        normalize_method=None,
        mask_type="default",
    ):
        # 1) doğrulama
        if not self.validate_attention_scores(attention_scores):
            raise ValueError("[ERROR] Attention scores contain NaN or infinity values before optimization.")
        if self.verbose:
            print("[INFO] Başlangıç tensörü doğrulandı. Optimization başlıyor.")

        # 2) maske
        if attention_mask is not None:
            try:
                attention_scores = self.mask_attention(attention_scores, attention_mask, mask_type)
                if self.verbose:
                    print(f"[INFO] {mask_type} maskesi uygulandı.")
            except Exception as e:
                raise RuntimeError(f"[ERROR] Mask uygulaması sırasında hata oluştu: {e}")

        # 3) ölçeklendirme
        if scaling_factor is not None:
            try:
                attention_scores = self.scale_attention(attention_scores, scaling_factor)
                if self.verbose:
                    print(f"[INFO] {scaling_factor} ölçek faktörü ile ölçeklendirildi.")
            except Exception as e:
                raise RuntimeError(f"[ERROR] Ölçeklendirme sırasında hata oluştu: {e}")

        # 4) clipping
        if clip_value is not None:
            try:
                attention_scores = self.clip_attention(attention_scores, clip_value)
                if self.verbose:
                    print(f"[INFO] {clip_value} clip değeri ile sınırlandırıldı.")
            except Exception as e:
                raise RuntimeError(f"[ERROR] Clipping işlemi sırasında hata oluştu: {e}")

        # 5) normalize
        try:
            optimized_attention = self.normalize_attention(attention_scores, method=normalize_method)
            if self.verbose:
                print(f"[INFO] Normalizasyon tamamlandı ({(normalize_method or self.default_scaling_method)})")
        except Exception as e:
            raise RuntimeError(f"[ERROR] Normalizasyon işlemi sırasında hata oluştu: {e}")

        # 6) çıkış doğrulama
        if not self.validate_attention_scores(optimized_attention):
            raise ValueError("[ERROR] Optimize edilen dikkat tensörleri NaN/Inf içeriyor.")

        if self.verbose:
            print("[INFO] Optimize edilen tensörler doğrulandı ve geçerli.")
            self.log_tensor_info(optimized_attention, "Optimized Attention")

        return optimized_attention

    def forward(
        self,
        attention_scores,
        attention_mask=None,
        scaling_factor=None,
        clip_value=None,
        normalize_method=None,
        mask_type="default",
    ):
        return self.optimize(attention_scores, attention_mask, scaling_factor, clip_value, normalize_method, mask_type)

    # ------------------------ Diagnostics ------------------------ #
    def check_for_nan(self, attention_scores, replace_with_zero=True):
        if not isinstance(attention_scores, torch.Tensor):
            raise TypeError("[ERROR] Attention scores must be a PyTorch tensor.")
        nan_exists = torch.isnan(attention_scores).any()
        if nan_exists and replace_with_zero:
            attention_scores = torch.nan_to_num(attention_scores, nan=0.0)
            if self.verbose:
                print("[INFO] NaN values replaced with 0.")
        return nan_exists.item(), attention_scores

    def check_for_inf(self, attention_scores, replace_with_max=True, clip_value=1e9):
        if not isinstance(attention_scores, torch.Tensor):
            raise TypeError("[ERROR] Attention scores must be a PyTorch tensor.")
        inf_exists = torch.isinf(attention_scores).any()
        if inf_exists and replace_with_max:
            attention_scores = torch.nan_to_num(attention_scores, posinf=clip_value, neginf=-clip_value)
            if self.verbose:
                print(f"[INFO] Infinity values replaced with ±{clip_value}.")
        return inf_exists.item(), attention_scores

    def validate_attention_scores(self, attention_scores):
        try:
            if not isinstance(attention_scores, torch.Tensor):
                raise TypeError("[ERROR] Attention scores must be a PyTorch tensor.")
            if attention_scores.dim() not in [3, 4]:
                raise ValueError(f"[ERROR] Invalid tensor dimensions: {attention_scores.dim()}. Expected: 3D or 4D tensor.")
            if torch.isnan(attention_scores).any():
                raise ValueError("[ERROR] Attention scores contain NaN values.")
            if torch.isinf(attention_scores).any():
                raise ValueError("[ERROR] Attention scores contain infinite values.")

            # opsiyonel hızlı örnekleme kontrolü
            if attention_scores.numel() > 1e6:
                sampled = attention_scores.view(-1)[torch.randint(0, attention_scores.numel(), (1000,))]
                if torch.isnan(sampled).any() or torch.isinf(sampled).any():
                    raise ValueError("[ERROR] Sampled values contain NaN/Inf values.")
            return True
        except (TypeError, ValueError) as e:
            if self.verbose:
                print(f"{e}")
            return False

    def extra_repr(self):
        return (
            f"epsilon={self.epsilon}, "
            f"verbose={self.verbose}, "
            f"default_scaling_method={self.default_scaling_method}, "
            f"default_clipping_value={self.default_clipping_value}, "
            f"supported_methods=['softmax', 'sigmoid', 'zscore', 'sqrt']"
        )


class MaskingHelper:
    """
    Dikkat tensörlerine maske uygulama işlemlerini yöneten yardımcı sınıf.
    Broadcast edilebilir maske şekillerini destekler.
    """

    @staticmethod
    def _broadcast_mask(attention_mask: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """
        Maskeyi hedef şekle broadcast eder.
        Örnek desteklenen şekiller:
          - (B, S) -> (B, 1, 1, S)
          - (B, 1, 1, S) -> (B, H, S, S) (S->son eksen maskesi)
          - (B, S, S) -> (B, 1, S, S) -> (B, H, S, S)
        """
        if not isinstance(attention_mask, torch.Tensor):
            raise TypeError("[ERROR] 'attention_mask' bir PyTorch tensörü olmalıdır.")

        mask = attention_mask

        # bool'a çevir (0/1 durumları için)
        if mask.dtype != torch.bool:
            mask = mask != 0

        # hedef 3D/4D beklenir (S,S) veya (B,S,S) gibi maskeler de desteklenir
        while mask.dim() < len(target_shape):
            # B, H eksenleri için eksen ekle (ikinci eksene eklemek genellikle daha doğal)
            mask = mask.unsqueeze(1)

        # boyut uyumu: 1 olan eksenler expand edilebilir
        try:
            expanded = mask.expand(*target_shape)
        except Exception:
            # son çare: (B,1,1,S) kalıbına uydurmayı deneyelim (4D hedef için)
            if len(target_shape) == 4:
                B, H, S1, S2 = target_shape
                # Eğer mask  (B,S) gibi ise -> (B,1,1,S2)
                if mask.dim() >= 2 and mask.size(-1) == S2:
                    base = mask
                    for _ in range(4 - mask.dim()):
                        base = base.unsqueeze(1)
                    expanded = base.expand(B, H, 1, S2)
                else:
                    raise ValueError(
                        f"[ERROR] attention_mask hedef şekle broadcast edilemedi. "
                        f"mask={tuple(attention_mask.shape)}, target={tuple(target_shape)}"
                    )
            else:
                raise ValueError(
                    f"[ERROR] attention_mask hedef şekle broadcast edilemedi. "
                    f"mask={tuple(attention_mask.shape)}, target={tuple(target_shape)}"
                )
        return expanded

    @staticmethod
    def validate_attention_inputs(attention_scores, attention_mask, mask_type):
        if not isinstance(attention_scores, torch.Tensor):
            raise TypeError("[ERROR] 'attention_scores' bir PyTorch tensörü olmalıdır.")
        if attention_scores.dim() not in (3, 4):
            raise ValueError("[ERROR] 'attention_scores' 3D veya 4D olmalıdır.")
        if mask_type not in ["default", "causal"]:
            raise ValueError(f"[ERROR] Geçersiz maske türü: {mask_type}. Geçerli türler: ['default', 'causal']")

        # default için maske gereklidir; broadcast edilebilirliği daha sonra kontrol edeceğiz
        if mask_type == "default":
            if attention_mask is None:
                raise ValueError("[ERROR] 'default' maske türü için 'attention_mask' gereklidir.")
            if not isinstance(attention_mask, torch.Tensor):
                raise TypeError("[ERROR] 'attention_mask' bir PyTorch tensörü olmalıdır.")

    @staticmethod
    def create_causal_mask(seq_len, device):
        if not isinstance(seq_len, int) or seq_len <= 0:
            raise ValueError("[ERROR] 'seq_len' pozitif bir tamsayı olmalıdır.")
        if not isinstance(device, torch.device):
            raise TypeError("[ERROR] 'device' bir torch.device olmalıdır.")
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    @staticmethod
    def apply_default_mask(attention_scores, attention_mask, verbose=False):
        if attention_mask is None:
            raise ValueError("[ERROR] Varsayılan maskeyi uygulamak için 'attention_mask' gereklidir.")

        # hedef şekil
        target_shape = attention_scores.shape
        mask = MaskingHelper._broadcast_mask(attention_mask, target_shape)

        # Not: mask True -> keep, False -> mask. masked_fill(~mask, -inf)
        masked_scores = attention_scores.masked_fill(~mask, float("-inf"))

        if verbose:
            print("[INFO] Varsayılan maske uygulandı.")
        return masked_scores

    @staticmethod
    def apply_causal_mask(attention_scores, verbose=False):
        # attention_scores: [B, H, S, S] veya [B, S, S]
        S = attention_scores.size(-1)
        device = attention_scores.device
        causal_mask = MaskingHelper.create_causal_mask(S, device)
        # 2D/3D/4D'e uygula
        while causal_mask.dim() < attention_scores.dim():
            causal_mask = causal_mask.unsqueeze(0)
        causal_mask = causal_mask.expand_as(attention_scores)
        masked_scores = attention_scores.masked_fill(causal_mask, float("-inf"))
        if verbose:
            print("[INFO] Causal maske uygulandı.")
        return masked_scores

    @staticmethod
    def clean_tensor(tensor, nan_value=0.0, posinf_value=1e6, neginf_value=-1e6, verbose=False):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("[ERROR] Temizlenecek veri bir PyTorch tensörü olmalıdır.")
        cleaned = torch.nan_to_num(tensor, nan=nan_value, posinf=posinf_value, neginf=neginf_value)
        if verbose:
            print("[INFO] Tensor temizlendi.")
        return cleaned

    @staticmethod
    def log_tensor_details(tensor, label="Tensor"):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("[ERROR] Loglanacak veri bir PyTorch tensörü olmalıdır.")
        # Debug print'leri kaldırıldı - gereksiz tensor bilgisi yazdırılmıyor
        pass
