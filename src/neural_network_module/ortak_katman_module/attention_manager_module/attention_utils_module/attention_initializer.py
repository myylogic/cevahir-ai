# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: attention_initializer.py
Modül: src/neural_network_module/ortak_katman_module/attention_manager_module/attention_utils_module
Görev: Attention Initializer - Dikkat mekanizmaları için parametre başlatıcı
       sınıfı. Desteklenen tipler: "xavier", "he", "uniform", "normal",
       "constant". Seed ve verbose parametreleri ile genişletilebilir.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (attention parametre başlatma)
- Design Patterns: Initializer Pattern (parametre başlatma)
- Endüstri Standartları: Weight initialization best practices

KULLANIM:
- Attention parametrelerini başlatmak için
- Farklı initialization methodları için
- Seed kontrolü için

BAĞIMLILIKLAR:
- torch.nn: Initialization fonksiyonları
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
from training_management.training_logger import TrainingLogger

logger = TrainingLogger()


class AttentionInitializer:
    """
    Dikkat mekanizmaları için parametre başlatıcı sınıfı.
    Desteklenen tipler: "xavier", "he", "uniform", "normal", "constant".
    """

    def __init__(self, initialization_type: str = "xavier", seed: int | None = None, verbose: bool = False):
        """
        Args:
            initialization_type: "xavier", "he", "uniform", "normal", "constant"
            seed: Rastgelelik için tohum (opsiyonel)
            verbose: Detaylı konsol çıktıları
        """
        self.logger = logger
        self.verbose = bool(verbose)

        if not isinstance(initialization_type, str):
            raise TypeError("`initialization_type` string olmalıdır.")
        self.initialization_type = initialization_type.lower()

        # Desteklenen tipler ve eşanlamlılar
        self.supported_initializations = ["xavier", "he", "uniform", "normal", "constant"]
        alias = {"kaiming": "he"}
        self.initialization_type = alias.get(self.initialization_type, self.initialization_type)

        if self.initialization_type not in self.supported_initializations:
            raise ValueError(
                f"Geçersiz başlatma tipi: {self.initialization_type}. "
                f"Desteklenen türler: {', '.join(self.supported_initializations)}"
            )

        # Tohum
        if seed is not None:
            if not isinstance(seed, int):
                raise TypeError("`seed` bir int olmalıdır.")
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            if self.verbose:
                print(f"[AttentionInitializer] Rastgele tohum ayarlandı: {seed}")

        self.logger.debug(
            f"[AttentionInitializer] init → type={self.initialization_type}, verbose={self.verbose}"
        )

    # ---------------------- Public API ---------------------- #
    def initialize_weights(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Verilen tensörü seçilen başlatma yöntemiyle YERİNDE (in-place) başlatır.

        Note:
            PyTorch init fonksiyonları **float** tensör bekler. Int/bool tensörlerde hata döner.
        """
        t0 = time.time()

        # Tip ve dtype doğrulama
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Başlatma için verilen veri bir PyTorch tensörü olmalıdır.")
        if not torch.is_floating_point(tensor):
            raise TypeError(
                f"Başlatma yalnızca float tensörlerde uygulanabilir. Alınan dtype={tensor.dtype}."
            )
        if tensor.numel() == 0:
            raise ValueError("Boş tensör başlatılamaz (numel()==0).")

        self.logger.debug(
            f"[AttentionInitializer] initialize_weights: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"
        )

        try:
            # 2D altı tensörler için (bias gibi) Xavier/He kullanımı teorik olarak uygun değildir.
            # Bu durumda güvenli bir dağılıma düşeriz ve uyarı logları bırakırız.
            if tensor.dim() < 2 and self.initialization_type in {"xavier", "he"}:
                self.logger.warning(
                    "[AttentionInitializer] Xavier/He başlatma için tensör en az 2D olmalı. "
                    "Bias/vektör için 'normal' dağılıma güvenli fallback uygulanıyor (std=0.02)."
                )
                nn.init.normal_(tensor, mean=0.0, std=0.02)

            else:
                if self.initialization_type == "xavier":
                    nn.init.xavier_uniform_(tensor)
                elif self.initialization_type == "he":
                    # Kaiming Uniform (He), ReLU varsayılarak
                    nn.init.kaiming_uniform_(tensor, nonlinearity="relu")
                elif self.initialization_type == "uniform":
                    nn.init.uniform_(tensor, a=-0.1, b=0.1)
                elif self.initialization_type == "normal":
                    nn.init.normal_(tensor, mean=0.0, std=0.02)
                elif self.initialization_type == "constant":
                    nn.init.constant_(tensor, 0.1)
                else:
                    # buraya gelmemeli, fakat savunmacı programlama:
                    raise ValueError(f"Desteklenmeyen başlatma tipi: {self.initialization_type}")

            # Geçerlilik kontrolü
            if not torch.isfinite(tensor).all():
                raise ValueError("Başlatılmış tensör NaN/Inf içeriyor.")

            dt = time.time() - t0
            self.logger.debug(
                f"[AttentionInitializer] {self.initialization_type} ile başlatıldı (elapsed={dt:.6f}s)."
            )

            if self.verbose:
                print(
                    f"[AttentionInitializer] Başlatıldı ({self.initialization_type}): "
                    f"shape={tuple(tensor.shape)}, "
                    f"min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, "
                    f"mean={tensor.mean().item():.6f}, std={tensor.std().item():.6f}"
                )

            return tensor

        except Exception as e:
            self.logger.error(f"[AttentionInitializer] Tensör başlatma sırasında hata: {e}", exc_info=True)
            raise RuntimeError(f"Tensör başlatma sırasında hata oluştu: {str(e)}") from e

    def initialize_tensors(self, tensor: torch.Tensor) -> torch.Tensor:
        """Geriye dönük uyumluluk: tek tensörü başlatır (alias)."""
        return self.initialize_weights(tensor)

    def initialize_param_matrix(self, input_dim: int, output_dim: int) -> torch.Tensor:
        """
        Ağırlık matrislerini başlatır (2D); Xavier/He için uygundur.
        """
        if not (isinstance(input_dim, int) and isinstance(output_dim, int)):
            raise TypeError("input_dim ve output_dim tamsayı olmalıdır.")
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("Giriş ve çıkış boyutları pozitif olmalıdır.")

        weights = torch.empty(input_dim, output_dim, dtype=torch.float32)
        return self.initialize_weights(weights)

    def initialize_bias(self, size: int) -> torch.Tensor:
        """
        Bias vektörünü başlatır (1D). Varsayılan olarak 0'lar.
        """
        if not isinstance(size, int):
            raise TypeError("Bias boyutu tamsayı olmalıdır.")
        if size <= 0:
            raise ValueError("Bias boyutu pozitif olmalıdır.")
        bias = torch.zeros(size, dtype=torch.float32)
        if self.verbose:
            # Debug print kaldırıldı - gereksiz tensor bilgisi
            pass
        return bias

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Fonksiyonel kullanım: initializer(tensor)."""
        return self.initialize_weights(tensor)

    def log_initialization_details(self, tensor: torch.Tensor, description: str = "") -> None:
        """İsteğe bağlı: başlatma sonrası özet log/print."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Loglama için verilen veri bir PyTorch tensörü olmalıdır.")
        msg = (
            f"[AttentionInitializer] {description} başlatıldı → "
            f"shape={tuple(tensor.shape)}, "
            f"min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, "
            f"mean={tensor.mean().item():.6f}, std={tensor.std().item():.6f}"
        )
        self.logger.debug(msg)
        if self.verbose:
            print(msg)

    def validate_tensor(self, tensor: torch.Tensor) -> bool:
        """
        Başlatma sonrası tensörün temel geçerlemesini yapar.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Kontrol için verilen veri bir PyTorch tensörü olmalıdır.")
        if tensor.numel() == 0:
            raise ValueError("Tensör boş.")
        if torch.isnan(tensor).any():
            raise ValueError("Tensör NaN değerler içeriyor.")
        if torch.isinf(tensor).any():
            raise ValueError("Tensör sonsuz değerler içeriyor.")
        return True
