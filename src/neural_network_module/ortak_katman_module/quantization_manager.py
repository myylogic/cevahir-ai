# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: quantization_manager.py
Modül: src/neural_network_module/ortak_katman_module
Görev: Quantization Manager - Eğitim tamamlandıktan SONRA modeli quantize ederek
       inference hızını ve bellek verimliliğini artırır.

       Desteklenen tipler:
         "none"         → Quantization yok (varsayılan, eğitim için)
         "fp16"         → FP16 yarı-hassasiyet (2x bellek tasarrufu, GPU gerektirir)
         "bf16"         → BF16 yarı-hassasiyet (FP16'ya göre daha stabil, Ampere+ GPU)
         "int8_dynamic" → INT8 dinamik quantization (ağırlıklar INT8, aktivasyonlar float)
                          Calibration gerekmez. Endüstri standardı. (Önerilen)
         "int8"         → INT8 statik quantization (calibration gerekli,
                          model içinde QuantStub/DeQuantStub katmanı gerektirir)

       ÖNEMLİ KULLANIM NOTU:
         Quantization __init__ sırasında değil, eğitim bittikten sonra
         CevahirNeuralNetwork.apply_quantization() ile uygulanır.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (quantization işlemleri)
- Design Patterns: Manager Pattern
- Endüstri Standartları: torch.ao.quantization (PyTorch 2.x modern API)

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

# PyTorch 2.x: torch.ao.quantization (modern API)
# PyTorch 1.x: torch.quantization (eski API — fallback)
try:
    import torch.ao.quantization as _tq
    _QUANTIZATION_API = "torch.ao.quantization"
except ImportError:
    import torch.quantization as _tq  # type: ignore[no-redef]
    _QUANTIZATION_API = "torch.quantization (legacy)"

_SUPPORTED_TYPES = {"none", "fp16", "bf16", "int8_dynamic", "int8"}


class QuantizationManager:
    """
    Model quantization yöneticisi.

    Eğitim tamamlandıktan sonra modeli quantize eder.
    Doğrudan çağrılmaz; CevahirNeuralNetwork.apply_quantization() üzerinden kullanılır.

    Desteklenen tipler:
        "none"         → İşlem yok.
        "fp16"         → model.half()  — GPU'da 2x bellek tasarrufu.
        "bf16"         → model.bfloat16() — Ampere+ GPU'larda daha stabil FP16 alternatifi.
        "int8_dynamic" → torch.ao.quantization.quantize_dynamic() — Önerilen yöntem.
                         Ağırlıklar INT8, aktivasyonlar float. Calibration gerekmez.
        "int8"         → Statik INT8. Model içinde QuantStub/DeQuantStub gerektirir.
                         Mevcut mimari ile sınırlı destek; int8_dynamic tercih edilmeli.
    """

    def __init__(
        self,
        quantization_type: str = "none",
        log_level: int = logging.INFO,
    ):
        """
        Args:
            quantization_type: Quantization tipi. Geçerli değerler: "none", "fp16",
                                "bf16", "int8_dynamic", "int8".
            log_level:         Logger seviyesi.
        """
        _qt = quantization_type.lower().strip()
        if _qt not in _SUPPORTED_TYPES:
            raise ValueError(
                f"Geçersiz quantization_type: '{quantization_type}'. "
                f"Geçerli seçenekler: {sorted(_SUPPORTED_TYPES)}"
            )
        self.quantization_type = _qt

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        if self.quantization_type != "none":
            self.logger.info(
                f"[QuantizationManager] Başlatıldı: type='{self.quantization_type}', "
                f"api='{_QUANTIZATION_API}'"
            )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def quantize_model(
        self,
        model: nn.Module,
        calibration_data: Optional[list] = None,
    ) -> nn.Module:
        """
        Modeli seçilen stratejiye göre quantize eder.

        Bu metodu doğrudan çağırmak yerine CevahirNeuralNetwork.apply_quantization()
        kullanın; o metod eval() ve boyut loglamasını otomatik yönetir.

        Args:
            model:            Quantize edilecek PyTorch modeli.
            calibration_data: INT8 static quantization için kalibrasyon verisi
                              (her eleman Tensor veya (Tensor, ...) tuple'ı).

        Returns:
            Quantize edilmiş model (in-place işlem değil; yeni nesne döner).
        """
        if self.quantization_type == "none":
            return model

        if model.training:
            self.logger.warning(
                "[QuantizationManager] Model eğitim modundayken quantize edilmeye çalışıldı. "
                "Quantization öncesi model.eval() çağrılmalı."
            )

        try:
            if self.quantization_type == "fp16":
                return self._apply_fp16(model)

            elif self.quantization_type == "bf16":
                return self._apply_bf16(model)

            elif self.quantization_type == "int8_dynamic":
                return self._apply_int8_dynamic(model)

            elif self.quantization_type == "int8":
                return self._apply_int8_static(model, calibration_data)

        except Exception as exc:
            self.logger.error(
                f"[QuantizationManager] Quantization hatası ({self.quantization_type}): {exc}. "
                "Model quantize edilmedi.",
                exc_info=True,
            )

        return model

    def dequantize_model(self, model: nn.Module) -> nn.Module:
        """
        FP16/BF16 modeli FP32'ye döndürür.

        NOT: INT8 quantize edilmiş modeller (int8_dynamic / int8) bu işlemle
        orijinal hassasiyete döndürülemez. INT8 dönüşümü kalıcıdır ve lossless
        değildir. Orijinal modeli saklamak için eğitim checkpoint'ini kullanın.
        """
        if self.quantization_type == "none":
            return model

        try:
            if self.quantization_type in {"fp16", "bf16"}:
                model = model.float()
                self.logger.info(
                    f"[QuantizationManager] Model FP32'ye dönüştürüldü "
                    f"({self.quantization_type} → float32)."
                )
                return model

            elif self.quantization_type in {"int8_dynamic", "int8"}:
                self.logger.warning(
                    "[QuantizationManager] INT8 quantize edilmiş model FP32'ye döndürülemiyor. "
                    "INT8 dönüşümü kalıcıdır. Orijinal modeli checkpoint'ten yükleyin."
                )
                return model

        except Exception as exc:
            self.logger.error(
                f"[QuantizationManager] Dequantization hatası: {exc}", exc_info=True
            )

        return model

    def is_quantized(self, model: nn.Module) -> bool:
        """
        Modelin quantize edilip edilmediğini kontrol eder.

        FP16/BF16 için: İlk parametrenin dtype'ını kontrol eder.
        INT8 için: Quantize edilmiş ağırlık paketine (_packed_params) sahip
                   modül arar — bu, torch.ao.quantize_dynamic çıktısının işareti.
        """
        if self.quantization_type == "none":
            return False

        try:
            if self.quantization_type == "fp16":
                first_param = next(model.parameters(), None)
                return first_param is not None and first_param.dtype == torch.float16

            elif self.quantization_type == "bf16":
                first_param = next(model.parameters(), None)
                return first_param is not None and first_param.dtype == torch.bfloat16

            elif self.quantization_type in {"int8_dynamic", "int8"}:
                # Dinamik quantized Linear modülleri _packed_params taşır
                for module in model.modules():
                    if hasattr(module, "_packed_params"):
                        return True
                return False

        except Exception as exc:
            self.logger.debug(f"[QuantizationManager] is_quantized kontrol hatası: {exc}")

        return False

    def get_model_size_mb(self, model: nn.Module) -> float:
        """
        Modelin yaklaşık parametre bellek boyutunu MB cinsinden döndürür.

        Quantization öncesi ve sonrası karşılaştırma için kullanılır.
        INT8 quantized modellerde _packed_params üzerinden boyut hesaplanır.
        """
        total_bytes = 0
        try:
            for module in model.modules():
                # Standart parametreler
                for param in module.parameters(recurse=False):
                    try:
                        total_bytes += param.nelement() * param.element_size()
                    except Exception:
                        pass
                # INT8 quantized ağırlıklar (_packed_params)
                if hasattr(module, "_packed_params"):
                    try:
                        w, b = module._packed_params._weight_bias()
                        if w is not None:
                            total_bytes += w.nelement() * w.element_size()
                        if b is not None:
                            total_bytes += b.nelement() * b.element_size()
                    except Exception:
                        pass
        except Exception as exc:
            self.logger.debug(f"[QuantizationManager] get_model_size_mb hatası: {exc}")

        return total_bytes / (1024 * 1024)

    # ------------------------------------------------------------------ #
    #  Internal strategies                                                 #
    # ------------------------------------------------------------------ #

    def _apply_fp16(self, model: nn.Module) -> nn.Module:
        """FP16 yarı-hassasiyet. GPU gerektirir; CPU'da performans kaybı olabilir."""
        if not torch.cuda.is_available():
            self.logger.warning(
                "[QuantizationManager] FP16 için CUDA önerilir. "
                "CPU'da FP16 performans iyileştirmesi sağlamayabilir."
            )
        model = model.half()
        self.logger.info("[QuantizationManager] Model FP16'ya dönüştürüldü.")
        return model

    def _apply_bf16(self, model: nn.Module) -> nn.Module:
        """BF16 yarı-hassasiyet. Ampere+ GPU'larda (A100, RTX 3090+) önerilir."""
        if not torch.cuda.is_available():
            self.logger.warning(
                "[QuantizationManager] BF16 için CUDA önerilir. "
                "CPU'da BF16 performans iyileştirmesi sağlamayabilir."
            )
        elif not torch.cuda.is_bf16_supported():
            self.logger.warning(
                "[QuantizationManager] Mevcut GPU BF16 desteklemiyor; FP16 kullanmayı deneyin."
            )
        model = model.bfloat16()
        self.logger.info("[QuantizationManager] Model BF16'ya dönüştürüldü.")
        return model

    def _apply_int8_dynamic(self, model: nn.Module) -> nn.Module:
        """
        INT8 dinamik quantization (önerilen yöntem).

        Ağırlıklar INT8 olarak depolanır, aktivasyonlar çalışma zamanında
        dinamik olarak quantize edilir. Calibration gerekmez.
        Linear katmanlar quantize edilir; diğerleri (Norm, Embedding) korunur.
        """
        model = _tq.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8,
        )
        self.logger.info(
            "[QuantizationManager] INT8 dinamik quantization uygulandı "
            "(Linear katmanlar → INT8, aktivasyonlar → float)."
        )
        return model

    def _apply_int8_static(
        self, model: nn.Module, calibration_data: Optional[list]
    ) -> nn.Module:
        """
        INT8 statik quantization.

        Model içinde QuantStub/DeQuantStub katmanları gerektirir.
        Mevcut CevahirNeuralNetwork mimarisinde bu katmanlar bulunmadığından
        int8_dynamic'e otomatik fallback uygulanır.
        Gelecekte QuantStub entegrasyonu eklenirse bu path devreye alınır.
        """
        if calibration_data is None:
            self.logger.warning(
                "[QuantizationManager] INT8 statik quantization için calibration_data gerekli. "
                "Ayrıca model mimarisinde QuantStub/DeQuantStub katmanları gerekir. "
                "Mevcut mimari ile uyumlu int8_dynamic'e fallback uygulanıyor."
            )
            return self._apply_int8_dynamic(model)

        # QuantStub varlığını kontrol et
        has_quant_stub = any(
            isinstance(m, torch.quantization.QuantStub) for m in model.modules()
        )
        if not has_quant_stub:
            self.logger.warning(
                "[QuantizationManager] Model QuantStub/DeQuantStub içermiyor. "
                "Statik INT8 quantization düzgün çalışmayacak. "
                "int8_dynamic'e fallback uygulanıyor."
            )
            return self._apply_int8_dynamic(model)

        # Statik quantization pipeline
        model.eval()
        model.qconfig = _tq.get_default_qconfig("fbgemm")
        _tq.prepare(model, inplace=True)

        with torch.no_grad():
            for data in calibration_data:
                try:
                    if isinstance(data, torch.Tensor):
                        model(data)
                    elif isinstance(data, (tuple, list)):
                        model(*data)
                    else:
                        model(data)
                except Exception as exc:
                    self.logger.warning(
                        f"[QuantizationManager] Calibration adımında hata (atlandı): {exc}"
                    )

        _tq.convert(model, inplace=True)
        self.logger.info("[QuantizationManager] INT8 statik quantization uygulandı.")
        return model
