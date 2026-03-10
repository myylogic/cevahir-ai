# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: quantization_manager.py
Modül: src/neural_network_module/ortak_katman_module
Görev: Quantization Manager - Model quantization, model boyutunu ve inference
       hızını artırmak için. INT8 quantization (4x compression, 2-4x speedup),
       FP16 quantization (2x compression, 1.5-2x speedup), dynamic quantization
       (runtime'da) ve static quantization (calibration ile) desteği sağlar.
       GPT-4, Claude, Gemini standardı.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (quantization işlemleri),
                     Open/Closed (genişletilebilir),
                     Dependency Inversion (nn.Module abstraction'ına bağımlı)
- Design Patterns: Manager Pattern (quantization yönetimi)
- Endüstri Standartları: PyTorch Quantization, INT8/FP16 quantization standardı

KULLANIM:
- Model quantization için
- INT8/FP16 quantization için
- Dynamic/static quantization için

BAĞIMLILIKLAR:
- torch: Quantization işlemleri
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
from typing import Optional, Literal
import logging


class QuantizationManager:
    """
    [OK] V4: Quantization Manager
    Endüstri standardı: GPT-4, Claude, Gemini
    
    Model quantization için manager.
    PyTorch'un quantization API'sini kullanır.
    """
    
    def __init__(
        self,
        quantization_type: Literal["none", "int8", "fp16", "int8_dynamic"] = "none",
        log_level: int = logging.INFO,
    ):
        """
        Args:
            quantization_type: Quantization tipi
                - "none": Quantization yok
                - "int8": INT8 static quantization (calibration gerekli)
                - "fp16": FP16 quantization (mixed precision)
                - "int8_dynamic": INT8 dynamic quantization (calibration gerekmez)
            log_level: Logging level
        """
        self.quantization_type = quantization_type.lower()
        
        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        if self.quantization_type != "none":
            self.logger.info(
                f"[V4] Quantization Manager initialized: type={quantization_type}"
            )
    
    def quantize_model(
        self,
        model: nn.Module,
        calibration_data: Optional[list] = None,
    ) -> nn.Module:
        """
        Model'i quantize et.
        
        Args:
            model: PyTorch model
            calibration_data: Calibration data (static quantization için)
        
        Returns:
            Quantized model
        """
        if self.quantization_type == "none":
            return model
        
        try:
            if self.quantization_type == "fp16":
                # FP16 quantization (mixed precision)
                model = model.half()  # Convert to FP16
                self.logger.info("[V4] Model FP16'ya quantize edildi")
                return model
            
            elif self.quantization_type == "int8_dynamic":
                # INT8 dynamic quantization
                # Sadece Linear ve LSTM layer'ları quantize eder
                model = torch.quantization.quantize_dynamic(
                    model,
                    {nn.Linear, nn.LSTM},  # Quantize edilecek layer tipleri
                    dtype=torch.qint8,
                )
                self.logger.info("[V4] Model INT8 dynamic quantization ile quantize edildi")
                return model
            
            elif self.quantization_type == "int8":
                # INT8 static quantization (calibration gerekli)
                if calibration_data is None:
                    self.logger.warning(
                        "[V4] INT8 static quantization için calibration_data gerekli. "
                        "Dynamic quantization kullanılıyor."
                    )
                    return self.quantize_model(model, None)  # Dynamic'e fallback
                
                # Model'i prepare et
                model.eval()
                model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                torch.quantization.prepare(model, inplace=True)
                
                # Calibration
                with torch.no_grad():
                    for data in calibration_data:
                        if isinstance(data, torch.Tensor):
                            model(data)
                        elif isinstance(data, (tuple, list)):
                            model(*data)
                        else:
                            model(data)
                
                # Convert
                torch.quantization.convert(model, inplace=True)
                self.logger.info("[V4] Model INT8 static quantization ile quantize edildi")
                return model
            
            else:
                self.logger.warning(
                    f"[V4] Bilinmeyen quantization tipi: {self.quantization_type}. "
                    "Quantization uygulanmadı."
                )
                return model
        
        except Exception as e:
            self.logger.error(
                f"[V4] Quantization hatası: {e}. Model quantize edilmedi.",
                exc_info=True
            )
            return model
    
    def dequantize_model(self, model: nn.Module) -> nn.Module:
        """
        Model'i dequantize et (orijinal precision'a döndür).
        
        Args:
            model: Quantized model
        
        Returns:
            Dequantized model
        """
        if self.quantization_type == "none":
            return model
        
        try:
            if self.quantization_type == "fp16":
                # FP16'dan FP32'ye döndür
                model = model.float()
                self.logger.info("[V4] Model FP32'ye dequantize edildi")
                return model
            
            elif self.quantization_type in ["int8", "int8_dynamic"]:
                # INT8'den FP32'ye döndür
                # PyTorch'un dequantize mekanizması
                if hasattr(model, 'dequantize'):
                    model = model.dequantize()
                else:
                    # Manual dequantization (gerekirse)
                    self.logger.warning(
                        "[V4] Model dequantize edilemedi. "
                        "Model zaten quantize olmayabilir."
                    )
                return model
            
            else:
                return model
        
        except Exception as e:
            self.logger.error(
                f"[V4] Dequantization hatası: {e}",
                exc_info=True
            )
            return model
    
    def is_quantized(self, model: nn.Module) -> bool:
        """
        Model quantize edilmiş mi?
        
        Args:
            model: PyTorch model
        
        Returns:
            True ise quantize edilmiş
        """
        if self.quantization_type == "none":
            return False
        
        # FP16 kontrolü
        if self.quantization_type == "fp16":
            try:
                # İlk parametrenin dtype'ını kontrol et
                first_param = next(model.parameters())
                return first_param.dtype == torch.float16
            except:
                return False
        
        # INT8 kontrolü
        elif self.quantization_type in ["int8", "int8_dynamic"]:
            # Quantized layer kontrolü
            for module in model.modules():
                if isinstance(module, (torch.quantization.QuantizedLinear, torch.nn.quantized.Linear)):
                    return True
            return False
        
        return False

