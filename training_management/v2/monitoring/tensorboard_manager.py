# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: tensorboard_manager.py
Modül: training_management/v2/monitoring
Görev: TensorBoard Manager - TensorBoard logging yönetimi. TensorBoard'a scalar,
       histogram, figure, text ve hparams logging işlemlerini yapar. TensorBoard
       kullanılabilirliğini kontrol eder ve opsiyonel kullanım sağlar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (TensorBoard logging)
- Design Patterns: Manager Pattern (TensorBoard yönetimi)
- Endüstri Standartları: TensorBoard logging best practices

KULLANIM:
- TensorBoard logging için
- Scalar, histogram, figure, text logging için
- HParams logging için

BAĞIMLILIKLAR:
- torch.utils.tensorboard: SummaryWriter (opsiyonel)

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from typing import Optional, Any, Dict
import os

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None  # type: ignore


class TensorBoardManager:
    """
    TensorBoard logging manager.
    
    Responsibilities:
    - Log scalars (loss, accuracy, etc.)
    - Log histograms
    - Log images
    - Manage TensorBoard writer
    
    SOLID: Single Responsibility Principle
    """
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        enabled: bool = True,
        logger: Optional[Any] = None
    ):
        """
        Initialize TensorBoardManager.
        
        Args:
            log_dir: Log directory (default: "./runs")
            enabled: Whether TensorBoard is enabled
            logger: Optional logger instance
        """
        self.log_dir = log_dir or "./runs"
        self.enabled = enabled and TENSORBOARD_AVAILABLE
        self.logger = logger
        self.writer: Optional[Any] = None
        
        if self.enabled and TENSORBOARD_AVAILABLE:
            try:
                os.makedirs(self.log_dir, exist_ok=True)
                self.writer = SummaryWriter(log_dir=self.log_dir)
                if self.logger:
                    self.logger.info(f"[TensorBoard] Writer initialized: {self.log_dir}")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"[TensorBoard] Failed to initialize: {e}")
                self.enabled = False
                self.writer = None
    
    def log_scalar(
        self,
        tag: str,
        value: float,
        step: int
    ) -> None:
        """
        Log scalar value.
        
        Args:
            tag: Tag name
            value: Scalar value
            step: Step number
        """
        if self.enabled and self.writer:
            try:
                self.writer.add_scalar(tag, value, step)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"[TensorBoard] Failed to log scalar '{tag}': {e}")
    
    def log_histogram(
        self,
        tag: str,
        values: Any,
        step: int
    ) -> None:
        """
        Log histogram.
        
        Args:
            tag: Tag name
            values: Values tensor
            step: Step number
        """
        if self.enabled and self.writer:
            try:
                self.writer.add_histogram(tag, values, step)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"[TensorBoard] Failed to log histogram '{tag}': {e}")
    
    def log_image(
        self,
        tag: str,
        img_tensor: Any,
        step: int
    ) -> None:
        """
        Log image.
        
        Args:
            tag: Tag name
            img_tensor: Image tensor
            step: Step number
        """
        if self.enabled and self.writer:
            try:
                self.writer.add_image(tag, img_tensor, step)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"[TensorBoard] Failed to log image '{tag}': {e}")
    
    def log_text(
        self,
        tag: str,
        text: str,
        step: int
    ) -> None:
        """
        Log text.
        
        Args:
            tag: Tag name
            text: Text content
            step: Step number
        """
        if self.enabled and self.writer:
            try:
                self.writer.add_text(tag, text, step)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"[TensorBoard] Failed to log text '{tag}': {e}")
    
    def flush(self) -> None:
        """Flush TensorBoard writer."""
        if self.enabled and self.writer:
            try:
                self.writer.flush()
            except Exception:
                pass
    
    def close(self) -> None:
        """Close TensorBoard writer."""
        if self.writer:
            try:
                self.writer.close()
                if self.logger:
                    self.logger.info("[TensorBoard] Writer closed")
            except Exception:
                pass
            self.writer = None
            self.enabled = False
