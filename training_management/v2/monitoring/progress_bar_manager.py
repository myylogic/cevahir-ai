# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: progress_bar_manager.py
Modül: training_management/v2/monitoring
Görev: Progress Bar Manager - Progress bar yönetimi. Progress bar oluşturma,
       güncelleme ve tqdm kullanılabilirliğini yönetir. tqdm yoksa fallback
       sağlar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (progress bar yönetimi)
- Design Patterns: Manager Pattern (progress bar yönetimi)
- Endüstri Standartları: Progress bar best practices

KULLANIM:
- Progress bar oluşturmak için
- Progress bar güncellemek için
- tqdm kullanılabilirliğini yönetmek için

BAĞIMLILIKLAR:
- tqdm: Progress bar (opsiyonel)

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from typing import Optional, Any, Iterable


class ProgressBarManager:
    """
    Progress bar manager.
    
    Responsibilities:
    - Create and manage progress bars
    - Update progress
    - Handle tqdm availability
    
    SOLID: Single Responsibility Principle
    """
    
    def __init__(
        self,
        enabled: bool = True,
        logger: Optional[Any] = None
    ):
        """
        Initialize ProgressBarManager.
        
        Args:
            enabled: Whether progress bar is enabled
            logger: Optional logger instance
        """
        self.enabled = enabled
        self.logger = logger
        
        # Check tqdm availability
        try:
            from tqdm import tqdm
            self.tqdm = tqdm
            self.tqdm_available = True
        except ImportError:
            self.tqdm_available = False
            if logger:
                logger.log_warning("tqdm not available, progress bars disabled")
    
    def create_progress_bar(
        self,
        iterable: Iterable,
        desc: str = "",
        total: Optional[int] = None,
    ):
        """
        Create progress bar for iterable.
        
        Args:
            iterable: Iterable to wrap
            desc: Description string
            total: Total number of items (if known)
            
        Returns:
            Progress bar wrapper or original iterable
        """
        if not self.enabled or not self.tqdm_available:
            return iterable
        
        return self.tqdm(iterable, desc=desc, total=total, leave=False)

