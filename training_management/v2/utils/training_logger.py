# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: training_logger.py
Modül: training_management/v2/utils
Görev: Training Logger - Eğitim sırasında metrikler, hatalar ve önemli olayları
       tek merkezden loglar. Rotating file logs (info/errors), opsiyonel JSONL
       event log, TensorBoard entegrasyonu (scalar/histogram/figure/text/hparams)
       ve çift handler engelleme işlemlerini yapar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (logging yönetimi)
- Design Patterns: Logger Pattern (logging yönetimi)
- Endüstri Standartları: Training logging best practices

KULLANIM:
- Eğitim metriklerini loglamak için
- Hataları loglamak için
- TensorBoard entegrasyonu için
- Rotating file logs için

BAĞIMLILIKLAR:
- logging: Python logging modülü
- torch.utils.tensorboard: TensorBoard entegrasyonu (opsiyonel)
- json: JSONL event log

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""
from __future__ import annotations

import os
import sys
import json
import logging
import time
import platform
from datetime import datetime
from typing import Any, Dict, Optional, Iterable, Union

from logging.handlers import RotatingFileHandler


def _detect_file_locker(filepath: str) -> Optional[str]:
    """
    Windows'ta hangi process'in dosyayı kullandığını tespit etmeye çalışır.
    Returns: Process bilgisi veya None
    """
    if platform.system() != "Windows":
        return None
    
    try:
        # psutil kullanarak dosyayı açan process'leri bul
        import psutil
        locked_by = []
        for proc in psutil.process_iter(['pid', 'name', 'exe']):
            try:
                for file in proc.open_files():
                    if os.path.abspath(file.path) == os.path.abspath(filepath):
                        locked_by.append(f"PID={proc.info['pid']}, Name={proc.info['name']}")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        if locked_by:
            return "; ".join(locked_by)
    except ImportError:
        # psutil yoksa handle.exe kullanmayı dene (Windows Sysinternals)
        try:
            import subprocess
            result = subprocess.run(
                ['handle.exe', filepath],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    except Exception:
        pass
    
    return None


class SafeRotatingFileHandler(RotatingFileHandler):
    """
    Windows'ta dosya kilitli olduğunda rotate hatasını yakalayan güvenli handler.
    PermissionError durumunda alternatif dosyaya yazmaya geçer ve durumu loglar.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._alternate_file = None
        self._alternate_handler = None
        self._rotate_warning_logged = False
        self._lock_info_logged = False
    
    def _switch_to_alternate_file(self):
        """Ana dosya kilitli olduğunda alternatif dosyaya geç"""
        if self._alternate_file:
            return  # Zaten alternatif dosyaya geçmiş
        
        try:
            base_path = self.baseFilename
            dir_name = os.path.dirname(base_path)
            base_name = os.path.basename(base_path)
            name, ext = os.path.splitext(base_name)
            
            # Timestamp ile alternatif dosya adı oluştur
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._alternate_file = os.path.join(dir_name, f"{name}_locked_{timestamp}{ext}")
            
            # Alternatif dosyaya yazmak için yeni handler oluştur
            self._alternate_handler = logging.FileHandler(
                self._alternate_file,
                encoding="utf-8",
                mode='a'
            )
            self._alternate_handler.setLevel(self.level)
            self._alternate_handler.setFormatter(self.formatter)
            
            # Lock bilgisini alternatif dosyaya yaz (sadece bir kez)
            if not self._lock_info_logged:
                lock_info = _detect_file_locker(self.baseFilename)
                lock_msg = f"[TrainingLogger] Ana log dosyası kilitli: {self.baseFilename}"
                if lock_info:
                    lock_msg += f"\n[TrainingLogger] Dosyayı kullanan process: {lock_info}"
                else:
                    lock_msg += "\n[TrainingLogger] Dosyayı kullanan process tespit edilemedi (psutil veya handle.exe gerekli)"
                lock_msg += f"\n[TrainingLogger] Alternatif dosyaya geçildi: {self._alternate_file}"
                lock_msg += f"\n[TrainingLogger] Tarih: {datetime.now().isoformat()}\n"
                
                # LogRecord oluştur (logging modülü için gerekli parametreler)
                record = logging.LogRecord(
                    name="TrainingLogger",
                    level=logging.WARNING,
                    pathname="",
                    lineno=0,
                    msg=lock_msg,
                    args=(),
                    exc_info=None,
                    func="",
                    sinfo=None
                )
                self._alternate_handler.emit(record)
                
                # Console'a da yaz (sadece bir kez)
                print(lock_msg, file=sys.stderr)
                self._lock_info_logged = True
            
        except Exception as e:
            print(f"[TrainingLogger] Alternatif dosya oluşturulamadı: {e}", file=sys.stderr)
    
    def emit(self, record):
        """Log kaydını yaz - alternatif dosya varsa oraya yaz"""
        if self._alternate_handler:
            try:
                self._alternate_handler.emit(record)
            except Exception:
                self.handleError(record)
        else:
            try:
                super().emit(record)
            except (PermissionError, OSError, IOError):
                # İlk hata - alternatif dosyaya geç
                self._switch_to_alternate_file()
                if self._alternate_handler:
                    try:
                        self._alternate_handler.emit(record)
                    except Exception:
                        self.handleError(record)
            except Exception:
                self.handleError(record)
    
    def doRollover(self):
        """Rotate işlemini güvenli şekilde yap - hata durumunda alternatif dosyaya geç"""
        # ✅ OPTİMİZASYON: Alternatif dosyaya geçilmişse rotate'i atla (zaten alternatif dosyaya yazıyoruz)
        if self._alternate_file:
            return  # Alternatif dosyaya geçilmiş, rotate gerekmez
        
        try:
            super().doRollover()
        except (PermissionError, OSError, IOError) as e:
            # Windows'ta dosya kilitli olduğunda rotate başarısız olabilir
            # Alternatif dosyaya geç ve lock bilgisini logla (sadece bir kez)
            if not self._rotate_warning_logged:
                lock_info = _detect_file_locker(self.baseFilename)
                error_msg = f"[TrainingLogger] Log rotate başarısız (dosya kilitli): {e}"
                if lock_info:
                    error_msg += f"\n[TrainingLogger] Dosyayı kullanan process: {lock_info}"
                else:
                    error_msg += "\n[TrainingLogger] Dosyayı kullanan process tespit edilemedi"
                print(error_msg, file=sys.stderr)
                self._rotate_warning_logged = True
            
            # Alternatif dosyaya geç (sadece bir kez)
            if not self._alternate_file:
                self._switch_to_alternate_file()
        except Exception as e:
            # Diğer hatalar için de alternatif dosyaya geç
            if not hasattr(self, '_rotate_error_logged'):
                print(f"[TrainingLogger] Log rotate hatası: {e}", file=sys.stderr)
                self._rotate_error_logged = True
            if not self._alternate_file:
                self._switch_to_alternate_file()
    
    def close(self):
        """Handler'ı kapat - alternatif handler'ı da kapat"""
        if self._alternate_handler:
            try:
                self._alternate_handler.close()
            except Exception:
                pass
        super().close()

# -- Güvenli import & varsayılanlar
# V2: config.parameters artık yok, direkt default değerleri kullanıyoruz
LOGGING_PATH = os.path.abspath("./logs")
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5MB
BACKUP_COUNT = 5

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # torch kurulu değilse TB opsiyonel olsun
    SummaryWriter = None  # type: ignore


class TrainingLogger:
    """
    Eğitim logger'ı: dosya + (opsiyonel) TensorBoard + (opsiyonel) JSONL.
    Singleton pattern ile tek instance garantisi.

    Eski API ile uyum:
      - info/warning/error/debug/critical
      - log_info/log_warning/log_error/log_debug/log_critical
      - log_metrics(epoch, training_loss, validation_loss=None, accuracy=None)
    Yeni API:
      - start_tb(tb_log_dir=None, run_name=None)
      - log_scalar/log_histogram/log_figure/log_text
      - add_hparams(hparams, metrics)
      - log_event(dict) → JSONL
      - close()
    """
    
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        """Singleton pattern - tek instance garantisi."""
        if cls._instance is None:
            cls._instance = super(TrainingLogger, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        *,
        run_name: Optional[str] = None,
        log_dir: Optional[str] = None,
        tb_log_dir: Optional[str] = None,
        enable_tb: bool = True,
        enable_console: bool = True,
        enable_jsonl: bool = True,
        enable_file_logging: bool = True,  # ✅ YENİ: Dosya logging'i tamamen kapatabilme
        jsonl_filename: str = "training_events.jsonl",
        level: int = logging.INFO,
    ) -> None:
        # Tek seferlik initialization
        if self._initialized:
            return
            
        self.run_name = run_name or "default-run"
        self.log_dir = os.path.abspath(log_dir or LOGGING_PATH)
        self.tb_log_dir = os.path.abspath(tb_log_dir) if tb_log_dir else None
        self.enable_tb = bool(enable_tb and SummaryWriter is not None)
        self.enable_console = bool(enable_console)
        self.enable_jsonl = bool(enable_jsonl)
        self.enable_file_logging = bool(enable_file_logging)  # ✅ YENİ
        self.jsonl_filename = jsonl_filename
        self.level = level

        # ✅ DOSYA LOGGING DEVRE DIŞI: Sadece console logger oluştur
        if self.enable_file_logging:
            os.makedirs(self.log_dir, exist_ok=True)
            # Ana logger (bilgi)
            self.logger = self._initialize_logger(
                name="training_logger",
                filename=os.path.join(self.log_dir, "training.log"),
                level=self.level,
                to_console=self.enable_console,
            )
            # Hata logger'ı
            self.error_logger = self._initialize_logger(
                name="error_logger",
                filename=os.path.join(self.log_dir, "errors.log"),
                level=logging.WARNING,
                to_console=self.enable_console,
            )
        else:
            # ✅ DOSYA LOGGING KAPALI: Sadece console logger (basit)
            self.logger = logging.getLogger("training_logger")
            self.logger.propagate = False
            self.logger.setLevel(self.level)
            # Mevcut handler'ları temizle
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)
            # Sadece console handler ekle
            if self.enable_console:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(self.level)
                console_format = logging.Formatter("%(message)s")
                console_handler.setFormatter(console_format)
                self.logger.addHandler(console_handler)
            # Error logger = aynı logger (dosya yok)
            self.error_logger = self.logger

        # JSONL writer
        self._jsonl_fp = None
        if self.enable_jsonl:
            try:
                # [OK] FIX: log_dir dizinini oluştur (JSONL için de gerekli)
                os.makedirs(self.log_dir, exist_ok=True)
                jsonl_path = os.path.join(self.log_dir, self.jsonl_filename)
                self._jsonl_fp = open(jsonl_path, "a", encoding="utf-8")
            except Exception as e:
                self.logger.warning(f"[TrainingLogger] JSONL açılamadı: {e}")
                self._jsonl_fp = None

        # TensorBoard
        self.writer: Optional[SummaryWriter] = None
        if self.enable_tb:
            self.start_tb(tb_log_dir=self.tb_log_dir, run_name=self.run_name)
            
        self._initialized = True

    # ------------------------------------------------------------------ core setup
    def _initialize_logger(
        self,
        name: str,
        filename: str,
        level: int = logging.INFO,
        to_console: bool = True,
    ) -> logging.Logger:
        """
        Logger oluştur - DOSYA LOGGING KAPALI (sadece console).
        """
        logger = logging.getLogger(name)
        logger.propagate = False
        logger.setLevel(level)

        # Mevcut handler'ları temizle (çakışma önleme)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            if hasattr(handler, 'close'):
                try:
                    handler.close()
                except Exception:
                    pass

        # ✅ DOSYA LOGGING TAMAMEN KAPALI - Sadece console handler ekle
        if to_console and not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_format = logging.Formatter("%(message)s")
            console_handler.setFormatter(console_format)
            logger.addHandler(console_handler)

        return logger

    # ---------------------------------------------------------------- TensorBoard
    def start_tb(self, *, tb_log_dir: Optional[str] = None, run_name: Optional[str] = None) -> None:
        """TensorBoard SummaryWriter’ı başlat."""
        if not self.enable_tb or SummaryWriter is None:
            self.logger.info("[TrainingLogger] TensorBoard devre dışı (torch/summarywriter yok ya da disable).")
            return
        base = tb_log_dir or os.path.join(self.log_dir, "tb")
        run = run_name or self.run_name
        os.makedirs(base, exist_ok=True)
        logdir = os.path.join(base, run)
        os.makedirs(logdir, exist_ok=True)
        try:
            self.writer = SummaryWriter(log_dir=logdir)
            self.logger.info(f"[TrainingLogger] TensorBoard aktif → {logdir}")
        except Exception as e:
            self.writer = None
            self.logger.warning(f"[TrainingLogger] TensorBoard başlatılamadı: {e}")

    # ------------------------------------------------------------- std logging API
    def info(self, message: str) -> None:
        self.logger.info(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str, exc_info: bool = False) -> None:
        if exc_info:
            self.error_logger.error(message, exc_info=True)
            self.logger.error(message, exc_info=True)
        else:
            self.error_logger.error(message)
            self.logger.error(message)

    def debug(self, message: str) -> None:
        self.logger.debug(message)

    def critical(self, message: str) -> None:
        self.logger.critical(message)
        self.error_logger.critical(message)

    # ------------------------------------------------------------- legacy aliases
    def log_info(self, message: str) -> None:
        self.info(message)

    def log_warning(self, message: str) -> None:
        self.warning(message)

    def log_error(self, message: str, exc_info: bool = False) -> None:
        self.error(message, exc_info=exc_info)

    def log_critical(self, message: str) -> None:
        self.critical(message)

    def log_debug(self, message: str) -> None:
        self.debug(message)

    # -------------------------------------------------------------------- metrics
    def log_metrics(
        self,
        epoch: int,
        training_loss: float,
        validation_loss: Optional[float] = None,
        accuracy: Optional[float] = None,
        *,
        step: Optional[int] = None,
    ) -> None:
        """Epoch metriklerini hem dosyaya hem TensorBoard'a yazar."""
        msg = f"Epoch {epoch} - Training Loss: {training_loss:.4f}"
        if validation_loss is not None:
            msg += f", Validation Loss: {validation_loss:.4f}"
        if accuracy is not None:
            msg += f", Accuracy: {accuracy:.2f}%"
        self.info(msg)

        # JSONL event
        self.log_event(
            {
                "type": "epoch_metrics",
                "epoch": epoch,
                "train_loss": training_loss,
                "val_loss": validation_loss,
                "accuracy": accuracy,
                "step": step,
            }
        )

        # TensorBoard
        if self.writer:
            self.writer.add_scalar("Loss/Train", training_loss, epoch if step is None else step)
            if validation_loss is not None:
                self.writer.add_scalar("Loss/Validation", validation_loss, epoch if step is None else step)
            if accuracy is not None:
                self.writer.add_scalar("Accuracy", accuracy, epoch if step is None else step)

    def log_validation_metrics(
        self,
        epoch: int,
        validation_loss: float,
        validation_accuracy: Optional[float] = None,
        validation_ppl: Optional[float] = None,
        *,
        step: Optional[int] = None,
    ) -> None:
        """Validation metriklerini özel olarak loglar."""
        msg = f"[VALIDATION] Epoch {epoch} - Val Loss: {validation_loss:.4f}"
        if validation_accuracy is not None:
            msg += f", Val Acc: {validation_accuracy:.2f}%"
        if validation_ppl is not None:
            msg += f", Val PPL: {validation_ppl:.2f}"
        self.info(msg)

        # JSONL event
        self.log_event(
            {
                "type": "validation_metrics",
                "epoch": epoch,
                "val_loss": validation_loss,
                "val_accuracy": validation_accuracy,
                "val_ppl": validation_ppl,
                "step": step,
            }
        )

        # TensorBoard
        if self.writer:
            self.writer.add_scalar("Validation/Loss", validation_loss, epoch if step is None else step)
            if validation_accuracy is not None:
                self.writer.add_scalar("Validation/Accuracy", validation_accuracy, epoch if step is None else step)
            if validation_ppl is not None:
                self.writer.add_scalar("Validation/Perplexity", validation_ppl, epoch if step is None else step)

    # --------------------------------------------------------------- TB utilities
    def log_scalar(self, name: str, value: Union[int, float], step: int) -> None:
        if self.writer:
            self.writer.add_scalar(name, value, step)

    def log_histogram(self, name: str, values: Any, step: int) -> None:
        if self.writer:
            try:
                self.writer.add_histogram(name, values, step)
            except Exception as e:
                self.warning(f"[TrainingLogger] Histogram yazılamadı ({name}): {e}")

    def log_figure(self, name: str, figure: Any, step: int) -> None:
        if self.writer:
            try:
                self.writer.add_figure(name, figure, step)
            except Exception as e:
                self.warning(f"[TrainingLogger] Figure yazılamadı ({name}): {e}")

    def log_text(self, name: str, text: str, step: int) -> None:
        if self.writer:
            try:
                self.writer.add_text(name, text, step)
            except Exception as e:
                self.warning(f"[TrainingLogger] Text yazılamadı ({name}): {e}")

    def add_hparams(self, hparams: Dict[str, Any], metrics: Dict[str, float]) -> None:
        if self.writer:
            try:
                self.writer.add_hparams(hparam_dict=hparams, metric_dict=metrics)
            except Exception as e:
                # add_hparams bazı TB sürümlerinde sorunlu olabilir; sessiz degrade
                self.warning(f"[TrainingLogger] add_hparams başarısız: {e}")

    # ------------------------------------------------------------------- JSONL API
    def log_event(self, event: Dict[str, Any]) -> None:
        """Serbest yapıda bir olayı JSONL’e yazar."""
        if not self._jsonl_fp:
            return
        try:
            self._jsonl_fp.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._jsonl_fp.flush()
        except Exception as e:
            self.warning(f"[TrainingLogger] JSONL yazılamadı: {e}")

    # ----------------------------------------------------------------------- close
    def close(self) -> None:
        """Kaynakları kapat (TB writer, JSONL dosyası, logger handlers)."""
        try:
            if self.writer:
                self.writer.flush()
                self.writer.close()
                self.writer = None
        except Exception as e:
            print(f"[TrainingLogger] TensorBoard kapatma hatası: {e}")
            
        try:
            if self._jsonl_fp:
                self._jsonl_fp.close()
                self._jsonl_fp = None
        except Exception as e:
            print(f"[TrainingLogger] JSONL kapatma hatası: {e}")
            
        # Logger handler'ları güvenli kapatma
        try:
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)
                if hasattr(handler, 'close'):
                    handler.close()
        except Exception as e:
            print(f"[TrainingLogger] Logger handler kapatma hatası: {e}")
            
        try:
            for handler in self.error_logger.handlers[:]:
                self.error_logger.removeHandler(handler)
                if hasattr(handler, 'close'):
                    handler.close()
        except Exception as e:
            print(f"[TrainingLogger] Error logger handler kapatma hatası: {e}")

    def __del__(self):
        """Destructor - otomatik temizlik."""
        try:
            self.close()
        except Exception:
            pass
    
    @classmethod
    def reset(cls):
        """Singleton'ı sıfırla - test/debug için."""
        if cls._instance:
            try:
                cls._instance.close()
            except Exception:
                pass
        cls._instance = None
        cls._initialized = False
