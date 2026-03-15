# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: exceptions.py
Modül: model_management
Görev: model_management modülüne özel exception hiyerarşisi.
       Tüm hatalar bu hiyerarşi üzerinden fırlatılır; böylece try/except
       bloklarında granüler yakalama yapılabilir ve hata kaynağı netleşir.

       CevahirModelError
       ├── ModelNotInitializedError   → initialize() çağrılmadan kullanım
       ├── ModelBuildError            → model / optimizer / scheduler oluşturma
       │    └── QuantizationError     → INT8/INT4 quantization başarısızlığı
       ├── CheckpointError            → checkpoint I/O
       │    ├── CheckpointNotFoundError
       │    ├── CheckpointCorruptError → SHA-256 / format tutarsızlığı
       │    └── CheckpointVersionError → model/checkpoint versiyon uyumsuzluğu
       ├── ForwardError               → model forward pass
       │    └── OOMRecoveryError      → CUDA OOM → kurtarma başarısız
       ├── DeviceError                → device seçimi / transfer
       │    └── DeviceMismatchError   → tensor device uyumsuzluğu
       ├── ShapeError                 → tensor şekil uyumsuzluğu
       │    └── VocabSizeMismatchError→ vocab_size checkpoint vs model
       ├── DistributedSetupError      → DDP / FSDP kurulum hatası
       └── HealthCheckError           → model sağlık testi başarısız

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
================================================================================
"""

from __future__ import annotations

from typing import Any, Optional


# ══════════════════════════════════════════════════════════════════════════════
# Kök İstisna
# ══════════════════════════════════════════════════════════════════════════════

class CevahirModelError(Exception):
    """
    model_management modülündeki tüm özel exception'ların tabanı.

    Attributes:
        message  : İnsan okunabilir hata açıklaması.
        context  : Opsiyonel ek bağlam sözlüğü (dosya yolu, şekil, epoch vb.).
    """

    def __init__(self, message: str, *, context: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = context or {}

    def __str__(self) -> str:
        base = self.message
        if self.context:
            ctx_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{base} [{ctx_str}]"
        return base

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.message!r}, context={self.context!r})"


# ══════════════════════════════════════════════════════════════════════════════
# Başlatma / Model Oluşturma Hataları
# ══════════════════════════════════════════════════════════════════════════════

class ModelNotInitializedError(CevahirModelError):
    """
    Model, optimizer veya başka bir bileşen initialize() çağrılmadan kullanılmaya
    çalışıldığında fırlatılır.

    Example:
        raise ModelNotInitializedError("optimizer", hint="initialize(build_optimizer=True)")
    """

    def __init__(self, component: str = "model", *, hint: str = "initialize() çağırın") -> None:
        super().__init__(
            f"'{component}' henüz hazır değil; {hint}.",
            context={"component": component},
        )
        self.component = component


class ModelBuildError(CevahirModelError):
    """
    Model, optimizer, criterion veya scheduler oluşturulurken meydana gelen hata.
    """

    def __init__(self, component: str, reason: str, *, context: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            f"{component} oluşturulamadı: {reason}",
            context={"component": component, **(context or {})},
        )
        self.component = component


class QuantizationError(ModelBuildError):
    """
    INT8 / INT4 quantization başarısız olduğunda fırlatılır.
    bitsandbytes eksik veya uyumsuz donanım gibi durumlar.
    """

    def __init__(self, quant_type: str, reason: str) -> None:
        super().__init__(
            component=f"Quantization[{quant_type}]",
            reason=reason,
            context={"quant_type": quant_type},
        )
        self.quant_type = quant_type


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint Hataları
# ══════════════════════════════════════════════════════════════════════════════

class CheckpointError(CevahirModelError):
    """Checkpoint okuma/yazma işlemleriyle ilgili tüm hatalar için taban."""

    def __init__(self, message: str, *, path: Optional[str] = None, context: Optional[dict[str, Any]] = None) -> None:
        ctx = {"path": path, **(context or {})} if path else (context or {})
        super().__init__(message, context=ctx)
        self.path = path


class CheckpointNotFoundError(CheckpointError):
    """Belirtilen dosya yolunda checkpoint bulunamadığında fırlatılır."""

    def __init__(self, path: str) -> None:
        super().__init__(f"Checkpoint dosyası bulunamadı: '{path}'", path=path)


class CheckpointCorruptError(CheckpointError):
    """
    Checkpoint dosyasının SHA-256 toplamı eşleşmediğinde ya da beklenen
    anahtar yapısı bozuk olduğunda fırlatılır.
    """

    def __init__(self, path: str, reason: str = "bütünlük kontrolü başarısız") -> None:
        super().__init__(
            f"Checkpoint bozuk veya güvenilmez ({reason}): '{path}'",
            path=path,
            context={"reason": reason},
        )
        self.reason = reason


class CheckpointVersionError(CheckpointError):
    """
    Checkpoint'in kayıt versiyonu ile beklenen versiyon uyuşmadığında fırlatılır.
    Örneğin: MoE etkin checkpoint → MoE etsiz model.
    """

    def __init__(
        self,
        path: str,
        saved_version: Any,
        expected_version: Any,
    ) -> None:
        super().__init__(
            f"Checkpoint versiyon uyumsuzluğu: kayıt={saved_version!r}, beklenen={expected_version!r}",
            path=path,
            context={"saved_version": saved_version, "expected_version": expected_version},
        )
        self.saved_version = saved_version
        self.expected_version = expected_version


# ══════════════════════════════════════════════════════════════════════════════
# Forward / Inference Hataları
# ══════════════════════════════════════════════════════════════════════════════

class ForwardError(CevahirModelError):
    """
    Model forward() geçişi sırasında meydana gelen hata.
    Tensor şekli, dtype, veya hesaplama hataları.
    """

    def __init__(self, reason: str, *, input_shape: Optional[tuple] = None) -> None:
        ctx: dict[str, Any] = {}
        if input_shape is not None:
            ctx["input_shape"] = input_shape
        super().__init__(f"Forward pass hatası: {reason}", context=ctx)


class OOMRecoveryError(ForwardError):
    """
    CUDA Out-Of-Memory hatası gerçekleşti ve otomatik kurtarma başarısız oldu.
    """

    def __init__(
        self,
        allocated_gb: float,
        required_gb: float,
        recovery_attempted: bool = True,
    ) -> None:
        super().__init__(
            f"CUDA OOM: tahsis={allocated_gb:.2f} GB, istek={required_gb:.2f} GB, "
            f"kurtarma={'denendi' if recovery_attempted else 'denenmedi'}",
            input_shape=None,
        )
        self.allocated_gb = allocated_gb
        self.required_gb = required_gb
        self.recovery_attempted = recovery_attempted


# ══════════════════════════════════════════════════════════════════════════════
# Device Hataları
# ══════════════════════════════════════════════════════════════════════════════

class DeviceError(CevahirModelError):
    """Device seçimi veya tensor transferiyle ilgili hatalar."""

    def __init__(self, message: str, *, device: Optional[str] = None) -> None:
        super().__init__(message, context={"device": device} if device else {})
        self.device = device


class DeviceMismatchError(DeviceError):
    """Tensor'lar farklı device'larda olduğunda fırlatılır."""

    def __init__(self, expected: str, got: str) -> None:
        super().__init__(
            f"Device uyumsuzluğu: beklenen={expected!r}, gelen={got!r}",
            device=expected,
        )
        self.expected = expected
        self.got = got


# ══════════════════════════════════════════════════════════════════════════════
# Şekil / Vocab Hataları
# ══════════════════════════════════════════════════════════════════════════════

class ShapeError(CevahirModelError):
    """Tensor şekli beklenenden farklı olduğunda fırlatılır."""

    def __init__(self, expected: Any, got: Any, *, where: str = "") -> None:
        location = f" ({where})" if where else ""
        super().__init__(
            f"Tensor şekli uyumsuz{location}: beklenen={expected}, gelen={got}",
            context={"expected": expected, "got": got, "where": where},
        )
        self.expected = expected
        self.got = got


class VocabSizeMismatchError(ShapeError):
    """
    Checkpoint'teki vocab_size ile modelin embedding boyutu uyuşmuyor.
    Bu genellikle yanlış checkpoint-model çifti yüklendiğinde olur.
    """

    def __init__(self, model_vocab: int, checkpoint_vocab: int) -> None:
        super().__init__(
            expected=model_vocab,
            got=checkpoint_vocab,
            where="embedding.weight",
        )
        self.model_vocab = model_vocab
        self.checkpoint_vocab = checkpoint_vocab


# ══════════════════════════════════════════════════════════════════════════════
# Dağıtık Eğitim Hataları
# ══════════════════════════════════════════════════════════════════════════════

class DistributedSetupError(CevahirModelError):
    """
    DDP (DistributedDataParallel) veya FSDP (FullyShardedDataParallel)
    kurulumu sırasında meydana gelen hata.
    """

    def __init__(self, backend: str, reason: str) -> None:
        super().__init__(
            f"Dağıtık eğitim ({backend}) kurulamadı: {reason}",
            context={"backend": backend, "reason": reason},
        )
        self.backend = backend


# ══════════════════════════════════════════════════════════════════════════════
# Model Sağlık Testi Hataları
# ══════════════════════════════════════════════════════════════════════════════

class HealthCheckError(CevahirModelError):
    """
    Model sağlık testi (gradient, ağırlık, attention entropy) başarısız olduğunda
    veya kritik bir patoloji tespit edildiğinde fırlatılır.
    """

    def __init__(self, check_name: str, details: str) -> None:
        super().__init__(
            f"Sağlık testi başarısız [{check_name}]: {details}",
            context={"check": check_name, "details": details},
        )
        self.check_name = check_name
        self.details = details
