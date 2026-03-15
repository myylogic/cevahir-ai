"""
training_management/v3/safety/__init__.py
==========================================
Cevahir Türkçe Dil Modeli - Eğitim Güvenlik Modülleri (v3)

Bu paket, Cevahir modelinin eğitim süreci için sayısal kararlılık,
anomali tespiti ve checkpoint bütünlüğünü sağlayan güvenlik bileşenlerini içerir.

Bileşenler:
    NaNRecovery        : NaN/Inf tespiti ve otomatik kurtarma
    LossSpikeDetector  : Loss spike tespiti ve müdahale
    DivergenceDetector : Train/val divergence ve convergence takibi
    CheckpointVerifier : Checkpoint bütünlük doğrulaması

Enum Tipleri:
    NaNRecoveryAction  : NaN kurtarma eylem tipleri
    ConvergenceStatus  : Convergence durum tipleri

Hızlı Başlangıç:
    >>> from training_management.v3.safety import (
    ...     NaNRecovery, LossSpikeDetector,
    ...     DivergenceDetector, CheckpointVerifier,
    ...     ConvergenceStatus, NaNRecoveryAction,
    ... )
    >>> nan_recovery = NaNRecovery(optimizer, nan_tolerance=3)
    >>> spike_detector = LossSpikeDetector(optimizer, window_size=10)
    >>> divergence_detector = DivergenceDetector(patience=20)
    >>> verifier = CheckpointVerifier(device="cpu")
"""

from training_management.v3.safety.checkpoint_verifier import CheckpointVerifier
from training_management.v3.safety.divergence_detector import (
    ConvergenceStatus,
    DivergenceDetector,
)
from training_management.v3.safety.loss_spike_detector import LossSpikeDetector
from training_management.v3.safety.nan_recovery import NaNRecovery, NaNRecoveryAction

__all__ = [
    # Ana sınıflar
    "NaNRecovery",
    "LossSpikeDetector",
    "DivergenceDetector",
    "CheckpointVerifier",
    # Enum tipleri
    "NaNRecoveryAction",
    "ConvergenceStatus",
]

__version__ = "3.0.0"
__author__ = "Cevahir AI Geliştirme Ekibi"
