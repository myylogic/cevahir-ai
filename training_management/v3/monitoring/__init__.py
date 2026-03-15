"""
training_management/v3/monitoring/__init__.py
=============================================
Cevahir V3 Eğitim Sistemi — İzleme (monitoring) paketi.

Bu paket, eğitim sürecini kapsamlı biçimde izlemek için
dört temel bileşen sağlar:

Bileşenler:
-----------
GradientHealthMonitor
    Per-layer gradient sağlık izleme sistemi.
    Norm, varyans, dead neuron oranı ve flow skoru hesaplar.
    Vanishing/exploding gradient erken uyarısı verir.

TokenDistributionMonitor
    Model çıktısının token dağılımını izler.
    EOS oranı, unigram entropy ve TTR ile collapse tespiti yapar.

InferenceQualityProbe
    Her N epoch'ta gerçek inference çalıştırır.
    Teacher forcing'in gizlediği collapse'ı tespit eder.
    Sabit Türkçe/İngilizce test prompt'larıyla kalite ölçer.

TensorBoardManager
    Yukarıdaki monitörlerin TensorBoard entegrasyonunu yönetir.
    V2'ye kıyasla: Entropy, GradHealth, TokenDist, InferenceQuality,
    LLRD, EMA, Safety ve Curriculum metriklerini destekler.

Örnek Kullanım:
---------------
    from training_management.v3.monitoring import (
        GradientHealthMonitor,
        TokenDistributionMonitor,
        InferenceQualityProbe,
        TensorBoardManager,
        InferenceQualityMetrics,
    )

    # TensorBoard yöneticisi başlat
    with TensorBoardManager("runs/cevahir_v3") as tb:
        grad_monitor = GradientHealthMonitor(model, tb.writer)
        token_monitor = TokenDistributionMonitor(tokenizer=tok, eos_id=2)
        probe = InferenceQualityProbe(
            model_manager=manager,
            tokenizer=tok,
            probe_interval=5,
            tensorboard_writer=tb.writer,
        )

        for epoch in range(num_epochs):
            for batch in dataloader:
                loss = train_step(batch)
                token_monitor.update(logits)

            # Gradient sağlık kontrolü
            grad_health = grad_monitor.compute()
            grad_monitor.log_to_tensorboard(global_step)

            # Inference probe (her 5 epoch'ta)
            if probe.should_probe(epoch):
                metrics = probe.run(epoch)

            # Epoch metrikleri
            tb.log_epoch(
                epoch=epoch,
                train_metrics={"loss": train_loss, "accuracy": acc},
                val_metrics={"loss": val_loss},
                lr=optimizer.param_groups[0]["lr"],
                gradient_health=grad_monitor.get_summary(),
                token_dist=token_monitor.get_stats(),
            )

Yazar: Cevahir Sinir Sistemi V3
Tarih: 2026
"""

from training_management.v3.monitoring.gradient_health_monitor import (
    GradientHealthMonitor,
    LayerGradientStats,
    VANISHING_NORM_THRESHOLD,
    EXPLODING_NORM_THRESHOLD,
    DEAD_NEURON_THRESHOLD,
)

from training_management.v3.monitoring.token_distribution_monitor import (
    TokenDistributionMonitor,
    COLLAPSE_EOS_RATIO,
    COLLAPSE_ENTROPY_THRESHOLD,
)

from training_management.v3.monitoring.inference_quality_probe import (
    InferenceQualityProbe,
    InferenceQualityMetrics,
    COLLAPSE_MAX_AVG_LENGTH,
    COLLAPSE_MIN_ENTROPY,
)

from training_management.v3.monitoring.tensorboard_manager import (
    TensorBoardManager,
)

# ---------------------------------------------------------------------------
# Paket düzeyinde dışa aktarım listesi
# ---------------------------------------------------------------------------

__all__ = [
    # Ana sınıflar
    "GradientHealthMonitor",
    "TokenDistributionMonitor",
    "InferenceQualityProbe",
    "TensorBoardManager",

    # TypedDict
    "InferenceQualityMetrics",

    # Yardımcı sınıflar
    "LayerGradientStats",

    # Sabitler (isteğe bağlı, dışarıdan özelleştirme için)
    "VANISHING_NORM_THRESHOLD",
    "EXPLODING_NORM_THRESHOLD",
    "DEAD_NEURON_THRESHOLD",
    "COLLAPSE_EOS_RATIO",
    "COLLAPSE_ENTROPY_THRESHOLD",
    "COLLAPSE_MAX_AVG_LENGTH",
    "COLLAPSE_MIN_ENTROPY",
]

# ---------------------------------------------------------------------------
# Paket meta verisi
# ---------------------------------------------------------------------------

__version__ = "3.0.0"
__author__ = "Cevahir Sinir Sistemi"
__description__ = "Cevahir V3 — Kapsamlı eğitim izleme ve TensorBoard entegrasyon paketi"
