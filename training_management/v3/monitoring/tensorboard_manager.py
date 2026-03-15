"""
tensorboard_manager.py
=======================
Cevahir V3 Eğitim Sistemi — TensorBoard entegrasyon yöneticisi.

V2'ye kıyasla genişletilmiş metrik kategorileri:
- Entropy, TokenDist, GradHealth, InferenceQuality
- Loss ayrıntıları (EntropyReg, Focal, Auxiliary)
- LLRD per-grup öğrenme hızları
- EMA model performansı
- Safety (NaN/Spike) ve Curriculum metrikleri

Yazar: Cevahir Sinir Sistemi V3
Tarih: 2026
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TensorBoard import — opsiyonel (kurulu değilse sessizce devre dışı kalır)
# ---------------------------------------------------------------------------

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    SummaryWriter = None  # type: ignore
    _TB_AVAILABLE = False
    logger.warning(
        "TensorBoard kurulu değil. 'pip install tensorboard' komutunu çalıştırın. "
        "Logging devre dışı bırakıldı."
    )


class TensorBoardManager:
    """
    Cevahir V3 TensorBoard entegrasyonu — kapsamlı metrik logging yöneticisi.

    V2'ye göre yeni metrikler:
    - Entropy/Train_Avg, Entropy/Layer_{i}
    - TokenDist/EOS_Ratio, TokenDist/ContentRatio, TokenDist/UnigramEntropy
    - GradHealth/DeadNeuronRatio, GradHealth/OverallScore
    - InferenceQuality/* (probe sonuçları)
    - Loss/EntropyReg, Loss/Focal, Loss/Auxiliary, Loss/Total
    - LR/PerGroup_{i} (LLRD)
    - EMA/ValLoss (EMA model performance)
    - Safety/NaNCount, Safety/SpikeCount
    - Curriculum/TeacherForcingProb, Curriculum/CurrentMaxLen
    """

    def __init__(self, log_dir: str, enabled: bool = True) -> None:
        """
        Args:
            log_dir : TensorBoard log dizini (oluşturulmazsa oluşturulur).
            enabled : False ise tüm logging işlemleri sessizce atlanır.
                      TensorBoard kurulu değilse de otomatik devre dışı kalır.
        """
        self.log_dir = log_dir
        self.enabled = enabled and _TB_AVAILABLE

        self._writer: Optional[Any] = None  # SummaryWriter örneği

        if self.enabled:
            try:
                os.makedirs(log_dir, exist_ok=True)
                self._writer = SummaryWriter(log_dir=log_dir)
                logger.info(
                    "TensorBoardManager başlatıldı: log_dir='%s'", log_dir
                )
            except Exception as e:
                logger.error(
                    "TensorBoard SummaryWriter oluşturulamadı: %s. "
                    "Logging devre dışı.",
                    e,
                )
                self.enabled = False
                self._writer = None
        else:
            if not _TB_AVAILABLE:
                logger.info("TensorBoardManager: TensorBoard kurulu değil, devre dışı.")
            else:
                logger.info("TensorBoardManager: enabled=False, logging atlanacak.")

    # ------------------------------------------------------------------
    # Dahili yardımcılar
    # ------------------------------------------------------------------

    def _safe_add_scalar(
        self, tag: str, value: float, step: int
    ) -> None:
        """
        NaN/Inf değerleri güvenli şekilde filtrelerek scalar yaz.

        Args:
            tag   : TensorBoard tag adı.
            value : Yazılacak sayısal değer.
            step  : Global adım numarası.
        """
        if self._writer is None:
            return
        if value is None:
            return
        import math
        if math.isnan(value) or math.isinf(value):
            logger.debug(
                "TensorBoard: NaN/Inf değer atlandı — tag='%s', value=%s",
                tag, value,
            )
            return
        try:
            self._writer.add_scalar(tag, float(value), step)
        except Exception as e:
            logger.debug("TensorBoard scalar yazma hatası (tag=%s): %s", tag, e)

    # ------------------------------------------------------------------
    # Ana logging metodları
    # ------------------------------------------------------------------

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Dict,
        lr: float,
        gradient_health: Optional[Dict] = None,
        token_dist: Optional[Dict] = None,
        loss_breakdown: Optional[Dict] = None,
        safety_stats: Optional[Dict] = None,
        curriculum_stats: Optional[Dict] = None,
        ema_val_loss: Optional[float] = None,
    ) -> None:
        """
        Epoch sonu için tüm metrikleri TensorBoard'a yaz.

        Args:
            epoch            : Mevcut epoch numarası.
            train_metrics    : Eğitim metrikleri dict'i.
                               Beklenen anahtarlar: loss, accuracy, perplexity,
                               avg_entropy, lr_list (opsiyonel)
            val_metrics      : Doğrulama metrikleri dict'i.
                               Beklenen anahtarlar: loss, accuracy, perplexity
            lr               : Mevcut öğrenme hızı (skaler veya ana grup).
            gradient_health  : GradientHealthMonitor.get_summary() çıktısı.
            token_dist       : TokenDistributionMonitor.get_stats() çıktısı.
            loss_breakdown   : Ayrıntılı kayıp bileşenleri.
                               Beklenen: total, cross_entropy, entropy_reg,
                                         focal, auxiliary
            safety_stats     : Güvenlik istatistikleri.
                               Beklenen: nan_count, spike_count, grad_clip_count
            curriculum_stats : Curriculum learning istatistikleri.
                               Beklenen: teacher_forcing_prob, current_max_len,
                                         difficulty_level
            ema_val_loss     : EMA model validation loss'u.
        """
        if not self.enabled or self._writer is None:
            return

        step = epoch  # Epoch bazlı grafik

        # --- Temel Eğitim Metrikleri ---
        self._safe_add_scalar("Loss/Train", train_metrics.get("loss", 0.0), step)
        self._safe_add_scalar("Loss/Val", val_metrics.get("loss", 0.0), step)
        self._safe_add_scalar("Accuracy/Train", train_metrics.get("accuracy", 0.0), step)
        self._safe_add_scalar("Accuracy/Val", val_metrics.get("accuracy", 0.0), step)
        self._safe_add_scalar(
            "Perplexity/Train", train_metrics.get("perplexity", 0.0), step
        )
        self._safe_add_scalar(
            "Perplexity/Val", val_metrics.get("perplexity", 0.0), step
        )

        # --- Öğrenme Hızı ---
        self._safe_add_scalar("LR/Main", lr, step)

        # Per-group öğrenme hızları (LLRD)
        lr_list = train_metrics.get("lr_list", [])
        for i, group_lr in enumerate(lr_list):
            self._safe_add_scalar(f"LR/PerGroup_{i}", group_lr, step)

        # --- Entropy ---
        self._safe_add_scalar(
            "Entropy/Train_Avg", train_metrics.get("avg_entropy", 0.0), step
        )
        # Per-layer entropy (varsa)
        layer_entropies = train_metrics.get("layer_entropies", [])
        for i, ent in enumerate(layer_entropies):
            self._safe_add_scalar(f"Entropy/Layer_{i}", ent, step)

        # --- Loss Ayrıntıları ---
        if loss_breakdown is not None:
            self._safe_add_scalar(
                "Loss/Total", loss_breakdown.get("total", 0.0), step
            )
            self._safe_add_scalar(
                "Loss/CrossEntropy", loss_breakdown.get("cross_entropy", 0.0), step
            )
            self._safe_add_scalar(
                "Loss/EntropyReg", loss_breakdown.get("entropy_reg", 0.0), step
            )
            self._safe_add_scalar(
                "Loss/Focal", loss_breakdown.get("focal", 0.0), step
            )
            self._safe_add_scalar(
                "Loss/Auxiliary", loss_breakdown.get("auxiliary", 0.0), step
            )

        # --- EMA Model Performansı ---
        if ema_val_loss is not None:
            self._safe_add_scalar("EMA/ValLoss", ema_val_loss, step)

        # --- Gradient Sağlık Metrikleri ---
        if gradient_health is not None:
            self._safe_add_scalar(
                "GradHealth/OverallScore",
                gradient_health.get("overall_health_score", 0.0),
                step,
            )
            self._safe_add_scalar(
                "GradHealth/DeadLayerCount",
                float(len(gradient_health.get("dead_layers", []))),
                step,
            )
            self._safe_add_scalar(
                "GradHealth/ExplodingLayerCount",
                float(len(gradient_health.get("exploding_layers", []))),
                step,
            )
            self._safe_add_scalar(
                "GradHealth/VanishingLayerCount",
                float(len(gradient_health.get("vanishing_layers", []))),
                step,
            )
            # Dead neuron oranı (tüm katmanlar ortalaması, varsa)
            dead_ratio = gradient_health.get("dead_neuron_ratio_avg")
            if dead_ratio is not None:
                self._safe_add_scalar(
                    "GradHealth/DeadNeuronRatio", dead_ratio, step
                )

        # --- Token Dağılım Metrikleri ---
        if token_dist is not None:
            self._safe_add_scalar(
                "TokenDist/EOS_Ratio", token_dist.get("eos_ratio", 0.0), step
            )
            self._safe_add_scalar(
                "TokenDist/ContentRatio", token_dist.get("content_ratio", 0.0), step
            )
            self._safe_add_scalar(
                "TokenDist/UnigramEntropy",
                token_dist.get("unigram_entropy", 0.0),
                step,
            )
            self._safe_add_scalar(
                "TokenDist/TypeTokenRatio",
                token_dist.get("type_token_ratio", 0.0),
                step,
            )
            self._safe_add_scalar(
                "TokenDist/IsCollapsed",
                1.0 if token_dist.get("is_collapsed", False) else 0.0,
                step,
            )

        # --- Safety İstatistikleri ---
        if safety_stats is not None:
            self._safe_add_scalar(
                "Safety/NaNCount",
                float(safety_stats.get("nan_count", 0)),
                step,
            )
            self._safe_add_scalar(
                "Safety/SpikeCount",
                float(safety_stats.get("spike_count", 0)),
                step,
            )
            self._safe_add_scalar(
                "Safety/GradClipCount",
                float(safety_stats.get("grad_clip_count", 0)),
                step,
            )

        # --- Curriculum İstatistikleri ---
        if curriculum_stats is not None:
            self._safe_add_scalar(
                "Curriculum/TeacherForcingProb",
                curriculum_stats.get("teacher_forcing_prob", 0.0),
                step,
            )
            self._safe_add_scalar(
                "Curriculum/CurrentMaxLen",
                float(curriculum_stats.get("current_max_len", 0)),
                step,
            )
            self._safe_add_scalar(
                "Curriculum/DifficultyLevel",
                float(curriculum_stats.get("difficulty_level", 0)),
                step,
            )

    def log_inference_quality(self, epoch: int, metrics: Dict) -> None:
        """
        InferenceQualityProbe sonuçlarını TensorBoard'a yaz.

        Tag'ler:
        InferenceQuality/AvgResponseLength
        InferenceQuality/AvgEntropy
        InferenceQuality/EOSRatio
        InferenceQuality/TypeTokenRatio
        InferenceQuality/IsCollapsed

        Args:
            epoch   : Epoch numarası.
            metrics : InferenceQualityProbe.run() döndürdüğü dict.
        """
        if not self.enabled or self._writer is None:
            return

        self._safe_add_scalar(
            "InferenceQuality/AvgResponseLength",
            metrics.get("avg_response_length", 0.0),
            epoch,
        )
        self._safe_add_scalar(
            "InferenceQuality/AvgEntropy",
            metrics.get("avg_entropy", 0.0),
            epoch,
        )
        self._safe_add_scalar(
            "InferenceQuality/EOSRatio",
            metrics.get("avg_eos_ratio", 0.0),
            epoch,
        )
        self._safe_add_scalar(
            "InferenceQuality/TypeTokenRatio",
            metrics.get("type_token_ratio", 0.0),
            epoch,
        )
        self._safe_add_scalar(
            "InferenceQuality/IsCollapsed",
            1.0 if metrics.get("is_collapsed", False) else 0.0,
            epoch,
        )

        # Örnek yanıtları metin olarak kaydet
        responses = metrics.get("responses", [])
        if responses and self._writer is not None:
            text_lines = []
            for i, resp in enumerate(responses[:8]):  # İlk 8 yanıt
                text_lines.append(f"**Yanıt {i+1}:** {resp or '(boş)'}")
            try:
                self._writer.add_text(
                    "InferenceQuality/SampleResponses",
                    "\n\n".join(text_lines),
                    epoch,
                )
            except Exception as e:
                logger.debug("TensorBoard add_text hatası: %s", e)

    def log_batch(
        self,
        global_step: int,
        loss: float,
        lr: float,
        grad_norm: float,
        tokens_per_sec: float,
    ) -> None:
        """
        Batch-level metrikler (daha sık, epoch'tan bağımsız).

        Args:
            global_step    : Global batch adım numarası.
            loss           : Mevcut batch loss değeri.
            lr             : Mevcut öğrenme hızı.
            grad_norm      : Gradient norm (clipping öncesi).
            tokens_per_sec : Saniyede işlenen token sayısı (throughput).
        """
        if not self.enabled or self._writer is None:
            return

        self._safe_add_scalar("Batch/Loss", loss, global_step)
        self._safe_add_scalar("Batch/LR", lr, global_step)
        self._safe_add_scalar("Batch/GradNorm", grad_norm, global_step)
        self._safe_add_scalar("Batch/TokensPerSec", tokens_per_sec, global_step)

    def log_histogram(
        self, tag: str, values: torch.Tensor, global_step: int
    ) -> None:
        """
        Histogram logging — gradient dağılımları, ağırlık dağılımları vb.

        Args:
            tag         : TensorBoard tag adı.
            values      : Histogram verisi (herhangi boyutlu Tensor).
            global_step : Global adım numarası.
        """
        if not self.enabled or self._writer is None:
            return

        try:
            # Sonsuz veya NaN içeriyorsa atla
            finite_vals = values[torch.isfinite(values)]
            if finite_vals.numel() == 0:
                logger.debug(
                    "log_histogram: Tüm değerler NaN/Inf, atlandı (tag=%s)", tag
                )
                return
            self._writer.add_histogram(tag, finite_vals.cpu(), global_step)
        except Exception as e:
            logger.debug("TensorBoard histogram yazma hatası (tag=%s): %s", tag, e)

    def log_text(self, tag: str, text: str, global_step: int) -> None:
        """
        Metin logging — inference yanıtları, config özetleri vb.

        Args:
            tag         : TensorBoard tag adı.
            text        : Yazılacak metin (Markdown desteklenir).
            global_step : Global adım numarası.
        """
        if not self.enabled or self._writer is None:
            return

        try:
            self._writer.add_text(tag, text, global_step)
        except Exception as e:
            logger.debug("TensorBoard metin yazma hatası (tag=%s): %s", tag, e)

    def log_gradient_health_detail(
        self,
        global_step: int,
        layer_metrics: Dict,
    ) -> None:
        """
        GradientHealthMonitor'ün per-layer metriklerini yaz.

        Args:
            global_step   : Global adım numarası.
            layer_metrics : GradientHealthMonitor.compute() çıktısı.
        """
        if not self.enabled or self._writer is None:
            return

        for layer_name, stats in layer_metrics.items():
            tb_name = layer_name.replace(".", "/")
            self._safe_add_scalar(
                f"GradHealth/{tb_name}/Norm", stats.get("norm", 0.0), global_step
            )
            self._safe_add_scalar(
                f"GradHealth/{tb_name}/DeadRatio",
                stats.get("dead_ratio", 0.0),
                global_step,
            )
            self._safe_add_scalar(
                f"GradHealth/{tb_name}/FlowScore",
                stats.get("flow_score", 0.0),
                global_step,
            )
            self._safe_add_scalar(
                f"GradHealth/{tb_name}/Variance",
                stats.get("variance", 0.0),
                global_step,
            )

    def log_learning_rates(
        self, global_step: int, lr_groups: List[float]
    ) -> None:
        """
        LLRD (Layer-wise Learning Rate Decay) için per-group LR logla.

        Args:
            global_step : Global adım numarası.
            lr_groups   : Her optimizer grubu için öğrenme hızı listesi.
        """
        if not self.enabled or self._writer is None:
            return

        for i, group_lr in enumerate(lr_groups):
            self._safe_add_scalar(f"LR/PerGroup_{i}", group_lr, global_step)

    def log_model_stats(
        self,
        global_step: int,
        total_params: int,
        trainable_params: int,
    ) -> None:
        """
        Model parametre istatistiklerini logla.

        Args:
            global_step      : Global adım numarası.
            total_params     : Toplam parametre sayısı.
            trainable_params : Eğitilebilir parametre sayısı.
        """
        if not self.enabled or self._writer is None:
            return

        self._safe_add_scalar("Model/TotalParams", float(total_params), global_step)
        self._safe_add_scalar(
            "Model/TrainableParams", float(trainable_params), global_step
        )
        if total_params > 0:
            ratio = trainable_params / total_params
            self._safe_add_scalar("Model/TrainableRatio", ratio, global_step)

    def flush(self) -> None:
        """
        TensorBoard yazma tamponunu temizle (disk'e yaz).
        Epoch sonunda veya checkpoint'lerde çağır.
        """
        if self._writer is not None:
            try:
                self._writer.flush()
            except Exception as e:
                logger.debug("TensorBoard flush hatası: %s", e)

    def close(self) -> None:
        """
        TensorBoard writer'ı kapat ve kaynakları serbest bırak.
        Eğitim tamamlandığında çağır.
        """
        if self._writer is not None:
            try:
                self._writer.flush()
                self._writer.close()
                logger.info("TensorBoardManager kapatıldı.")
            except Exception as e:
                logger.warning("TensorBoard kapatma hatası: %s", e)
            finally:
                self._writer = None

    @contextmanager
    def as_context(self) -> Generator["TensorBoardManager", None, None]:
        """
        Context manager desteği — `with tb_manager:` syntax'ı.

        Blok çıkışında writer otomatik olarak kapatılır.

        Kullanım::

            with TensorBoardManager("runs/exp1") as tb:
                for epoch in range(100):
                    tb.log_epoch(epoch, ...)
        """
        try:
            yield self
        finally:
            self.close()

    def __enter__(self) -> "TensorBoardManager":
        """with statement giriş noktası."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """with statement çıkış noktası — writer'ı kapat."""
        self.close()

    @property
    def is_enabled(self) -> bool:
        """TensorBoard logging aktif mi?"""
        return self.enabled and self._writer is not None

    @property
    def writer(self) -> Optional[Any]:
        """Altta yatan SummaryWriter örneği (ileri düzey kullanım için)."""
        return self._writer

    def get_log_dir(self) -> str:
        """Log dizin yolunu döndür."""
        return self.log_dir

    def __repr__(self) -> str:
        status = "aktif" if self.is_enabled else "devre dışı"
        return (
            f"TensorBoardManager("
            f"log_dir='{self.log_dir}', "
            f"status={status})"
        )
