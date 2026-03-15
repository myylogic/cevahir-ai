"""
gradient_health_monitor.py
==========================
Cevahir V3 Eğitim Sistemi — Per-layer gradient sağlık izleme modülü.

Her katman için gradient norm, varyans, ölü nöron oranı ve
gradient akış skoru hesaplanır. TensorBoard entegrasyonu mevcuttur.

Yazar: Cevahir Sinir Sistemi V3
Tarih: 2026
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sabitler ve eşik değerleri
# ---------------------------------------------------------------------------

# Gradient norm eşik değerleri
VANISHING_NORM_THRESHOLD: float = 1e-7   # Bu değerin altı → vanishing
EXPLODING_NORM_THRESHOLD: float = 10.0   # Bu değerin üstü → exploding

# Ölü nöron oranı eşiği
DEAD_NEURON_THRESHOLD: float = 0.3       # Bu değerin üstü → katman donuyor

# Flow skoru hesabı için kullanılan loglama tabanı
_LOG_BASE: float = math.e


class LayerGradientStats:
    """Tek bir katmanın gradient istatistikleri."""

    __slots__ = ("norm", "variance", "dead_ratio", "flow_score", "status",
                 "param_count", "nonzero_count")

    def __init__(
        self,
        norm: float,
        variance: float,
        dead_ratio: float,
        flow_score: float,
        status: str,
        param_count: int,
        nonzero_count: int,
    ) -> None:
        self.norm = norm
        self.variance = variance
        self.dead_ratio = dead_ratio
        self.flow_score = flow_score
        self.status = status          # "healthy" | "vanishing" | "exploding" | "dead"
        self.param_count = param_count
        self.nonzero_count = nonzero_count

    def to_dict(self) -> Dict:
        return {
            "norm": self.norm,
            "variance": self.variance,
            "dead_ratio": self.dead_ratio,
            "flow_score": self.flow_score,
            "status": self.status,
            "param_count": self.param_count,
            "nonzero_count": self.nonzero_count,
        }


class GradientHealthMonitor:
    """
    Per-layer gradient sağlık izleme sistemi.

    İzlenen metrikler (her layer için):
    - Gradient norm (L2)
    - Gradient variance
    - Dead neuron oranı (sıfır gradient olan param oranı)
    - Gradient flow skoru (0=donmuş, 1=sağlıklı)

    Uyarı durumları:
    - dead_ratio > 0.3  : katman donuyor uyarısı
    - norm < 1e-7       : vanishing gradient uyarısı
    - norm > 10.0       : exploding gradient uyarısı

    TensorBoard'a per-layer metrikler yazar.
    """

    def __init__(
        self,
        model: nn.Module,
        tensorboard_writer=None,
        dead_threshold: float = 1e-8,
        log_every_n_batches: int = 50,
    ) -> None:
        """
        Args:
            model               : İzlenecek PyTorch modeli.
            tensorboard_writer  : SummaryWriter örneği (opsiyonel).
            dead_threshold      : Gradient abs değerinin altında olduğunda
                                  parametre "ölü" sayılır.
            log_every_n_batches : Kaç batch'te bir TensorBoard'a yazılır.
        """
        self.model = model
        self.writer = tensorboard_writer
        self.dead_threshold = dead_threshold
        self.log_every_n_batches = log_every_n_batches

        # Son hesaplanan metrikler
        self._metrics: Dict[str, Dict] = {}

        # Tarihsel özet (son N ölçüm)
        self._history: deque = deque(maxlen=200)

        # Batch sayacı (TensorBoard log sıklığı için)
        self._batch_counter: int = 0

        # İlk çağrıda katman adlarını keşfet
        self._layer_names: List[str] = self._discover_layers()

        logger.info(
            "GradientHealthMonitor başlatıldı: %d katman izlenecek.",
            len(self._layer_names),
        )

    # ------------------------------------------------------------------
    # Dahili yardımcılar
    # ------------------------------------------------------------------

    def _discover_layers(self) -> List[str]:
        """Gradient'ı olan tüm katman adlarını listele."""
        names = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Katman adı: son iki bileşen (weight/bias dahil)
                layer_key = ".".join(name.split(".")[:2]) if "." in name else name
                if layer_key not in names:
                    names.append(layer_key)
        return names

    def _classify_layer(self, norm: float, dead_ratio: float) -> str:
        """
        Katmanın sağlık durumunu sınıflandır.

        Returns:
            "healthy"   : Normal gradient akışı
            "vanishing" : Gradient norm çok küçük
            "exploding" : Gradient norm çok büyük
            "dead"      : Ölü nöron oranı çok yüksek
        """
        # Öncelik sırasına göre kontrol
        if dead_ratio > DEAD_NEURON_THRESHOLD:
            return "dead"
        if norm < VANISHING_NORM_THRESHOLD:
            return "vanishing"
        if norm > EXPLODING_NORM_THRESHOLD:
            return "exploding"
        return "healthy"

    def _compute_flow_score(self, norm: float, dead_ratio: float) -> float:
        """
        Gradient flow skoru hesapla: [0.0, 1.0] aralığında.

        Skor hesaplama mantığı:
        - Dead ratio cezası: (1 - dead_ratio)
        - Norm cezası: sigmoid benzeri, ideal norm ~0.1-1.0 aralığında

        Returns:
            float: 0.0 (tamamen donmuş) ile 1.0 (sağlıklı) arası.
        """
        if dead_ratio > DEAD_NEURON_THRESHOLD:
            return max(0.0, 1.0 - dead_ratio)

        # Norm'u logaritmik ölçekte değerlendir
        # İdeal norm log aralığı: [-2, 1] (0.01 ile 10 arası)
        if norm <= 0.0:
            norm_score = 0.0
        else:
            log_norm = math.log10(max(norm, 1e-12))
            # [-7, -2] → düşük, [-2, 0] → iyi, [0, 1] → kabul edilebilir, [1, +∞] → yüksek
            if log_norm < -7:
                norm_score = 0.0
            elif log_norm < -2:
                # Lineer interpolasyon: -7→0, -2→0.6
                norm_score = (log_norm + 7) / 5 * 0.6
            elif log_norm <= 1:
                # İdeal bölge: tam skor
                norm_score = 1.0
            else:
                # Çok yüksek: ceza uygula
                norm_score = max(0.0, 1.0 - (log_norm - 1) * 0.3)

        # Ölü nöron cezası
        dead_penalty = 1.0 - dead_ratio

        # Ağırlıklı ortalama
        flow_score = 0.6 * norm_score + 0.4 * dead_penalty
        return float(max(0.0, min(1.0, flow_score)))

    # ------------------------------------------------------------------
    # Ana API
    # ------------------------------------------------------------------

    def compute(self, model: Optional[nn.Module] = None) -> Dict[str, Dict]:
        """
        Tüm katmanlar için gradient sağlık metrikleri hesapla.

        Parametreler gruplara ayrılır (üst modül adına göre),
        her grup için ayrı istatistik hesaplanır.

        Args:
            model: İzlenecek model. None ise __init__'deki model kullanılır.

        Returns:
            Dict[layer_name, {norm, variance, dead_ratio, flow_score, status}]
        """
        target_model = model if model is not None else self.model

        # Katman → gradient listesi haritası
        layer_grads: Dict[str, List[torch.Tensor]] = defaultdict(list)

        for name, param in target_model.named_parameters():
            if param.grad is None:
                continue
            # Katman grubu adını belirle
            parts = name.split(".")
            layer_key = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
            layer_grads[layer_key].append(param.grad.detach().float())

        result: Dict[str, Dict] = {}

        for layer_name, grads in layer_grads.items():
            # Tüm parametrelerin gradientlarını tek tensöre birleştir
            flat = torch.cat([g.reshape(-1) for g in grads])

            # --- L2 norm ---
            norm = float(flat.norm(2).item())

            # --- Varyans ---
            variance = float(flat.var().item()) if flat.numel() > 1 else 0.0

            # --- Ölü nöron oranı ---
            total_params = flat.numel()
            dead_count = int((flat.abs() < self.dead_threshold).sum().item())
            dead_ratio = dead_count / max(total_params, 1)

            # --- Flow skoru ---
            flow_score = self._compute_flow_score(norm, dead_ratio)

            # --- Durum sınıflandırması ---
            status = self._classify_layer(norm, dead_ratio)

            stats = LayerGradientStats(
                norm=norm,
                variance=variance,
                dead_ratio=dead_ratio,
                flow_score=flow_score,
                status=status,
                param_count=total_params,
                nonzero_count=total_params - dead_count,
            )
            result[layer_name] = stats.to_dict()

            # Uyarı mesajları
            if status == "dead":
                logger.warning(
                    "GRADIENT UYARI [dead]     → Katman: %-40s | dead_ratio=%.3f",
                    layer_name, dead_ratio,
                )
            elif status == "vanishing":
                logger.warning(
                    "GRADIENT UYARI [vanishing] → Katman: %-40s | norm=%.2e",
                    layer_name, norm,
                )
            elif status == "exploding":
                logger.warning(
                    "GRADIENT UYARI [exploding] → Katman: %-40s | norm=%.4f",
                    layer_name, norm,
                )

        self._metrics = result
        self._history.append(result)
        self._batch_counter += 1

        return result

    def log_to_tensorboard(self, global_step: int) -> None:
        """
        Mevcut metrikleri TensorBoard'a yaz.

        Her katman için ayrı tag'ler altında:
        GradHealth/{LayerName}/Norm
        GradHealth/{LayerName}/DeadRatio
        GradHealth/{LayerName}/Variance
        GradHealth/{LayerName}/FlowScore

        Args:
            global_step: TensorBoard global adım numarası.
        """
        if self.writer is None:
            return
        if not self._metrics:
            logger.debug("log_to_tensorboard: henüz metrik yok, atlanıyor.")
            return

        for layer_name, stats in self._metrics.items():
            # Tag'lerde nokta yerine eğik çizgi kullan (TB ağaç yapısı için)
            tb_name = layer_name.replace(".", "/")

            self.writer.add_scalar(
                f"GradHealth/{tb_name}/Norm", stats["norm"], global_step
            )
            self.writer.add_scalar(
                f"GradHealth/{tb_name}/DeadRatio", stats["dead_ratio"], global_step
            )
            self.writer.add_scalar(
                f"GradHealth/{tb_name}/Variance", stats["variance"], global_step
            )
            self.writer.add_scalar(
                f"GradHealth/{tb_name}/FlowScore", stats["flow_score"], global_step
            )

        # Özet metrikler
        summary = self.get_summary()
        self.writer.add_scalar(
            "GradHealth/OverallScore", summary["overall_health_score"], global_step
        )
        self.writer.add_scalar(
            "GradHealth/DeadLayerCount", len(summary["dead_layers"]), global_step
        )
        self.writer.add_scalar(
            "GradHealth/ExplodingLayerCount", len(summary["exploding_layers"]), global_step
        )
        self.writer.add_scalar(
            "GradHealth/VanishingLayerCount", len(summary["vanishing_layers"]), global_step
        )

    def get_summary(self) -> Dict:
        """
        Tüm katmanların özet raporu.

        Returns:
            Dict içeren:
            - overall_health_score (float, 0-1): Ağırlıklı ortalama flow skoru
            - dead_layers (List[str])    : Donmuş katman adları
            - exploding_layers (List[str]): Patlayan gradient katmanları
            - vanishing_layers (List[str]): Kaybolan gradient katmanları
            - healthy_count (int)
            - total_layers (int)
        """
        if not self._metrics:
            return {
                "overall_health_score": 0.0,
                "dead_layers": [],
                "exploding_layers": [],
                "vanishing_layers": [],
                "healthy_count": 0,
                "total_layers": 0,
            }

        dead_layers: List[str] = []
        exploding_layers: List[str] = []
        vanishing_layers: List[str] = []
        healthy_count: int = 0
        flow_scores: List[float] = []

        for name, stats in self._metrics.items():
            flow_scores.append(stats["flow_score"])
            status = stats["status"]
            if status == "dead":
                dead_layers.append(name)
            elif status == "exploding":
                exploding_layers.append(name)
            elif status == "vanishing":
                vanishing_layers.append(name)
            else:
                healthy_count += 1

        overall_health_score = (
            sum(flow_scores) / len(flow_scores) if flow_scores else 0.0
        )

        return {
            "overall_health_score": float(overall_health_score),
            "dead_layers": dead_layers,
            "exploding_layers": exploding_layers,
            "vanishing_layers": vanishing_layers,
            "healthy_count": healthy_count,
            "total_layers": len(self._metrics),
        }

    def should_log(self) -> bool:
        """Bu batch'te TensorBoard'a yazılmalı mı?"""
        return self._batch_counter % self.log_every_n_batches == 0

    def get_layer_names(self) -> List[str]:
        """İzlenen katman adlarını döndür."""
        return list(self._metrics.keys())

    def get_layer_stats(self, layer_name: str) -> Optional[Dict]:
        """Belirli bir katmanın son metriklerini döndür."""
        return self._metrics.get(layer_name)

    def reset(self) -> None:
        """Mevcut metrikleri ve sayacı sıfırla (yeni epoch için)."""
        self._metrics = {}
        self._batch_counter = 0
        logger.debug("GradientHealthMonitor sıfırlandı.")

    def __repr__(self) -> str:
        n_layers = len(self._metrics)
        summary = self.get_summary() if n_layers > 0 else {}
        score = summary.get("overall_health_score", 0.0)
        return (
            f"GradientHealthMonitor("
            f"layers={n_layers}, "
            f"health_score={score:.3f}, "
            f"dead_threshold={self.dead_threshold})"
        )
