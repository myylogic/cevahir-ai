# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: health_monitor.py
Modül: model_management
Görev: Model sağlık izleme — gradient akışı, ağırlık dağılımı ve dikkat
       entropisi patolojilerini tespit eder; eğitim stabilitesini korur.

       ModelHealthMonitor  → Ana sağlık izleme sınıfı
       GradientHealth      → Gradient sağlık raporu
       WeightHealth        → Ağırlık dağılım raporu
       AttentionHealth     → Dikkat entropisi raporu
       HealthReport        → Birleşik sağlık raporu

TEŞHİS EDİLEN PATOLOJİLER:
  • Gradient vanishing  → |grad| < 1e-8 (ölü nöron / vanishing gradient)
  • Gradient explosion  → |grad| > 1e4  (patlayan gradient)
  • NaN/Inf gradient    → eğitim patlaması
  • Dead weights        → |weight| < 1e-9 (ölü ağırlık)
  • Weight explosion    → |weight| > 1e3  (ağırlık patlaması)
  • Attention collapse  → entropy < 0.05 (tüm dikkat tek token'a)
  • Attention uniform   → entropy > 0.99 (dikkat öğrenilmemiş)

KULLANIM:
    report = ModelHealthMonitor.full_health_check(model)
    if not report.is_healthy:
        print(report.summary())

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
================================================================================
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

health_logger = logging.getLogger("ModelHealthMonitor")


# ══════════════════════════════════════════════════════════════════════════════
# Eşik Sabitleri
# ══════════════════════════════════════════════════════════════════════════════

_GRAD_VANISH_THRESH   = 1e-8    # Bu değerin altındaki gradient normu → vanishing
_GRAD_EXPLODE_THRESH  = 1e4     # Bu değerin üzerindeki gradient normu → exploding
_WEIGHT_DEAD_THRESH   = 1e-9    # Bu değerin altındaki ağırlık → ölü
_WEIGHT_EXPLODE_THRESH = 1e3    # Bu değerin üzerindeki ağırlık → patlama riski
_ATTN_COLLAPSE_THRESH = 0.05   # Normalize entropy < bu değer → attention collapse
_ATTN_UNIFORM_THRESH  = 0.99   # Normalize entropy > bu değer → attention öğrenmemiş
_NAN_BUDGET           = 0       # Tolere edilen NaN sayısı (0 = hiç tolerans yok)


# ══════════════════════════════════════════════════════════════════════════════
# Veri Sınıfları
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GradientHealth:
    """Gradient sağlık raporu."""
    has_nan: bool = False
    has_inf: bool = False
    nan_layers: List[str] = field(default_factory=list)
    inf_layers: List[str] = field(default_factory=list)
    vanishing_layers: List[str] = field(default_factory=list)
    exploding_layers: List[str] = field(default_factory=list)
    max_norm: float = 0.0
    min_norm: float = float("inf")
    mean_norm: float = 0.0
    total_params_with_grad: int = 0

    @property
    def is_healthy(self) -> bool:
        return (
            not self.has_nan
            and not self.has_inf
            and not self.exploding_layers
        )

    @property
    def severity(self) -> str:
        if self.has_nan or self.has_inf:
            return "CRITICAL"
        if self.exploding_layers:
            return "WARNING"
        if self.vanishing_layers:
            return "INFO"
        return "OK"

    def summary(self) -> str:
        lines = [f"GradientHealth [{self.severity}]"]
        if self.has_nan:
            lines.append(f"  ❌ NaN gradients: {self.nan_layers}")
        if self.has_inf:
            lines.append(f"  ❌ Inf gradients: {self.inf_layers}")
        if self.exploding_layers:
            lines.append(f"  ⚠️  Exploding gradients: {self.exploding_layers}")
        if self.vanishing_layers:
            lines.append(f"  ℹ️  Vanishing gradients: {self.vanishing_layers}")
        lines.append(
            f"  norm: max={self.max_norm:.3e}, min={self.min_norm:.3e}, "
            f"mean={self.mean_norm:.3e}, params_with_grad={self.total_params_with_grad}"
        )
        return "\n".join(lines)


@dataclass
class WeightHealth:
    """Ağırlık dağılım sağlık raporu."""
    has_nan: bool = False
    has_inf: bool = False
    nan_layers: List[str] = field(default_factory=list)
    dead_layers: List[str] = field(default_factory=list)
    """Ortalaması sıfıra yakın, std'si 1e-9'dan küçük katmanlar."""
    exploding_layers: List[str] = field(default_factory=list)

    # Genel istatistikler
    global_mean: float = 0.0
    global_std: float = 0.0
    global_max_abs: float = 0.0

    layer_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    """Her katman için mean / std / max_abs."""

    @property
    def is_healthy(self) -> bool:
        return (
            not self.has_nan
            and not self.has_inf
            and not self.exploding_layers
        )

    @property
    def severity(self) -> str:
        if self.has_nan or self.has_inf:
            return "CRITICAL"
        if self.exploding_layers:
            return "WARNING"
        if self.dead_layers:
            return "INFO"
        return "OK"

    def summary(self) -> str:
        lines = [f"WeightHealth [{self.severity}]"]
        if self.has_nan:
            lines.append(f"  ❌ NaN ağırlıklar: {self.nan_layers}")
        if self.has_inf:
            lines.append(f"  ❌ Inf ağırlıklar")
        if self.exploding_layers:
            lines.append(f"  ⚠️  Patlayan ağırlıklar: {self.exploding_layers}")
        if self.dead_layers:
            lines.append(f"  ℹ️  Ölü ağırlıklar (std ≈ 0): {self.dead_layers}")
        lines.append(
            f"  global: mean={self.global_mean:.4f}, "
            f"std={self.global_std:.4f}, "
            f"max_abs={self.global_max_abs:.4f}"
        )
        return "\n".join(lines)


@dataclass
class AttentionHealth:
    """Dikkat entropisi sağlık raporu."""
    collapse_layers: List[str] = field(default_factory=list)
    """entropy < _ATTN_COLLAPSE_THRESH → tüm dikkat tek token'a yığılmış."""
    uniform_layers: List[str] = field(default_factory=list)
    """entropy > _ATTN_UNIFORM_THRESH → dikkat hiç öğrenmemiş, düzgün dağılım."""

    mean_entropy: float = 0.0
    min_entropy: float = 1.0
    max_entropy: float = 0.0

    layer_entropies: Dict[str, float] = field(default_factory=dict)

    @property
    def is_healthy(self) -> bool:
        return not self.collapse_layers

    @property
    def severity(self) -> str:
        if self.collapse_layers:
            return "WARNING"
        if self.uniform_layers:
            return "INFO"
        return "OK"

    def summary(self) -> str:
        lines = [f"AttentionHealth [{self.severity}]"]
        if self.collapse_layers:
            lines.append(f"  ⚠️  Attention collapse katmanları: {self.collapse_layers}")
        if self.uniform_layers:
            lines.append(f"  ℹ️  Düzgün attention (öğrenmemiş): {self.uniform_layers}")
        lines.append(
            f"  entropy: mean={self.mean_entropy:.3f}, "
            f"min={self.min_entropy:.3f}, max={self.max_entropy:.3f}"
        )
        return "\n".join(lines)


@dataclass
class HealthReport:
    """Birleşik model sağlık raporu."""
    gradient: GradientHealth = field(default_factory=GradientHealth)
    weight: WeightHealth = field(default_factory=WeightHealth)
    attention: AttentionHealth = field(default_factory=AttentionHealth)

    @property
    def is_healthy(self) -> bool:
        return (
            self.gradient.is_healthy
            and self.weight.is_healthy
            and self.attention.is_healthy
        )

    @property
    def overall_severity(self) -> str:
        severities = {
            "CRITICAL": 3,
            "WARNING": 2,
            "INFO": 1,
            "OK": 0,
        }
        worst = max(
            severities.get(self.gradient.severity, 0),
            severities.get(self.weight.severity, 0),
            severities.get(self.attention.severity, 0),
        )
        return {v: k for k, v in severities.items()}[worst]

    def summary(self) -> str:
        icon = "✅" if self.is_healthy else ("🔴" if "CRITICAL" in self.overall_severity else "⚠️")
        lines = [
            f"{'═'*60}",
            f"  {icon} MODEL SAĞLIK RAPORU — {self.overall_severity}",
            f"{'═'*60}",
            self.gradient.summary(),
            "",
            self.weight.summary(),
            "",
            self.attention.summary(),
            f"{'═'*60}",
        ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Ana Monitor Sınıfı
# ══════════════════════════════════════════════════════════════════════════════

class ModelHealthMonitor:
    """
    Cevahir model sağlık izleme aracı.
    Tüm metodlar statik — instance gerektirmez.
    """

    # ── 1. Gradient Sağlığı ──────────────────────────────────────────────────

    @staticmethod
    def check_gradient_flow(
        model: nn.Module,
        *,
        vanish_thresh: float = _GRAD_VANISH_THRESH,
        explode_thresh: float = _GRAD_EXPLODE_THRESH,
        log: bool = True,
    ) -> GradientHealth:
        """
        Gradient'ların NaN/Inf, vanishing ve exploding durumunu kontrol eder.
        Backward() çağrısından SONRA çağrılmalıdır.
        """
        health = GradientHealth()
        norms: List[float] = []

        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            health.total_params_with_grad += 1
            grad = param.grad.detach()

            # NaN/Inf kontrolü
            if torch.isnan(grad).any():
                health.has_nan = True
                health.nan_layers.append(name)
            if torch.isinf(grad).any():
                health.has_inf = True
                health.inf_layers.append(name)

            # Norm hesapla (NaN/Inf varsa atlat)
            if not (health.has_nan or health.has_inf):
                try:
                    norm = float(grad.norm(2).item())
                    norms.append(norm)
                    if norm < vanish_thresh:
                        health.vanishing_layers.append(name)
                    if norm > explode_thresh:
                        health.exploding_layers.append(name)
                except Exception:
                    pass

        if norms:
            health.max_norm = max(norms)
            health.min_norm = min(norms)
            health.mean_norm = sum(norms) / len(norms)

        if log and health.severity != "OK":
            health_logger.warning(f"[HealthMonitor]\n{health.summary()}")
        elif log:
            health_logger.debug(
                f"[HealthMonitor] Gradient sağlığı OK — "
                f"max_norm={health.max_norm:.3e}"
            )

        return health

    # ── 2. Ağırlık Sağlığı ───────────────────────────────────────────────────

    @staticmethod
    def check_weight_distribution(
        model: nn.Module,
        *,
        dead_std_thresh: float = _WEIGHT_DEAD_THRESH,
        explode_thresh: float = _WEIGHT_EXPLODE_THRESH,
        log: bool = True,
    ) -> WeightHealth:
        """
        Parametre ağırlıklarının NaN/Inf, ölü ve patlama durumlarını kontrol eder.
        Herhangi bir zamanda çağrılabilir.
        """
        health = WeightHealth()

        all_weights: List[torch.Tensor] = []

        for name, param in model.named_parameters():
            data = param.data.detach().float()

            # NaN/Inf
            if torch.isnan(data).any():
                health.has_nan = True
                health.nan_layers.append(name)
                continue
            if torch.isinf(data).any():
                health.has_inf = True
                continue

            w_mean = float(data.mean().item())
            w_std = float(data.std().item())
            w_max = float(data.abs().max().item())

            health.layer_stats[name] = {
                "mean": w_mean,
                "std": w_std,
                "max_abs": w_max,
            }

            if w_std < dead_std_thresh:
                health.dead_layers.append(name)
            if w_max > explode_thresh:
                health.exploding_layers.append(name)

            all_weights.append(data.flatten())

        if all_weights:
            combined = torch.cat(all_weights)
            health.global_mean = float(combined.mean().item())
            health.global_std = float(combined.std().item())
            health.global_max_abs = float(combined.abs().max().item())

        if log and health.severity != "OK":
            health_logger.warning(f"[HealthMonitor]\n{health.summary()}")
        elif log:
            health_logger.debug(
                f"[HealthMonitor] Ağırlık sağlığı OK — "
                f"global_std={health.global_std:.4f}"
            )

        return health

    # ── 3. Dikkat Entropisi ──────────────────────────────────────────────────

    @staticmethod
    def check_attention_entropy(
        model: nn.Module,
        sample_input: Optional[torch.Tensor] = None,
        *,
        device: Optional[str] = None,
        collapse_thresh: float = _ATTN_COLLAPSE_THRESH,
        uniform_thresh: float = _ATTN_UNIFORM_THRESH,
        log: bool = True,
    ) -> AttentionHealth:
        """
        Modelin dahili `_last_attn_entropy` değerlerini (eğer kayıt ediyorsa)
        okuyarak dikkat entropisi patolojilerini tespit eder.

        Not: Sinir ağı _last_attn_entropy attribute'unu kaydetmiyorsa
        sample_input üzerinden bir forward geçişi yapılarak tahmin yapılır.
        """
        health = AttentionHealth()
        entropies: Dict[str, float] = {}

        # 1) Model dahili entropi değerlerini oku
        for name, module in model.named_modules():
            if hasattr(module, "_last_attn_entropy"):
                val = module._last_attn_entropy
                if val is not None:
                    try:
                        entropy_val = float(val) if not isinstance(val, torch.Tensor) else float(val.item())
                        entropies[name] = entropy_val
                    except Exception:
                        pass

        # 2) Değer yoksa ve sample_input verilmişse basit forward geçişi yap
        if not entropies and sample_input is not None:
            try:
                dev = device or str(sample_input.device)
                model.eval()
                with torch.no_grad():
                    model(sample_input.to(dev))

                # Tekrar dene
                for name, module in model.named_modules():
                    if hasattr(module, "_last_attn_entropy"):
                        val = module._last_attn_entropy
                        if val is not None:
                            try:
                                entropies[name] = float(val) if not isinstance(val, torch.Tensor) else float(val.item())
                            except Exception:
                                pass
            except Exception as exc:
                health_logger.debug(f"[HealthMonitor] Forward geçişi başarısız: {exc}")

        # 3) Analiz
        if entropies:
            entropy_vals = list(entropies.values())
            health.mean_entropy = sum(entropy_vals) / len(entropy_vals)
            health.min_entropy = min(entropy_vals)
            health.max_entropy = max(entropy_vals)
            health.layer_entropies = entropies

            for layer_name, ent in entropies.items():
                if ent < collapse_thresh:
                    health.collapse_layers.append(layer_name)
                elif ent > uniform_thresh:
                    health.uniform_layers.append(layer_name)

        if log and health.severity != "OK":
            health_logger.warning(f"[HealthMonitor]\n{health.summary()}")
        elif log and entropies:
            health_logger.debug(
                f"[HealthMonitor] Attention sağlığı OK — "
                f"mean_entropy={health.mean_entropy:.3f}"
            )

        return health

    # ── 4. Birleşik Sağlık Kontrolü ──────────────────────────────────────────

    @staticmethod
    def full_health_check(
        model: nn.Module,
        sample_input: Optional[torch.Tensor] = None,
        *,
        check_gradients: bool = True,
        check_weights: bool = True,
        check_attention: bool = True,
        log: bool = True,
        raise_on_critical: bool = False,
    ) -> HealthReport:
        """
        Tüm sağlık kontrollerini birleştirir ve HealthReport döndürür.

        Args:
            model           : İncelenecek model.
            sample_input    : Attention entropy için örnek input (opsiyonel).
            check_gradients : Gradient akışını kontrol et.
            check_weights   : Ağırlık dağılımını kontrol et.
            check_attention : Dikkat entropisini kontrol et.
            log             : INFO/WARNING olarak loga yaz.
            raise_on_critical: True ise CRITICAL durum HealthCheckError fırlatır.

        Returns:
            HealthReport birleşik raporu.
        """
        from model_management.exceptions import HealthCheckError

        report = HealthReport()

        if check_gradients:
            report.gradient = ModelHealthMonitor.check_gradient_flow(model, log=False)

        if check_weights:
            report.weight = ModelHealthMonitor.check_weight_distribution(model, log=False)

        if check_attention:
            report.attention = ModelHealthMonitor.check_attention_entropy(
                model, sample_input, log=False
            )

        if log:
            if report.is_healthy:
                health_logger.info(
                    f"[HealthMonitor] ✅ Model sağlıklı — "
                    f"grad_ok={report.gradient.is_healthy}, "
                    f"weight_ok={report.weight.is_healthy}, "
                    f"attn_ok={report.attention.is_healthy}"
                )
            else:
                health_logger.warning(f"[HealthMonitor]\n{report.summary()}")

        if raise_on_critical and report.overall_severity == "CRITICAL":
            raise HealthCheckError(
                check_name="full_health_check",
                details=f"CRITICAL patoloji tespit edildi:\n{report.summary()}",
            )

        return report

    # ── 5. Eğitim Sırasında Hızlı Kontrol ───────────────────────────────────

    @staticmethod
    def quick_gradient_check(model: nn.Module) -> Tuple[bool, str]:
        """
        Hızlı NaN/Inf gradient kontrolü — her batch sonrası çağrılabilir.
        Tam check_gradient_flow'dan daha hızlı (only NaN/Inf).

        Returns:
            (is_safe, message) — is_safe=False ise batch'i atla.
        """
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if torch.isnan(param.grad).any():
                return False, f"NaN gradient: {name}"
            if torch.isinf(param.grad).any():
                return False, f"Inf gradient: {name}"
        return True, "OK"

    @staticmethod
    def log_gradient_norms(
        model: nn.Module,
        step: int,
        *,
        tb_writer: Optional[Any] = None,
        top_n: int = 5,
    ) -> Dict[str, float]:
        """
        Her parametrenin gradient normunu hesaplar; TensorBoard'a yazar.
        Epoch sonu debug için uygundur.

        Returns:
            {param_name: grad_norm}
        """
        norms: Dict[str, float] = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                try:
                    norms[name] = float(param.grad.norm(2).item())
                except Exception:
                    pass

        if tb_writer is not None:
            for name, norm_val in norms.items():
                try:
                    tb_writer.add_scalar(f"grad_norm/{name}", norm_val, step)
                except Exception:
                    pass

        # En yüksek normlara sahip katmanları logla
        top = sorted(norms.items(), key=lambda x: x[1], reverse=True)[:top_n]
        if top:
            top_str = ", ".join(f"{n}={v:.2e}" for n, v in top)
            health_logger.debug(f"[HealthMonitor] Top-{top_n} grad norms: {top_str}")

        return norms
