# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: profiler.py
Modül: model_management
Görev: Model profiling — parametre sayımı, bellek ölçümü, FLOP tahmini ve
       torch.profiler entegrasyonu.

       ModelProfiler   → Ana profiler sınıfı (statik metodlar)
       ParamStats      → Parametre istatistikleri veri sınıfı
       MemorySnapshot  → Anlık bellek durumu
       ProfileResult   → torch.profiler çalışma sonucu

KULLANIM:
    from model_management.profiler import ModelProfiler

    stats = ModelProfiler.count_parameters(model)
    print(stats)  # trainable=125M, frozen=0, total=125M

    mem = ModelProfiler.memory_snapshot()
    print(mem)    # allocated=3.72 GB, reserved=4.00 GB

    with ModelProfiler.profile_context(model) as prof:
        _ = model(sample)
    print(prof.key_averages())

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
================================================================================
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

profiler_logger = logging.getLogger("ModelProfiler")


# ══════════════════════════════════════════════════════════════════════════════
# Veri Sınıfları
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ParamStats:
    """Model parametre istatistikleri."""
    total: int = 0
    trainable: int = 0
    frozen: int = 0
    trainable_mb: float = 0.0
    """Eğitilebilir parametrelerin float32 boyutu (MB)."""

    by_layer: Dict[str, int] = field(default_factory=dict)
    """Her üst-modül adı için toplam parametre sayısı."""

    def __str__(self) -> str:
        def _fmt(n: int) -> str:
            if n >= 1_000_000_000:
                return f"{n / 1e9:.2f}B"
            if n >= 1_000_000:
                return f"{n / 1e6:.2f}M"
            if n >= 1_000:
                return f"{n / 1e3:.1f}K"
            return str(n)

        return (
            f"ParamStats("
            f"total={_fmt(self.total)}, "
            f"trainable={_fmt(self.trainable)}, "
            f"frozen={_fmt(self.frozen)}, "
            f"trainable_mem={self.trainable_mb:.1f} MB)"
        )

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class MemorySnapshot:
    """Anlık GPU bellek durumu (MB cinsinden)."""
    allocated_mb: float = 0.0
    reserved_mb: float = 0.0
    free_mb: float = 0.0
    total_mb: float = 0.0
    device: str = "cpu"

    @property
    def allocated_gb(self) -> float:
        return self.allocated_mb / 1024

    @property
    def reserved_gb(self) -> float:
        return self.reserved_mb / 1024

    @property
    def utilization_pct(self) -> float:
        """Tahsis edilen / toplam (%)."""
        if self.total_mb == 0:
            return 0.0
        return 100.0 * self.allocated_mb / self.total_mb

    def __str__(self) -> str:
        return (
            f"MemorySnapshot({self.device}) "
            f"alloc={self.allocated_gb:.2f} GB / "
            f"total={self.total_mb / 1024:.2f} GB "
            f"({self.utilization_pct:.1f}%)"
        )


@dataclass
class FlopEstimate:
    """İleri geçiş FLOP tahmini."""
    total_flops: int = 0
    """Toplam kayan nokta işlemi sayısı."""

    attention_flops: int = 0
    ffn_flops: int = 0
    embedding_flops: int = 0

    seq_len: int = 0
    batch_size: int = 1

    @property
    def gflops(self) -> float:
        return self.total_flops / 1e9

    @property
    def tflops(self) -> float:
        return self.total_flops / 1e12

    def __str__(self) -> str:
        return (
            f"FlopEstimate(total={self.gflops:.1f} GFLOPs, "
            f"attn={self.attention_flops/1e9:.1f}, "
            f"ffn={self.ffn_flops/1e9:.1f}, "
            f"seq={self.seq_len}, batch={self.batch_size})"
        )


@dataclass
class TimingResult:
    """Model forward geçiş zamanlama sonucu."""
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    n_runs: int = 0
    tokens_per_second: float = 0.0
    seq_len: int = 0
    batch_size: int = 1

    def __str__(self) -> str:
        return (
            f"TimingResult("
            f"mean={self.mean_ms:.2f}ms ±{self.std_ms:.2f}, "
            f"tok/s={self.tokens_per_second:.0f}, "
            f"runs={self.n_runs})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Ana Profiler Sınıfı
# ══════════════════════════════════════════════════════════════════════════════

class ModelProfiler:
    """
    Cevahir model profiling aracı.

    Tüm metodlar statik olduğundan instance oluşturmaya gerek yoktur:
        stats = ModelProfiler.count_parameters(model)
    """

    # ── 1. Parametre Sayımı ──────────────────────────────────────────────────

    @staticmethod
    def count_parameters(model: nn.Module, *, log: bool = True) -> ParamStats:
        """
        Modelin trainable / frozen / toplam parametre sayılarını hesaplar.

        Args:
            model: İncelenecek model.
            log  : True ise INFO seviyesinde loga yazar.

        Returns:
            ParamStats veri sınıfı.
        """
        total = 0
        trainable = 0

        # Üst-modül bazlı sayım
        by_layer: Dict[str, int] = {}
        for name, module in model.named_children():
            layer_params = sum(p.numel() for p in module.parameters())
            by_layer[name] = layer_params

        for p in model.parameters():
            n = p.numel()
            total += n
            if p.requires_grad:
                trainable += n

        frozen = total - trainable
        # float32 → 4 byte / parametre
        trainable_mb = trainable * 4 / (1024 ** 2)

        stats = ParamStats(
            total=total,
            trainable=trainable,
            frozen=frozen,
            trainable_mb=trainable_mb,
            by_layer=by_layer,
        )

        if log:
            profiler_logger.info(f"[Profiler] {stats}")
            if by_layer:
                top5 = sorted(by_layer.items(), key=lambda x: x[1], reverse=True)[:5]
                for layer_name, count in top5:
                    profiler_logger.info(
                        f"  └─ {layer_name}: {count / 1e6:.2f}M params"
                    )

        return stats

    # ── 2. Bellek Ölçümü ─────────────────────────────────────────────────────

    @staticmethod
    def memory_snapshot(device: Optional[str] = None, *, log: bool = True) -> MemorySnapshot:
        """
        Anlık GPU (veya CPU) bellek durumunu ölçer.

        Args:
            device: 'cuda', 'cuda:0', ... None → otomatik seçim.
            log   : INFO seviyesinde loga yaz.

        Returns:
            MemorySnapshot veri sınıfı.
        """
        if not torch.cuda.is_available():
            snap = MemorySnapshot(device="cpu")
            if log:
                profiler_logger.info("[Profiler] CUDA yok — CPU bellek ölçümü atlandı.")
            return snap

        dev = device or "cuda"
        try:
            allocated = torch.cuda.memory_allocated(dev)
            reserved = torch.cuda.memory_reserved(dev)
            try:
                free, total = torch.cuda.mem_get_info(dev)
            except Exception:
                free, total = 0, reserved

            snap = MemorySnapshot(
                allocated_mb=allocated / (1024 ** 2),
                reserved_mb=reserved / (1024 ** 2),
                free_mb=free / (1024 ** 2),
                total_mb=total / (1024 ** 2),
                device=dev,
            )
            if log:
                profiler_logger.info(f"[Profiler] {snap}")
            return snap
        except Exception as exc:
            profiler_logger.warning(f"[Profiler] Bellek ölçümü başarısız: {exc}")
            return MemorySnapshot(device=dev)

    # ── 3. Model Boyutu ──────────────────────────────────────────────────────

    @staticmethod
    def estimate_model_size_mb(model: nn.Module) -> float:
        """
        Modelin state_dict'ini disk/bellekte kaç MB yer kapladığını tahmin eder.
        Gerçek disk boyutu overhead'den dolayı biraz daha büyük olabilir.
        """
        total_bytes = 0
        for p in model.parameters():
            total_bytes += p.numel() * p.element_size()
        for b in model.buffers():
            total_bytes += b.numel() * b.element_size()
        return total_bytes / (1024 ** 2)

    # ── 4. FLOP Tahmini ──────────────────────────────────────────────────────

    @staticmethod
    def estimate_flops(
        model: nn.Module,
        seq_len: int,
        batch_size: int = 1,
        *,
        embed_dim: Optional[int] = None,
        num_heads: Optional[int] = None,
        num_layers: Optional[int] = None,
        ffn_dim: Optional[int] = None,
        log: bool = True,
    ) -> FlopEstimate:
        """
        Transformer forward geçişinin teorik FLOP sayısını hesaplar.
        Gerçek profiling yerine hızlı tahmin içindir.

        Formüller (Kaplan et al. 2020 / PaLM paper'dan):
          Attention: 4 × B × L × T × D  (QKV + Output projeksiyon)
          Softmax:   2 × B × L × H × T²  (dikkat skoru)
          FFN:       2 × B × L × T × 4D  (iki doğrusal katman)

        B=batch, L=num_layers, T=seq_len, D=embed_dim, H=num_heads
        """
        # Config'ten parametre çıkarmaya çalış
        cfg: Dict[str, Any] = {}
        if hasattr(model, "config"):
            cfg = model.config if isinstance(model.config, dict) else {}

        D = embed_dim or cfg.get("embed_dim", 512)
        L = num_layers or cfg.get("num_layers", 8)
        H = num_heads or cfg.get("num_heads", 8)
        F = ffn_dim or cfg.get("ffn_dim") or (4 * D)
        B = batch_size
        T = seq_len

        # QKV + Output proj: 4 matmul × B × T × D² × 2 (çarpma + toplama)
        attn_proj_flops = 4 * B * L * T * D * D * 2
        # Attention scores: B × L × H × T × (T × (D//H)) × 2
        attn_score_flops = 2 * B * L * H * T * T * (D // H)
        attention_flops = attn_proj_flops + attn_score_flops

        # FFN: 2 matmul × B × L × T × D × F × 2
        ffn_flops = 2 * B * L * T * D * F * 2

        # Embedding lookup (trivial — atlanabilir ama dahil edelim)
        V = getattr(model, "vocab_size", None) or cfg.get("vocab_size", 60000)
        embedding_flops = B * T * D  # lookup + positional

        total = attention_flops + ffn_flops + embedding_flops

        est = FlopEstimate(
            total_flops=total,
            attention_flops=attention_flops,
            ffn_flops=ffn_flops,
            embedding_flops=embedding_flops,
            seq_len=T,
            batch_size=B,
        )
        if log:
            profiler_logger.info(f"[Profiler] {est}")
        return est

    # ── 5. Zamanlama ─────────────────────────────────────────────────────────

    @staticmethod
    @torch.no_grad()
    def benchmark_forward(
        model: nn.Module,
        sample_input: torch.Tensor,
        *,
        n_warmup: int = 3,
        n_runs: int = 10,
        device: Optional[str] = None,
        log: bool = True,
    ) -> TimingResult:
        """
        Model forward geçişini n_runs kez çalıştırarak ortalama süre ölçer.

        Args:
            model       : Ölçülecek model (eval moduna geçirilir).
            sample_input: Örnek input tensörü.
            n_warmup    : CUDA grafiği ısınması için ön çalıştırma sayısı.
            n_runs      : Gerçek ölçüm sayısı.
            device      : 'cuda' veya 'cpu'; None → input'un device'ı.
            log         : INFO seviyesinde loga yaz.

        Returns:
            TimingResult veri sınıfı.
        """
        import statistics

        dev = device or str(sample_input.device)
        model.eval()
        inp = sample_input.to(dev)

        use_cuda = dev.startswith("cuda") and torch.cuda.is_available()

        # Isınma
        for _ in range(n_warmup):
            _ = model(inp)
        if use_cuda:
            torch.cuda.synchronize()

        times_ms: List[float] = []
        for _ in range(n_runs):
            if use_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(inp)
                end.record()
                torch.cuda.synchronize()
                times_ms.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                _ = model(inp)
                times_ms.append((time.perf_counter() - t0) * 1000)

        B, T = inp.shape[0], inp.shape[1] if inp.ndim >= 2 else 1
        mean_ms = statistics.mean(times_ms)
        tok_per_sec = (B * T) / (mean_ms / 1000) if mean_ms > 0 else 0.0

        result = TimingResult(
            mean_ms=mean_ms,
            std_ms=statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
            min_ms=min(times_ms),
            max_ms=max(times_ms),
            n_runs=n_runs,
            tokens_per_second=tok_per_sec,
            seq_len=T,
            batch_size=B,
        )
        if log:
            profiler_logger.info(f"[Profiler] {result}")
        return result

    # ── 6. torch.profiler Entegrasyonu ───────────────────────────────────────

    @staticmethod
    def profile_context(
        model: nn.Module,
        *,
        activities: Optional[List[Any]] = None,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
        output_path: Optional[str] = None,
    ):
        """
        torch.profiler context manager döndürür. Forward geçişini bu context
        içinde çalıştırarak detaylı profil çıktısı alın.

        Kullanım:
            with ModelProfiler.profile_context(model) as prof:
                logits, _ = model(inputs)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

        Args:
            output_path: Chrome trace JSON export yolu (opsiyonel).
        """
        try:
            from torch.profiler import (
                profile,
                ProfilerActivity,
                record_function,
                tensorboard_trace_handler,
            )

            _activities = activities or [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                _activities.append(ProfilerActivity.CUDA)

            schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)

            ctx = profile(
                activities=_activities,
                schedule=schedule,
                record_shapes=record_shapes,
                profile_memory=profile_memory,
                with_stack=with_stack,
                on_trace_ready=(
                    tensorboard_trace_handler(output_path) if output_path else None
                ),
            )
            return ctx

        except ImportError:
            profiler_logger.warning("[Profiler] torch.profiler mevcut değil; dummy context döndürülüyor.")
            from contextlib import nullcontext
            return nullcontext()

    # ── 7. Özet Rapor ────────────────────────────────────────────────────────

    @staticmethod
    def full_report(
        model: nn.Module,
        *,
        seq_len: int = 512,
        batch_size: int = 1,
        device: Optional[str] = None,
        run_timing: bool = False,
    ) -> Dict[str, Any]:
        """
        Parametre sayımı + bellek + FLOP + opsiyonel timing'i tek dict'te döndürür.
        ModelManager.build_model() sonrasında otomatik çağrılır.

        Returns:
            {
              "params":  ParamStats,
              "memory":  MemorySnapshot,
              "flops":   FlopEstimate,
              "timing":  TimingResult | None,
              "size_mb": float,
            }
        """
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

        params = ModelProfiler.count_parameters(model, log=False)
        memory = ModelProfiler.memory_snapshot(dev, log=False)
        flops = ModelProfiler.estimate_flops(model, seq_len=seq_len, batch_size=batch_size, log=False)
        size_mb = ModelProfiler.estimate_model_size_mb(model)

        timing: Optional[TimingResult] = None
        if run_timing:
            try:
                vocab = getattr(model, "vocab_size", None) or 60000
                sample = torch.randint(0, min(vocab, 1000), (batch_size, seq_len)).to(dev)
                timing = ModelProfiler.benchmark_forward(model, sample, log=False)
            except Exception as exc:
                profiler_logger.warning(f"[Profiler] Timing ölçümü başarısız: {exc}")

        report = {
            "params": params,
            "memory": memory,
            "flops": flops,
            "timing": timing,
            "size_mb": size_mb,
        }

        # Loga yaz
        profiler_logger.info(
            f"[Profiler] ══ Model Raporu ══\n"
            f"  Parametreler : {params}\n"
            f"  Model boyutu : {size_mb:.1f} MB\n"
            f"  Bellek       : {memory}\n"
            f"  FLOP (T={seq_len}): {flops}\n"
            + (f"  Zamanlama    : {timing}" if timing else "  Zamanlama    : atlandı")
        )
        return report
