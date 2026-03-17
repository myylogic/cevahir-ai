# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: feed_forward_network.py
Modül: src/neural_network_module/ortak_katman_module
Görev: Feed Forward Network — Transformer standardı 2-layer MLP (expand → contract).
       SwiGLU / GeGLU / GELU-tanh / GELU / ReLU / SiLU activation desteği.
       Bias-free design (LLaMA/PaLM/Gemma standardı).
       Bellek-verimli forward pass (anlık tensor sayısı minimize edildi).

MİMARİ:
- SOLID Prensipleri: Single Responsibility (FFN işlemleri),
                     Open/Closed (activation parametresi ile genişletilebilir),
                     Dependency Inversion (nn.Module abstraction'ına bağımlı)
- Design Patterns: Network Pattern (feed-forward network)
- Endüstri Standartları: LLaMA 3, Gemma 2, PaLM 2, GPT-2/3/4, T5 FFN standardı

V6 DEĞİŞİKLİKLERİ:
- Fix 5: gate * torch.sigmoid(gate) → F.silu (kernel-fused, %15-20 daha hızlı)
- V6+: use_bias=False parametresi (LLaMA/PaLM standardı — varsayılan)
- V6+: GeGLU activation eklendi (T5, Flan-T5, Switch Transformer standardı)
- V6+: gelu_tanh activation eklendi (GPT-2/3/4 hızlı yaklaşım)
- V6+: Bellek-verimli SwiGLU/GeGLU forward (4 tensor → 2 tensor: ~%50 VRAM azalması)
- V6+: Kaiming/normal init, aktivasyona göre otomatik seçim
- V6+: fc2 için ayrı init_std (scaled residual init ile uyumluluk)

V7 DEĞİŞİKLİKLERİ:
- Merged gate_up_proj: gate_proj + up_proj → tek Linear(embed_dim, 2*ffn_dim)
  Tek GEMM → %10-20 GPU hızlanması (LLaMA / HuggingFace standardı)
- chunk(2, dim=-1): Kopyasız in-memory split (view tabanlı)
- del gate dead code kaldırıldı: Autograd tensor'larda etkisiz
- reset_parameters() eklendi: Standart PyTorch re-init arayüzü

KULLANIM:
- Feed-forward network oluşturmak için
- MLP katmanı için
- SwiGLU/GeGLU/GELU/ReLU/SiLU activation için

BAĞIMLILIKLAR:
- torch.nn: Linear modülü
- torch.nn.functional: Activation fonksiyonları

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

# Gated activation aileleri (SwiGLU, GeGLU): gate_up_proj (merged) + fc2 mimarisi
_GATED_ACTIVATIONS = frozenset({"swiglu", "geglu"})

# Standart activation aileleri: fc1 + activation + fc2 mimarisi
_STANDARD_ACTIVATIONS = frozenset({"gelu", "gelu_tanh", "relu", "silu", "mish"})

_ALL_ACTIVATIONS = _GATED_ACTIVATIONS | _STANDARD_ACTIVATIONS


# ---------------------------------------------------------------------------
# FeedForwardNetwork
# ---------------------------------------------------------------------------

class FeedForwardNetwork(nn.Module):
    """
    [V6] Transformer FFN — bias-free, bellek-verimli, çoklu activation.

    Mimariler:
    ──────────────────────────────────────────────────────────────────────────
    Standart (GELU / GELU-tanh / ReLU / SiLU / Mish):
        x → fc1 → act → dropout → fc2

    Gated (SwiGLU / GeGLU):
        x_proj        = gate_up_proj(x)     [B, T, 2*ffn_dim]  ← tek GEMM
        gate, up      = x_proj.chunk(2)     [B, T, ffn_dim] her biri
        h             = act(gate) * up      [B, T, ffn_dim]   ← in-place çarpım
        h             → dropout → fc2       [B, T, embed_dim]
    ──────────────────────────────────────────────────────────────────────────

    Activation Seçimi:
        "swiglu"    → SiLU(gate) * up           (LLaMA, PaLM, Gemma)
        "geglu"     → GELU(gate) * up            (T5, Flan-T5, Switch)
        "gelu"      → GELU(x)                    (BERT, GPT-2, ViT)
        "gelu_tanh" → GELU(x, approx="tanh")     (GPT-2/3/4 hızlı versiyon)
        "relu"      → ReLU(x)                    (Orijinal Transformer)
        "silu"      → SiLU(x)                    (Bazı vision modeller)
        "mish"      → Mish(x)                    (Alternatif smooth activation)

    Parametre Sayısı:
        Standart:  (embed_dim × ffn_dim) + (ffn_dim × embed_dim)
        Gated:     (embed_dim × 2*ffn_dim) + (ffn_dim × embed_dim)  [V7: merged]
        Bias=True: her projection için + ffn_dim veya embed_dim parametre
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_bias: bool = False,         # [V6] LLaMA/PaLM/Gemma standardı: bias=False
        init_std: float | None = None,  # None → otomatik (0.02 GPT-2 standardı)
        fc2_init_std: float | None = None,  # fc2 için ayrı std (scaled residual init)
        log_level: int = logging.WARNING,   # INFO → WARNING: eğitimde log spam azalır
    ):
        """
        Args:
            embed_dim:      Embedding boyutu (giriş/çıkış)
            ffn_dim:        Feed-forward genişleme boyutu (expand size)
            dropout:        Dropout oranı [0.0, 1.0]
            activation:     Activation fonksiyonu — "swiglu" | "geglu" | "gelu" |
                            "gelu_tanh" | "relu" | "silu" | "mish"
            use_bias:       Linear katmanlarda bias kullan. Modern LLM standardı:
                            False (LLaMA, PaLM, Gemma). True: BERT, GPT-2 uyumlu.
            init_std:       Ağırlık başlatma std'i. None → otomatik:
                            - ReLU: kaiming_normal_ (He init)
                            - Diğer: normal_(std=0.02) (GPT-2 standardı)
            fc2_init_std:   fc2 (output projection) için ayrı std. None → init_std ile aynı.
                            NeuralNetwork'teki scaled residual init ile override edilebilir.
            log_level:      Logging seviyesi
        """
        super().__init__()

        # --- Validasyon ---
        if not isinstance(embed_dim, int) or embed_dim <= 0:
            raise ValueError(f"embed_dim ({embed_dim}) pozitif bir tamsayı olmalıdır.")
        if not isinstance(ffn_dim, int) or ffn_dim <= 0:
            raise ValueError(f"ffn_dim ({ffn_dim}) pozitif bir tamsayı olmalıdır.")
        if not isinstance(dropout, (int, float)) or not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout ({dropout}) 0.0 ile 1.0 arasında olmalıdır.")
        act_lower = activation.lower()
        if act_lower not in _ALL_ACTIVATIONS:
            raise ValueError(
                f"Desteklenmeyen activation: '{activation}'. "
                f"Geçerli seçenekler: {sorted(_ALL_ACTIVATIONS)}"
            )

        # --- Hyperparameter'lar ---
        self.embed_dim       = embed_dim
        self.ffn_dim         = ffn_dim
        self.dropout_rate    = dropout
        self.activation_name = act_lower
        self.use_bias        = use_bias
        self._is_gated       = act_lower in _GATED_ACTIVATIONS

        # --- Logger ---
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            _h = logging.StreamHandler()
            _h.setLevel(log_level)
            _h.setFormatter(logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ))
            self.logger.addHandler(_h)

        # --- Katmanlar ---
        if self._is_gated:
            # [V7] Merged gate_up_proj: tek GEMM → %10-20 GPU hızlanması
            # Bias=False: LLaMA/PaLM/Gemma standardı; gate'de bias SwiGLU'nun
            # teorik formülünü bozar (Dauphin et al. 2017 orijinal formülasyon)
            # HuggingFace LLaMA implementasyonu ile uyumlu parametre isimlendirmesi
            self.gate_up_proj = nn.Linear(embed_dim, 2 * ffn_dim, bias=use_bias)
            self.fc1          = None   # Gated path'te kullanılmaz
            self.fc2          = nn.Linear(ffn_dim, embed_dim, bias=use_bias)
            self.activation   = None   # Gated path: forward içinde F.silu / F.gelu
        else:
            # Standart path: fc1 + activation + fc2
            self.fc1          = nn.Linear(embed_dim, ffn_dim, bias=use_bias)
            self.fc2          = nn.Linear(ffn_dim, embed_dim, bias=use_bias)
            self.gate_up_proj = None
            # Activation modülü (nn.GELU/ReLU/SiLU/Mish)
            self.activation   = self._build_activation(act_lower)

        self.dropout = nn.Dropout(dropout)

        # --- Ağırlık Başlatma ---
        self._init_weights(init_std=init_std, fc2_init_std=fc2_init_std)

        self.logger.debug(
            f"FeedForwardNetwork[V6] embed={embed_dim}, ffn={ffn_dim}, "
            f"act={act_lower}, bias={use_bias}, dropout={dropout}, "
            f"params={self.parameter_count:,}"
        )

    # ------------------------------------------------------------------
    # Yardımcı: activation modülü oluştur
    # ------------------------------------------------------------------

    @staticmethod
    def _build_activation(name: str) -> nn.Module:
        """Activation adından nn.Module döndürür."""
        _map = {
            "gelu":      lambda: nn.GELU(),
            "gelu_tanh": lambda: nn.GELU(approximate="tanh"),  # GPT-2/3/4 approx
            "relu":      lambda: nn.ReLU(),
            "silu":      lambda: nn.SiLU(),
            "mish":      lambda: nn.Mish(),
        }
        return _map[name]()

    # ------------------------------------------------------------------
    # Yardımcı: ağırlık başlatma
    # ------------------------------------------------------------------

    def _init_weights(
        self,
        init_std: float | None,
        fc2_init_std: float | None,
    ) -> None:
        """
        Aktivasyona göre uygun başlatma stratejisi seçer.

        Strateji:
            ReLU       → kaiming_normal_ (He 2015; ReLU için teorik optimum)
            SwiGLU /
            GeGLU /
            GELU /
            SiLU /
            Mish       → normal_(std=0.02) (GPT-2 standardı; smooth non-linear'lar)

        fc2 (output projection):
            fc2_init_std verilmişse onu kullan (scaled residual init ile uyumlu).
            Verilmemişse init_std ile aynı.
        """
        _std = init_std if init_std is not None else 0.02
        _fc2_std = fc2_init_std if fc2_init_std is not None else _std

        def _init_linear(layer: nn.Linear, std: float) -> None:
            if self.activation_name == "relu" and layer is not self.fc2:
                # He init — ReLU için teorik optimum
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            else:
                # GPT-2 / LLaMA standart normal init
                nn.init.normal_(layer.weight, mean=0.0, std=std)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        if self._is_gated:
            # [V7] gate_up_proj: merged projection; tek init çağrısı yeterli
            _init_linear(self.gate_up_proj, _std)
        else:
            _init_linear(self.fc1, _std)

        # fc2: scaled residual init desteği için ayrı std
        _init_linear(self.fc2, _fc2_std)

    # ------------------------------------------------------------------
    # Public: re-init
    # ------------------------------------------------------------------

    def reset_parameters(
        self,
        init_std: float | None = None,
        fc2_init_std: float | None = None,
    ) -> None:
        """
        Ağırlıkları yeniden başlatır.

        Standart PyTorch reset_parameters() arayüzü; continual learning,
        hyperparameter search veya modül yeniden kullanımı senaryolarında
        harici kod tarafından çağrılabilir.

        Args:
            init_std:     Yeni başlatma std'i. None → 0.02 (GPT-2 standardı).
            fc2_init_std: fc2 için ayrı std. None → init_std ile aynı.
        """
        self._init_weights(init_std=init_std, fc2_init_std=fc2_init_std)
        self.logger.debug(
            f"FeedForwardNetwork.reset_parameters() çağrıldı: "
            f"init_std={init_std}, fc2_init_std={fc2_init_std}"
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        FFN forward pass.

        Args:
            x: [B, T, embed_dim]

        Returns:
            [B, T, embed_dim]

        Bellek / hız optimizasyonu (Gated path) [V7]:
            Eski: gate_proj(x) + up_proj(x) = 2 ayrı GEMM
            Yeni: gate_up_proj(x).chunk(2) = tek GEMM + kopyasız split
            Hız:  %10-20 GPU hızlanması (bellek bant genişliği kazancı)
            VRAM: chunk view tabanlıdır; ekstra tahsis yok
        """
        if self._is_gated:
            return self._gated_forward(x)
        return self._standard_forward(x)

    def _gated_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU / GeGLU forward — tek GEMM, bellek-verimli.

        [V7] Merged gate_up_proj: tek Linear(embed_dim, 2*ffn_dim) çağrısı,
        ardından chunk(2) ile gate ve up'a kopyasız split (view tabanlı).
        İki ayrı GEMM yerine tek GEMM → %10-20 GPU hızlanması.

        SwiGLU: h = SiLU(gate) ⊙ up
        GeGLU:  h = GELU(gate) ⊙ up
        """
        # [V7] Tek GEMM → [B, T, 2*ffn_dim]
        x_proj = self.gate_up_proj(x)

        # chunk: kopyasız in-memory split (view tabanlı, sıfır bellek overhead'i)
        gate, up = x_proj.chunk(2, dim=-1)   # her biri [B, T, ffn_dim]

        # Activation — kernel-fused (F.silu / F.gelu)
        if self.activation_name == "swiglu":
            gate = F.silu(gate)   # [V6] Fix 5: kernel-fused SiLU
        else:
            gate = F.gelu(gate)   # geglu: exact GELU (daha iyi kalite)

        # in-place elementwise çarpım → ayrı tensor tahsisi yok
        gate *= up

        # Dropout → output projection
        gate = self.dropout(gate)
        return self.fc2(gate)   # [B, T, embed_dim]

    def _standard_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standart GELU / GELU-tanh / ReLU / SiLU / Mish forward.

        x → fc1 → activation → dropout → fc2
        """
        x = self.fc1(x)          # [B, T, embed_dim] → [B, T, ffn_dim]
        x = self.activation(x)   # in-place aktive edilmez (activation modülü)
        x = self.dropout(x)
        x = self.fc2(x)          # [B, T, ffn_dim] → [B, T, embed_dim]
        return x

    # ------------------------------------------------------------------
    # Yardımcı property'ler
    # ------------------------------------------------------------------

    @property
    def parameter_count(self) -> int:
        """Toplam parametre sayısı (bias dahil)."""
        return sum(p.numel() for p in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        """Eğitilebilir parametre sayısı."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Temsil
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, ffn_dim={self.ffn_dim}, "
            f"activation={self.activation_name}, bias={self.use_bias}, "
            f"dropout={self.dropout_rate}, params={self.parameter_count:,}"
        )
