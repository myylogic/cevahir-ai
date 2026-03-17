# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: mixture_of_experts.py
Modül: src/neural_network_module/ortak_katman_module
Görev: Mixture of Experts (MoE) - Büyük modeller için sparse activation sağlar.
       N adet expert (FFN), router (her token için en uygun expert'leri seçer),
       top-k routing (her token için k expert seçilir) ve load balancing
       (expert'lerin dengeli kullanımı) işlemlerini yapar. GPT-4, Gemini, GShard
       standardı. Referans: "Outrageously Large Neural Networks" (2017),
       "GShard: Scaling Giant Models" (2020),
       "Switch Transformer" (Fedus et al. 2021).

MİMARİ:
- SOLID Prensipleri: Single Responsibility (MoE işlemleri),
                     Open/Closed (genişletilebilir),
                     Dependency Inversion (nn.Module abstraction'ına bağımlı)
- Design Patterns: Expert Pattern (mixture of experts)
- Endüstri Standartları: GPT-4, Gemini, GShard, Switch Transformer MoE standardı

V8 DEĞİŞİKLİKLERİ:
- Bug Fix: Load balancing loss formülü düzeltildi (Switch Transformer standardı)
  Eski: num_experts * (p_i^2).sum()  — f_i terimi eksikti
  Yeni: alpha * N * sum(f_i * p_i)  — Fedus et al. 2021 formülü
- load_balance_alpha parametresi eklendi (varsayılan: 0.01, Switch Transformer)
- Router jitter noise eklendi (training stabilizasyonu için, GPT-4 standardı)
- top_k validasyonu eklendi (top_k <= num_experts kontrolü)
- del dead code kaldırıldı (autograd tensor'larında etkisiz)
- Dispatch loop vektörleştirildi (k-loop → tek matris çarpımı ile ağırlık hesabı)
- _compute_load_balance_loss ayrı metoda taşındı (test edilebilirlik)
- Router.forward: logits döndürür (loss computation MoE'da, Router'da değil)

KULLANIM:
- MoE layer oluşturmak için
- Sparse activation için
- Büyük model kapasitesi için (use_moe=True, transformer_encoder_layer.py)

BAĞIMLILIKLAR:
- torch.nn: Module base class
- torch.nn.functional: Activation fonksiyonları
- FeedForwardNetwork: Her expert için FFN modülü

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple
import math


class Router(nn.Module):
    """
    [V8] Expert Router — Jitter noise + doğru validation.

    Her token için router logit'lerinden top-k expert seçer ve
    renormalize edilmiş ağırlıklar döndürür.

    Jitter Noise (Switch Transformer / GPT-4):
        Eğitim sırasında router logit'lerine küçük uniform gürültü eklenir.
        Routing collapse'ı önler: başlangıçta birkaç güçlü expert'e yığılma
        riski azalır, geri kalan expert'ler de eğitilebilir hale gelir.
        jitter_noise=0.0 → gürültü yok (ablasyon / inference için).

    Forward Çıktısı:
        expert_weights : [B, T, top_k] — Renormalize ağırlıklar (softmax)
        expert_indices : [B, T, top_k] — Seçilen expert indeksleri
        router_logits  : [B, T, N]     — Ham logitler (loss hesabı için MoE'a iletilir)
    """

    def __init__(
        self,
        embed_dim: int,
        num_experts: int,
        top_k: int = 1,
        jitter_noise: float = 0.01,
        log_level: int = logging.INFO,
    ):
        """
        Args:
            embed_dim:    Embedding boyutu
            num_experts:  Expert sayısı
            top_k:        Her token için seçilecek expert sayısı (1 veya 2)
            jitter_noise: Eğitimde router logit'lerine eklenen uniform gürültü miktarı.
                          0.0 = kapalı. Switch Transformer önerisi: 0.01.
            log_level:    Logger seviyesi
        """
        super().__init__()

        if not isinstance(embed_dim, int) or embed_dim <= 0:
            raise ValueError(f"embed_dim ({embed_dim}) pozitif bir tamsayı olmalıdır.")
        if not isinstance(num_experts, int) or num_experts <= 0:
            raise ValueError(f"num_experts ({num_experts}) pozitif bir tamsayı olmalıdır.")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k ({top_k}) pozitif bir tamsayı olmalıdır.")
        if top_k > num_experts:
            raise ValueError(
                f"top_k ({top_k}) > num_experts ({num_experts}). "
                f"top_k en fazla num_experts kadar olabilir."
            )
        if jitter_noise < 0.0:
            raise ValueError(f"jitter_noise ({jitter_noise}) >= 0.0 olmalıdır.")

        self.embed_dim   = embed_dim
        self.num_experts = num_experts
        self.top_k       = top_k
        self.jitter_noise = float(jitter_noise)

        # Router projeksiyon: [embed_dim] → [num_experts]
        # bias=False: routing kararları sadece yön bilgisine dayanmalı
        self.router = nn.Linear(embed_dim, num_experts, bias=False)

        # Xavier init: başlangıçta tüm expert'ler eşit olasılıkla seçilsin
        nn.init.xavier_uniform_(self.router.weight)

        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info(
            f"[V8] Router initialized: embed_dim={embed_dim}, num_experts={num_experts}, "
            f"top_k={top_k}, jitter_noise={jitter_noise}"
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Router forward pass.

        Args:
            x: [B, T, embed_dim]

        Returns:
            expert_weights: [B, T, top_k] — Renormalize ağırlıklar (softmax)
            expert_indices: [B, T, top_k] — Seçilen expert indeksleri
            router_logits:  [B, T, N]     — Ham logitler (MoE'da loss hesabı için)
        """
        # Router logitler: [B, T, num_experts]
        router_logits = self.router(x)

        # [V8] Jitter noise — eğitimde routing collapse önlemi (Switch Transformer)
        # Gürültü logit uzayında eklenir (softmax öncesi) → etkili pertürbasyon
        if self.training and self.jitter_noise > 0.0:
            router_logits = router_logits + torch.empty_like(router_logits).uniform_(
                -self.jitter_noise, self.jitter_noise
            )

        # Top-k seçimi (logit uzayından)
        top_k_logits, expert_indices = torch.topk(
            router_logits, k=self.top_k, dim=-1
        )  # [B, T, top_k]

        # Seçilen top-k logit'ler üzerinde softmax → renormalize ağırlıklar
        # (Tüm N expert yerine sadece seçilen k expert arasında normalize)
        expert_weights = F.softmax(top_k_logits, dim=-1)  # [B, T, top_k]

        return expert_weights, expert_indices, router_logits


class MixtureOfExperts(nn.Module):
    """
    [V8] Mixture of Experts (MoE) — Switch Transformer load balancing düzeltmesi.

    N adet FeedForwardNetwork expert + Router ile sparse activation.
    Her token top_k expert'e yönlendirilir; diğer expert'ler o token için hesap yapmaz.

    Load Balancing Loss (Switch Transformer standardı, Fedus et al. 2021):
    ──────────────────────────────────────────────────────────────────────
        L_aux = alpha × N × Σ_i (f_i × P_i)

        f_i: Expert i'ye dispatch edilen token oranı (hard count, gradient yok)
             = (Expert i'ye atanan token sayısı) / (toplam token sayısı)
        P_i: Router'ın expert i için verdiği ortalama olasılık (differentiable)
             = mean(softmax(router_logits)[..., i])
        alpha: Load balancing katsayısı (varsayılan: 0.01)
        N:     Expert sayısı (normalleştirici)

        f_i × P_i terimi: Gerçek kullanım ile beklenen kullanım arasındaki
        korelasyonu minimize eder → expert'ler dengeli kullanılır.
    ──────────────────────────────────────────────────────────────────────

    Sparse Dispatch Akışı:
        1. Router: Her token için top-k expert seçer
        2. Dispatch: Her expert sadece kendisine atanan token'ları işler
        3. Aggregate: Expert çıktıları ağırlıklı toplanır
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        activation: str = "swiglu",
        # [V8] Load balancing katsayısı (Switch Transformer: 0.01, önerilen: 0.001–0.1)
        load_balance_alpha: float = 0.01,
        # [V8] Router jitter noise (0.0 = kapalı; önerilen: 0.01)
        jitter_noise: float = 0.01,
        log_level: int = logging.INFO,
    ):
        """
        Args:
            embed_dim:          Embedding boyutu (giriş/çıkış)
            ffn_dim:            Her expert FFN'nin iç boyutu
            num_experts:        Expert sayısı (GPT-4: 8, Gemini: 16)
            top_k:              Her token için seçilecek expert sayısı (GPT-4: 2)
            dropout:            Expert FFN'lerdeki dropout oranı
            activation:         Expert aktivasyon fonksiyonu ("swiglu" | "geglu" | "gelu" | ...)
            load_balance_alpha: Load balancing loss katsayısı.
                                Ana loss'a alpha × aux_loss şeklinde eklenir.
                                Çok büyük → routing quality bozulur.
                                Çok küçük → expert collapse riski artar.
                                Switch Transformer standardı: 0.01.
            jitter_noise:       Router jitter noise miktarı (0.0 = kapalı).
                                Eğitimde routing collapse önler. Switch Transformer: 0.01.
            log_level:          Logger seviyesi
        """
        super().__init__()

        # Validasyon
        if not isinstance(embed_dim, int) or embed_dim <= 0:
            raise ValueError(f"embed_dim ({embed_dim}) pozitif bir tamsayı olmalıdır.")
        if not isinstance(ffn_dim, int) or ffn_dim <= 0:
            raise ValueError(f"ffn_dim ({ffn_dim}) pozitif bir tamsayı olmalıdır.")
        if not isinstance(num_experts, int) or num_experts <= 0:
            raise ValueError(f"num_experts ({num_experts}) pozitif bir tamsayı olmalıdır.")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k ({top_k}) pozitif bir tamsayı olmalıdır.")
        if top_k > num_experts:
            raise ValueError(
                f"top_k ({top_k}) > num_experts ({num_experts}). "
                f"top_k en fazla num_experts kadar olabilir."
            )
        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout ({dropout}) [0.0, 1.0] aralığında olmalıdır.")
        if load_balance_alpha < 0.0:
            raise ValueError(f"load_balance_alpha ({load_balance_alpha}) >= 0.0 olmalıdır.")

        self.embed_dim         = embed_dim
        self.ffn_dim           = ffn_dim
        self.num_experts       = num_experts
        self.top_k             = top_k
        self.load_balance_alpha = float(load_balance_alpha)

        # Router
        self.router = Router(
            embed_dim=embed_dim,
            num_experts=num_experts,
            top_k=top_k,
            jitter_noise=jitter_noise,
            log_level=log_level,
        )

        # Expert FFN'ler
        from .feed_forward_network import FeedForwardNetwork
        self.experts = nn.ModuleList([
            FeedForwardNetwork(
                embed_dim=embed_dim,
                ffn_dim=ffn_dim,
                dropout=dropout,
                activation=activation,
                log_level=log_level,
            )
            for _ in range(num_experts)
        ])

        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        total_params = sum(p.numel() for p in self.parameters())
        self.logger.info(
            f"[V8] MixtureOfExperts initialized: embed_dim={embed_dim}, ffn_dim={ffn_dim}, "
            f"num_experts={num_experts}, top_k={top_k}, activation={activation}, "
            f"load_balance_alpha={load_balance_alpha}, jitter_noise={jitter_noise}, "
            f"total_params={total_params:,}"
        )

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MoE forward pass.

        Args:
            x: [B, T, embed_dim]

        Returns:
            output:               [B, T, embed_dim] — Ağırlıklı expert çıktıları
            load_balancing_loss:  Scalar             — Auxiliary load balancing loss
                                  (train.py'de: total_loss += alpha * aux_loss şeklinde değil,
                                   zaten içinde alpha çarpılmış; direkt total_loss += aux_loss)
        """
        B, T, D = x.shape

        # 1) Router — expert seçimi
        expert_weights, expert_indices, router_logits = self.router(x)
        # expert_weights: [B, T, top_k]
        # expert_indices: [B, T, top_k]
        # router_logits:  [B, T, num_experts]

        # 2) [V8] Switch Transformer load balancing loss
        load_balancing_loss = self._compute_load_balance_loss(
            router_logits=router_logits,
            expert_indices=expert_indices,
            total_tokens=B * T,
        )

        # 3) Sparse dispatch
        output = torch.zeros_like(x)  # [B, T, D]
        x_flat = x.view(B * T, D)    # [B*T, D] — flatten for indexing

        # Flatten routing tensors to token level
        flat_indices = expert_indices.view(B * T, self.top_k)   # [B*T, top_k]
        flat_weights = expert_weights.view(B * T, self.top_k)   # [B*T, top_k]

        for expert_idx in range(self.num_experts):
            # [B*T, top_k] — bu expert'e atanan (token, k) pozisyonları
            expert_mask = (flat_indices == expert_idx)

            if not expert_mask.any():
                continue  # Bu expert hiçbir token'a atanmamış → atla

            # [B*T] — bu expert'e atanan token'lar (herhangi bir k pozisyonunda)
            token_mask = expert_mask.any(dim=-1)
            selected_tokens = x_flat[token_mask]  # [N_sel, D]

            if selected_tokens.shape[0] == 0:
                continue

            # Expert forward — sadece seçili token'lar işlenir
            expert_out = self.experts[expert_idx](selected_tokens)  # [N_sel, D]

            # [V8] Vektörize ağırlık hesabı (eski k-loop yerine):
            # topk unique indices garantisi sayesinde her token'da bu expert
            # en fazla bir k pozisyonunda yer alır → sum over k = tek non-zero değer
            eff_weights = (
                flat_weights[token_mask] * expert_mask[token_mask].float()
            ).sum(dim=-1)  # [N_sel]

            # Seçili token'ların ağırlıklı çıktısını tam boyuta geri dağıt
            contrib = torch.zeros(B * T, D, dtype=x.dtype, device=x.device)
            contrib[token_mask] = expert_out * eff_weights.unsqueeze(-1)
            output = output + contrib.view(B, T, D)

        return output, load_balancing_loss

    # ------------------------------------------------------------------ #
    #  Load Balancing Loss                                                 #
    # ------------------------------------------------------------------ #

    def _compute_load_balance_loss(
        self,
        router_logits: torch.Tensor,
        expert_indices: torch.Tensor,
        total_tokens: int,
    ) -> torch.Tensor:
        """
        [V8] Switch Transformer auxiliary load balancing loss.

        Referans: Fedus et al. (2021) — https://arxiv.org/abs/2101.03961

        L_aux = alpha × N × Σ_i (f_i × P_i)

        f_i: Expert i'ye dispatch edilen token fraksiyonu.
             Hard count → gradient akışı yok (detach).
             = (Expert i'ye atanan token sayısı) / total_tokens

        P_i: Router'ın expert i için verdiği ortalama olasılık.
             Differentiable → gradient bu terim üzerinden akar.
             = mean(softmax(router_logits)[..., i])

        Args:
            router_logits:  [B, T, num_experts] veya [BT, num_experts]
            expert_indices: [B, T, top_k] veya [BT, top_k]
            total_tokens:   B × T (flatten boyutu)

        Returns:
            Scalar auxiliary loss tensörü.
        """
        N = self.num_experts

        # P_i: Differentiable, gradient akışı var
        router_probs = F.softmax(
            router_logits.view(total_tokens, N), dim=-1
        )  # [BT, N]
        P = router_probs.mean(dim=0)  # [N]

        # f_i: Hard dispatch count — gradient yok (count-based)
        flat = expert_indices.view(total_tokens, self.top_k)  # [BT, top_k]

        # Dispatch matrisi: dispatch[i, e] = 1 eğer token i expert e'ye atandıysa
        dispatch = torch.zeros(
            total_tokens, N, device=router_logits.device, dtype=router_probs.dtype
        )
        for k_idx in range(self.top_k):
            dispatch.scatter_(1, flat[:, k_idx : k_idx + 1], 1.0)
        # topk unique garantisi sayesinde clamp gerekmez ama savunmacı programlama:
        dispatch = dispatch.clamp(max=1.0)

        f = dispatch.mean(dim=0).detach()  # [N] — gradient kesildi

        # Switch Transformer formülü: alpha × N × Σ(f_i × P_i)
        return self.load_balance_alpha * N * (f * P).sum()

    # ------------------------------------------------------------------ #
    #  Temsil                                                              #
    # ------------------------------------------------------------------ #

    @property
    def parameter_count(self) -> int:
        """Toplam parametre sayısı."""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, ffn_dim={self.ffn_dim}, "
            f"num_experts={self.num_experts}, top_k={self.top_k}, "
            f"load_balance_alpha={self.load_balance_alpha}, "
            f"params={self.parameter_count:,}"
        )
