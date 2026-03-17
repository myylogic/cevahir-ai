# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: transformer_encoder_layer.py
Modül: src/neural_network_module/ortak_katman_module
Görev: Transformer Encoder Layer - Transformer standardı: Self-Attention + FFN +
       Residual + Pre-norm/Post-norm. Endüstri standardı: GPT-2/3/4 (Pre-norm),
       BERT (Post-norm), T5 (Pre-norm). Gradient checkpointing, advanced
       checkpointing ve RMSNorm desteği sağlar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (encoder layer işlemleri),
                     Open/Closed (Pre-norm/Post-norm seçeneği ile genişletilebilir),
                     Dependency Inversion (MultiHeadAttention, FeedForwardNetwork
                     abstraction'larına bağımlı)
- Design Patterns: Layer Pattern (Transformer encoder katmanı)
- Endüstri Standartları: GPT-2/3/4, BERT, T5 Transformer encoder standardı

KULLANIM:
- Transformer encoder katmanı oluşturmak için
- Self-attention + FFN kombinasyonu için
- Pre-norm/Post-norm seçimi için

BAĞIMLILIKLAR:
- MultiHeadAttention: Çok başlıklı dikkat
- FeedForwardNetwork: Feed-forward network
- RMSNorm: Root mean square normalization
- AdvancedCheckpointing: Gelişmiş checkpointing

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
import logging
from typing import Optional, Tuple, Union

# [OK] V3: Gradient Checkpointing (endüstri standardı: GPT-3+, Claude, Gemini)
from torch.utils.checkpoint import checkpoint

from .attention_manager_module.multi_head_attention import MultiHeadAttention
from .feed_forward_network import FeedForwardNetwork
# [OK] V4: RMSNorm (endüstri standardı: GPT-3+, LLaMA)
from .rms_norm import RMSNorm
# [OK] V4: Advanced Checkpointing (endüstri standardı: GPT-4, Claude, Gemini)
from .advanced_checkpointing import AdvancedCheckpointing, create_checkpointing_strategy


class TransformerEncoderLayer(nn.Module):
    """
    Transformer standardı: Self-Attention + FFN + Residual + Pre-norm/Post-norm
    Endüstri standardı: GPT-2/3/4 (Pre-norm), BERT (Post-norm), T5 (Pre-norm)
    
    PRE-NORM AKIŞI (pre_norm=True, GPT-2/3/4 standardı):
    ┌─────────────────────────────────────────────────────────┐
    │ Input: [B, T, embed_dim]                                │
    │   ↓                                                     │
    │ LayerNorm (PRE)                                         │
    │   ↓                                                     │
    │ Self-Attention                                          │
    │   ↓                                                     │
    │ Dropout                                                 │
    │   ↓                                                     │
    │ Residual (+) ←───────────┐                              │
    │   ↓                      │                              │
    │ [B, T, embed_dim]        │                              │
    │   ↓                      │                              │
    │ LayerNorm (PRE)          │                              │
    │   ↓                      │                              │
    │ FFN                      │                              │
    │   ↓                      │                              │
    │ Dropout                  │                              │
    │   ↓                      │                              │
    │ Residual (+) ←───────────┘                              │
    │   ↓                                                     │
    │ Output: [B, T, embed_dim]                               │
    └─────────────────────────────────────────────────────────┘
    
    POST-NORM AKIŞI (pre_norm=False, BERT standardı):
    ┌─────────────────────────────────────────────────────────┐
    │ Input: [B, T, embed_dim]                                │
    │   ↓                                                     │
    │ Self-Attention                                          │
    │   ↓                                                     │
    │ Dropout                                                 │
    │   ↓                                                     │
    │ Residual (+) ←───────────┐                              │
    │   ↓                      │                              │
    │ LayerNorm (POST)         │                              │
    │   ↓                      │                              │
    │ [B, T, embed_dim]        │                              │
    │   ↓                      │                              │
    │ FFN                      │                              │
    │   ↓                      │                              │
    │ Dropout                  │                              │
    │   ↓                      │                              │
    │ Residual (+) ←───────────┘                              │
    │   ↓                                                     │
    │ LayerNorm (POST)                                        │
    │   ↓                                                     │
    │ Output: [B, T, embed_dim]                               │
    └─────────────────────────────────────────────────────────┘
    
    FARK:
    - PRE-NORM: LayerNorm → Attention/FFN → Residual
    - POST-NORM: Attention/FFN → Residual → LayerNorm
    
    ENDÜSTRİ STANDARDI:
    - GPT-2/3/4: [OK] Pre-norm (daha stabil, deep network'lerde)
    - BERT: [OK] Post-norm (orijinal Transformer)
    - T5: [OK] Pre-norm (modern standard)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        pre_norm: bool = True,
        attention_type: str = "multi_head",
        normalization_type: str = "layer_norm",
        log_level: int = logging.INFO,
        # [OK] V3: Flash Attention 2.0 desteği (endüstri standardı)
        use_flash_attention: bool = False,
        # [OK] V3: Gradient Checkpointing desteği (endüstri standardı: GPT-3+, Claude, Gemini)
        use_gradient_checkpointing: bool = True,  # V4 aktif
        # [OK] V4: Advanced Checkpointing desteği (endüstri standardı: GPT-4, Claude, Gemini)
        use_advanced_checkpointing: bool = False,  # Advanced checkpointing kullan
        checkpointing_strategy: str = "selective",  # "selective" | "layer_wise" | "adaptive"
        # [OK] V4: RoPE (Rotary Position Embedding) desteği (endüstri standardı: GPT-3+, Claude, Gemini)
        use_rope: bool = False,
        positional_encoding=None,  # PositionalEncoding modülü referansı (RoPE için)
        # [OK] V4: RMSNorm desteği (endüstri standardı: GPT-3+, LLaMA)
        use_rmsnorm: bool = False,  # True ise RMSNorm, False ise LayerNorm
        # [OK] V4: KV Cache desteği (endüstri standardı: GPT-4, Claude, Gemini)
        use_kv_cache: bool = False,  # KV Cache kullan (inference için)
        max_cache_len: int = 2048,  # Maximum cache length
        # [OK] V4: MoE (Mixture of Experts) desteği (endüstri standardı: GPT-4, Gemini)
        use_moe: bool = False,  # MoE kullan
        num_experts: int = 8,  # Expert sayısı (GPT-4: 8, Gemini: 16)
        moe_top_k: int = 2,  # Her token için seçilecek expert sayısı (GPT-4: 2)
        # [OK] V5: GQA (Grouped Query Attention) desteği (LLaMA-2/3, Mistral standardı)
        num_kv_heads: Optional[int] = None,  # None → standart MHA
        # [OK] V5: Sliding Window Attention (Mistral-7B standardı)
        sliding_window: Optional[int] = None,  # None → full attention
        # [V6] F.scaled_dot_product_attention (PyTorch 2.0+) — MHA'ya iletilir
        use_pytorch_sdpa: bool = True,
        # [V6] QK-Norm (Gemma/PaLM 2) — MHA'ya iletilir
        use_qk_norm: bool = False,
        # [V6] Parallel Residual / Attention+FFN Paralel (GPT-J, PaLM standardı)
        # x = x + attn(norm(x)) + ffn(norm(x)) — tek paylaşımlı norm, paralel branch
        # Avantaj: Tek forward norm geçişi, azaltılmış latency
        parallel_residual: bool = False,
        # [V6] Attention Logit Soft-Cap (Gemma 2) — MHA'ya iletilir (0=kapalı)
        attn_logit_cap: float = 0.0,
        # [V7] Stochastic Depth / LayerDrop (Huang et al. 2016, DeiT standardı)
        # Her layer'ın residual path'ini training sırasında rastgele siler.
        # Derin katmanlar daha yüksek rate alır (lineer decay ile neural_network.py'den set edilir).
        # 0.0 = kapalı (default, geriye dönük uyumlu)
        drop_path_rate: float = 0.0,
        # FFN activation seçimi — FeedForwardNetwork'e iletilir
        # "swiglu" | "geglu" | "gelu" | "gelu_tanh" | "relu" | "silu" | "mish"
        ffn_activation: str = "gelu",
        # FFN bias — FeedForwardNetwork'e iletilir (LLaMA/PaLM standardı: False)
        ffn_use_bias: bool = False,
        # [V7] KV Cache eviction (StreamingLLM) — MHA'ya iletilir
        kv_eviction_strategy: str = "sliding_window",  # "none" | "sliding_window"
        kv_num_sink_tokens: int = 4,    # Attention sink token sayısı (Xiao et al. 2023)
        **kwargs,  # Geriye dönük uyumluluk için (artık ffn_activation buraya gelmez)
    ):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ffn_dim: Feed-forward network dimension
            dropout: Dropout rate
            pre_norm: True ise Pre-norm, False ise Post-norm
            attention_type: "multi_head", "self", "cross"
            normalization_type: "layer_norm", "batch_norm", "group_norm"
            log_level: Logging level
            use_rope: RoPE (Rotary Position Embedding) kullan
            positional_encoding: PositionalEncoding modülü referansı (RoPE için gerekli)
            use_rmsnorm: True ise RMSNorm, False ise LayerNorm kullan (GPT-3+, LLaMA standardı)
            use_kv_cache: KV Cache kullan (inference için)
            max_cache_len: Maximum cache length
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout
        self.pre_norm = pre_norm
        
        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # 1) Self-Attention
        # [OK] V3: Flash Attention 2.0 desteği (endüstri standardı)
        # [OK] V4: RoPE (Rotary Position Embedding) desteği (endüstri standardı: GPT-3+, Claude, Gemini)
        # [OK] V4: KV Cache desteği (endüstri standardı: GPT-4, Claude, Gemini)
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            normalization_type=normalization_type,
            debug=False,
            use_flash_attention=use_flash_attention,  # [OK] V3
            use_rope=use_rope,  # [OK] V4
            positional_encoding=positional_encoding,  # [OK] V4
            use_kv_cache=use_kv_cache,  # [OK] V4
            max_cache_len=max_cache_len,  # [OK] V4
            num_kv_heads=num_kv_heads,   # [OK] V5: GQA
            sliding_window=sliding_window,  # [OK] V5: Sliding Window
            use_pytorch_sdpa=use_pytorch_sdpa,       # [V6]: F.scaled_dot_product_attention
            use_qk_norm=use_qk_norm,               # [V6]: QK-Norm (Gemma/PaLM 2)
            attn_logit_cap=attn_logit_cap,         # [V6]: Attention Logit Soft-Cap
            kv_eviction_strategy=kv_eviction_strategy,  # [V7]: StreamingLLM eviction
            kv_num_sink_tokens=kv_num_sink_tokens,      # [V7]: Attention sink sayısı
        )

        # [V6] Feature C: Parallel Residual (GPT-J, PaLM standardı)
        self.parallel_residual = parallel_residual
        if parallel_residual:
            self.logger.info(
                "[V6] Parallel Residual etkinleştirildi (GPT-J/PaLM standardı): "
                "attn ve ffn aynı norm'dan besleniyor (tek norm geçişi)"
            )
        # [OK] V4: RMSNorm veya LayerNorm seçimi (GPT-3+, LLaMA standardı)
        if use_rmsnorm:
            self.norm1 = RMSNorm(embed_dim, eps=1e-6, log_level=log_level)
            self.norm2 = RMSNorm(embed_dim, eps=1e-6, log_level=log_level)
        else:
            # Endüstri standardı: LayerNorm epsilon (GPT-2/3/4: 1e-5)
            self.norm1 = nn.LayerNorm(embed_dim, eps=1e-5)
            self.norm2 = nn.LayerNorm(embed_dim, eps=1e-5)
        
        # 2) FFN veya MoE
        # [OK] V4: MoE (Mixture of Experts) desteği (endüstri standardı: GPT-4, Gemini)
        # use_moe, num_experts, moe_top_k parametreleri __init__ signature'ında tanımlı
        
        if use_moe:
            from .mixture_of_experts import MixtureOfExperts
            self.ffn = MixtureOfExperts(
                embed_dim=embed_dim,
                ffn_dim=ffn_dim,
                num_experts=num_experts,
                top_k=moe_top_k,
                dropout=dropout,
                activation=ffn_activation,  # [V7] explicit param
                log_level=log_level,
            )
            self.use_moe = True
        else:
            # [OK] V4: SwiGLU veya GELU activation seçimi (GPT-4, PaLM standardı)
            self.ffn = FeedForwardNetwork(
                embed_dim=embed_dim,
                ffn_dim=ffn_dim,
                dropout=dropout,
                activation=ffn_activation,  # [V7] explicit param (swiglu/geglu/gelu/...)
                use_bias=ffn_use_bias,       # [V7] bias kontrolü iletildi
                log_level=log_level,
            )
            self.use_moe = False
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # [OK] V3: Gradient Checkpointing (endüstri standardı: GPT-3+, Claude, Gemini)
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # [OK] V4: Advanced Checkpointing (endüstri standardı: GPT-4, Claude, Gemini)
        self.use_advanced_checkpointing = use_advanced_checkpointing
        self.checkpointing_strategy = checkpointing_strategy
        self.advanced_checkpointing: Optional[AdvancedCheckpointing] = None
        if use_advanced_checkpointing:
            from src.neural_network_module.ortak_katman_module.advanced_checkpointing import AdvancedCheckpointing
            self.advanced_checkpointing = AdvancedCheckpointing(
                strategy=checkpointing_strategy,
                log_level=log_level,
            )
            self.logger.info(
                f"[V4] Advanced Checkpointing etkinleştirildi (endüstri standardı: GPT-4, Claude, Gemini), "
                f"strategy={checkpointing_strategy}"
            )
        
        # [V8 Fix] MoE auxiliary loss: liste yerine scalar accumulation.
        # Eski tasarım: self._moe_losses = [] → her forward tensor ekleniyor,
        # get_and_reset_moe_loss() çağrılmazsa tüm computation graph'lar bellekte kalıyor.
        # Yeni tasarım: None veya scalar tensor. Birden fazla forward'da gradyan
        # akışı korunur (gradient accumulation), ancak N adımlık liste yoktur.
        self._moe_loss_accum: Optional[torch.Tensor] = None
        # Kaç adımda bir reset edilmeden accumulate edildiğini izle (debug/warning için)
        self._moe_accum_steps: int = 0

        # [V7] Stochastic Depth (Huang et al. 2016 — Deep Networks with Stochastic Depth)
        # drop_path_rate > 0 ise training'de residual path rastgele sıfırlanır.
        # layer_idx arttıkça oran artar (lineer decay, neural_network.py'den set edilir).
        self.drop_path_rate: float = float(drop_path_rate)
        if self.drop_path_rate > 0.0:
            self.logger.info(
                f"[V7] Stochastic Depth etkinleştirildi: drop_path_rate={self.drop_path_rate:.4f} "
                f"(Huang et al. 2016, DeiT standardı)"
            )

        self.logger.info(
            f"TransformerEncoderLayer initialized: embed_dim={embed_dim}, "
            f"num_heads={num_heads}, num_kv_heads={num_kv_heads}, "
            f"ffn_dim={ffn_dim}, ffn_activation={ffn_activation}, dropout={dropout}, "
            f"pre_norm={pre_norm}, use_gradient_checkpointing={use_gradient_checkpointing}, "
            f"use_rope={use_rope}, use_rmsnorm={use_rmsnorm}, sliding_window={sliding_window}, "
            f"[V6] pytorch_sdpa={use_pytorch_sdpa}, qk_norm={use_qk_norm}, "
            f"parallel_residual={parallel_residual}, attn_logit_cap={attn_logit_cap}, "
            f"[V7] drop_path_rate={drop_path_rate}, "
            f"kv_eviction='{kv_eviction_strategy}', kv_sink_tokens={kv_num_sink_tokens}"
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal_mask: bool = False,
        # [OK] V4: KV Cache parametreleri (endüstri standardı: GPT-4, Claude, Gemini)
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple]],
    ]:
        """
        Forward pass.
        
        Args:
            x: [B, T, embed_dim]
            mask: Optional attention mask
            causal_mask: True ise causal masking uygulanır
        
        Returns:
            x: [B, T, embed_dim]
            attn_weights: Optional attention weights
        
        PRE-NORM vs POST-NORM AÇIKLAMASI:
        ===================================
        
        PRE-NORM (pre_norm=True):
        --------------------------
        Adım 1: x_norm = LayerNorm(x)           # Normalize et
        Adım 2: attn_out = Attention(x_norm)   # Normalized ile attention
        Adım 3: x = x + Dropout(attn_out)      # Orijinal x ile topla
        
        POST-NORM (pre_norm=False):
        ----------------------------
        Adım 1: attn_out = Attention(x)        # Orijinal x ile attention
        Adım 2: x = LayerNorm(x + Dropout(attn_out))  # Topla, sonra normalize
        
        FARK:
        - Pre-norm: Normalize → İşlem → Residual
        - Post-norm: İşlem → Residual → Normalize
        
        ENDÜSTRİ STANDARDI:
        - GPT-2/3/4: Pre-norm [OK] (daha stabil)
        - BERT: Post-norm [OK] (orijinal)
        - T5: Pre-norm [OK] (modern)
        """
        # [OK] V3/V4: Gradient Checkpointing (endüstri standardı: GPT-3+/4, Claude, Gemini)
        # Memory-efficient training: activation'ları kaydetmek yerine backward'da yeniden hesapla
        if self.training:
            # [OK] V4: Advanced Checkpointing (endüstri standardı: GPT-4, Claude, Gemini)
            if self.use_advanced_checkpointing and self.advanced_checkpointing is not None:
                # Advanced checkpointing: Selective veya layer-wise strateji
                layer_idx = getattr(self, 'layer_idx', 0)
                total_layers = getattr(self, 'total_layers', 1)
                should_checkpoint = self.advanced_checkpointing.should_checkpoint(
                    layer_idx=layer_idx,
                    total_layers=total_layers,
                    training=self.training,
                )
                if should_checkpoint:
                    return self.advanced_checkpointing.checkpoint_forward(
                        self._forward_impl,
                        x, mask, causal_mask, False, None,
                        use_reentrant=False,
                    )
                else:
                    # Normal forward (checkpoint yok)
                    return self._forward_impl(x, mask, causal_mask, False, None)
            
            # [OK] V3: Standard Gradient Checkpointing
            elif self.use_gradient_checkpointing:
                # Gradient checkpointing: forward pass'i checkpoint ile wrap et
                # NOT: KV Cache ile gradient checkpointing birlikte kullanılamaz (training modunda cache yok)
                # Training modunda use_cache=False olmalı
                return checkpoint(
                    self._forward_impl, 
                    x, mask, causal_mask, False, None, 
                    use_reentrant=False
                )
        
        # Normal forward pass (inference veya checkpointing yok)
        return self._forward_impl(x, mask, causal_mask, use_cache, cache_position)
    
    # =========================================================================
    # [V7] Stochastic Depth — Huang et al. (2016) "Deep Networks with Stochastic Depth"
    # DeiT, RoBERTa, modern ViT implementasyonlarında kullanılan regularization tekniği.
    # =========================================================================

    @staticmethod
    def _stochastic_depth(
        residual: torch.Tensor,
        drop_rate: float,
        training: bool,
    ) -> torch.Tensor:
        """
        Residual path'e stochastic depth uygular.

        Training sırasında drop_rate olasılığıyla tüm residual branch sıfırlanır
        (batch boyutunda bağımsız Bernoulli örneklemesi). Beklenen değeri korumak
        için hayatta kalan örneğe 1/(1-drop_rate) ölçeği uygulanır.

        Inference'ta hiçbir şey sıfırlanmaz — davranış değişmez.

        Args:
            residual : Residual katkı tensörü [B, T, D]
            drop_rate: 0.0 → kapalı; (0, 1) → drop olasılığı
            training  : model.training ile aynı değer

        Returns:
            Ölçeklendirilmiş ve maskelenmiş (veya değişmemiş) residual tensör.

        Referans:
            Huang et al. (2016) — https://arxiv.org/abs/1603.09382
            Touvron et al. (2021) DeiT — DropPath implementasyonu
        """
        if not training or drop_rate == 0.0:
            return residual

        survival_prob = 1.0 - drop_rate
        # [B, 1, 1] şeklinde Bernoulli maskesi → tüm T ve D boyutlarına broadcast
        shape = (residual.shape[0], 1, 1)
        mask = torch.empty(shape, dtype=residual.dtype, device=residual.device)
        mask.bernoulli_(survival_prob)
        # Beklenen değeri koru: E[mask * x / p] = x
        return residual * mask / survival_prob

    def get_and_reset_moe_loss(self) -> Optional[torch.Tensor]:
        """
        [V8 Fix] MoE auxiliary loss'u döndürür ve accumulation'ı sıfırlar.
        Her training step'inde optimizer.step() öncesinde çağrılmalı.

        Eski tasarım (V6): Liste → her forward tensor ekliyordu; reset edilmezse
        computation graph'lar bellekte birikirdi (memory leak).
        Yeni tasarım (V8): Scalar accumulation → sadece tek tensor tutuluyor;
        gradient accumulation desteği korunuyor, leak riski yok.

        Kullanım:
            loss = model_loss + layer.get_and_reset_moe_loss()
            loss.backward()

        Returns:
            Birikmiş MoE load balancing loss (scalar tensor), yoksa None.
        """
        loss = self._moe_loss_accum
        self._moe_loss_accum = None   # Sıfırla — computation graph serbest bırakıldı
        self._moe_accum_steps = 0
        return loss

    def _forward_impl(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal_mask: bool = False,
        # [OK] V4: KV Cache parametreleri (endüstri standardı: GPT-4, Claude, Gemini)
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple]],
    ]:
        """
        Internal forward implementation (checkpointing için ayrı method).
        """
        # ============================================================
        # [V6] Feature C: Parallel Residual (GPT-J, PaLM standardı)
        # Tek paylaşımlı norm; attn ve ffn aynı normalized input'tan beslenir
        # x = x + attn(norm(x)) + ffn(norm(x))
        # ============================================================
        if self.parallel_residual and self.pre_norm:
            return self._parallel_forward_impl(x, mask, causal_mask, use_cache, cache_position)

        # ============================================================
        # 1) SELF-ATTENTION: Pre-norm vs Post-norm Akışı
        # ============================================================

        if self.pre_norm:
            # [OK] PRE-NORM AKIŞI (GPT-2/3/4, Modern Transformer'lar)
            # Akış: x → LayerNorm → Attention → Dropout → Residual (+)
            # 
            # Adım 1: LayerNorm önce (PRE)
            x_norm = self.norm1(x)  # [B, T, embed_dim]
            # 
            # Adım 2: Attention (normalized input ile)
            # [OK] V4: KV Cache desteği (endüstri standardı: GPT-4, Claude, Gemini)
            attn_result = self.attn(
                x_norm, x_norm, x_norm,
                mask=mask,
                causal_mask=causal_mask,
                return_attention_weights=True,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            # KV Cache kullanılıyorsa: (output, attn_weights, kv_cache)
            # Normal mode: (output, attn_weights)
            if use_cache and len(attn_result) == 3:
                attn_output, attn_weights, kv_cache = attn_result
            else:
                attn_output, attn_weights = attn_result
                kv_cache = None
            # 
            # Adım 3: Dropout + Stochastic Depth + Residual (orijinal x ile topla)
            # [V7] Stochastic depth: training'de residual path rastgele sıfırlanabilir
            _attn_residual = self._stochastic_depth(
                self.dropout(attn_output), self.drop_path_rate, self.training
            )
            x = x + _attn_residual  # [B, T, embed_dim]
            #
            # PRE-NORM AVANTAJLARI:
            # - Daha stabil gradient flow (deep network'lerde)
            # - LayerNorm input'u normalize eder, attention daha iyi çalışır
            # - Modern Transformer'ların standardı (GPT-2/3/4, T5)
        else:
            # [OK] POST-NORM AKIŞI (Orijinal Transformer, BERT)
            # Akış: x → Attention → Dropout → Residual (+) → LayerNorm
            #
            # Adım 1: Attention (orijinal input ile)
            # [OK] V4: KV Cache desteği (endüstri standardı: GPT-4, Claude, Gemini)
            attn_result = self.attn(
                x, x, x,
                mask=mask,
                causal_mask=causal_mask,
                return_attention_weights=True,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            # KV Cache kullanılıyorsa: (output, attn_weights, kv_cache)
            # Normal mode: (output, attn_weights)
            if use_cache and len(attn_result) == 3:
                attn_output, attn_weights, kv_cache = attn_result
            else:
                attn_output, attn_weights = attn_result
                kv_cache = None
            #
            # Adım 2: Dropout + Stochastic Depth + Residual + LayerNorm (POST)
            # [V7] Stochastic depth: training'de residual path rastgele sıfırlanabilir
            _attn_residual = self._stochastic_depth(
                self.dropout(attn_output), self.drop_path_rate, self.training
            )
            x = self.norm1(x + _attn_residual)  # [B, T, embed_dim]
            # 
            # POST-NORM AVANTAJLARI:
            # - Orijinal Transformer paper'ına uygun
            # - BERT gibi encoder modellerinde kullanılır
            # - Daha basit implementasyon
        
        # ============================================================
        # 2) FFN: Pre-norm vs Post-norm Akışı
        # ============================================================
        
        if self.pre_norm:
            # [OK] PRE-NORM AKIŞI (GPT-2/3/4, Modern Transformer'lar)
            # Akış: x → LayerNorm → FFN → Dropout → Residual (+)
            # 
            # Adım 1: LayerNorm önce (PRE)
            x_norm = self.norm2(x)  # [B, T, embed_dim]
            # 
            # Adım 2: FFN veya MoE (normalized input ile)
            # [OK] V4: MoE desteği (endüstri standardı: GPT-4, Gemini)
            if self.use_moe:
                ffn_output, moe_load_balancing_loss = self.ffn(x_norm)  # [B, T, embed_dim], scalar
                # [V8 Fix] Scalar accumulation (liste değil) — memory leak yok
                self._moe_loss_accum = (
                    moe_load_balancing_loss if self._moe_loss_accum is None
                    else self._moe_loss_accum + moe_load_balancing_loss
                )
                self._moe_accum_steps += 1
            else:
                ffn_output = self.ffn(x_norm)  # [B, T, embed_dim]
            #
            # Adım 3: Dropout + Stochastic Depth + Residual (orijinal x ile topla)
            # [V7] Stochastic depth FFN branch
            _ffn_residual = self._stochastic_depth(
                self.dropout(ffn_output), self.drop_path_rate, self.training
            )
            x = x + _ffn_residual  # [B, T, embed_dim]
        else:
            # [OK] POST-NORM AKIŞI (Orijinal Transformer, BERT)
            # Akış: x → FFN → Dropout → Residual (+) → LayerNorm
            #
            # Adım 1: FFN veya MoE (orijinal input ile)
            # [OK] V4: MoE desteği (endüstri standardı: GPT-4, Gemini)
            if self.use_moe:
                ffn_output, moe_load_balancing_loss = self.ffn(x)  # [B, T, embed_dim], scalar
                # [V8 Fix] Scalar accumulation
                self._moe_loss_accum = (
                    moe_load_balancing_loss if self._moe_loss_accum is None
                    else self._moe_loss_accum + moe_load_balancing_loss
                )
                self._moe_accum_steps += 1
            else:
                ffn_output = self.ffn(x)  # [B, T, embed_dim]
            #
            # Adım 2: Dropout + Stochastic Depth + Residual + LayerNorm (POST)
            # [V7] Stochastic depth FFN branch
            _ffn_residual = self._stochastic_depth(
                self.dropout(ffn_output), self.drop_path_rate, self.training
            )
            x = self.norm2(x + _ffn_residual)  # [B, T, embed_dim]
        
        # [OK] V4: KV Cache döndür (use_cache=True ise)
        if use_cache:
            return x, attn_weights, kv_cache
        return x, attn_weights
    
    def _parallel_forward_impl(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal_mask: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple]],
    ]:
        """
        [V6] Feature C: Parallel Residual forward (GPT-J, PaLM standardı)

        Akış:
            x_norm = norm1(x)                        # Tek paylaşımlı norm
            attn_out = attn(x_norm)                  # Attention branch
            ffn_out  = ffn(x_norm)                   # FFN branch (aynı x_norm)
            x = x + dropout(attn_out) + dropout(ffn_out)  # Paralel residual

        Avantaj: Sequential pre-norm'a kıyasla tek norm geçişi → daha düşük latency.
        Referans: GPT-J (EleutherAI), PaLM (Google).
        """
        # Tek paylaşımlı norm (norm1 kullanılır; norm2 bu path'te kullanılmaz)
        x_norm = self.norm1(x)  # [B, T, embed_dim]

        # Attention branch
        attn_result = self.attn(
            x_norm, x_norm, x_norm,
            mask=mask,
            causal_mask=causal_mask,
            return_attention_weights=True,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        if use_cache and len(attn_result) == 3:
            attn_output, attn_weights, kv_cache = attn_result
        else:
            attn_output, attn_weights = attn_result
            kv_cache = None

        # FFN branch (aynı x_norm'dan)
        if self.use_moe:
            ffn_output, moe_load_balancing_loss = self.ffn(x_norm)
            # [V8 Fix] Scalar accumulation
            self._moe_loss_accum = (
                moe_load_balancing_loss if self._moe_loss_accum is None
                else self._moe_loss_accum + moe_load_balancing_loss
            )
            self._moe_accum_steps += 1
        else:
            ffn_output = self.ffn(x_norm)

        # Paralel residual: her iki branch aynı anda eklenir
        # [V7] Stochastic depth: her iki branch için aynı drop_path_rate uygulanır
        _attn_res = self._stochastic_depth(
            self.dropout(attn_output), self.drop_path_rate, self.training
        )
        _ffn_res = self._stochastic_depth(
            self.dropout(ffn_output), self.drop_path_rate, self.training
        )
        x = x + _attn_res + _ffn_res  # [B, T, embed_dim]

        if use_cache:
            return x, attn_weights, kv_cache
        return x, attn_weights

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"ffn_dim={self.ffn_dim}, dropout={self.dropout_rate}, pre_norm={self.pre_norm}, "
            f"parallel_residual={self.parallel_residual}, "
            f"drop_path_rate={self.drop_path_rate:.4f}"
        )

