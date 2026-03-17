# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: neural_network.py
Modül: src
Görev: Cevahir Neural Network - Ana sinir ağı modeli. Dil katmanı (DilKatmani),
       Transformerencoder layer'ları (TransformerEncoderLayer), memory manager
       ve output layer normalization (RMSNorm) ile tam Transformermimarisi
       sağlar. Transformer , BERT, T5 standardında Transformermodeli.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (ana model koordinasyonu),
                     Dependency Inversion (modül abstraksiyonlarına bağımlı)
- Design Patterns: Composite Pattern (katmanların birleşimi)
- Endüstri Standartları: Transformer , BERT, T5 Transformermimarisi

KULLANIM:
- Model oluşturmak için
- Model eğitimi için
- Model inference için
- Standalone model olarak kullanım için

BAĞIMLILIKLAR:
- DilKatmani: Dil katmanı (embedding + positional encoding)
- TransformerEncoderLayer: Transformerencoder katmanları
- MemoryManager: Bellek yönetimi
- RMSNorm: Output layer normalization

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

import sys
import os
import math
import logging
from typing import Optional, Dict, Any, Protocol, runtime_checkable

import torch
import torch.nn as nn

# Proje kökünü ve src'yi path'e ekle (sessiz; print yok)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# [OK] ENDÜSTRİ STANDARDI REFACTOR: DilKatmani deprecated, doğrudan alt modüller kullanılıyor
from neural_network_module.dil_katmani_module.language_embedding import LanguageEmbedding
from neural_network_module.dil_katmani_module.positional_encoding import PositionalEncoding
# [OK] V-2/V-3/V-4: TransformerEncoderLayer kullanılıyor (NeuralLayerProcessor ve TensorProcessingManager yerine)
from neural_network_module.ortak_katman_module.transformer_encoder_layer import TransformerEncoderLayer
# ❌ DEPRECATED: MemoryManager kaldırıldı (2026-01-04)
# from neural_network_module.ortak_katman_module.memory_manager import MemoryManager
# [OK] ENDÜSTRİ STANDARDI: Output layer normalization için RMSNorm/LayerNorm
from neural_network_module.ortak_katman_module.rms_norm import RMSNorm

# --- TensorBoard (opsiyonel) -------------------------------------------------
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # ortamda tensorboard olmayabilir
    SummaryWriter = None  # type: ignore


@runtime_checkable
class _SummaryWriterLike(Protocol):
    """TensorBoard SummaryWriter'a benzer minimal arayüz (Pylance için güvenli)."""
    def add_scalar(self, tag: str, scalar_value: float, global_step: Optional[int] = ...) -> None: ...
    def add_histogram(self, tag: str, values: Any, global_step: Optional[int] = ...) -> None: ...
    def add_image(self, tag: str, img_tensor: Any, global_step: Optional[int] = ...) -> None: ...
    def close(self) -> None: ...


class CevahirNeuralNetwork(nn.Module):
    """
    Cevahir sinir ağı ana modülü (V-2/V-3/V-4).
    
    Mimari:
    - LanguageEmbedding + PositionalEncoding (doğrudan kullanılıyor)
    - TransformerEncoderLayer (N adet) - MultiHeadAttention + FeedForwardNetwork
    - Output layer (vocab projection)
    
    V-2 Özellikleri:
    - Layer stacking (12+ layer)
    - Pre-norm/Post-norm desteği
    - Causal masking
    
    V-3 Optimizasyonları:
    - Flash Attention 2.0 (opsiyonel)
    - RoPE (Rotary Position Embedding)
    - Gradient Checkpointing
    - Weight Tying
    
    V-4 İyileştirmeleri:
    - RoPE entegrasyonu tamamlandı (MultiHeadAttention içinde)
    
    Opsiyonel: TensorBoard telemetri (dahili veya harici SummaryWriter).
    """

    def __init__(
        self,
        learning_rate: float,
        dropout: float,
        vocab_size: int,
        embed_dim: int,
        seq_proj_dim: int,
        num_heads: int,
        attention_type: str = "multi_head",
        normalization_type: str = "layer_norm",
        device: str = "cpu",
        log_level: int = logging.INFO,
        # [OK] YENİ PARAMETRELER (V-2):
        num_layers: int = 12,  # Layer stacking için (endüstri standardı: 12-24)
        ffn_dim: Optional[int] = None,  # FFN dimension (None ise 4x seq_proj_dim)
        pre_norm: bool = True,  # Pre-norm (Transformer ) veya Post-norm (BERT)
        causal_mask: bool = True,  # Autoregressive için causal masking
        # [OK] V3: Flash Attention 2.0 desteği (endüstri standardı: Popüler Kuzenleri Gibi)
        use_flash_attention: bool = False,  # Flash Attention 2.0 kullan (varsa)
        # [OK] V3: RoPE (Rotary Position Embedding) desteği (endüstri standardı: Yakın Kuzenlerine Benzer)
        pe_mode: str = "rope",  # "sinusoidal" | "learned" | "rope" (default: rope - V4 aktif)
        # [OK] V3: Gradient Checkpointing desteği (endüstri standardı: Yakın Kuzenlerine Benzer)
        use_gradient_checkpointing: bool = True,  # Memory-efficient training için (V4 aktif)
        # [OK] V3: Weight Tying desteği (endüstri standardı: Transformer , BERT, T5)
        tie_weights: bool = True,  # Input embedding ve output layer weight sharing (V4 aktif)
        # [OK] V4: RMSNorm desteği (endüstri standardı: GPT-3+, LLaMA)
        use_rmsnorm: bool = True,  # True ise RMSNorm, False ise LayerNorm (V4 aktif)
        # [OK] V4: SwiGLU activation desteği (endüstri standardı: GPT-4, PaLM)
        use_swiglu: bool = True,  # True ise SwiGLU, False ise GELU (V4 aktif)
        # [OK] V4: KV Cache desteği (endüstri standardı: Popüler Kuzenleri Gibi)
        use_kv_cache: bool = True,  # KV Cache kullan (inference için) (V4 aktif)
        max_cache_len: int = 2048,  # Maximum cache length
        # [OK] V4: Advanced Checkpointing desteği (endüstri standardı: Popüler Kuzenleri Gibi)
        use_advanced_checkpointing: bool = False,  # Advanced checkpointing kullan (opsiyonel)
        checkpointing_strategy: str = "selective",  # "selective" | "layer_wise" | "adaptive"
        # [OK] V4: Quantization desteği (endüstri standardı: Popüler Kuzenleri Gibi)
        quantization_type: str = "none",  # "none" | "int8" | "fp16" | "int8_dynamic"
        # [OK] V4: MoE (Mixture of Experts) desteği (endüstri standardı: GPT-4, Gemini)
        use_moe: bool = False,  # MoE kullan (büyük modeller için)
        num_experts: int = 8,  # Expert sayısı (GPT-4: 8, Gemini: 16)
        moe_top_k: int = 2,  # Her token için seçilecek expert sayısı (GPT-4: 2)
        # [OK] V5: GQA (Grouped Query Attention) — LLaMA-2/3, Mistral, Gemini standardı
        # None       → standart MHA (num_kv_heads = num_heads)
        # 1          → MQA (Multi-Query Attention, en hızlı)
        # 2..n-1     → GQA (verimli denge) — Önerilen: num_heads // 4 veya // 8
        # Örnek: num_heads=8, num_kv_heads=2 → %75 KV cache azalması
        num_kv_heads: Optional[int] = None,
        # [OK] V5: Sliding Window Attention — Mistral-7B, Gemma standardı
        # None → full attention (default, geriye dönük uyumluluk)
        # int  → her token yalnızca önceki N token'a attend eder (long-context verimli)
        # Önerilen: 2048 (orta uzunluk) veya 4096 (Mistral standardı)
        sliding_window: Optional[int] = None,
        # [OK] V5: YaRN RoPE genişletme (uzun context desteği)
        # "none"   → standart RoPE (default)
        # "yarn"   → YaRN NTK-by-parts (LLaMA-3.1 standardı, önerilir)
        # "linear" → Position Interpolation (PI, basit)
        rope_scaling_type: str = "none",
        # Context uzatma faktörü (hedef / orijinal)
        # Örnek: 8192 / 2048 = 4.0 (4x context uzatma)
        rope_scaling_factor: float = 1.0,
        # [V6] F.scaled_dot_product_attention (PyTorch 2.0+) — tüm layer'lara iletilir
        use_pytorch_sdpa: bool = True,
        # [V6] QK-Norm (Gemma/PaLM 2) — attention logit patlamasını önler
        use_qk_norm: bool = False,
        # [V6] Parallel Residual / Attention+FFN Paralel (GPT-J, PaLM)
        # x = x + attn(norm(x)) + ffn(norm(x)) — tek norm, paralel branch
        parallel_residual: bool = False,
        # [V6] Output Logit Soft-Cap (Gemma 2): logits = tanh(logits/cap)*cap
        # 0.0 = devre dışı; 30.0 = önerilen (mevcut 1/sqrt(d) yerine)
        logit_soft_cap: float = 30.0,
        # [V6] Attention Logit Soft-Cap (Gemma 2) — tüm layer'lara iletilir (0=kapalı)
        attn_logit_cap: float = 0.0,
        # [V7] Stochastic Depth / LayerDrop (Huang et al. 2016, DeiT standardı)
        # Training sırasında her layer'ın residual path'i drop_rate olasılığıyla sıfırlanır.
        # Lineer decay: layer 0 → 0.0, son layer → drop_path_rate değeri.
        # Inference'ta etki yoktur; geriye dönük uyumlu (default=0.0 = kapalı).
        # Önerilen aralık: 0.05 – 0.20 (model derinliğine göre)
        drop_path_rate: float = 0.0,
        # ---- TensorBoard / Telemetri seçenekleri ----
        use_tensorboard: bool = False,
        tb_writer: Optional[_SummaryWriterLike] = None,
        tb_log_dir: Optional[str] = None,
        tb_log_every_n: int = 10,            # her N forward'ta bir log
        tb_log_histograms: bool = True,      # histogram ekle
        tb_log_attention_image: bool = True, # attn görüntü örneği
        tb_flush_secs: int = 10,
        # İleriye dönük uyumluluk
        **kwargs,
    ):
        super(CevahirNeuralNetwork, self).__init__()

        # --- ctor konfigini sakla (pickling için gerekli) ---
        self._ctor_cfg = {
            "learning_rate": learning_rate,
            "dropout": dropout,
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "seq_proj_dim": seq_proj_dim,
            "num_heads": num_heads,
            "attention_type": attention_type,
            "normalization_type": normalization_type,
            "device": device,
            "log_level": log_level,
            # [OK] YENİ PARAMETRELER (V-2):
            "num_layers": num_layers,
            "ffn_dim": ffn_dim,
            "pre_norm": pre_norm,
            "causal_mask": causal_mask,
            # [OK] V3: Flash Attention 2.0
            "use_flash_attention": use_flash_attention,
            # [OK] V3: RoPE
            "pe_mode": pe_mode,
            # [OK] V3: Gradient Checkpointing
            "use_gradient_checkpointing": use_gradient_checkpointing,
            # [OK] V3: Weight Tying
            "tie_weights": tie_weights,
            # [OK] V4: RMSNorm
            "use_rmsnorm": use_rmsnorm,
            # [OK] V4: SwiGLU
            "use_swiglu": use_swiglu,
            # [OK] V4: KV Cache
            "use_kv_cache": use_kv_cache,
            "max_cache_len": max_cache_len,
            # [OK] V4: Advanced Checkpointing
            "use_advanced_checkpointing": use_advanced_checkpointing,
            "checkpointing_strategy": checkpointing_strategy,
            # [OK] V4: Quantization
            "quantization_type": quantization_type,
            # [OK] V4: MoE
            "use_moe": use_moe,
            "num_experts": num_experts,
            "moe_top_k": moe_top_k,
            # [OK] V5: GQA, Sliding Window, YaRN
            "num_kv_heads": num_kv_heads,
            "sliding_window": sliding_window,
            "rope_scaling_type": rope_scaling_type,
            "rope_scaling_factor": rope_scaling_factor,
            # [V6] Yeni özellikler
            "use_pytorch_sdpa": use_pytorch_sdpa,
            "use_qk_norm": use_qk_norm,
            "parallel_residual": parallel_residual,
            "logit_soft_cap": logit_soft_cap,
            "attn_logit_cap": attn_logit_cap,
            # [V7] Stochastic Depth
            "drop_path_rate": drop_path_rate,
            "use_tensorboard": use_tensorboard,
            "tb_log_dir": tb_log_dir,
            "tb_log_every_n": tb_log_every_n,
            "tb_log_histograms": tb_log_histograms,
            "tb_log_attention_image": tb_log_attention_image,
            "tb_flush_secs": tb_flush_secs,
        }

        # [OK] ENDÜSTRİ STANDARDI: Parametre validasyonu (Transformer , BERT, T5)
        # Kritik parametreler için validation eklenmeli
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            raise ValueError(f"vocab_size ({vocab_size}) pozitif bir tamsayı olmalıdır.")
        if not isinstance(embed_dim, int) or embed_dim <= 0:
            raise ValueError(f"embed_dim ({embed_dim}) pozitif bir tamsayı olmalıdır.")
        if not isinstance(seq_proj_dim, int) or seq_proj_dim <= 0:
            raise ValueError(f"seq_proj_dim ({seq_proj_dim}) pozitif bir tamsayı olmalıdır.")
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(f"num_heads ({num_heads}) pozitif bir tamsayı olmalıdır.")
        if seq_proj_dim % num_heads != 0:
            raise ValueError(f"seq_proj_dim ({seq_proj_dim}) num_heads ({num_heads}) ile tam bölünmelidir.")
        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError(f"num_layers ({num_layers}) pozitif bir tamsayı olmalıdır (en az 1).")
        if not isinstance(dropout, (int, float)) or not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout ({dropout}) 0.0 ile 1.0 arasında olmalıdır.")
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            raise ValueError(f"learning_rate ({learning_rate}) pozitif bir sayı olmalıdır.")

        self.learning_rate = learning_rate
        self.dropout = dropout

        # --- Logger (picklenmeyecek) ---
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)

        # --- Modüller ---
        # [OK] V3: Weight Tying kontrolü (embed_dim == seq_proj_dim olmalı)
        if tie_weights and embed_dim != seq_proj_dim:
            self.logger.warning(
                f"[V3] Weight tying için embed_dim ({embed_dim}) == seq_proj_dim ({seq_proj_dim}) olmalı. "
                f"Weight tying devre dışı bırakıldı."
            )
            tie_weights = False
        
        self.tie_weights = tie_weights
        
        # [OK] ENDÜSTRİ STANDARDI REFACTOR: DilKatmani deprecated, doğrudan alt modüller kullanılıyor
        # [OK] Transformer standardı: Embedding + PositionalEncoding doğrudan initialize edilir
        
        # SeqProjection kontrolü: Eğer embed_dim != seq_proj_dim ise uyarı ver
        if embed_dim != seq_proj_dim:
            self.logger.warning(
                f"[WARNING] embed_dim ({embed_dim}) != seq_proj_dim ({seq_proj_dim}). "
                f"Endüstri standardında genellikle eşittir. SeqProjection kaldırıldı, embed_dim kullanılıyor."
            )
            # SeqProjection kaldırıldı, embed_dim kullanıyoruz
            effective_dim = embed_dim
        else:
            effective_dim = embed_dim
        
        # 1) Language Embedding (Transformer standardı)
        # [OK] FIX: Xavier Normal initialization (endüstri standardı: GPT-3/4, LLaMA)
        # Normal init (std=0.02) büyük vocab_size (60k+) için weight norm yüksek (110.90)
        # Xavier Normal weight norm 3.5× daha düşük (31.82) - gradient explosion önleme
        self.embedding = LanguageEmbedding(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            init_method="xavier_normal",  # [OK] Endüstri standardı: Gradient explosion önleme
            scale_by_sqrt=False,   # [OK] Gradient explosion önleme
            log_level=log_level,
        )
        
        # 2) Positional Encoding (RoPE, Sinusoidal, Learned)
        # [OK] V5: YaRN RoPE desteği — uzun context için
        pe_max_len = kwargs.get("pe_max_len", 2048)
        self.pos_encoding = PositionalEncoding(
            embed_dim=embed_dim,
            max_len=pe_max_len,
            dropout=kwargs.get("pe_dropout", 0.0),
            mode=pe_mode,
            num_heads=num_heads if pe_mode.lower() == "rope" else None,  # [OK] RoPE için num_heads
            log_level=log_level,
            # [OK] V5: YaRN RoPE ölçekleme
            rope_scaling_type=rope_scaling_type,
            rope_scaling_factor=rope_scaling_factor,
            rope_original_max_len=kwargs.get("rope_original_max_len", 2048),
        )
        
        # 3) Embedding Dropout (Transformer  standardı)
        self.embed_dropout = nn.Dropout(dropout)
        
        # [V6] Fix 6: SwiGLU ffn_dim otomatik hesaplama (LLaMA standardı, parametre paritesi)
        # SwiGLU iki gate kullanır; standart 4x FFN ile eş parametre için 2/3 oranı gerekir
        # 256'nın katına yuvarla → tensor core optimal (A100/H100)
        if use_swiglu and ffn_dim is None:
            raw = int(2 / 3 * 4 * effective_dim)
            ffn_dim = (raw + 255) // 256 * 256  # embed_dim=512 → 1536
            self.logger.info(
                f"[V6] SwiGLU ffn_dim otomatik hesaplandı: {ffn_dim} "
                f"(= round_256(2/3*4*{effective_dim}), LLaMA standardı)"
            )
        elif ffn_dim is None:
            ffn_dim = effective_dim * 4  # GPT standardı: 4x embed_dim

        # [V6] Output Logit Soft-Cap (Feature D) — forward'da kullanılır
        self.logit_soft_cap = float(logit_soft_cap)
        
        # [OK] YENİ (V-2): Layer stacking - N adet TransformerEncoderLayer
        # [OK] V3: Flash Attention 2.0 desteği (endüstri standardı)
        # [OK] V4: RoPE (Rotary Position Embedding) desteği (endüstri standardı: Yakın Kuzenlerine Benzer)
        # [OK] V4: RMSNorm ve SwiGLU desteği (endüstri standardı: Yakın Kuzenlerine Benzer)
        self.num_layers = num_layers
        use_rope = (pe_mode.lower() == "rope")
        positional_encoding_ref = self.pos_encoding if use_rope else None  # [OK] REFACTOR: Doğrudan pos_encoding kullan
        
        # [OK] V4: FFN activation seçimi (SwiGLU veya GELU)
        ffn_activation = "swiglu" if use_swiglu else "gelu"
        
        # [V7] Stochastic Depth — lineer decay rate listesi (DeiT / Touvron et al. 2021 standardı)
        # İlk layer: rate=0, son layer: rate=drop_path_rate
        # Derin layer'lar daha fazla regularize edilir → en kritik erken katmanlar korunur.
        _dpr: list = [
            float(x) for x in
            torch.linspace(0.0, float(drop_path_rate), num_layers)
        ] if drop_path_rate > 0.0 else [0.0] * num_layers
        if drop_path_rate > 0.0:
            self.logger.info(
                f"[V7] Stochastic Depth lineer decay: 0.0 → {drop_path_rate:.4f} "
                f"({num_layers} layer, DeiT standardı)"
            )

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=effective_dim,  # [OK] REFACTOR: seq_proj_dim yerine effective_dim (embed_dim)
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                pre_norm=pre_norm,
                attention_type=attention_type,
                normalization_type=normalization_type,
                use_flash_attention=use_flash_attention,  # [OK] V3
                use_gradient_checkpointing=use_gradient_checkpointing,  # [OK] V3
                use_rope=use_rope,  # [OK] V4
                positional_encoding=positional_encoding_ref,  # [OK] V4
                use_rmsnorm=use_rmsnorm,  # [OK] V4
                ffn_activation=ffn_activation,  # [OK] V4
                use_kv_cache=use_kv_cache,  # [OK] V4
                max_cache_len=max_cache_len,  # [OK] V4
                use_advanced_checkpointing=use_advanced_checkpointing,  # [OK] V4
                checkpointing_strategy=checkpointing_strategy,  # [OK] V4
                use_moe=use_moe,  # [OK] V4
                num_experts=num_experts,  # [OK] V4
                moe_top_k=moe_top_k,  # [OK] V4
                num_kv_heads=num_kv_heads,      # [OK] V5: GQA
                sliding_window=sliding_window,  # [OK] V5: Sliding Window
                use_pytorch_sdpa=use_pytorch_sdpa,   # [V6]: F.scaled_dot_product_attention
                use_qk_norm=use_qk_norm,             # [V6]: QK-Norm (Gemma/PaLM 2)
                parallel_residual=parallel_residual,  # [V6]: Parallel Residual (GPT-J/PaLM)
                attn_logit_cap=attn_logit_cap,       # [V6]: Attention Logit Soft-Cap
                drop_path_rate=_dpr[_i],             # [V7]: Stochastic Depth (lineer decay)
            ) for _i in range(num_layers)
        ])

        # [V7] Bug Fix: layer_idx ve total_layers her layer'a set edilmeli.
        # Advanced checkpointing "selective" ve "adaptive" stratejileri bu değerlere
        # dayanır; set edilmezse tüm layer'lar idx=0, total=1 varsayar → yanlış strateji.
        for _i, _layer in enumerate(self.layers):
            _layer.layer_idx = _i
            _layer.total_layers = num_layers

        self.logger.info(
            f"[V7] layer_idx (0..{num_layers-1}) ve total_layers={num_layers} "
            f"tüm TransformerEncoderLayer'lara set edildi (advanced checkpointing fix)"
        )

        # [V6] Feature E: Scaled Residual Projection Init (LLaMA / DeepNet standardı)
        # Derinlikle birlikte varyans büyümesini önler; eğitim stabilitesini artırır
        # out_proj ve fc2 ağırlıkları 1/sqrt(2*N) ile ölçeklenir (N = num_layers)
        _residual_init_scale = 1.0 / math.sqrt(2.0 * num_layers)
        for _layer in self.layers:
            if hasattr(_layer, 'attn') and hasattr(_layer.attn, 'out_proj'):
                nn.init.normal_(_layer.attn.out_proj.weight, mean=0.0, std=_residual_init_scale)
            if hasattr(_layer, 'ffn') and hasattr(_layer.ffn, 'fc2') and _layer.ffn.fc2 is not None:
                nn.init.normal_(_layer.ffn.fc2.weight, mean=0.0, std=_residual_init_scale)
        self.logger.info(
            f"[V6] Scaled Residual Init uygulandı: out_proj + fc2 ağırlıkları "
            f"1/sqrt(2*{num_layers}) = {_residual_init_scale:.4f} ile başlatıldı (LLaMA/DeepNet)"
        )
        
        # [OK] V-2/V-3/V-4: TransformerEncoderLayer kullanılıyor (eski NeuralLayerProcessor ve TensorProcessingManager yerine)
        
        # ❌ DEPRECATED: MemoryManager kaldırıldı (2026-01-04)
        # MemoryManager gereksiz bir abstraction - PyTorch'un built-in bellek yönetimi yeterli
        # KV Cache ve Gradient Checkpointing zaten TransformerEncoderLayer içinde yönetiliyor
        # Endüstri standardında (Transformer) MemoryManager yok
        # self.memory_manager = MemoryManager()  # ❌ Kaldırıldı
        
        # [OK] ENDÜSTRİ STANDARDI: Output layer normalization (Transformer , LLaMA standardı)
        # Output layer'dan önce normalization ekle - logits aralığını kontrol altına alır
        # Bu, accuracy takılı kalma sorununu çözer (logits çok geniş aralıkta olmaz)
        if use_rmsnorm:
            self.output_norm = RMSNorm(effective_dim, eps=1e-6, log_level=log_level)  # [OK] REFACTOR: effective_dim kullan
            # [OK] DÜZELTME: RMSNorm scale initialization'ı normal bırak (1.0)
            # Scale=0.1 çok küçük, gradient'leri büyütüyor (output_norm.scale gradient: 1614.67)
            # Output layer weight initialization zaten düzeltildi (gain=0.05), scale normal kalmalı
            self.logger.info("[V4] Output layer normalization: RMSNorm (GPT-3+, LLaMA standardı)")
        else:
            self.output_norm = nn.LayerNorm(effective_dim, eps=1e-6)  # [OK] REFACTOR: effective_dim kullan
            self.logger.info("[V4] Output layer normalization: LayerNorm (GPT-2 standardı)")
        
        # [OK] Output layer (endüstri standardı: Xavier initialization)
        # [OK] V3: Weight Tying (endüstri standardı: Transformer , BERT, T5)
        if self.tie_weights:
            # Weight tying: output layer weight'ini embedding weight'ine bağla
            # [OK] REFACTOR: embed_dim kullanıyoruz, weight tying doğrudan çalışır
            self.output_layer = nn.Linear(effective_dim, vocab_size, bias=False)  # [OK] REFACTOR: effective_dim kullan
            # Weight tying: output_layer.weight = embedding.weight (aynı referans)
            # Linear layer weight shape: [out_features, in_features] = [vocab_size, effective_dim]
            # Embedding weight shape: [vocab_size, embed_dim] = [vocab_size, effective_dim] (eşit)
            self.output_layer.weight = self.embedding.embedding.weight  # [OK] REFACTOR: Doğrudan self.embedding kullan
            self.logger.info("[V3] Weight tying aktif: Input embedding ve output layer weight'leri paylaşılıyor.")
        else:
            # Normal output layer: [vocab_size, effective_dim]
            self.output_layer = nn.Linear(effective_dim, vocab_size)  # [OK] REFACTOR: effective_dim kullan
            # [OK] ENDÜSTRİ STANDARDI: Proper weight initialization (Xavier/Kaiming)
            # [OK] KRİTİK DÜZELTME: Gain'i daha da küçült (gradient explosion önleme)
            nn.init.xavier_uniform_(self.output_layer.weight, gain=0.05)  # 0.1 → 0.05 (logits kontrolü + gradient explosion önleme)
            if self.output_layer.bias is not None:
                nn.init.zeros_(self.output_layer.bias)
        
        # [OK] V-2 parametreleri (testler için gerekli)
        self.pre_norm = pre_norm
        self.causal_mask = causal_mask
        
        # GPU desteği ekle
        if device and device != 'cpu':
            # self.memory_manager = self.memory_manager.to(device)  # ❌ Kaldırıldı
            self.output_norm = self.output_norm.to(device)
            self.output_layer = self.output_layer.to(device)
            # layers (nn.ModuleList) otomatik GPU'ya taşınır

        # --- TB/telemetri durumu ---
        self._global_step: int = 0
        self._last_snapshot: Dict[str, Any] = {}
        self._last_attn_entropy: Optional[float] = None  # Attention entropy monitoring
        self._tb_log_every_n = int(tb_log_every_n) if tb_log_every_n and tb_log_every_n > 0 else 10
        self._tb_log_histograms = bool(tb_log_histograms)
        self._tb_log_attention_image = bool(tb_log_attention_image)
        self._tb_flush_secs = int(tb_flush_secs) if tb_flush_secs and tb_flush_secs > 0 else 10
        self._tb_writer: Optional[_SummaryWriterLike] = None

        if tb_writer is not None:
            # dışarıdan writer enjekte edildi (manager tarafından kontrol ediliyor)
            self._tb_writer = tb_writer
            self.logger.info("[INIT] Harici TensorBoard writer kullanılıyor.")
        elif use_tensorboard:
            # dahili writer oluştur
            if SummaryWriter is None:
                self.logger.warning("[INIT] TensorBoard bulunamadı. 'pip install tensorboard' ile kurabilirsiniz.")
            else:
                log_dir = tb_log_dir or os.path.join(BASE_DIR, "runs", self.__class__.__name__)
                self._tb_writer = SummaryWriter(log_dir=log_dir, flush_secs=self._tb_flush_secs)  # type: ignore[call-arg]
                self.logger.info(f"[INIT] Dahili TensorBoard etkin. log_dir={log_dir}")

        # [OK] V4: Quantization Manager (endüstri standardı: Popüler Kuzenleri Gibi)
        from src.neural_network_module.ortak_katman_module.quantization_manager import QuantizationManager
        self.quantization_manager = QuantizationManager(
            quantization_type=quantization_type,
            log_level=log_level,
        )
        
        self.logger.info(
            f"[INIT] Cevahir sinir ağı başlatıldı (V6 — GQA + YaRN + Sliding Window + F.sdpa + QK-Norm + Soft-Cap): "
            f"vocab_size={vocab_size}, embed_dim={embed_dim}, effective_dim={effective_dim}, "
            f"num_heads={num_heads}, num_kv_heads={num_kv_heads}, "
            f"num_layers={num_layers}, ffn_dim={ffn_dim}, pre_norm={pre_norm}, "
            f"causal_mask={causal_mask}, sliding_window={sliding_window}, "
            f"use_flash_attention={use_flash_attention}, pe_mode={pe_mode}, "
            f"rope_scaling={rope_scaling_type}(x{rope_scaling_factor:.1f}), "
            f"use_gradient_checkpointing={use_gradient_checkpointing}, tie_weights={tie_weights}, "
            f"use_rope={use_rope}, use_rmsnorm={use_rmsnorm}, use_swiglu={use_swiglu}, "
            f"use_kv_cache={use_kv_cache}, max_cache_len={max_cache_len}, "
            f"quantization_type={quantization_type}, use_moe={use_moe}, num_experts={num_experts}, "
            f"[V6] pytorch_sdpa={use_pytorch_sdpa}, qk_norm={use_qk_norm}, "
            f"parallel_residual={parallel_residual}, logit_soft_cap={logit_soft_cap}, "
            f"attn_logit_cap={attn_logit_cap}"
        )
        self.logger.info(
            "[INIT] [OK] ENDÜSTRİ STANDARDI REFACTOR: DilKatmani deprecated, embedding ve positional encoding doğrudan kullanılıyor. "
            "SeqProjection kaldırıldı (Transformer standardı)."
        )
        
        # Quantization __init__ içinde uygulanmaz (self.training bu noktada daima True).
        # Eğitim tamamlandıktan sonra apply_quantization() çağrılmalıdır.

    # ------------------ Pickling-safe ------------------ #
    def __getstate__(self):
        """
        Pickle sırasında sadece güvenli state'i dön:
        - model ağırlıkları (state_dict)
        - yeniden inşa için ctor konfigi
        - train/eval modu
        - global step
        (Logger ve TensorBoard writer picklenmez.)
        """
        return {
            "state_dict": self.state_dict(),
            "ctor_cfg": getattr(self, "_ctor_cfg", {}),
            "train_mode": bool(self.training),
            "global_step": int(self._global_step),
        }

    def set_tb_config(
        self,
        *,
        log_every_n: Optional[int] = None,
        log_histograms: Optional[bool] = None,
        log_attention_image: Optional[bool] = None,
    ) -> None:
        if log_every_n is not None:
            self._tb_log_every_n = int(log_every_n)
        if log_histograms is not None:
            self._tb_log_histograms = bool(log_histograms)
        if log_attention_image is not None:
            self._tb_log_attention_image = bool(log_attention_image)

    def set_global_step(self, step: int) -> None:
        self._global_step = int(step)
        
    def __setstate__(self, state):
        """
        Unpickle sırasında modeli yeniden kur ve ağırlıkları yükle.
        """
        ctor_cfg = state.get("ctor_cfg", {})
        type(self).__init__(self, **ctor_cfg)
        self.load_state_dict(state["state_dict"])
        self.train(bool(state.get("train_mode", True)))
        self._global_step = int(state.get("global_step", 0))
        # [OK] V-2 attribute'ları restore et (pre_norm, causal_mask)
        if "pre_norm" in ctor_cfg:
            self.pre_norm = ctor_cfg["pre_norm"]
        if "causal_mask" in ctor_cfg:
            self.causal_mask = ctor_cfg["causal_mask"]
        # writer/log yeniden kurulmaz (gerekirse dışarıdan set edilmelidir)

    # ------------------ TensorBoard kontrol/yardımcıları ------------------ #
    def set_tb_writer(self, writer: Optional[_SummaryWriterLike]):
        """Dışarıdan SummaryWriter enjekte et (veya None yaparak kapat)."""
        self._tb_writer = writer
        if writer is None:
            self.logger.info("[TB] TensorBoard writer kapatıldı (None).")
        else:
            self.logger.info("[TB] TensorBoard writer atandı (harici).")

    def close_tb(self):
        """Dahili writer varsa kapat."""
        if self._tb_writer is not None and hasattr(self._tb_writer, "close"):
            try:
                self._tb_writer.close()
                self.logger.info("[TB] TensorBoard writer kapatıldı.")
            except Exception:
                pass
        self._tb_writer = None

    @staticmethod
    def _tensor_stats(t: torch.Tensor) -> Dict[str, float]:
        t_ = t.detach()
        return {
            "min": float(t_.min().item()),
            "max": float(t_.max().item()),
            "mean": float(t_.mean().item()),
            "std": float(t_.std().item()),
        }
        # GEREKSIZ TENSOR PRINT'LERI KALDIRILDI

    def _tb_log_tensor(self, tag: str, t: torch.Tensor, step: int):
        """TensorBoard'a histogram + scalar istatistikleri ekler (opsiyonel)."""
        if self._tb_writer is None:
            return
        t_cpu = t.detach().to("cpu")
        stats = self._tensor_stats(t_cpu)
        self._tb_writer.add_scalar(f"{tag}/min", stats["min"], step)
        self._tb_writer.add_scalar(f"{tag}/max", stats["max"], step)
        self._tb_writer.add_scalar(f"{tag}/mean", stats["mean"], step)
        self._tb_writer.add_scalar(f"{tag}/std", stats["std"], step)
        if self._tb_log_histograms:
            self._tb_writer.add_histogram(f"{tag}/hist", t_cpu, step)
        # GEREKSIZ TENSOR PRINT'LERI KALDIRILDI

    def _tb_log_attention_image_safe(self, tag: str, attn: torch.Tensor, step: int):
        """
        Dikkat ağırlıklarından örnek bir 2D haritayı image olarak loglar (opsiyonel).
        Farklı şekilleri yönetmek için son iki boyutu (L,S) hedefler.
        """
        if self._tb_writer is None or not self._tb_log_attention_image:
            return
        try:
            attn_ = attn.detach().to("cpu")
            if attn_.dim() < 2:
                return
            if attn_.dim() >= 4:
                img = attn_[0, 0]   # (L,S)
            elif attn_.dim() == 3:
                img = attn_[0]      # (L,S)
            else:
                img = attn_

            imin = float(img.min())
            imax = float(img.max())
            if imax - imin > 1e-12:
                img_norm = (img - imin) / (imax - imin)
            else:
                img_norm = torch.zeros_like(img)
            img_norm = img_norm.unsqueeze(0)  # [1, H, W]
            self._tb_writer.add_image(tag, img_norm, step)
        except Exception as e:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"[TB] Attention image loglanamadı: {e}")
        # GEREKSIZ TENSOR PRINT'LERI KALDIRILDI

    def log_gradients(self, step: Optional[int] = None):
        """
        Eğitim döngüsünde backward sonrasında çağrılabilir. Parametre/gradient istatistiklerini TB'ye yollar.
        (Manager/Trainer tarafından kullanılmak üzere.)
        """
        if self._tb_writer is None:
            return
        step = int(self._global_step if step is None else step)
        for name, p in self.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach().to("cpu")
            stats = self._tensor_stats(g)
            self._tb_writer.add_scalar(f"grads/{name}/norm", float(g.norm().item()), step)
            self._tb_writer.add_scalar(f"grads/{name}/mean", stats["mean"], step)
            if self._tb_log_histograms:
                self._tb_writer.add_histogram(f"grads/{name}/hist", g, step)
        # GEREKSIZ TENSOR PRINT'LERI KALDIRILDI

    def get_last_snapshot(self) -> Dict[str, Any]:
        """Son forward’tan özet istatistikler (panel için hızlı erişim)."""
        return dict(self._last_snapshot)

    # ------------------ Forward ------------------ #
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None, 
        causal_mask: Optional[bool] = None,
        # [OK] V4: KV Cache parametreleri (endüstri standardı: Popüler Kuzenleri Gibi)
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
    ):
        """
        İleri yönlü hesaplama (Transformerstandardı - V-2).
        
        Args:
            x: [B, T] token IDs
            mask: Optional attention mask (padding mask vb.)
            causal_mask: Optional causal mask override (None ise self.causal_mask kullanılır)
        
        Returns:
            final_output: [B, T, vocab_size]
            attn_weights: Optional attention weights (son layer'dan)
        """
        step = self._global_step

        try:
            # 1) Giriş doğrulama
            if not isinstance(x, torch.Tensor):
                self.logger.error(f"[FORWARD] Hatalı giriş türü: {type(x)}. torch.Tensor bekleniyordu.")
                raise TypeError(f"Beklenen giriş türü torch.Tensor, ancak {type(x)} alındı.")
            self.logger.debug(f"[FORWARD] Giriş -> shape={x.shape}, dtype={x.dtype}, device={x.device}")

            # 2) Embedding + Positional Encoding (ENDÜSTRİ STANDARDI: Transformer Yakın Kuzenlerine Benzer)
            # [OK] REFACTOR: DilKatmani deprecated, doğrudan embedding ve positional encoding kullanılıyor
            # Akış: Embedding → PositionalEncoding → Dropout
            embedded = self.embedding(x)  # [B, T] → [B, T, embed_dim]
            embedded = self.pos_encoding(embedded)  # [B, T, embed_dim] (RoPE, Sinusoidal, Learned)
            embedded = self.embed_dropout(embedded)  # [B, T, embed_dim]
            if not isinstance(embedded, torch.Tensor):
                raise TypeError("Embedding çıktısı geçerli bir tensör değil!")

            # [OK] YENİ (V-2): Layer stacking - N adet TransformerEncoderLayer
            # Causal mask belirleme
            use_causal = causal_mask if causal_mask is not None else self.causal_mask

            # Layer stacking: Her layer'da Attention + FFN + Residual
            x = embedded
            attn_weights = None
            kv_cache_output = None  # [OK] V4: KV Cache çıktısı için
            _layer_attn_entropies = []  # Attention entropy per layer for collapse detection

            for i, layer in enumerate(self.layers):
                layer_result = layer(
                    x,
                    mask=mask,
                    causal_mask=use_causal,
                    use_cache=use_cache,  # [OK] V4
                    cache_position=cache_position,  # [OK] V4
                )
                # KV Cache kullanılıyorsa: (x, attn_weights, kv_cache)
                # Normal mode: (x, attn_weights)
                if use_cache and len(layer_result) == 3:
                    x, layer_attn_weights, kv_cache_output = layer_result
                else:
                    x, layer_attn_weights = layer_result
                    kv_cache_output = None  # Cache kullanılmıyorsa None
                # Son layer'ın attention weights'ini sakla
                if i == len(self.layers) - 1:
                    attn_weights = layer_attn_weights

                # Attention entropy monitoring (collapse / uniformity detection)
                if layer_attn_weights is not None and self.training:
                    try:
                        # attn_weights: [B, H, T_q, T_k]
                        p = layer_attn_weights.detach().float().clamp(min=1e-10)
                        log_p = torch.log(p)
                        # Shannon entropy over key dimension: [B, H, T_q]
                        h = -(p * log_p).sum(dim=-1).mean().item()
                        # Max entropy for uniform distribution over T_k tokens
                        t_k = layer_attn_weights.shape[-1]
                        max_h = math.log(t_k) if t_k > 1 else 1.0
                        normalized_h = h / max_h if max_h > 0 else 0.0
                        _layer_attn_entropies.append(normalized_h)
                        # Collapse detection: entropy < 0.05 → near-total focus on one token
                        if normalized_h < 0.05:
                            self.logger.warning(
                                f"[ATTN] Attention collapse detected! layer={i}, "
                                f"entropy={normalized_h:.4f} (step={step})"
                            )
                    except Exception:
                        pass

            # Store mean attention entropy across all layers in this step
            self._last_attn_entropy = (
                sum(_layer_attn_entropies) / len(_layer_attn_entropies)
                if _layer_attn_entropies else None
            )

            # ❌ ESKİ (V-1): Kaldırıldı
            # attention_output, attn_weights = self.layer_processor(embedded, key=embedded, value=embedded)
            # projected_output = self.tensor_processing_manager.project(attention_output)

            # 5) Çıktı Katmanı (ENDÜSTRİ STANDARDI: Transformer , LLaMA)
            # [OK] KRİTİK DÜZELTME: Output layer'dan önce normalization (logits aralığını kontrol altına alır)
            # Bu, accuracy takılı kalma sorununu çözer (logits çok geniş aralıkta olmaz)
            # Transformer  ve LLaMA'da output layer'dan önce normalization kullanılır
            x_normalized = self.output_norm(x)  # [B, T, embed_dim]  # [OK] REFACTOR: effective_dim kullanılıyor

            # [OK] V3: Weight tying durumunda da output_layer [vocab_size, embed_dim] bekliyor
            # [OK] REFACTOR: embed_dim kullanıyoruz, weight tying doğrudan çalışır
            logits_raw = self.output_layer(x_normalized)  # [B, T, vocab_size]

            # [V6] Feature D: Output Logit Soft-Cap (Gemma 2 standardı)
            # tanh(logits / cap) * cap → logit'leri [-cap, +cap] aralığına sıkıştırır
            # Önceki: logits_raw / sqrt(d) — sabit bölme, daha kaba
            # Yeni: tanh soft-cap → sürekli türevli, eğitim stabilitesi daha iyi
            if self.logit_soft_cap > 0.0:
                final_output = torch.tanh(logits_raw / self.logit_soft_cap) * self.logit_soft_cap
            else:
                final_output = logits_raw  # Cap kapalı — sınırsız logit

            if not isinstance(final_output, torch.Tensor):
                raise TypeError("Çıktı katmanı geçerli bir tensör döndürmüyor!")

            # 7) Snapshot (panel için özet)
            try:
                self._last_snapshot = {
                    "step": step,
                    "input": {"shape": tuple(x.shape), "dtype": str(x.dtype)},
                    "embedded": {
                        "shape": tuple(embedded.shape),
                        "stats": self._tensor_stats(embedded),
                    },
                    "layer_output": {
                        "shape": tuple(x.shape),
                        "stats": self._tensor_stats(x),
                    },
                    "final_output": {
                        "shape": tuple(final_output.shape),
                        "stats": self._tensor_stats(final_output),
                    },
                    "attn_weights": None if attn_weights is None else {
                        "shape": tuple(attn_weights.shape),
                        "min": float(attn_weights.detach().min().item()),
                        "max": float(attn_weights.detach().max().item()),
                        "mean": float(attn_weights.detach().mean().item()),
                    },
                    "attn_entropy": {
                        "mean_normalized": self._last_attn_entropy,
                        "collapse_threshold": 0.05,
                        "status": (
                            "collapse" if self._last_attn_entropy is not None and self._last_attn_entropy < 0.05
                            else "uniform" if self._last_attn_entropy is not None and self._last_attn_entropy > 0.95
                            else "normal" if self._last_attn_entropy is not None
                            else "not_computed"
                        ),
                    },
                }
            except Exception:
                # snapshot hiçbir zaman modeli düşürmesin
                pass

            # step'i artır
            self._global_step += 1
            
            # [OK] V4: KV Cache kullanılıyorsa 3 değer döndür
            if use_cache:
                return final_output, attn_weights, kv_cache_output
            else:
                return final_output, attn_weights

        except Exception as e:
            self.logger.error(f"[FORWARD] Hata oluştu: {e}", exc_info=True)
            raise

    def clear_kv_cache(self) -> None:
        """
        Tüm layer'lardaki KV cache'i temizler.
        Her yeni generate() öncesi çağrılmalı; aksi halde önceki turun cache'i
        kaldığı için ikinci soruda scores/mask boyut uyuşmazlığı (18 vs 12) oluşur.
        """
        for layer in self.layers:
            if hasattr(layer, "attn") and getattr(layer.attn, "kv_cache", None) is not None:
                layer.attn.kv_cache.clear()
        self.logger.debug("[V4] KV Cache tüm layer'larda temizlendi.")

    def apply_quantization(self, calibration_data: list | None = None) -> None:
        """
        Eğitim tamamlandıktan sonra modeli quantize eder.

        __init__ içinde değil, burada çağrılır; çünkü __init__ sırasında
        self.training=True olduğundan quantization anlamlı değildir.

        Kullanım:
            model = CevahirNeuralNetwork(..., quantization_type="int8_dynamic")
            # ... eğitim döngüsü ...
            model.eval()
            model.apply_quantization()   # ← eğitim bitti, deploy öncesi
            logits = model(x)

        Args:
            calibration_data: Yalnızca quantization_type="int8" için gerekli.
                              Her eleman torch.Tensor veya (Tensor, ...) tuple'ı.
        """
        qt = self.quantization_manager.quantization_type
        if qt == "none":
            self.logger.info("[QUANTIZATION] quantization_type='none', işlem atlandı.")
            return

        if self.training:
            self.logger.warning(
                "[QUANTIZATION] Model eğitim modunda. "
                "Quantization öncesi eval() moduna alınıyor."
            )
            self.eval()

        before_mb = self.quantization_manager.get_model_size_mb(self)
        self.logger.info(
            f"[QUANTIZATION] Başlıyor: type='{qt}', "
            f"model boyutu={before_mb:.1f} MB"
        )

        self.quantization_manager.quantize_model(self, calibration_data=calibration_data)

        after_mb = self.quantization_manager.get_model_size_mb(self)
        savings_pct = (1.0 - after_mb / before_mb) * 100 if before_mb > 0 else 0.0
        self.logger.info(
            f"[QUANTIZATION] Tamamlandı: {before_mb:.1f} MB → {after_mb:.1f} MB "
            f"({savings_pct:.1f}% tasarruf)"
        )

    def get_quantization_info(self) -> dict:
        """
        Quantization durumu hakkında özet bilgi döndürür.

        Returns:
            dict: quantization_type, is_quantized, model_size_mb alanlarını içerir.
        """
        return {
            "quantization_type": self.quantization_manager.quantization_type,
            "is_quantized": self.quantization_manager.is_quantized(self),
            "model_size_mb": round(self.quantization_manager.get_model_size_mb(self), 2),
        }
