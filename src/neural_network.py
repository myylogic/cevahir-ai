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
import time
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
        self.pos_encoding = PositionalEncoding(
            embed_dim=embed_dim,
            max_len=kwargs.get("pe_max_len", 2048),
            dropout=kwargs.get("pe_dropout", 0.0),
            mode=pe_mode,
            num_heads=num_heads if pe_mode.lower() == "rope" else None,  # [OK] RoPE için num_heads
            log_level=log_level,
        )
        
        # 3) Embedding Dropout (Transformer  standardı)
        self.embed_dropout = nn.Dropout(dropout)
        
        # [OK] YENİ (V-2): FFN dimension hesapla
        if ffn_dim is None:
            ffn_dim = effective_dim * 4  # GPT standardı: 4x embed_dim
        
        # [OK] YENİ (V-2): Layer stacking - N adet TransformerEncoderLayer
        # [OK] V3: Flash Attention 2.0 desteği (endüstri standardı)
        # [OK] V4: RoPE (Rotary Position Embedding) desteği (endüstri standardı: Yakın Kuzenlerine Benzer)
        # [OK] V4: RMSNorm ve SwiGLU desteği (endüstri standardı: Yakın Kuzenlerine Benzer)
        self.num_layers = num_layers
        use_rope = (pe_mode.lower() == "rope")
        positional_encoding_ref = self.pos_encoding if use_rope else None  # [OK] REFACTOR: Doğrudan pos_encoding kullan
        
        # [OK] V4: FFN activation seçimi (SwiGLU veya GELU)
        ffn_activation = "swiglu" if use_swiglu else "gelu"
        
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
            ) for _ in range(num_layers)
        ])
        
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
            f"[INIT] Cevahir sinir ağı başlatıldı (V-2/V-3/V-4 + ENDÜSTRİ STANDARDI REFACTOR): vocab_size={vocab_size}, "
            f"embed_dim={embed_dim}, effective_dim={effective_dim}, num_heads={num_heads}, "
            f"num_layers={num_layers}, ffn_dim={ffn_dim}, pre_norm={pre_norm}, causal_mask={causal_mask}, "
            f"use_flash_attention={use_flash_attention}, pe_mode={pe_mode}, "
            f"use_gradient_checkpointing={use_gradient_checkpointing}, tie_weights={tie_weights}, "
            f"use_rope={use_rope}, use_rmsnorm={use_rmsnorm}, use_swiglu={use_swiglu}, "
            f"use_kv_cache={use_kv_cache}, max_cache_len={max_cache_len}, "
            f"quantization_type={quantization_type}, use_moe={use_moe}, num_experts={num_experts}"
        )
        self.logger.info(
            "[INIT] [OK] ENDÜSTRİ STANDARDI REFACTOR: DilKatmani deprecated, embedding ve positional encoding doğrudan kullanılıyor. "
            "SeqProjection kaldırıldı (Transformer standardı)."
        )
        
        # [OK] V4: Model'i quantize et (eğer quantization aktifse ve inference modundaysa)
        if quantization_type != "none" and not self.training:
            self.logger.info(f"[V4] Model quantization uygulanıyor: {quantization_type}")
            self.quantization_manager.quantize_model(self)

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
        t_start = time.time()
        step = self._global_step

        try:
            # 1) Giriş doğrulama
            if not isinstance(x, torch.Tensor):
                self.logger.error(f"[FORWARD] Hatalı giriş türü: {type(x)}. torch.Tensor bekleniyordu.")
                raise TypeError(f"Beklenen giriş türü torch.Tensor, ancak {type(x)} alındı.")
            self.logger.debug(f"[FORWARD] Giriş -> shape={x.shape}, dtype={x.dtype}, device={x.device}")
            t_input = time.time()

            # 2) Embedding + Positional Encoding (ENDÜSTRİ STANDARDI: Transformer Yakın Kuzenlerine Benzer)
            # [OK] REFACTOR: DilKatmani deprecated, doğrudan embedding ve positional encoding kullanılıyor
            # Akış: Embedding → PositionalEncoding → Dropout
            embedded = self.embedding(x)  # [B, T] → [B, T, embed_dim]
            embedded = self.pos_encoding(embedded)  # [B, T, embed_dim] (RoPE, Sinusoidal, Learned)
            embedded = self.embed_dropout(embedded)  # [B, T, embed_dim]
            if not isinstance(embedded, torch.Tensor):
                raise TypeError("Embedding çıktısı geçerli bir tensör değil!")
            t_embedding = time.time()

            # [OK] YENİ (V-2): Layer stacking - N adet TransformerEncoderLayer
            # Causal mask belirleme
            use_causal = causal_mask if causal_mask is not None else self.causal_mask
            
            # Layer stacking: Her layer'da Attention + FFN + Residual
            x = embedded
            attn_weights = None
            kv_cache_output = None  # [OK] V4: KV Cache çıktısı için
            
            t_layers_start = time.time()
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
            t_layers_end = time.time()
            t_attention = t_layers_end  # Timing için

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
            # [AUDIT FIX] Logit scale: output_norm.scale eğitimde büyüyebiliyor (-14..28 aralığı),
            # logit'ler patlıyor → softmax aşırı sivri, mode collapse. 1/sqrt(d) ile sınırla.
            final_output = logits_raw / math.sqrt(self.output_layer.in_features)
            if not isinstance(final_output, torch.Tensor):
                raise TypeError("Çıktı katmanı geçerli bir tensör döndürmüyor!")
            
            t_output = time.time()

            # 6) Bellek Yönetimi
            # ❌ DEPRECATED: MemoryManager kaldırıldı (2026-01-04)
            # PyTorch'un built-in bellek yönetimi yeterli
            # KV Cache zaten TransformerEncoderLayer içinde yönetiliyor (use_kv_cache=True)
            # Gradient Checkpointing zaten var (use_gradient_checkpointing=True)
            t_memory = time.time()

            # 7) Snapshot (panel için özet) - GEREKSIZ TENSOR PRINT'LERI KALDIRILDI
            try:
                self._last_snapshot = {
                    "step": step,
                    "input": {"shape": tuple(x.shape), "dtype": str(x.dtype)},
                    "embedded": {
                        "shape": tuple(embedded.shape),
                        "stats": self._tensor_stats(embedded),
                    },
                    "layer_output": {  # [OK] YENİ: Layer stacking output
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
                    "timings": {
                        "input": t_input - t_start,
                        "embedding": t_embedding - t_input,
                        "layers": t_layers_end - t_layers_start,  # [OK] YENİ: Layer stacking timing
                        "output": t_output - t_layers_end,
                        "memory": t_memory - t_output,
                        "total": t_memory - t_start,
                    },
                }
            except Exception:
                # snapshot hiçbir zaman modeli düşürmesin
                pass

            # 8) TensorBoard telemetri KAPATILDI - gereksiz loglar
            # TensorBoard logging devre dışı bırakıldı

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "[FORWARD] Zamanlar (V-2) | "
                    f"Input:{t_input - t_start:.4f}s, Emb:{t_embedding - t_input:.4f}s, "
                    f"Layers:{t_layers_end - t_layers_start:.4f}s ({self.num_layers} layer), "
                    f"Out:{t_output - t_layers_end:.4f}s, Mem:{t_memory - t_output:.4f}s, "
                    f"Toplam:{t_memory - t_start:.4f}s"
                )

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
