# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: multi_head_attention.py
Modül: src/neural_network_module/ortak_katman_module/attention_manager_module
Görev: Multi-Head Attention - Çok başlıklı dikkat mekanizması. KV Cache desteği
       (GPT-4, Claude, Gemini standardı), Flash Attention 2.0 desteği (opsiyonel),
       scaled dot-product attention ve multi-head attention işlemlerini yapar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (multi-head attention işlemleri)
- Design Patterns: Attention Pattern (çok başlıklı dikkat)
- Endüstri Standartları: GPT-4, Claude, Gemini attention standardı

KULLANIM:
- Multi-head attention oluşturmak için
- KV Cache ile inference için
- Flash Attention optimizasyonu için

BAĞIMLILIKLAR:
- KVCache: Key-value cache (opsiyonel)
- flash_attn: Flash Attention (opsiyonel)
- torch.nn: Attention modülleri

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
import math
import logging
import sys
import os
from typing import Optional, Tuple

# [OK] V4: KV Cache (endüstri standardı: GPT-4, Claude, Gemini)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
try:
    from src.neural_network_module.ortak_katman_module.kv_cache import KVCache
except ImportError:
    # Fallback: KV Cache modülü yoksa None olarak işaretle
    KVCache = None  # type: ignore

# [V6] QK-Norm için RMSNorm import (Gemma/PaLM 2 standardı)
try:
    from src.neural_network_module.ortak_katman_module.rms_norm import RMSNorm as _RMSNorm
    _RMSNORM_AVAILABLE = True
except ImportError:
    _RMSNorm = None  # type: ignore
    _RMSNORM_AVAILABLE = False

# [OK] V3: Flash Attention 2.0 desteği (endüstri standardı: GPT-4, Claude, Gemini)
# Flash Attention opsiyonel dependency - yoksa standard SDPA kullanılır
try:
    import flash_attn  # type: ignore[reportMissingImports]
    from flash_attn import flash_attn_func  # type: ignore[reportMissingImports]
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    flash_attn_func = None

# [OK] Standart logging kullan (TrainingLogger değil - sinir ağı modülünde gereksiz)
logger = logging.getLogger("MultiHeadAttention")
logger.propagate = False  # [OK] Root logger'a gitmesin (tekrar önleme)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)  # [OK] INFO mesajlarını gizle (çok fazla log)


class MultiHeadAttention(nn.Module):
    """
    Çok başlıklı dikkat mekanizması sınıfı.
    Giriş tensörleri üzerinde farklı başlıklar altında dikkat hesaplaması yapar.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout,
        normalization_type="layer_norm",
        debug=False,
        log_level=None,
        # [OK] V3: Flash Attention 2.0 desteği (endüstri standardı)
        use_flash_attention: bool = False,
        # [OK] V4: RoPE (Rotary Position Embedding) desteği (endüstri standardı: GPT-3+, Claude, Gemini)
        use_rope: bool = False,
        positional_encoding=None,  # PositionalEncoding modülü referansı (RoPE için)
        # [OK] V4: KV Cache desteği (endüstri standardı: GPT-4, Claude, Gemini)
        use_kv_cache: bool = False,       # KV Cache kullan (inference için)
        max_cache_len: int = 2048,        # Maximum cache length
        # [V5] StreamingLLM / Attention Sink eviction parametreleri
        kv_eviction_strategy: str = "sliding_window",  # "none" | "sliding_window"
        kv_num_sink_tokens: int = 4,      # Attention sink token sayısı (Xiao et al. 2023)
        # [OK] V5: GQA (Grouped Query Attention) desteği (endüstri standardı: LLaMA-2/3, Mistral, Gemini)
        # num_kv_heads < num_heads: GQA (çoklu query, az KV head)
        # num_kv_heads == 1: MQA (Multi-Query Attention)
        # num_kv_heads == num_heads: standart MHA (default)
        num_kv_heads: Optional[int] = None,
        # [OK] V5: Sliding Window Attention (endüstri standardı: Mistral-7B, Gemma)
        # Her token sadece önceki `sliding_window` token'a attend eder (long-context efficiency)
        # None ise devre dışı (full attention)
        sliding_window: Optional[int] = None,
        # [V6] F.scaled_dot_product_attention (PyTorch 2.0+): Flash/memory-efficient/math otomatik
        # Harici flash_attn bağımlılığı yok; PyTorch 2.0+ Flash Attention backend kullanır
        use_pytorch_sdpa: bool = True,
        # [V6] QK-Norm (Gemma/PaLM 2): Uzun context'te attention logit patlamasını önler
        # Q ve K head'lerine RoPE öncesi RMSNorm uygulanır
        use_qk_norm: bool = False,
        # [V6] Attention Logit Soft-Cap (Gemma 2): scores = tanh(s/cap)*cap, softmax öncesi
        # 0.0 = devre dışı; önerilen: 50.0 (use_pytorch_sdpa=False ile uyumlu)
        attn_logit_cap: float = 0.0,
    ):
        """
        MultiHeadAttention sınıfını başlatır.

        Args:
            embed_dim (int): Giriş tensörlerinin gömme boyutu.
            num_heads (int): Query dikkat başlığı sayısı.
            dropout (float): Tüm katmanlarda ortak kullanılacak Dropout oranı.
            normalization_type (str): Normalizasyon türü ('layer_norm', 'batch_norm', 'instance_norm', 'group_norm').
            debug (bool): Hata ayıklama modu.
            log_level (int, optional): Logger seviyesi (geriye dönük uyumluluk için).
            use_flash_attention (bool): Flash Attention 2.0 kullan (varsa). Default: False.
            use_rope (bool): RoPE (Rotary Position Embedding) kullan. Default: False.
            positional_encoding: PositionalEncoding modülü referansı (RoPE için gerekli).
            use_kv_cache (bool): KV Cache kullan (inference için). Default: False.
            max_cache_len (int): Maximum cache length. Default: 2048.
            num_kv_heads (int, optional): GQA/MQA için KV head sayısı.
                None → standart MHA (num_kv_heads = num_heads).
                1   → MQA (Multi-Query Attention, maksimum verim).
                2..num_heads-1 → GQA (Grouped Query Attention, denge).
                LLaMA-2 70B: num_heads=64, num_kv_heads=8.
            sliding_window (int, optional): Sliding Window Attention pencere boyutu.
                None → full attention (default).
                int  → her token sadece önceki sliding_window token'a attend eder.
                Mistral-7B standardı: 4096 token penceresi.
            use_pytorch_sdpa (bool): F.scaled_dot_product_attention kullan (PyTorch 2.0+). Default: True.
            use_qk_norm (bool): QK-Norm uygula (Gemma/PaLM 2 standardı). Default: False.
            attn_logit_cap (float): Attention score tanh cap. 0.0=devre dışı. Default: 0.0.
        """
        super(MultiHeadAttention, self).__init__()

        # --- Parametre kontrolleri ---
        if not isinstance(embed_dim, int) or embed_dim <= 0:
            raise ValueError(f"embed_dim ({embed_dim}) pozitif bir tamsayı olmalıdır.")
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(f"num_heads ({num_heads}) pozitif bir tamsayı olmalıdır.")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Gömme boyutu ({embed_dim}) çok başlık sayısına ({num_heads}) tam bölünmelidir."
            )
        if not (isinstance(dropout, float) and 0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout ({dropout}) 0..1 aralığında float olmalıdır.")
        if normalization_type not in ["layer_norm", "batch_norm", "instance_norm", "group_norm"]:
            raise ValueError(
                f"Desteklenmeyen normalizasyon türü: {normalization_type}. "
                f"Geçerli seçenekler: 'layer_norm', 'batch_norm', 'instance_norm', 'group_norm'."
            )

        # --- Özellikler ---
        self.logger = logger
        self.logger.debug("MultiHeadAttention __init__ çağrıldı.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.debug = debug

        # [OK] V5: GQA (Grouped Query Attention) — LLaMA-2/3, Mistral, Gemini standardı
        # num_kv_heads == num_heads → standart MHA (geriye dönük uyumluluk)
        # num_kv_heads < num_heads  → GQA (KV cache daha küçük, daha hızlı inference)
        # num_kv_heads == 1         → MQA (en az KV, maksimum hız)
        if num_kv_heads is None:
            self.num_kv_heads = num_heads  # Standart MHA
        else:
            if num_kv_heads <= 0:
                raise ValueError(f"num_kv_heads ({num_kv_heads}) pozitif olmalıdır.")
            if num_heads % num_kv_heads != 0:
                raise ValueError(
                    f"num_heads ({num_heads}) num_kv_heads ({num_kv_heads}) ile tam bölünmelidir. "
                    f"GQA: num_kv_groups = num_heads // num_kv_heads tam sayı olmalı."
                )
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads  # KV head repetition factor
        self.kv_dim = self.num_kv_heads * self.head_dim      # KV projeksiyon boyutu

        gqa_mode = (
            "MHA" if self.num_kv_heads == num_heads
            else "MQA" if self.num_kv_heads == 1
            else f"GQA(kv_heads={self.num_kv_heads})"
        )
        self.logger.info(
            f"[V5] Attention modu: {gqa_mode} | "
            f"num_heads={num_heads}, num_kv_heads={self.num_kv_heads}, "
            f"num_kv_groups={self.num_kv_groups}"
        )

        # [OK] V5: Sliding Window Attention — Mistral-7B, Gemma standardı
        # Uzun dizilerde local attention → O(n*w) yerine O(n²) azaltır
        self.sliding_window = sliding_window
        if sliding_window is not None:
            if sliding_window <= 0:
                raise ValueError(f"sliding_window ({sliding_window}) pozitif olmalıdır.")
            self.logger.info(
                f"[V5] Sliding Window Attention aktif: window_size={sliding_window} "
                f"(Mistral-7B standardı)"
            )

        # [V6] F.scaled_dot_product_attention (PyTorch 2.0+)
        # use_pytorch_sdpa=True → Flash/memory-efficient/math backend otomatik seçilir
        # use_pytorch_sdpa=False → Mevcut manuel SDPA (fallback veya Flash Attn 2)
        self.use_pytorch_sdpa = use_pytorch_sdpa

        # [V6] Attention Logit Soft-Cap (Gemma 2 standardı)
        # Sadece use_pytorch_sdpa=False (manuel SDPA) path'inde aktif
        self.attn_logit_cap = float(attn_logit_cap)

        # [V6] QK-Norm (Gemma/PaLM 2 standardı)
        # Q ve K head'lerine RoPE öncesi RMSNorm uygulanır; attention logit patlamasını önler
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            if not _RMSNORM_AVAILABLE:
                raise ImportError(
                    "[V6] QK-Norm için RMSNorm gerekli ama import edilemedi. "
                    "rms_norm.py dosyasının ortak_katman_module içinde olduğundan emin olun."
                )
            self.q_norm = _RMSNorm(self.head_dim)
            self.k_norm = _RMSNorm(self.head_dim)
            self.logger.info(
                f"[V6] QK-Norm etkinleştirildi: head_dim={self.head_dim} (Gemma/PaLM 2 standardı)"
            )
        else:
            self.q_norm = None
            self.k_norm = None

        # [OK] V3: Flash Attention 2.0 desteği (endüstri standardı)
        self.use_flash_attention = use_flash_attention and FLASH_ATTENTION_AVAILABLE
        if use_flash_attention and not FLASH_ATTENTION_AVAILABLE:
            self.logger.warning(
                "[V3] Flash Attention 2.0 isteniyor ama yüklü değil. "
                "Standard SDPA kullanılacak. 'pip install flash-attn' ile yükleyebilirsiniz."
            )
        if self.use_flash_attention:
            self.logger.info("[V3] Flash Attention 2.0 etkinleştirildi (endüstri standardı: GPT-4, Claude, Gemini)")

        # [OK] Flash Attention dtype uyarısı için flag (sadece bir kez uyarı göster)
        self._flash_attn_dtype_warning_shown = False

        # [OK] V4: RoPE (Rotary Position Embedding) desteği (endüstri standardı: GPT-3+, Claude, Gemini)
        self.use_rope = use_rope
        self.positional_encoding = positional_encoding
        if use_rope:
            if positional_encoding is None:
                self.logger.warning(
                    "[V4] RoPE isteniyor ama positional_encoding referansı yok. "
                    "RoPE devre dışı bırakıldı."
                )
                self.use_rope = False
            elif not hasattr(positional_encoding, 'apply_rotary_pos_emb'):
                self.logger.warning(
                    "[V4] positional_encoding'de apply_rotary_pos_emb metodu yok. "
                    "RoPE devre dışı bırakıldı."
                )
                self.use_rope = False
            elif positional_encoding.mode != "rope":
                self.logger.warning(
                    f"[V4] positional_encoding modu 'rope' değil ({positional_encoding.mode}). "
                    "RoPE devre dışı bırakıldı."
                )
                self.use_rope = False
            else:
                self.logger.info("[V4] RoPE etkinleştirildi (endüstri standardı: GPT-3+, Claude, Gemini)")

        # [OK] V4: KV Cache desteği (endüstri standardı: GPT-4, Claude, Gemini)
        self.use_kv_cache = use_kv_cache
        self.max_cache_len = max_cache_len
        # [V5] StreamingLLM eviction parametreleri (KVCache yapıcısına iletilir)
        self.kv_eviction_strategy = kv_eviction_strategy
        self.kv_num_sink_tokens = kv_num_sink_tokens
        self.kv_cache: Optional[KVCache] = None
        if self.use_kv_cache:
            self.logger.info(
                f"[V5] KV Cache etkinleştirildi: max_cache_len={max_cache_len}, "
                f"eviction='{kv_eviction_strategy}', num_sink_tokens={kv_num_sink_tokens}"
            )

        # --- Dropout ---
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(self.dropout_rate)

        # --- Projeksiyon katmanları ---
        # [OK] V5: GQA desteği — Query full embed_dim, Key/Value kv_dim (küçük)
        # MHA (standart): Q/K/V = embed_dim = num_heads * head_dim
        # GQA:            Q     = embed_dim  (num_heads * head_dim)
        #                 K/V   = kv_dim     (num_kv_heads * head_dim) — daha küçük
        # Inference'ta: K/V cache %{100 - (num_kv_heads/num_heads)*100} daha küçük
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj   = nn.Linear(embed_dim, self.kv_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, self.kv_dim, bias=False)
        self.out_proj   = nn.Linear(embed_dim, embed_dim, bias=False)

        # [OK] V5: Endüstri standardı: bias=False (LLaMA-2/3, Mistral standardı)
        # Bias'sız projeksiyonlar: daha az parametre, daha hızlı, regularization etkisi

        # [OK] V5: Gelişmiş weight initialization — scaled Xavier (LLaMA/GPT-3 standardı)
        # Q projeksiyon: Xavier uniform (standart)
        nn.init.xavier_uniform_(self.query_proj.weight)
        # K/V projeksiyonları: Xavier uniform
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        # Output projeksiyonu: küçük gain (gradient stability, özellikle deep models)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=math.sqrt(1.0 / num_heads))

        # --- Normalizasyon ---
        self.normalization_type = normalization_type
        if normalization_type == "batch_norm":
            self.norm = nn.BatchNorm1d(embed_dim, eps=1e-5, momentum=0.9, track_running_stats=True)
        elif normalization_type == "layer_norm":
            self.norm = nn.LayerNorm(embed_dim, eps=1e-5)
        elif normalization_type == "instance_norm":
            self.norm = nn.InstanceNorm1d(embed_dim, eps=1e-5, affine=True, track_running_stats=True)
        else:  # group_norm
            # pratik seçim: grup sayısını başlık sayısına eşitle
            self.norm = nn.GroupNorm(num_groups=self.num_heads, num_channels=embed_dim, eps=1e-5)

        if self.debug:
            print(
                f"MultiHeadAttention Başlatıldı: embed_dim={embed_dim}, num_heads={num_heads}, "
                f"num_kv_heads={self.num_kv_heads}, num_kv_groups={self.num_kv_groups}, "
                f"head_dim={self.head_dim}, kv_dim={self.kv_dim}\n"
                f"Normalizasyon: {normalization_type}, Dropout: {self.dropout_rate}, "
                f"sliding_window={self.sliding_window}\n"
                f"[V6] use_pytorch_sdpa={use_pytorch_sdpa}, use_qk_norm={use_qk_norm}, "
                f"attn_logit_cap={attn_logit_cap}"
            )

    # ------------------------ Yardımcılar ------------------------ #
    def _apply_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        LayerNorm: [B, L, E] -> direkt.
        Batch/Instance/GroupNorm: (N,C,L) bekler; (B,L,E) -> (B,E,L) -> norm -> (B,L,E)
        """
        if self.normalization_type == "layer_norm":
            return self.norm(x)

        if x.dim() != 3:
            raise ValueError(f"Normalization için 3D tensör beklenir, alındı: {x.shape}")
        B, L, E = x.shape
        if E != self.embed_dim:
            raise ValueError(f"Normalization: embed_dim uyuşmuyor: {E} != {self.embed_dim}")

        xt = x.transpose(1, 2)  # (B,E,L)
        yt = self.norm(xt)      # (B,E,L)
        y = yt.transpose(1, 2)  # (B,L,E)
        return y

    def _prepare_attention_mask(self, mask: torch.Tensor, batch_size: int, tgt_len: int, src_len: int, device):
        """
        Scaled-dot-prod softmax öncesi skor maskesi hazırlar.
        Desteklenen girişler:
          - bool: True=engelle
          - float: additive maske (engellenen yerlere -inf eklenir)
          - {0,1} float/bool benzeri: 0=engelle -> -inf
        Şekiller:
          - (tgt_len, src_len)
          - (batch_size, tgt_len, src_len)
          - (batch_size, 1/num_heads, tgt_len, src_len) -> yayınlanabilir
        Çıkış: additive float maske (B, H, T, S) ile broadcast edilebilir.
        """
        m = mask.to(device)
        # (B,L,S) veya (L,S) kabul edelim; (B,1,L,S)/(B,H,L,S) zaten broadcast olur
        if m.dim() == 2:
            if m.size(0) != tgt_len or m.size(1) != src_len:
                raise ValueError(f"attention mask boyutu (L,S)=({tgt_len},{src_len}) beklenir, alındı {tuple(m.shape)}")
            m = m.unsqueeze(0).unsqueeze(0)  # (1,1,L,S)
        elif m.dim() == 3:
            if m.size(0) != batch_size or m.size(1) != tgt_len or m.size(2) != src_len:
                raise ValueError(f"attention mask (B,L,S)=({batch_size},{tgt_len},{src_len}) beklenir, alındı {tuple(m.shape)}")
            m = m.unsqueeze(1)  # (B,1,L,S)
        elif m.dim() == 4:
            # (B,1,L,S) ya da (B,H,L,S) ise olduğu gibi bırak
            if not (m.size(0) == batch_size and m.size(2) == tgt_len and m.size(3) == src_len):
                raise ValueError(f"attention mask (B,*,L,S) beklenir, alındı {tuple(m.shape)}")
        else:
            raise ValueError(f"attention mask 2D/3D/4D olmalı, alındı: {tuple(m.shape)}")

        # dtype semantiği
        if m.dtype == torch.bool:
            # True engellenecek yerler: additive -inf'e çevir
            add = torch.zeros_like(m, dtype=torch.float32)
            add[m] = float("-inf")
            return add
        else:
            # Eğer (0,1) aralığında ise 0 engelle -> -inf
            if torch.isfinite(m).all() and m.min() >= 0.0 and m.max() <= 1.0:
                keep = (m > 0.5)
                add = torch.zeros_like(m, dtype=torch.float32)
                add[~keep] = float("-inf")
                return add
            # aksi halde additive maske olarak kabul et
            return m.to(torch.float32)

    def _check_tensor_values(self, name, t: torch.Tensor):
        # [OK] JIT tracing/scripting sırasında validation'ları atla (TracerWarning önleme)
        try:
            is_tracing = torch._C._get_tracing_state() is not None
        except (AttributeError, RuntimeError):
            is_tracing = False

        if torch.jit.is_scripting() or is_tracing:
            return

        # NaN/Inf kontrolleri (tensor boolean'ı Python boolean'a çevir)
        if torch.isnan(t).any().item():
            raise ValueError(f"[ERROR] {name} içinde NaN bulundu.")
        if torch.isinf(t).any().item():
            raise ValueError(f"[ERROR] {name} içinde Inf bulundu.")

        # Debug logging (sadece DEBUG modunda)
        if self.debug and self.logger.isEnabledFor(logging.DEBUG):
            t_cpu = t.detach().cpu()
            stats = {
                "min": float(t_cpu.min().item()),
                "max": float(t_cpu.max().item()),
                "mean": float(t_cpu.mean().item()),
                "std": float(t_cpu.std().item()),
            }
            self.logger.debug(
                f"[MHA:check] {name} -> min={stats['min']:.6f} max={stats['max']:.6f} "
                f"mean={stats['mean']:.6f} std={stats['std']:.6f}"
            )

    # ---------------- Scaled Dot-Product Attention ---------------- #
    def scaled_dot_product_attention(self, query, key, value, mask=None, temperature=1.0, apply_dropout=True, causal_mask=False):
        """
        Ölçeklenmiş nokta çarpımı dikkat mekanizması.
        [V6] Routing: use_pytorch_sdpa → F.sdpa → Flash/memory-efficient/math otomatik
        [OK] V3: Flash Attention 2.0 desteği (endüstri standardı: GPT-4, Claude, Gemini)

        Args:
            query, key, value: (B, H, T, D) - Multi-head format
            mask: additive maske ile broadcast edilebilir (B, H/1, T, S) veya (1,1,T,S)
            temperature: Scaling factor
            apply_dropout: Dropout uygula
            causal_mask: Causal mask uygula (Flash Attention için)
        Returns:
            output, attn_weights
        """
        if self.debug:
            self.logger.debug(
                f"[SDPA] q={tuple(query.shape)} k={tuple(key.shape)} v={tuple(value.shape)} "
                f"pytorch_sdpa={self.use_pytorch_sdpa} flash_attn={self.use_flash_attention}"
            )

        # [V6] Feature A: F.scaled_dot_product_attention (PyTorch 2.0+)
        # Flash/memory-efficient/math backend otomatik seçilir; harici bağımlılık yok
        if self.use_pytorch_sdpa:
            return self._pytorch_sdpa_forward(query, key, value, mask=mask, temperature=temperature, apply_dropout=apply_dropout, causal_mask=causal_mask)

        # [OK] V3: Flash Attention 2.0 (endüstri standardı)
        if self.use_flash_attention:
            return self._flash_attention_forward(query, key, value, mask, temperature, apply_dropout, causal_mask)

        # Standard SDPA (geriye dönük uyumluluk)
        return self._standard_sdpa_forward(query, key, value, mask, temperature, apply_dropout)

    def _pytorch_sdpa_forward(self, query, key, value, mask=None, temperature=1.0, apply_dropout=True, causal_mask=False):
        """
        [V6] F.scaled_dot_product_attention (PyTorch 2.0+)
        Flash Attention / memory-efficient / math backend otomatik seçilir.
        Harici flash_attn bağımlılığı yok.

        Desteklenen PyTorch 2.0+: Flash Attention (CUDA), memory-efficient (xformers), math (CPU).
        """
        dropout_p = self.dropout_rate if (apply_dropout and self.training) else 0.0
        # temperature != 1.0 ise özel scale; aksi halde None → F.sdpa 1/sqrt(D) hesaplar
        scale = (1.0 / math.sqrt(query.size(-1))) * float(temperature) if temperature != 1.0 else None

        # mask varsa: is_causal=False, attn_mask ile birleştir
        # mask yoksa + causal_mask=True: is_causal=True (Flash backend tam optimize)
        if mask is None:
            attn_output = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=causal_mask,
                scale=scale,
            )
        else:
            # Mask ve is_causal birlikte F.sdpa'da desteklenmez; attn_mask kullan
            attn_output = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=mask,
                dropout_p=dropout_p,
                is_causal=False,
                scale=scale,
            )

        # F.sdpa attention weights döndürmez (Flash backend — memory efficient)
        return attn_output, None

    def _flash_attention_forward(self, query, key, value, mask, temperature, apply_dropout, causal_mask):
        """
        [OK] V3: Flash Attention 2.0 forward pass (endüstri standardı: GPT-4, Claude, Gemini)
        Memory-efficient attention: O(n) memory complexity (vs O(n²) standard)
        """
        # [OK] KRİTİK: Flash Attention sadece fp16/bf16 destekler, float32 desteklemez
        # Eğer float32 ise, Flash Attention'ı hiç deneme, direkt standard SDPA'ya geç
        if query.dtype == torch.float32:
            if not self._flash_attn_dtype_warning_shown:
                self.logger.debug(
                    "[V3] Flash Attention atlandı: float32 dtype desteklenmiyor "
                    "(sadece fp16/bf16 desteklenir). Standard SDPA kullanılıyor."
                )
                self._flash_attn_dtype_warning_shown = True
            return self._standard_sdpa_forward(query, key, value, mask, temperature, apply_dropout)

        B, H, Lq, D = query.shape
        _, _, Lk, _ = key.shape

        # Flash Attention format: (batch, seq_len, num_heads, head_dim)
        # Mevcut format: (batch, num_heads, seq_len, head_dim) -> transpose
        q = query.transpose(1, 2).contiguous()  # (B, Lq, H, D)
        k = key.transpose(1, 2).contiguous()    # (B, Lk, H, D)
        v = value.transpose(1, 2).contiguous()  # (B, Lk, H, D)

        # Flash Attention parametreleri
        dropout_p = self.dropout_rate if (apply_dropout and self.training) else 0.0
        softmax_scale = 1.0 / math.sqrt(D) * float(temperature)

        # Causal mask: Flash Attention boolean mask kullanır
        is_causal = causal_mask

        if mask is not None:
            # mask shape: (B, H, Lq, Lk) veya (1, 1, Lq, Lk)
            if mask.dtype == torch.bool:
                if mask.dim() == 4:
                    mask_bool = mask[:, 0, :, :] if mask.size(1) > 1 else mask.squeeze(1)
                else:
                    mask_bool = mask.squeeze() if mask.dim() > 2 else mask
            else:
                if mask.dim() == 4:
                    mask_bool = mask[:, 0, :, :] < -1e9 if mask.size(1) > 1 else mask.squeeze(1) < -1e9
                else:
                    mask_bool = (mask.squeeze() < -1e9) if mask.dim() > 2 else (mask < -1e9)
        else:
            mask_bool = None

        try:
            output = flash_attn_func(
                q, k, v,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=is_causal,
                window_size=(-1, -1),
            )

            # Output format: (B, Lq, H, D) -> (B, H, Lq, D)
            output = output.transpose(1, 2).contiguous()

            # Flash Attention attention weights döndürmüyor (memory efficient)
            attn_weights = None

            if self.debug:
                self.logger.debug(
                    f"[FLASH-ATTN] out={tuple(output.shape)} memory_efficient=O(n)"
                )

            return output, attn_weights

        except Exception as e:
            # Fallback to standard SDPA if Flash Attention fails
            if not self._flash_attn_dtype_warning_shown:
                self.logger.debug(f"[V3] Flash Attention hatası, standard SDPA'ya geçiliyor: {e}")
                self._flash_attn_dtype_warning_shown = True
            return self._standard_sdpa_forward(query, key, value, mask, temperature, apply_dropout)

    def _standard_sdpa_forward(self, query, key, value, mask, temperature, apply_dropout):
        """
        Standard Scaled Dot-Product Attention (geriye dönük uyumluluk)
        [V6] Feature F: Attention Logit Soft-Cap (Gemma 2 standardı)
        """
        # Handle empty sequences (sequence length = 0)
        seq_len = query.size(-2)  # T dimension
        if seq_len == 0:
            # Return zero output with same shape as expected
            batch_size = query.size(0)
            num_heads = query.size(1)
            head_dim = query.size(-1)
            output = torch.zeros((batch_size, num_heads, 0, head_dim),
                               dtype=query.dtype, device=query.device)
            attn_weights = torch.zeros((batch_size, num_heads, 0, 0),
                                     dtype=query.dtype, device=query.device)
            return output, attn_weights

        d_k = query.size(-1)
        scale = math.sqrt(d_k) * float(temperature)

        # (B,H,T,D) x (B,H,D,S) -> (B,H,T,S)
        scores = torch.matmul(query, key.transpose(-2, -1)) / max(scale, 1e-6)

        # [V6] Feature F: Attention Logit Soft-Cap (Gemma 2 standardı)
        # scores = tanh(scores / cap) * cap → [-cap, cap] aralığına sıkıştır
        # Özellikle uzun context'te attention spike'larını önler
        if self.attn_logit_cap > 0.0:
            scores = torch.tanh(scores / self.attn_logit_cap) * self.attn_logit_cap

        # additive mask uygula
        if mask is not None:
            scores = scores + mask  # mask -inf içeriyorsa engeller

        # sayısal kararlılık
        if scores.size(-1) > 0:  # Ensure sequence dimension is not zero
            max_scores, _ = scores.max(dim=-1, keepdim=True)
            scores = scores - max_scores
        else:
            max_scores = torch.zeros_like(scores)

        scores = torch.nan_to_num(scores, nan=0.0, posinf=1e9, neginf=-1e9)

        attn_weights = F.softmax(scores, dim=-1)

        if apply_dropout and self.training:
            attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, value)

        if self.debug:
            self.logger.debug(
                f"[SDPA] out={tuple(output.shape)} wmin={attn_weights.min().item():.6f} "
                f"wmax={attn_weights.max().item():.6f}"
            )

        return output, attn_weights

    # ---------------------------- Forward ---------------------------- #
    def forward(
        self,
        query,
        key=None,
        value=None,
        mask=None,
        causal_mask=False,
        return_attention_weights=False,
        apply_dropout=True,
        # [OK] V4: KV Cache parametreleri (endüstri standardı: GPT-4, Claude, Gemini)
        use_cache: bool = False,  # Bu forward'da cache kullan
        cache_position: Optional[torch.Tensor] = None,  # Cache pozisyonları (incremental generation için)
    ):
        """
        Uygulanmış Çok Başlıklı Dikkat (MultiHeadAttention) işlemi.

        Args:
            query (torch.Tensor): (B, L, E) - Zorunlu
            key (torch.Tensor, optional):   (B, S, E) - None ise query kullanılır (self-attention)
            value (torch.Tensor, optional): (B, S, E) - None ise query kullanılır (self-attention)
            mask (torch.Tensor, optional): (L,S) veya (B,L,S) ya da (B,*,L,S)
            causal_mask (bool): Causal mask uygula (autoregressive için)
            return_attention_weights (bool): True ise attention ağırlıkları da döndürülür.
            apply_dropout (bool): Dropout'u uygula.
            use_cache (bool): KV Cache kullan (inference için). Default: False.
            cache_position (torch.Tensor, optional): Cache pozisyonları [new_len] (incremental generation için).

        Returns:
            output (torch.Tensor): (B, L, E)  — Sadece projeksiyon çıktısı (residual+norm TransformerEncoderLayer'da)
            attn_weights (torch.Tensor, optional): (B, H, L, S) - return_attention_weights=True ise
            kv_cache (tuple, optional): (key_cache, value_cache) - use_cache=True ise
        """
        # [OK] ENDÜSTRİ STANDARDI: Self-attention desteği (GPT-2/3/4, BERT, T5)
        # Eğer key ve value None ise, query'yi kullan (self-attention)
        if key is None:
            key = query
        if value is None:
            value = query

        # --- Girdi doğrulama ---
        if query is None:
            raise ValueError("[ERROR] Query tensörü None olamaz!")
        if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
            raise ValueError(
                f"[ERROR] Giriş tensörleri 3D olmalıdır! "
                f"Query: {query.shape}, Key: {key.shape}, Value: {value.shape}"
            )
        # [OK] JIT tracing/scripting sırasında shape kontrollerini atla (TracerWarning önleme)
        try:
            is_tracing = torch._C._get_tracing_state() is not None
        except (AttributeError, RuntimeError):
            is_tracing = False

        if not (torch.jit.is_scripting() or is_tracing):
            if (int(query.size(-1)) != self.embed_dim or
                int(key.size(-1)) != self.embed_dim or
                int(value.size(-1)) != self.embed_dim):
                raise ValueError(
                    f"[ERROR] Giriş tensörlerinin son boyutu embed_dim ({self.embed_dim}) ile eşleşmelidir! "
                    f"Got: Q {int(query.size(-1))}, K {int(key.size(-1))}, V {int(value.size(-1))}"
                )

        B, Lq, _ = query.size()
        Sk = key.size(1)
        device = query.device

        self.logger.debug(
            f"[MHA FORWARD] Input shapes -> Q:{tuple(query.shape)} K:{tuple(key.shape)} V:{tuple(value.shape)}"
        )

        # --- Projeksiyonlar: (B,L,E) -> (B,H,L,D) ---
        # [OK] V5: GQA desteği — Q full heads, K/V kv_heads (sonra expand)

        # Query: (B, Lq, E) -> (B, num_heads, Lq, head_dim)
        query_proj = self.query_proj(query)
        query_proj = query_proj.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        if self.debug:
            self._check_tensor_values("QueryProj", query_proj)

        # Key: (B, Sk, E) -> (B, num_kv_heads, Sk, head_dim)
        key_proj = self.key_proj(key)
        key_proj = key_proj.view(B, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.debug:
            self._check_tensor_values("KeyProj", key_proj)

        # Value: (B, Sk, E) -> (B, num_kv_heads, Sk, head_dim)
        value_proj = self.value_proj(value)
        value_proj = value_proj.view(B, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.debug:
            self._check_tensor_values("ValueProj", value_proj)

        # [V6] Feature B: QK-Norm (Gemma/PaLM 2 standardı)
        # Q ve K head'lerine RoPE öncesi RMSNorm uygulanır
        # Uzun context'te attention logit patlamasını önler
        if self.use_qk_norm and self.q_norm is not None and self.k_norm is not None:
            query_proj = self.q_norm(query_proj)  # [B, H, T, head_dim]
            key_proj = self.k_norm(key_proj)      # [B, num_kv_heads, S, head_dim]

        # [OK] V5: GQA — KV heads expand (KV cache güncellemesinden ÖNCE expand)
        # KV Cache ile kullanıldığında: önce cache'e ekle (num_kv_heads boyutunda),
        # sonra expand (num_heads boyutuna). Bu şekilde cache küçük kalır.
        # use_cache=True ise: expand cache güncellemesinden sonra yapılacak.
        # use_cache=False ise: şimdi expand et.
        if self.num_kv_groups > 1 and not (use_cache and self.use_kv_cache and not self.training):
            # (B, num_kv_heads, Sk, head_dim) -> (B, num_heads, Sk, head_dim)
            key_proj   = key_proj.repeat_interleave(self.num_kv_groups, dim=1)
            value_proj = value_proj.repeat_interleave(self.num_kv_groups, dim=1)

        # [OK] V4: RoPE (Rotary Position Embedding) uygulama (endüstri standardı: GPT-3+, Claude, Gemini)
        # RoPE query ve key'e uygulanır (value'ya değil); QK-Norm'dan SONRA uygulanır
        if self.use_rope and self.positional_encoding is not None:
            try:
                # Query ve key'e RoPE uygula: [B, H, T, D] formatında
                query_proj = self.positional_encoding.apply_rotary_pos_emb(query_proj)
                key_proj = self.positional_encoding.apply_rotary_pos_emb(key_proj)
                if self.debug:
                    self.logger.debug("[V4] RoPE uygulandı: query_proj ve key_proj")
            except Exception as e:
                self.logger.warning(f"[V4] RoPE uygulanırken hata oluştu: {e}. RoPE atlandı.")

        # [OK] V4/V5: KV Cache yönetimi — GQA uyumlu (num_kv_heads boyutunda cache)
        kv_cache_output = None
        if use_cache and self.use_kv_cache and not self.training:
            # KV Cache'i başlat veya güncelle
            if self.kv_cache is None:
                # [OK] V5 GQA: Cache num_kv_heads boyutunda tutulur (hafıza tasarrufu)
                self.kv_cache = KVCache(
                    batch_size=B,
                    num_heads=self.num_kv_heads,       # [V5] GQA: kv_heads boyutunda cache
                    head_dim=self.head_dim,
                    max_cache_len=self.max_cache_len,
                    device=device,
                    dtype=query.dtype,
                    log_level=self.logger.level,
                    # [V5] StreamingLLM parametreleri
                    eviction_strategy=self.kv_eviction_strategy,
                    num_sink_tokens=self.kv_num_sink_tokens,
                )
                self.logger.debug(
                    f"[V5] KV Cache başlatıldı (GQA uyumlu): "
                    f"num_kv_heads={self.num_kv_heads}, head_dim={self.head_dim}"
                )

            # [V5 GQA] Cache güncelleme: key/value num_kv_heads boyutunda (expand öncesi)
            # Önce cache'e ekle (küçük boyut), sonra expand
            key_proj_cached, value_proj_cached = self.kv_cache.update(
                key_proj, value_proj, cache_position
            )

            # [V5 GQA] Cache sonrası expand: num_kv_heads -> num_heads
            if self.num_kv_groups > 1:
                key_proj   = key_proj_cached.repeat_interleave(self.num_kv_groups, dim=1)
                value_proj = value_proj_cached.repeat_interleave(self.num_kv_groups, dim=1)
            else:
                key_proj   = key_proj_cached
                value_proj = value_proj_cached

            # KV Cache output (num_kv_heads boyutunda — kompakt)
            kv_cache_output = (
                self.kv_cache.key_cache[:, :, :self.kv_cache.cache_len],
                self.kv_cache.value_cache[:, :, :self.kv_cache.cache_len]
            )
            self.logger.debug(
                f"[V5] KV Cache güncellendi. Cache length: {self.kv_cache.cache_len}, "
                f"key shape: {key_proj.shape}"
            )

        # --- Maske hazırlama ---
        add_mask = None

        # [OK] V5: Sliding Window Attention maskesi (Mistral-7B standardı)
        # Her token yalnızca [pos - window + 1, pos] aralığına attend eder
        if self.sliding_window is not None:
            rows = torch.arange(Lq, device=device, dtype=torch.long).unsqueeze(1)   # (Lq, 1)
            cols = torch.arange(Sk, device=device, dtype=torch.long).unsqueeze(0)   # (1, Sk)
            outside_window = (rows - cols) >= self.sliding_window  # (Lq, Sk) bool
            sw_float = torch.where(outside_window, float("-inf"), 0.0)  # (Lq, Sk)
            sw_float = sw_float.unsqueeze(0).unsqueeze(0)  # (1, 1, Lq, Sk) — broadcast
            add_mask = sw_float

        # [OK] YENİ: Causal mask ekle (GPT-2/3/4 standardı)
        # [V6 OOM FIX] F.sdpa path + sadece causal (padding/sliding_window yok):
        #   is_causal=True kullan → dense [B,H,T,T] tensor oluşturma (Flash backend O(1) bellek)
        #   Eski kod: [B, H, Lq, Sk] × float32 → batch_size=32, seq=512 için ~256 MB gereksiz tahsis
        _skip_dense_causal = (
            self.use_pytorch_sdpa
            and causal_mask
            and mask is None            # padding mask yok
            and self.sliding_window is None  # sliding window yok → add_mask=None garantili
        )
        if causal_mask and not _skip_dense_causal:
            # Dense causal mask — F.sdpa olmayan path veya sliding window + causal birleştirme
            # [V6 OOM FIX] (1, 1, Lq, Sk) broadcast kullan — [B, H, Lq, Sk] materializasyonundan kaçın
            #   Bellek: [1, 1, 512, 512] × 4 bytes = 1 MB  vs.  [32, 8, 512, 512] = 256 MB (256x daha az)
            causal_mask_tensor = torch.triu(
                torch.ones(Lq, Sk, device=device, dtype=torch.bool),
                diagonal=1
            )  # (Lq, Sk) — True = engelle, False = izin ver

            # [OK] ENDÜSTRİ STANDARDI: Boolean mask'i float mask'e çevir (-inf = engelle, 0.0 = izin ver)
            # (1, 1, Lq, Sk) → F.sdpa ve manual SDPA her ikisi broadcast eder
            causal_mask_float = torch.where(
                causal_mask_tensor.unsqueeze(0).unsqueeze(0),  # (1, 1, Lq, Sk) broadcast
                torch.tensor(float('-inf'), device=device, dtype=torch.float32),
                torch.tensor(0.0, device=device, dtype=torch.float32),
            )

            if add_mask is None:
                add_mask = causal_mask_float
            else:
                if add_mask.dtype == torch.bool:
                    add_mask = torch.where(add_mask, float('-inf'), 0.0)
                add_mask = add_mask + causal_mask_float

        # Mevcut mask (padding mask vb.) ekle
        if mask is not None:
            mask_prepared = self._prepare_attention_mask(mask, B, Lq, Sk, device)
            if mask_prepared.dim() == 4 and mask_prepared.size(1) == 1:
                mask_prepared = mask_prepared.expand(B, self.num_heads, Lq, Sk)

            if mask_prepared.dtype == torch.bool:
                mask_prepared = torch.where(mask_prepared, float('-inf'), 0.0)

            if add_mask is None:
                add_mask = mask_prepared
            else:
                if add_mask.dtype == torch.bool:
                    add_mask = torch.where(add_mask, float('-inf'), 0.0)
                add_mask = add_mask + mask_prepared

        # --- SDPA (PyTorch F.sdpa / Flash Attention 2.0 / Standard) ---
        # [V6] Routing: use_pytorch_sdpa → _pytorch_sdpa_forward
        # [V3] Fallback: use_flash_attention → _flash_attention_forward
        # Fallback: _standard_sdpa_forward (causal_mask için Flash parametresi)
        use_causal_for_sdpa = causal_mask if (self.use_flash_attention and not self.use_pytorch_sdpa) else causal_mask
        attn_output, attn_weights = self.scaled_dot_product_attention(
            query_proj, key_proj, value_proj,
            mask=add_mask if not (self.use_flash_attention and not self.use_pytorch_sdpa) else None,
            temperature=1.0,
            apply_dropout=apply_dropout,
            causal_mask=use_causal_for_sdpa
        )

        # --- Birleştir, çıkış projeksiyon ---
        # [V6] Fix 1: MHA sadece ham projeksiyon çıktısını döndürür
        # Residual ve norm TransformerEncoderLayer tarafından uygulanır (pre-norm pattern)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Lq, self.embed_dim)
        final_projection = self.out_proj(attn_output)

        if self.debug:
            self._check_tensor_values("FinalProjection", final_projection)
            self.logger.debug(f"[MHA FORWARD] Output shape: {tuple(final_projection.shape)}")

        # [V6] Fix 1: Residual+Norm kaldırıldı — TransformerEncoderLayer halleder
        # ESKİ: output = self._apply_norm(final_projection + query)
        output = final_projection  # Ham projeksiyon — residual yok, norm yok

        # [OK] V4: KV Cache kullanılıyorsa 3 değer döndür
        if use_cache and self.use_kv_cache and kv_cache_output is not None:
            if return_attention_weights:
                return output, attn_weights, kv_cache_output
            else:
                return output, None, kv_cache_output
        elif return_attention_weights:
            return output, attn_weights
        else:
            return output
