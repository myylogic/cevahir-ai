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
import time
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
        use_kv_cache: bool = False,  # KV Cache kullan (inference için)
        max_cache_len: int = 2048,  # Maximum cache length
    ):
        """
        MultiHeadAttention sınıfını başlatır.

        Args:
            embed_dim (int): Giriş tensörlerinin gömme boyutu.
            num_heads (int): Dikkat başlığı sayısı.
            dropout (float): Tüm katmanlarda ortak kullanılacak Dropout oranı.
            normalization_type (str): Normalizasyon türü ('layer_norm', 'batch_norm', 'instance_norm', 'group_norm').
            debug (bool): Hata ayıklama modu.
            log_level (int, optional): Logger seviyesi (geriye dönük uyumluluk için).
            use_flash_attention (bool): Flash Attention 2.0 kullan (varsa). Default: False (geriye dönük uyumluluk).
            use_rope (bool): RoPE (Rotary Position Embedding) kullan. Default: False (geriye dönük uyumluluk).
            positional_encoding: PositionalEncoding modülü referansı (RoPE için gerekli).
            use_kv_cache (bool): KV Cache kullan (inference için). Default: False (geriye dönük uyumluluk).
            max_cache_len (int): Maximum cache length. Default: 2048.
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
        self.use_kv_cache = use_kv_cache  # KV Cache kullanım bayrağı (training/inference kontrolü forward'da yapılır)
        self.max_cache_len = max_cache_len
        self.kv_cache: Optional[KVCache] = None
        if self.use_kv_cache:
            self.logger.info(
                f"[V4] KV Cache etkinleştirildi (endüstri standardı: GPT-4, Claude, Gemini), "
                f"max_cache_len={max_cache_len}"
            )

        # --- Dropout ---
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(self.dropout_rate)

        # --- Projeksiyon katmanları ---
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # [OK] YENİ: Endüstri standardı Xavier initialization (GPT-2/3/4, BERT, T5)
        # Q/K/V projeksiyonları için standart Xavier
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        # Output projeksiyonu için özel gain (num_heads'e göre normalize)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=math.sqrt(1.0 / num_heads))
        
        # Bias'lar için zero initialization
        if self.query_proj.bias is not None:
            nn.init.zeros_(self.query_proj.bias)
        if self.key_proj.bias is not None:
            nn.init.zeros_(self.key_proj.bias)
        if self.value_proj.bias is not None:
            nn.init.zeros_(self.value_proj.bias)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

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
                f"MultiHeadAttention Başlatıldı: embed_dim={embed_dim}, num_heads={num_heads}, head_dim={self.head_dim}\n"
                f"Normalizasyon: {normalization_type}, Dropout: {self.dropout_rate}"
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
        start_time = time.time()
        if self.debug:
            self.logger.debug(
                f"[SDPA] q={tuple(query.shape)} k={tuple(key.shape)} v={tuple(value.shape)} "
                f"flash_attn={self.use_flash_attention}"
            )

        # [OK] V3: Flash Attention 2.0 (endüstri standardı)
        if self.use_flash_attention:
            return self._flash_attention_forward(query, key, value, mask, temperature, apply_dropout, causal_mask)
        else:
            # Standard SDPA (geriye dönük uyumluluk)
            return self._standard_sdpa_forward(query, key, value, mask, temperature, apply_dropout)
    
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
        
        start_time = time.time()  # [OK] Debug için
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
        
        # Padding mask: Flash Attention için boolean mask'e çevir
        # mask None ise veya zaten boolean ise direkt kullan
        # mask float ise (additive), boolean'a çevir (-inf -> True, 0.0 -> False)
        if mask is not None:
            # mask shape: (B, H, Lq, Lk) veya (1, 1, Lq, Lk)
            if mask.dtype == torch.bool:
                # Boolean mask: Flash Attention için uygun
                # Ancak Flash Attention (B, Lq, Lk) formatında boolean mask bekler
                # Multi-head mask'i head dimension'ından kaldır (her head için aynı mask)
                if mask.dim() == 4:
                    # (B, H, Lq, Lk) -> (B, Lq, Lk) (ilk head'i al)
                    mask_bool = mask[:, 0, :, :] if mask.size(1) > 1 else mask.squeeze(1)
                else:
                    mask_bool = mask.squeeze() if mask.dim() > 2 else mask
            else:
                # Float mask (additive): -inf -> True, 0.0 -> False
                if mask.dim() == 4:
                    mask_bool = mask[:, 0, :, :] < -1e9 if mask.size(1) > 1 else mask.squeeze(1) < -1e9
                else:
                    mask_bool = (mask.squeeze() < -1e9) if mask.dim() > 2 else (mask < -1e9)
        else:
            mask_bool = None
        
        try:
            # Flash Attention 2.0 call
            # flash_attn_func(q, k, v, dropout_p, softmax_scale, causal=is_causal, window_size=(-1, -1))
            output = flash_attn_func(
                q, k, v,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=is_causal,
                window_size=(-1, -1),  # No local attention window
            )
            
            # Output format: (B, Lq, H, D) -> (B, H, Lq, D)
            output = output.transpose(1, 2).contiguous()
            
            # Flash Attention attention weights döndürmüyor (memory efficient)
            # Dummy attention weights (test için gerekli olabilir)
            attn_weights = None  # Flash Attention memory efficient olduğu için weights saklamıyor
            
            if self.debug:
                elapsed = time.time() - start_time
                self.logger.debug(
                    f"[FLASH-ATTN] out={tuple(output.shape)} time={elapsed:.6f}s "
                    f"memory_efficient=O(n)"
                )
            
            return output, attn_weights
            
        except Exception as e:
            # Fallback to standard SDPA if Flash Attention fails
            # Uyarıyı sadece bir kez göster
            if not self._flash_attn_dtype_warning_shown:
                self.logger.debug(f"[V3] Flash Attention hatası, standard SDPA'ya geçiliyor: {e}")
                self._flash_attn_dtype_warning_shown = True
            return self._standard_sdpa_forward(query, key, value, mask, temperature, apply_dropout)
    
    def _standard_sdpa_forward(self, query, key, value, mask, temperature, apply_dropout):
        """
        Standard Scaled Dot-Product Attention (geriye dönük uyumluluk)
        """
        start_time = time.time()
        
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
            if self.debug:
                elapsed = time.time() - start_time
                self.logger.debug(f"[SDPA] Empty sequence detected, returning zero output (elapsed={elapsed:.4f}s)")
            return output, attn_weights
        
        d_k = query.size(-1)
        scale = math.sqrt(d_k) * float(temperature)

        # (B,H,T,D) x (B,H,D,S) -> (B,H,T,S)
        scores = torch.matmul(query, key.transpose(-2, -1)) / max(scale, 1e-6)

        # additive mask uygula
        if mask is not None:
            scores = scores + mask  # mask -inf içeriyorsa engeller

        # sayısal kararlılık
        # Check if scores has valid dimensions before max operation
        if scores.size(-1) > 0:  # Ensure sequence dimension is not zero
            max_scores, _ = scores.max(dim=-1, keepdim=True)
            scores = scores - max_scores
        else:
            # If sequence dimension is zero, create zero scores
            max_scores = torch.zeros_like(scores)
        
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1e9, neginf=-1e9)

        attn_weights = F.softmax(scores, dim=-1)

        if apply_dropout and self.training:
            attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, value)

        if self.debug:
            elapsed = time.time() - start_time
            self.logger.debug(
                f"[SDPA] out={tuple(output.shape)} wmin={attn_weights.min().item():.6f} "
                f"wmax={attn_weights.max().item():.6f} time={elapsed:.6f}s"
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
            output (torch.Tensor): (B, L, E)
            attn_weights (torch.Tensor, optional): (B, H, L, S) - return_attention_weights=True ise
            kv_cache (tuple, optional): (key_cache, value_cache) - use_cache=True ise
        """
        t_start = time.time()

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
        def apply_projection(tensor, projection_layer, name):
            proj = projection_layer(tensor)
            proj = proj.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,L/D,D)
            self._check_tensor_values(name, proj)
            return proj

        t_proj_start = time.time()
        query_proj = apply_projection(query, self.query_proj, "QueryProj")
        key_proj = apply_projection(key, self.key_proj, "KeyProj")
        value_proj = apply_projection(value, self.value_proj, "ValueProj")
        t_proj_end = time.time()
        self.logger.debug(f"[MHA FORWARD] Projection time: {t_proj_end - t_proj_start:.6f}s")

        # --- Q/K/V üzerine dropout (isteğe bağlı) ---
        if apply_dropout and self.training:
            query_proj = self.dropout(query_proj)
            key_proj = self.dropout(key_proj)
            value_proj = self.dropout(value_proj)

        # [OK] V4: RoPE (Rotary Position Embedding) uygulama (endüstri standardı: GPT-3+, Claude, Gemini)
        # RoPE query ve key'e uygulanır (value'ya değil)
        if self.use_rope and self.positional_encoding is not None:
            try:
                # Query ve key'e RoPE uygula: [B, H, T, D] formatında
                query_proj = self.positional_encoding.apply_rotary_pos_emb(query_proj)
                key_proj = self.positional_encoding.apply_rotary_pos_emb(key_proj)
                if self.debug:
                    self.logger.debug("[V4] RoPE uygulandı: query_proj ve key_proj")
            except Exception as e:
                self.logger.warning(f"[V4] RoPE uygulanırken hata oluştu: {e}. RoPE atlandı.")

        # [OK] V4: KV Cache yönetimi (endüstri standardı: GPT-4, Claude, Gemini)
        kv_cache_output = None
        if use_cache and self.use_kv_cache and not self.training:
            # KV Cache'i başlat veya güncelle
            if self.kv_cache is None:
                # İlk çağrıda cache'i başlat
                self.kv_cache = KVCache(
                    batch_size=B,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    max_cache_len=self.max_cache_len,
                    device=device,
                    dtype=query.dtype,
                    log_level=self.logger.level,
                )
                self.logger.debug("[V4] KV Cache başlatıldı.")
            
            # Key ve value'ları cache'e ekle (RoPE'den sonra, attention'dan önce)
            key_proj_cached, value_proj_cached = self.kv_cache.update(
                key_proj, value_proj, cache_position
            )
            # Cache'lenmiş key/value'ları kullan
            key_proj = key_proj_cached
            value_proj = value_proj_cached
            # KV Cache output'u hazırla (son layer için)
            kv_cache_output = (self.kv_cache.key_cache[:, :, :self.kv_cache.cache_len], 
                              self.kv_cache.value_cache[:, :, :self.kv_cache.cache_len])
            self.logger.debug(f"[V4] KV Cache güncellendi. Cache length: {self.kv_cache.cache_len}")
        
        # --- Maske hazırlama ---
        add_mask = None
        
        # [OK] YENİ: Causal mask ekle (GPT-2/3/4 standardı)
        if causal_mask:
            # Causal mask: Üst üçgen matris (future tokens'ları engelle)
            causal_mask_tensor = torch.triu(
                torch.ones(Lq, Sk, device=device, dtype=torch.bool),
                diagonal=1
            )  # (Lq, Sk) - True = engelle, False = izin ver
            
            # (B, H, Lq, Sk) formatına genişlet
            causal_mask_expanded = causal_mask_tensor.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, Lq, Sk)
            
            # [OK] ENDÜSTRİ STANDARDI: Boolean mask'i float mask'e çevir (-inf = engelle, 0.0 = izin ver)
            # scaled_dot_product_attention boolean mask'i doğru işlemez, float mask gerekiyor
            causal_mask_float = torch.where(
                causal_mask_expanded,
                float('-inf'),
                0.0
            )
            
            if add_mask is None:
                # Sadece causal mask var
                add_mask = causal_mask_float
            else:
                # Mevcut mask ile birleştir (ikisi de float olmalı)
                if add_mask.dtype == torch.bool:
                    # add_mask boolean ise float'a çevir
                    add_mask = torch.where(add_mask, float('-inf'), 0.0)
                add_mask = add_mask + causal_mask_float
        
        # Mevcut mask (padding mask vb.) ekle
        if mask is not None:
            mask_prepared = self._prepare_attention_mask(mask, B, Lq, Sk, device)
            # (B,1/H,L,S) ile (B,H,L,S) broadcast için H eksenini genişlet
            if mask_prepared.dim() == 4 and mask_prepared.size(1) == 1:
                mask_prepared = mask_prepared.expand(B, self.num_heads, Lq, Sk)
            
            # [OK] ENDÜSTRİ STANDARDI: Boolean mask'i float mask'e çevir
            if mask_prepared.dtype == torch.bool:
                mask_prepared = torch.where(mask_prepared, float('-inf'), 0.0)
            
            if add_mask is None:
                add_mask = mask_prepared
            else:
                # Mevcut mask ile birleştir (ikisi de float olmalı)
                if add_mask.dtype == torch.bool:
                    # add_mask boolean ise float'a çevir
                    add_mask = torch.where(add_mask, float('-inf'), 0.0)
                add_mask = add_mask + mask_prepared

        # --- SDPA (Standard veya Flash Attention 2.0) ---
        t_attn_start = time.time()
        # [OK] V3: Flash Attention için causal_mask'i direkt geçir (mask hazırlama Flash Attention içinde yapılacak)
        # Standard SDPA için add_mask kullan (zaten hazırlanmış)
        use_causal_for_flash = causal_mask if self.use_flash_attention else False
        attn_output, attn_weights = self.scaled_dot_product_attention(
            query_proj, key_proj, value_proj, 
            mask=add_mask if not self.use_flash_attention else None,  # Flash Attention mask'i kendi içinde işler
            temperature=1.0, 
            apply_dropout=apply_dropout,
            causal_mask=use_causal_for_flash  # [OK] V3: Flash Attention için causal mask
        )
        t_attn_end = time.time()
        attn_type = "Flash-Attn" if self.use_flash_attention else "Standard-SDPA"
        self.logger.debug(f"[MHA FORWARD] {attn_type} time: {t_attn_end - t_attn_start:.6f}s")

        # --- Birleştir, çıkış projeksiyon ve residual+norm ---
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Lq, self.embed_dim)
        final_projection = self.out_proj(attn_output)  # ek ölçek uygulamıyoruz; SDPA zaten ölçekli
        self._check_tensor_values("FinalProjection", final_projection)

        # Residual + Normalization
        output = self._apply_norm(final_projection + query)

        if self.debug:
            self._check_tensor_values("Output", output)
            self.logger.debug(f"[MHA FORWARD] Output shape: {tuple(output.shape)}")

        t_end = time.time()
        self.logger.info(f"[MHA FORWARD] Forward pass completed in {t_end - t_start:.6f} seconds.")

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
