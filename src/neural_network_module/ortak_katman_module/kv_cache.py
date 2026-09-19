# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: kv_cache.py
Modül: src/neural_network_module/ortak_katman_module
Görev: KV Cache (Key-Value Cache) - Autoregressive generation sırasında key ve
       value'ları cache'ler. Her yeni token için sadece yeni token'ın key/value'sunu
       hesaplar, önceki token'ların key/value'larını tekrar hesaplamadan cache'den
       kullanır.

       [V5] StreamingLLM / Attention Sink desteği eklendi:
       Cache dolduğunda RuntimeError yerine eviction stratejisi uygulanır.
       İlk num_sink_tokens token (attention sink) daima korunur; geri kalan
       bölümde kayan pencere (sliding window) ile en eski tokenlar çıkarılır.
       Bu sayede max_cache_len sınırı aşılmadan sonsuz uzunlukta generation
       yapılabilir.

       Referans: Xiao et al. 2023 — "Efficient Streaming Language Models with
       Attention Sinks" (https://arxiv.org/abs/2309.17453)

MİMARİ:
- SOLID Prensipleri: Single Responsibility (KV cache işlemleri),
                     Open/Closed (genişletilebilir),
                     Dependency Inversion (nn.Module abstraction'ına bağımlı)
- Design Patterns: Cache Pattern (key-value cache)
- Endüstri Standartları: Transformer KV cache standardı

KULLANIM:
- KV cache oluşturmak için
- Autoregressive generation için
- Inference optimizasyonu için
- StreamingLLM ile sonsuz uzunlukta generation için

BAĞIMLILIKLAR:
- torch.nn: Module base class

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
from typing import Optional, Tuple


class KVCache:
    """
    [V5] KV Cache — StreamingLLM / Attention Sink desteğiyle

    Autoregressive generation için key ve value'ları cache'ler.
    Her layer için ayrı cache tutulur.

    Cache yapısı:
        key_cache:   [B, num_heads, cache_len, head_dim]
        value_cache: [B, num_heads, cache_len, head_dim]

    Eviction stratejisi (eviction_strategy="sliding_window"):
        [sink_0 ... sink_{S-1}] [sliding window: son W token]
        - İlk S token (attention sink) daima korunur.
        - Cache dolduğunda en eski non-sink tokenlar çıkarılır.
        - Generation sınırı kalkar; max_cache_len sabit bellek kullanımı sağlar.

    Kullanım:
        1. İlk forward: Tüm sequence için key/value hesapla ve cache'le.
        2. Sonraki forward'lar: Sadece yeni token için key/value hesapla, cache'e ekle.
        3. Attention: Cached key/value + new key/value birleştirilmiş olarak döner.
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        max_cache_len: int = 2048,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        log_level: int = logging.INFO,
        # [V5] StreamingLLM parametreleri
        eviction_strategy: str = "sliding_window",  # "none" | "sliding_window"
        num_sink_tokens: int = 4,                   # Attention sink token sayısı
    ):
        """
        Args:
            batch_size:         Batch size (ilk update() çağrısında dinamik olarak güncellenir).
            num_heads:          Attention head sayısı (GQA kullanımında num_kv_heads geçilir).
            head_dim:           Her head'in boyutu.
            max_cache_len:      Maksimum cache uzunluğu.
            device:             Cache tensorleri için device.
            dtype:              Cache tensorleri için veri tipi.
            log_level:          Logger seviyesi.
            eviction_strategy:  "none"           → Cache dolduğunda RuntimeError.
                                "sliding_window" → StreamingLLM: sink + kayan pencere.
            num_sink_tokens:    Sliding window modunda daima korunan ilk token sayısı.
                                Xiao et al. 2023 — önerilen değer: 4.
        """
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_cache_len = max_cache_len
        self.device = device or torch.device("cpu")
        self.dtype = dtype or torch.float32
        if max_cache_len <= 0:
            raise ValueError("max_cache_len must be positive")
        self.positions = torch.empty(0, dtype=torch.long, device=self.device)
        self.attention_positions = self.positions

        # Eviction
        _valid_strategies = {"none", "sliding_window"}
        if eviction_strategy not in _valid_strategies:
            raise ValueError(
                f"Geçersiz eviction_strategy: '{eviction_strategy}'. "
                f"Geçerli seçenekler: {_valid_strategies}"
            )
        self.eviction_strategy = eviction_strategy
        self.num_sink_tokens = max(0, int(num_sink_tokens))

        if self.eviction_strategy == "sliding_window":
            # Kesin hata: sıfır non-sink alan kalıyor
            if self.num_sink_tokens >= max_cache_len:
                raise ValueError(
                    f"num_sink_tokens ({self.num_sink_tokens}) >= max_cache_len ({max_cache_len}). "
                    f"Sliding window için en az 1 non-sink slot gerekli."
                )
            # [V8 Fix] Zayıf boundary: 1 non-sink slot sliding window için pratik değil.
            # num_sink_tokens > max_cache_len * 0.75 → thrashing riski (her eviction'da
            # sadece 1-2 yeni token tutulabilir, fayda/maliyet oranı bozulur).
            _min_sliding = max(1, max_cache_len // 4)  # en az %25 non-sink alan
            if self.num_sink_tokens > max_cache_len - _min_sliding:
                raise ValueError(
                    f"num_sink_tokens ({self.num_sink_tokens}) çok büyük: "
                    f"max_cache_len={max_cache_len} için non-sink alan en az "
                    f"{_min_sliding} slot olmalı (önerilen: max_cache_len'in en fazla %%75'i). "
                    f"Maksimum güvenli num_sink_tokens: {max_cache_len - _min_sliding}."
                )

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

        # Cache storage: [B, H, cache_len, D]
        self.key_cache: Optional[torch.Tensor] = None
        self.value_cache: Optional[torch.Tensor] = None
        self.cache_len: int = 0
        self._initialized: bool = False

        # [V5] Toplam görülen token sayısı (prefill + decode)
        self._seen_tokens: int = 0

        self.logger.info(
            f"[V5] KVCache initialized: batch_size={batch_size}, num_heads={num_heads}, "
            f"head_dim={head_dim}, max_cache_len={max_cache_len}, "
            f"eviction='{eviction_strategy}', num_sink_tokens={self.num_sink_tokens}"
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cache'i güncelle ve birleştirilmiş key/value döndür.

        Args:
            key:            [B, H, new_len, D] — Yeni key tensörü.
            value:          [B, H, new_len, D] — Yeni value tensörü.
            cache_position: [new_len] — Cache'deki hedef pozisyonlar (None → append).

        Returns:
            key_out:   [B, H, cache_len, D] — Cached + yeni key.
            value_out: [B, H, cache_len, D] — Cached + yeni value.
        """
        if key.ndim != 4 or key.shape != value.shape:
            raise ValueError("key and value must have identical [B,H,T,D] shapes")
        B, H, new_len, D = key.shape
        if H != self.num_heads or D != self.head_dim:
            raise ValueError("KV head shape differs from cache configuration")
        if key.device != value.device or key.dtype != value.dtype:
            raise ValueError("key/value dtype and device must match")
        if (B != self.batch_size or key.device != self.device or key.dtype != self.dtype):
            self.clear()
            self.key_cache = self.value_cache = None
            self.batch_size, self.device, self.dtype = B, key.device, key.dtype
            self.positions = torch.empty(0, device=key.device, dtype=torch.long)
        if cache_position is None:
            positions = torch.arange(self._seen_tokens, self._seen_tokens + new_len,
                                     device=key.device, dtype=torch.long)
        else:
            if cache_position.ndim != 1 or cache_position.numel() != new_len:
                raise ValueError("cache_position must contain one absolute position per input token")
            if cache_position.dtype not in (torch.int32, torch.int64):
                raise ValueError("cache_position must be integer")
            positions = cache_position.to(device=key.device, dtype=torch.long)
            if positions.numel() and (positions.min() < 0 or (positions[1:] <= positions[:-1]).any()):
                raise ValueError("cache_position must be nonnegative and strictly increasing")

        # The usual decoding path only appends positions. Avoid sorting and
        # copying all retained K/V tensors while there is spare capacity.
        appending = not self.positions.numel() or not positions.numel() or int(positions[0]) > int(self.positions[-1])
        if appending and self.cache_len + new_len <= self.max_cache_len and not torch.is_grad_enabled():
            if self.key_cache is None:
                shape = (B, H, self.max_cache_len, D)
                self.key_cache = torch.zeros(shape, device=key.device, dtype=key.dtype)
                self.value_cache = torch.zeros_like(self.key_cache)
            end = self.cache_len + new_len
            self.key_cache[:, :, self.cache_len:end].copy_(key)
            self.value_cache[:, :, self.cache_len:end].copy_(value)
            self.cache_len = end
            self.positions = torch.cat((self.positions, positions))
            self.attention_positions = self.positions
            if positions.numel():
                self._seen_tokens = max(self._seen_tokens, int(positions[-1]) + 1)
            self._initialized = True
            return self.get()

        old_k, old_v = self.get()
        old_positions = self.positions
        if old_k is not None and len(old_positions):
            # Explicit writes may replace retained positions, but never create
            # zero-filled holes that would be mistaken for real context.
            keep_old = ~torch.isin(old_positions, positions)
            all_positions = torch.cat((old_positions[keep_old], positions))
            all_key = torch.cat((old_k[:, :, keep_old], key), dim=2)
            all_value = torch.cat((old_v[:, :, keep_old], value), dim=2)
            order = torch.argsort(all_positions)
            all_positions = all_positions[order]
            all_key, all_value = all_key[:, :, order], all_value[:, :, order]
        else:
            all_positions, all_key, all_value = positions, key, value

        total = all_positions.numel()
        if total > self.max_cache_len and self.eviction_strategy == "none":
            raise RuntimeError("KVCache capacity exceeded with eviction disabled")
        if total > self.max_cache_len:
            sinks = min(self.num_sink_tokens, total)
            selected = torch.cat((torch.arange(sinks, device=key.device),
                                  torch.arange(total - (self.max_cache_len - sinks), total,
                                               device=key.device)))
        else:
            selected = torch.arange(total, device=key.device)
        if self.key_cache is None:
            shape = (B, H, self.max_cache_len, D)
            self.key_cache = torch.zeros(shape, device=key.device, dtype=key.dtype)
            self.value_cache = torch.zeros_like(self.key_cache)
        self.cache_len = selected.numel()
        self.key_cache[:, :, :self.cache_len].copy_(all_key[:, :, selected].detach())
        self.value_cache[:, :, :self.cache_len].copy_(all_value[:, :, selected].detach())
        self.positions = all_positions[selected].clone()
        self.attention_positions = all_positions
        if positions.numel():
            self._seen_tokens = max(self._seen_tokens, int(positions[-1]) + 1)
        self._initialized = True
        # Attend to the complete incoming chunk before retaining bounded state.
        # This preserves chunked prefill causality even if the chunk is oversized.
        return all_key, all_value

    def get(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Mevcut cache içeriğini döndür."""
        if self.key_cache is None:
            return None, None
        return (
            self.key_cache[:, :, :self.cache_len],
            self.value_cache[:, :, :self.cache_len],
        )

    def clear(self) -> None:
        """
        Cache durumunu sıfırla.

        Buffer tahsisi korunur (GC baskısı ve bellek fragmentasyonu önlenir).
        Aynı batch_size ile tekrar kullanım için hızlı yeniden başlatma sağlar.
        Multi-turn konuşmalarda her tur sonunda çağrılmak üzere tasarlanmıştır.
        """
        if self.key_cache is not None:
            self.key_cache.zero_()
            self.value_cache.zero_()
        self.cache_len = 0
        self._seen_tokens = 0
        self.positions = torch.empty(0, dtype=torch.long, device=self.device)
        self.attention_positions = self.positions
        self.logger.debug("[V5] KVCache: Sıfırlandı (buffer yeniden kullanım için korundu).")

    def reset(self) -> None:
        """clear() için geriye dönük uyumluluk alias'ı."""
        self.clear()

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def seen_tokens(self) -> int:
        """
        Toplam görülen token sayısı (prefill + decode).

        cache_len'den farklıdır: eviction sonrasında cache_len küçülür
        ama seen_tokens monoton artar. Prefill ve decode aşamalarını
        ayırt etmek için kullanılır.
        """
        return self._seen_tokens

    @property
    def is_empty(self) -> bool:
        """Cache'in boş olup olmadığını döndür."""
        return self.cache_len == 0

    @property
    def is_full(self) -> bool:
        """Cache'in kapasitesine ulaşıp ulaşmadığını döndür."""
        return self.cache_len >= self.max_cache_len

    def __len__(self) -> int:
        """Cache'in şu anki dolu uzunluğunu döndür."""
        return self.cache_len

    def __repr__(self) -> str:
        return (
            f"KVCache(batch={self.batch_size}, heads={self.num_heads}, "
            f"head_dim={self.head_dim}, cache_len={self.cache_len}/"
            f"{self.max_cache_len}, seen={self._seen_tokens}, "
            f"eviction='{self.eviction_strategy}')"
        )

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _evict_sliding_window(self, incoming_len: int) -> None:
        """
        [V5] StreamingLLM eviction stratejisi (Xiao et al. 2023).

        Cache yapısı:
            [sink_0 ... sink_{S-1}] [eski tokenlar...] [yeni tokenlar için yer]

        Adımlar:
            1. Açılması gereken slot sayısını hesapla (needed).
            2. Sink tokenlar (ilk S) yerinde kalır.
            3. Non-sink bölüm sola kaydırılır: en eski `needed` token düşer.
            4. self.cache_len güncellenir; append için yer açılmış olur.

        Args:
            incoming_len: Eklenmek üzere olan yeni token sayısı.
        """
        S = min(self.num_sink_tokens, self.cache_len)
        needed = (self.cache_len + incoming_len) - self.max_cache_len

        non_sink_len = self.cache_len - S
        new_non_sink_len = max(0, non_sink_len - needed)

        if new_non_sink_len > 0:
            # [S + needed : cache_len] bölümünü [S : S + new_non_sink_len]'e kopyala
            src_start = S + needed
            src_end   = self.cache_len
            self.key_cache[:, :, S : S + new_non_sink_len].copy_(
                self.key_cache[:, :, src_start:src_end].clone()
            )
            self.value_cache[:, :, S : S + new_non_sink_len].copy_(
                self.value_cache[:, :, src_start:src_end].clone()
            )

        old_cache_len = self.cache_len
        self.cache_len = S + new_non_sink_len

        self.logger.debug(
            f"[V5][StreamingLLM] Eviction: sink={S}, evicted={needed}, "
            f"cache_len: {old_cache_len} → {self.cache_len} "
            f"(incoming={incoming_len}, max={self.max_cache_len})"
        )
