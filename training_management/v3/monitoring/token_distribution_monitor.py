"""
token_distribution_monitor.py
==============================
Cevahir V3 Eğitim Sistemi — Token dağılımı izleme modülü.

Eğitim sırasında model çıktısının token dağılımını izler.
Collapse tespiti için EOS oranı, entropy ve TTR hesaplar.

Yazar: Cevahir Sinir Sistemi V3
Tarih: 2026
"""

from __future__ import annotations

import logging
import math
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

# Collapse eşikleri
COLLAPSE_EOS_RATIO: float = 0.5     # EOS oranı bu değerin üstündeyse collapse
COLLAPSE_ENTROPY_THRESHOLD: float = 1.0  # Entropy bu değerin altındaysa collapse

# Çok sık görmek istemediğimiz maksimum EOS oranı (uyarı seviyesi)
WARNING_EOS_RATIO: float = 0.3


class TokenDistributionMonitor:
    """
    Eğitim sırasında model çıktısının token dağılımını izler.

    İzlenen metrikler:
    - EOS oranı (generated_eos / total_tokens) — yüksekse collapse
    - PAD oranı
    - Content token oranı
    - Top-10 en sık üretilen tokenlar
    - Unigram entropy: H(token_dist) — düşükse collapse
    - Type-token ratio (TTR): unique/total — çeşitlilik ölçüsü

    Her N batch'te batch'ten token örnekleri alır,
    argmax(logits) ile üretilen token dağılımını tahmin eder.
    """

    def __init__(
        self,
        tokenizer=None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        window_size: int = 1000,
        log_every_n_batches: int = 10,
    ) -> None:
        """
        Args:
            tokenizer           : HuggingFace veya uyumlu tokenizer (opsiyonel,
                                  token ID → string çevirimi için).
            eos_id              : End-of-sequence token ID'si. None ise
                                  tokenizer.eos_token_id kullanılır.
            pad_id              : Padding token ID'si.
            window_size         : Kaydırmalı pencere boyutu (son N token saklanır).
            log_every_n_batches : Kaç batch'te bir istatistik loglanır.
        """
        self.tokenizer = tokenizer
        self.pad_id = pad_id
        self.window_size = window_size
        self.log_every_n_batches = log_every_n_batches

        # EOS ID belirle
        if eos_id is not None:
            self.eos_id = eos_id
        elif tokenizer is not None and hasattr(tokenizer, "eos_token_id"):
            self.eos_id = tokenizer.eos_token_id
        else:
            self.eos_id = None
            logger.warning(
                "TokenDistributionMonitor: EOS token ID bulunamadı. "
                "EOS oranı hesaplanamayacak."
            )

        # Sliding window: son `window_size` kadar token ID
        self._token_window: deque = deque(maxlen=window_size)

        # Batch sayacı
        self._batch_counter: int = 0

        # Son hesaplanan istatistikler (cache)
        self._cached_stats: Optional[Dict] = None
        self._cache_dirty: bool = True

        logger.info(
            "TokenDistributionMonitor başlatıldı: "
            "window_size=%d, eos_id=%s, pad_id=%d",
            window_size, self.eos_id, pad_id,
        )

    # ------------------------------------------------------------------
    # Dahili yardımcılar
    # ------------------------------------------------------------------

    def _compute_entropy(self, counter: Counter) -> float:
        """
        Unigram entropi hesapla: H = -Σ p(x) * log2(p(x))

        Args:
            counter: Token ID → frekans sayacı.

        Returns:
            float: Bit cinsinden entropi. Counter boşsa 0.0.
        """
        total = sum(counter.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in counter.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        return float(entropy)

    def _token_id_to_str(self, token_id: int) -> str:
        """Token ID'yi string'e çevir (tokenizer varsa)."""
        if self.tokenizer is None:
            return str(token_id)
        try:
            return self.tokenizer.decode([token_id], skip_special_tokens=False)
        except Exception:
            return str(token_id)

    # ------------------------------------------------------------------
    # Ana API
    # ------------------------------------------------------------------

    def update(
        self,
        logits: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Batch logits'inden token dağılımı güncelle.

        Args:
            logits  : (B, T, V) şeklinde ham logit tensörü.
                      argmax ile her pozisyon için tahmin edilen token alınır.
            targets : (B, T) şeklinde hedef token ID'leri (opsiyonel, şu an
                      kullanılmıyor ancak gelecekte karşılaştırma için).
        """
        if logits.dim() != 3:
            logger.warning(
                "update(): logits boyutu (B,T,V) bekleniyor, alındı: %s",
                tuple(logits.shape),
            )
            return

        # Gradient takibini devre dışı bırak, CPU'ya taşı
        with torch.no_grad():
            # (B, T) — her pozisyon için argmax token ID'si
            predicted = logits.argmax(dim=-1).reshape(-1).cpu().tolist()

        # Pencereye ekle
        self._token_window.extend(predicted)
        self._batch_counter += 1
        self._cache_dirty = True  # İstatistikler yeniden hesaplanmalı

        # Periyodik uyarı kontrolü
        if self._batch_counter % self.log_every_n_batches == 0:
            stats = self.get_stats()
            if stats["is_collapsed"]:
                logger.error(
                    "TOKEN COLLAPSE TESPİT EDİLDİ! "
                    "eos_ratio=%.3f entropy=%.3f ttr=%.3f",
                    stats["eos_ratio"],
                    stats["unigram_entropy"],
                    stats["type_token_ratio"],
                )
            elif stats["eos_ratio"] > WARNING_EOS_RATIO:
                logger.warning(
                    "Yüksek EOS oranı uyarısı: eos_ratio=%.3f",
                    stats["eos_ratio"],
                )

    def get_stats(self) -> Dict:
        """
        Sliding window üzerinden token dağılım istatistiklerini hesapla.

        Returns:
            Dict içeren:
            - eos_ratio (float)        : EOS tokenlarının oranı
            - pad_ratio (float)        : PAD tokenlarının oranı
            - content_ratio (float)    : İçerik tokenlarının oranı
            - unigram_entropy (float)  : Bit cinsinden entropi
            - type_token_ratio (float) : unique/total token oranı
            - top10_tokens (List)      : [(token_id, frekans_oranı), ...]
            - is_collapsed (bool)      : Collapse tespiti
            - total_tokens (int)       : Penceredeki toplam token sayısı
        """
        # Cache kontrolü
        if not self._cache_dirty and self._cached_stats is not None:
            return self._cached_stats

        tokens = list(self._token_window)
        total = len(tokens)

        if total == 0:
            empty = {
                "eos_ratio": 0.0,
                "pad_ratio": 0.0,
                "content_ratio": 0.0,
                "unigram_entropy": 0.0,
                "type_token_ratio": 0.0,
                "top10_tokens": [],
                "is_collapsed": False,
                "total_tokens": 0,
            }
            self._cached_stats = empty
            self._cache_dirty = False
            return empty

        counter: Counter = Counter(tokens)

        # --- EOS oranı ---
        eos_count = counter.get(self.eos_id, 0) if self.eos_id is not None else 0
        eos_ratio = eos_count / total

        # --- PAD oranı ---
        pad_count = counter.get(self.pad_id, 0)
        pad_ratio = pad_count / total

        # --- Content token oranı ---
        special_count = eos_count + pad_count
        content_ratio = max(0.0, (total - special_count) / total)

        # --- Unigram entropy ---
        unigram_entropy = self._compute_entropy(counter)

        # --- Type-token ratio ---
        unique_tokens = len(counter)
        type_token_ratio = unique_tokens / total

        # --- Top-10 token ---
        top10_raw = counter.most_common(10)
        top10_tokens: List[Tuple[int, float]] = [
            (tid, cnt / total) for tid, cnt in top10_raw
        ]

        # --- Collapse tespiti ---
        is_collapsed = (
            eos_ratio > COLLAPSE_EOS_RATIO
            or unigram_entropy < COLLAPSE_ENTROPY_THRESHOLD
        )

        stats = {
            "eos_ratio": float(eos_ratio),
            "pad_ratio": float(pad_ratio),
            "content_ratio": float(content_ratio),
            "unigram_entropy": float(unigram_entropy),
            "type_token_ratio": float(type_token_ratio),
            "top10_tokens": top10_tokens,
            "is_collapsed": is_collapsed,
            "total_tokens": total,
        }

        self._cached_stats = stats
        self._cache_dirty = False
        return stats

    def reset_window(self) -> None:
        """
        Sliding window'u temizle.
        Epoch başında veya manual sıfırlama gerektiğinde çağır.
        """
        self._token_window.clear()
        self._cached_stats = None
        self._cache_dirty = True
        logger.debug("TokenDistributionMonitor penceresi sıfırlandı.")

    def log_to_tensorboard(self, writer: Any, global_step: int) -> None:
        """
        TensorBoard'a token dağılım metriklerini yaz.

        Yazılan tag'ler:
        TokenDist/EOS_Ratio
        TokenDist/PAD_Ratio
        TokenDist/ContentRatio
        TokenDist/UnigramEntropy
        TokenDist/TypeTokenRatio
        TokenDist/IsCollapsed
        TokenDist/UniqueTokenCount

        Args:
            writer      : torch.utils.tensorboard.SummaryWriter
            global_step : Global adım numarası.
        """
        if writer is None:
            return

        stats = self.get_stats()

        writer.add_scalar("TokenDist/EOS_Ratio", stats["eos_ratio"], global_step)
        writer.add_scalar("TokenDist/PAD_Ratio", stats["pad_ratio"], global_step)
        writer.add_scalar("TokenDist/ContentRatio", stats["content_ratio"], global_step)
        writer.add_scalar("TokenDist/UnigramEntropy", stats["unigram_entropy"], global_step)
        writer.add_scalar("TokenDist/TypeTokenRatio", stats["type_token_ratio"], global_step)
        writer.add_scalar(
            "TokenDist/IsCollapsed",
            1.0 if stats["is_collapsed"] else 0.0,
            global_step,
        )

        # Top token'ların metin formatında logu (opsiyonel)
        if stats["top10_tokens"]:
            top_lines = []
            for i, (tid, freq) in enumerate(stats["top10_tokens"][:5]):
                tok_str = self._token_id_to_str(tid)
                top_lines.append(f"#{i+1}: '{tok_str}' (id={tid}, %{freq*100:.1f})")
            writer.add_text(
                "TokenDist/Top5_Tokens",
                "\n".join(top_lines),
                global_step,
            )

    def get_formatted_report(self) -> str:
        """
        İnsan tarafından okunabilir metin raporu üret.

        Returns:
            str: Biçimlendirilmiş rapor metni.
        """
        stats = self.get_stats()
        lines = [
            "=" * 50,
            "TOKEN DAĞILIM RAPORU",
            "=" * 50,
            f"Toplam token sayısı : {stats['total_tokens']}",
            f"EOS oranı           : {stats['eos_ratio']:.4f}",
            f"PAD oranı           : {stats['pad_ratio']:.4f}",
            f"Content oranı       : {stats['content_ratio']:.4f}",
            f"Unigram entropi     : {stats['unigram_entropy']:.4f} bit",
            f"Type-Token Ratio    : {stats['type_token_ratio']:.4f}",
            f"Collapse durumu     : {'EVET (UYARI!)' if stats['is_collapsed'] else 'Hayır'}",
            "",
            "En sık 10 token:",
        ]
        for i, (tid, freq) in enumerate(stats["top10_tokens"]):
            tok_str = self._token_id_to_str(tid)
            lines.append(f"  {i+1:2d}. id={tid:6d}  '{tok_str}'  %{freq*100:.2f}")
        lines.append("=" * 50)
        return "\n".join(lines)

    @property
    def batch_count(self) -> int:
        """Şimdiye kadar işlenen batch sayısı."""
        return self._batch_counter

    def __repr__(self) -> str:
        return (
            f"TokenDistributionMonitor("
            f"window_size={self.window_size}, "
            f"batches_processed={self._batch_counter}, "
            f"current_window_len={len(self._token_window)})"
        )
