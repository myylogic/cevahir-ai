"""
inference_quality_probe.py
===========================
Cevahir V3 Eğitim Sistemi — Gerçek inference kalite ölçüm modülü.

Her N epoch'ta model üzerinde gerçek inference çalıştırır.
Teacher forcing'in gizlediği kalite sorunlarını tespit eder.

Yazar: Cevahir Sinir Sistemi V3
Tarih: 2026
"""

from __future__ import annotations

import logging
import math
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import torch

# Python 3.8 uyumluluğu için TypedDict
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tip tanımları
# ---------------------------------------------------------------------------

class InferenceQualityMetrics(TypedDict):
    """Tek bir probe çalışmasının sonuç metrikleri."""
    avg_response_length: float    # Ortalama yanıt uzunluğu (token sayısı)
    avg_entropy: float            # Çıktı token dağılımının ortalama entropisi
    avg_eos_ratio: float          # Her yanıttaki EOS oranının ortalaması
    type_token_ratio: float       # Tüm yanıtlardaki unique/total token oranı
    is_collapsed: bool            # Basit heuristic ile collapse tespiti
    responses: List[str]          # Örnek yanıtlar (ham string)


# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

# Collapse heuristic eşikleri
COLLAPSE_MAX_AVG_LENGTH: float = 3.0    # Yanıtlar bu kadar kısaysa collapse
COLLAPSE_MIN_ENTROPY: float = 0.5       # Entropy bu değerin altındaysa collapse
COLLAPSE_MAX_EOS_RATIO: float = 0.6     # EOS oranı yüksekse collapse

# Varsayılan generation parametreleri
DEFAULT_MAX_NEW_TOKENS: int = 50
DEFAULT_TEMPERATURE: float = 1.5
DEFAULT_TOP_K: int = 50


class InferenceQualityProbe:
    """
    Her N epoch'ta model üzerinde gerçek inference çalıştırır.
    Eğitim boyunca inference kalitesini izler.

    TEMELİ: Training metrikleri (accuracy, loss) aldatıcı olabilir
    (teacher forcing sayesinde yüksek görünebilir).
    Bu probe GERÇEK inference kalitesini ölçer.

    Sabit test prompt'larına yanıt üretir:
    - Türkçe temel selamlaşma: "selam", "merhaba nasılsın"
    - Türkçe soru: "adın ne", "nasıl çalışıyorsun"
    - İngilizce: "hello", "how are you"
    - Genel Türkçe: "bugün hava nasıl", "bana bir şey anlat"

    Metrikler:
    - Ortalama yanıt uzunluğu (collapse → hep kısa)
    - Output token entropy (collapse → düşük)
    - EOS oranı (collapse → yüksek)
    - TTR (type-token ratio): çeşitlilik
    - is_collapsed: basit heuristic
    """

    DEFAULT_PROMPTS: List[str] = [
        "selam",
        "merhaba nasılsın",
        "adın ne",
        "nasıl çalışıyorsun",
        "hello",
        "bugün hava nasıl",
        "bana bir şey anlat",
        "Türkiye'nin başkenti neresi",
    ]

    def __init__(
        self,
        cevahir_model: Any = None,
        model_manager: Any = None,
        tokenizer: Any = None,
        probe_interval: int = 5,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        custom_prompts: Optional[List[str]] = None,
        tensorboard_writer: Any = None,
    ) -> None:
        """
        Args:
            cevahir_model      : Cevahir model örneği. generate() metodu beklenir.
            model_manager      : ModelManager örneği (Cevahir yoksa alternatif).
            tokenizer          : HuggingFace uyumlu tokenizer.
            probe_interval     : Kaç epoch'ta bir probe çalıştırılır.
            max_new_tokens     : Her yanıt için maksimum üretilen token sayısı.
            temperature        : Sampling sıcaklığı (>1.0 → daha çeşitli).
            top_k              : Top-K sampling parametresi.
            custom_prompts     : Özelleştirilmiş test prompt listesi.
                                 None ise DEFAULT_PROMPTS kullanılır.
            tensorboard_writer : SummaryWriter örneği (opsiyonel).
        """
        # Model referansları
        self.cevahir_model = cevahir_model
        self.model_manager = model_manager
        self.tokenizer = tokenizer

        # Probe ayarları
        self.probe_interval = probe_interval
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.writer = tensorboard_writer

        # Test prompt listesi
        self.prompts: List[str] = custom_prompts if custom_prompts else self.DEFAULT_PROMPTS

        # Tarihsel sonuçlar
        self._history: List[Dict] = []

        # Model yoksa uyar
        if cevahir_model is None and model_manager is None:
            logger.warning(
                "InferenceQualityProbe: Ne cevahir_model ne de model_manager "
                "verildi. probe.run() çağrıldığında hata oluşabilir."
            )

        if tokenizer is None:
            logger.warning(
                "InferenceQualityProbe: tokenizer verilmedi. "
                "Metin çözümleme yapılamayacak."
            )

        logger.info(
            "InferenceQualityProbe başlatıldı: "
            "probe_interval=%d, prompts=%d, max_new_tokens=%d",
            probe_interval, len(self.prompts), max_new_tokens,
        )

    # ------------------------------------------------------------------
    # Dahili yardımcılar
    # ------------------------------------------------------------------

    def _compute_entropy_from_tokens(self, token_ids: List[int]) -> float:
        """
        Token ID listesinden unigram entropi hesapla.

        Args:
            token_ids: Token ID'lerinin listesi.

        Returns:
            float: Bit cinsinden entropi. Liste boşsa 0.0.
        """
        if not token_ids:
            return 0.0

        counter: Counter = Counter(token_ids)
        total = len(token_ids)
        entropy = 0.0
        for count in counter.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return float(entropy)

    def _compute_eos_ratio(self, token_ids: List[int]) -> float:
        """
        Token listesindeki EOS oranını hesapla.

        Args:
            token_ids: Token ID'lerinin listesi.

        Returns:
            float: EOS oranı (0.0 - 1.0).
        """
        if not token_ids:
            return 0.0

        eos_id = None
        if self.tokenizer is not None and hasattr(self.tokenizer, "eos_token_id"):
            eos_id = self.tokenizer.eos_token_id

        if eos_id is None:
            return 0.0

        eos_count = token_ids.count(eos_id)
        return eos_count / len(token_ids)

    def _decode_response(self, token_ids: List[int]) -> str:
        """
        Token ID listesini metne çevir.

        Args:
            token_ids: Token ID'lerinin listesi.

        Returns:
            str: Çözümlenmiş metin. Tokenizer yoksa ID'lerin string hali.
        """
        if self.tokenizer is None:
            return " ".join(str(t) for t in token_ids)
        try:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        except Exception as e:
            logger.debug("Token çözümleme hatası: %s", e)
            return " ".join(str(t) for t in token_ids)

    # ------------------------------------------------------------------
    # Ana API
    # ------------------------------------------------------------------

    def should_probe(self, epoch: int) -> bool:
        """
        Bu epoch'ta probe çalıştırılmalı mı?

        Args:
            epoch: Mevcut epoch numarası (0-indexed).

        Returns:
            bool: True ise probe çalıştırılmalı.
        """
        # İlk epoch her zaman çalışır
        if epoch == 0:
            return True
        return epoch % self.probe_interval == 0

    def _generate_response(self, prompt: str) -> str:
        """
        Tek bir prompt için yanıt üret.

        ModelManager veya Cevahir modelini dener. Her ikisi de yoksa
        boş string döner.

        Args:
            prompt: Giriş prompt metni.

        Returns:
            str: Üretilen yanıt metni.
        """
        # 1. ModelManager ile dene
        if self.model_manager is not None:
            try:
                response = self.model_manager.generate(
                    prompt=prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_k=self.top_k,
                )
                if isinstance(response, str):
                    return response
                if isinstance(response, list) and len(response) > 0:
                    return str(response[0])
            except Exception as e:
                logger.debug(
                    "ModelManager.generate() hatası (prompt='%s'): %s", prompt[:30], e
                )

        # 2. Cevahir modeli ile dene
        if self.cevahir_model is not None:
            try:
                # Tokenize et
                if self.tokenizer is not None:
                    input_ids = self.tokenizer.encode(
                        prompt, return_tensors="pt"
                    )
                    # GPU'ya taşı (model neredeyse)
                    device = next(self.cevahir_model.parameters()).device
                    input_ids = input_ids.to(device)

                    with torch.no_grad():
                        output = self.cevahir_model.generate(
                            input_ids,
                            max_new_tokens=self.max_new_tokens,
                            temperature=self.temperature,
                            top_k=self.top_k,
                            do_sample=True,
                        )
                    # Yalnızca yeni tokenları çöz
                    new_tokens = output[0][input_ids.shape[1]:].tolist()
                    return self._decode_response(new_tokens)
                else:
                    # Tokenizer yoksa doğrudan generate çağır
                    response = self.cevahir_model.generate(
                        prompt,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        top_k=self.top_k,
                    )
                    return str(response) if response else ""
            except Exception as e:
                logger.debug(
                    "Cevahir.generate() hatası (prompt='%s'): %s", prompt[:30], e
                )

        logger.warning(
            "_generate_response: Ne ModelManager ne Cevahir modeli çalışmadı. "
            "Prompt: '%s'",
            prompt[:40],
        )
        return ""

    def _compute_metrics(self, responses: List[str]) -> InferenceQualityMetrics:
        """
        Üretilen yanıtlardan quality metriklerini hesapla.

        Args:
            responses: Ham yanıt string'lerinin listesi.

        Returns:
            InferenceQualityMetrics dict'i.
        """
        if not responses:
            return InferenceQualityMetrics(
                avg_response_length=0.0,
                avg_entropy=0.0,
                avg_eos_ratio=0.0,
                type_token_ratio=0.0,
                is_collapsed=True,
                responses=[],
            )

        all_token_ids: List[int] = []
        per_response_lengths: List[float] = []
        per_response_entropies: List[float] = []
        per_response_eos_ratios: List[float] = []

        for resp in responses:
            # Token ID'lerine çevir
            if self.tokenizer is not None:
                try:
                    token_ids: List[int] = self.tokenizer.encode(
                        resp, add_special_tokens=False
                    )
                except Exception:
                    token_ids = []
            else:
                # Kelime bazında yaklaşım
                token_ids = list(range(len(resp.split())))

            per_response_lengths.append(float(len(token_ids)))
            per_response_entropies.append(self._compute_entropy_from_tokens(token_ids))
            per_response_eos_ratios.append(self._compute_eos_ratio(token_ids))
            all_token_ids.extend(token_ids)

        # Ortalamalar
        avg_response_length = (
            sum(per_response_lengths) / len(per_response_lengths)
        )
        avg_entropy = (
            sum(per_response_entropies) / len(per_response_entropies)
        )
        avg_eos_ratio = (
            sum(per_response_eos_ratios) / len(per_response_eos_ratios)
        )

        # Type-Token Ratio (tüm yanıtlar üzerinden)
        total_tokens = len(all_token_ids)
        unique_tokens = len(set(all_token_ids))
        type_token_ratio = unique_tokens / max(total_tokens, 1)

        # Collapse tespiti (basit heuristic)
        is_collapsed = (
            avg_response_length <= COLLAPSE_MAX_AVG_LENGTH
            or avg_entropy < COLLAPSE_MIN_ENTROPY
            or avg_eos_ratio > COLLAPSE_MAX_EOS_RATIO
        )

        return {
            "avg_response_length": float(avg_response_length),
            "avg_entropy": float(avg_entropy),
            "avg_eos_ratio": float(avg_eos_ratio),
            "type_token_ratio": float(type_token_ratio),
            "is_collapsed": bool(is_collapsed),
            "responses": responses,
        }

    def run(self, epoch: int) -> InferenceQualityMetrics:
        """
        Tüm test prompt'larında inference çalıştır.
        Metrikleri hesapla, history'ye ekle ve döndür.

        Args:
            epoch: Mevcut epoch numarası.

        Returns:
            InferenceQualityMetrics: Hesaplanan metrikler.
        """
        logger.info(
            "InferenceQualityProbe çalışıyor — Epoch %d, %d prompt...",
            epoch, len(self.prompts),
        )
        start_time = time.time()

        responses: List[str] = []
        for prompt in self.prompts:
            try:
                response = self._generate_response(prompt)
                responses.append(response)
                logger.debug(
                    "  Prompt: '%s' → Yanıt: '%s'",
                    prompt[:40],
                    response[:60] if response else "(boş)",
                )
            except Exception as e:
                logger.error("Prompt işlenirken hata: '%s' — %s", prompt[:40], e)
                responses.append("")

        elapsed = time.time() - start_time
        metrics = self._compute_metrics(responses)

        # History'ye kaydet
        history_entry = {
            "epoch": epoch,
            "elapsed_sec": elapsed,
            **{k: v for k, v in metrics.items() if k != "responses"},
            "response_count": len(responses),
        }
        self._history.append(history_entry)

        # Özet loglama
        collapse_flag = "COLLAPSE!" if metrics["is_collapsed"] else "OK"
        logger.info(
            "InferenceQualityProbe [Epoch %d] [%s] — "
            "avg_len=%.1f, entropy=%.3f, eos_ratio=%.3f, ttr=%.3f (%.1fs)",
            epoch,
            collapse_flag,
            metrics["avg_response_length"],
            metrics["avg_entropy"],
            metrics["avg_eos_ratio"],
            metrics["type_token_ratio"],
            elapsed,
        )

        if metrics["is_collapsed"]:
            logger.error(
                "MODEL COLLAPSE TESPİT EDİLDİ (Epoch %d)! "
                "Eğitim parametrelerini kontrol edin.",
                epoch,
            )

        # TensorBoard'a yaz
        if self.writer is not None:
            self.log_to_tensorboard(metrics, epoch)

        return metrics

    def log_to_tensorboard(
        self, metrics: InferenceQualityMetrics, epoch: int
    ) -> None:
        """
        TensorBoard'a inference quality metriklerini yaz.

        Yazılan tag'ler:
        InferenceQuality/AvgResponseLength
        InferenceQuality/AvgEntropy
        InferenceQuality/EOSRatio
        InferenceQuality/TypeTokenRatio
        InferenceQuality/IsCollapsed

        Ayrıca örnek yanıtlar metin olarak kaydedilir.

        Args:
            metrics : Hesaplanan metrikler.
            epoch   : Epoch numarası (global step olarak kullanılır).
        """
        if self.writer is None:
            return

        self.writer.add_scalar(
            "InferenceQuality/AvgResponseLength",
            metrics["avg_response_length"],
            epoch,
        )
        self.writer.add_scalar(
            "InferenceQuality/AvgEntropy",
            metrics["avg_entropy"],
            epoch,
        )
        self.writer.add_scalar(
            "InferenceQuality/EOSRatio",
            metrics["avg_eos_ratio"],
            epoch,
        )
        self.writer.add_scalar(
            "InferenceQuality/TypeTokenRatio",
            metrics["type_token_ratio"],
            epoch,
        )
        self.writer.add_scalar(
            "InferenceQuality/IsCollapsed",
            1.0 if metrics["is_collapsed"] else 0.0,
            epoch,
        )

        # Örnek yanıtları metin olarak kaydet
        if metrics["responses"]:
            sample_lines = []
            for i, (prompt, resp) in enumerate(
                zip(self.prompts, metrics["responses"])
            ):
                sample_lines.append(
                    f"**Q{i+1}:** `{prompt}`  \n**A:** {resp or '(boş yanıt)'}"
                )
            sample_text = "\n\n".join(sample_lines)
            self.writer.add_text(
                "InferenceQuality/SampleResponses",
                sample_text,
                epoch,
            )

    def get_history(self) -> List[Dict]:
        """
        Geçmiş tüm probe sonuçlarını döndür.

        Returns:
            List[Dict]: Her eleman bir probe çalışmasının özet verilerini içerir.
                        'responses' anahtarı (büyük veri) dahil değildir.
        """
        return list(self._history)

    def get_last_metrics(self) -> Optional[Dict]:
        """
        En son probe çalışmasının metriklerini döndür.

        Returns:
            Dict veya None (henüz probe çalıştırılmamışsa).
        """
        if not self._history:
            return None
        return self._history[-1]

    def get_collapse_epochs(self) -> List[int]:
        """
        Collapse tespit edilen epoch'ların listesini döndür.

        Returns:
            List[int]: Collapse olan epoch numaraları.
        """
        return [
            entry["epoch"]
            for entry in self._history
            if entry.get("is_collapsed", False)
        ]

    def add_prompt(self, prompt: str) -> None:
        """
        Çalışma zamanında yeni test prompt ekle.

        Args:
            prompt: Eklenecek prompt metni.
        """
        if prompt not in self.prompts:
            self.prompts.append(prompt)
            logger.info("Yeni test prompt eklendi: '%s'", prompt[:60])

    def remove_prompt(self, prompt: str) -> bool:
        """
        Prompt listesinden bir prompt'u kaldır.

        Args:
            prompt: Kaldırılacak prompt metni.

        Returns:
            bool: Kaldırma başarılıysa True.
        """
        if prompt in self.prompts:
            self.prompts.remove(prompt)
            logger.info("Test prompt kaldırıldı: '%s'", prompt[:60])
            return True
        return False

    def __repr__(self) -> str:
        last = self.get_last_metrics()
        last_epoch = last["epoch"] if last else "—"
        return (
            f"InferenceQualityProbe("
            f"prompts={len(self.prompts)}, "
            f"interval={self.probe_interval}, "
            f"probe_count={len(self._history)}, "
            f"last_epoch={last_epoch})"
        )
