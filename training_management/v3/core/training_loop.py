"""
Cevahir V3 - Eğitim Döngüsü
=============================
Bu modül, Cevahir Türkçe dil modelinin epoch bazlı eğitim ve doğrulama
döngülerini implemente eder.

Öne Çıkan Özellikler:
    1. Scheduled Sampling   : Bengio et al. (2015) öğretmen zorlaması bozunumu
    2. AMP Desteği          : torch.cuda.amp ile karışık hassasiyetli eğitim
    3. Gradyan Birikimi     : grad_accum_steps ile büyük efektif batch
    4. Token Dağılım İzleme : EOS/içerik token oranları izleme
    5. Gradyan Gürültüsü    : İsteğe bağlı Neelakantan 2015 gürültüsü

Referanslar:
    - Bengio et al. (2015): "Scheduled Sampling for Sequence Prediction
      with Recurrent Neural Networks" (NIPS 2015)
    - Micikevicius et al. (2018): "Mixed Precision Training" (AMP için)
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
# [FIX] torch.cuda.amp.GradScaler deprecated in PyTorch ≥2.0
# torch.amp.GradScaler('cuda') kullan
import torch.amp as _torch_amp
from typing_extensions import TypedDict

from training_management.v3.core.batch_processor import BatchProcessor
from training_management.v3.core.gradient_manager import GradientManager
from training_management.v3.core.loss_manager import (
    CompositeLossManager,
    LossOutput,
    compute_entropy,
)

logger = logging.getLogger(__name__)

__all__ = [
    "TrainingLoop",
    "TrainingLoopConfig",
    "EpochMetrics",
    "ScheduledSamplingMixin",
]


# ---------------------------------------------------------------------------
# Paylaşımlı Arayüz: EpochMetrics TypedDict
# ---------------------------------------------------------------------------

class EpochMetrics(TypedDict):
    """
    Bir eğitim/doğrulama epoch'unun sonuç metriklerini içeren TypedDict.

    Alanlar:
        loss           : Ortalama toplam kayıp değeri
        accuracy       : Token-seviyesinde doğruluk oranı [0, 1]
        perplexity     : exp(ce_loss), dil modeli değerlendirme metriği
        entropy        : Ortalama çıkış dağılımı Shannon entropisi
        gradient_norm  : Epoch boyunca ortalama gradyan L2 norm'u
        tokens_per_sec : İşlenen token hızı (verim metriği)
        loss_breakdown : CE/focal/entropy_reg/auxiliary kayıp bileşenleri
        token_dist     : EOS oranı, içerik token oranı vb. dağılım istatistikleri
    """
    loss: float
    accuracy: float
    perplexity: float
    entropy: float
    gradient_norm: float
    tokens_per_sec: float
    loss_breakdown: dict
    token_dist: dict


# ---------------------------------------------------------------------------
# Konfigürasyon Veri Sınıfı
# ---------------------------------------------------------------------------

@dataclass
class TrainingLoopConfig:
    """
    TrainingLoop için tüm parametreleri içeren konfigürasyon sınıfı.

    Scheduled Sampling Parametreleri:
        use_scheduled_sampling    : Zamanlanmış örnekleme aktif mi?
        scheduled_sampling_epochs : Öğretmen zorlaması bozunumu kaç epoch'ta tamamlanır
        min_teacher_forcing_prob  : Minimum öğretmen zorlaması olasılığı
        initial_teacher_forcing   : Başlangıç öğretmen zorlaması oranı (genellikle 1.0)

    AMP ve Verimlilik Parametreleri:
        use_amp              : Karışık hassasiyetli eğitim (Float16 + Float32)
        grad_accum_steps     : Gradyan birikimi adım sayısı
        max_grad_norm        : Gradyan kırpma eşiği (None → kırpma yok)
        use_adaptive_clip    : Standart yerine AGC kullan (Brock 2021)

    Gürültü ve Düzenlileştirme:
        use_gradient_noise   : Gradyan gürültüsü enjeksiyonu
        noise_eta            : Gürültü ölçeği η (Neelakantan 2015)

    Token Dağılım İzleme:
        log_token_dist_every : Bu kadar batch'te bir token dağılımını raporla
        eos_token_id         : EOS token ID'si (dağılım istatistikleri için)
        pad_token_id         : PAD token ID'si
    """
    # Scheduled Sampling (Bengio et al. 2015)
    use_scheduled_sampling: bool = True
    scheduled_sampling_epochs: int = 50
    min_teacher_forcing_prob: float = 0.1
    initial_teacher_forcing: float = 1.0

    # AMP (Karışık Hassasiyetli Eğitim)
    use_amp: bool = True
    amp_dtype: torch.dtype = torch.float16

    # Gradyan Birikimi
    grad_accum_steps: int = 1

    # Gradyan Kırpma
    max_grad_norm: Optional[float] = 1.0
    use_adaptive_clip: bool = False
    agc_lambda: float = 0.01

    # Gradyan Gürültüsü (Neelakantan et al. 2015)
    use_gradient_noise: bool = False
    noise_eta: float = 0.01

    # Token Dağılım İzleme
    log_token_dist_every: int = 10
    eos_token_id: int = 2
    pad_token_id: int = 0

    # Genel
    device: str = "cuda"
    log_every_n_batches: int = 50


# ---------------------------------------------------------------------------
# Zamanlanmış Örnekleme Mixin
# ---------------------------------------------------------------------------

class ScheduledSamplingMixin:
    """
    Zamanlanmış Örnekleme Mixin Sınıfı (Bengio et al. 2015)
    =========================================================
    NIPS 2015 "Scheduled Sampling for Sequence Prediction with RNNs"

    Yöntem:
    Eğitim başlarında öğretmen zorlaması (teacher forcing) kullanılır:
        inputs = gerçek hedef tokenlar (oracle)

    Eğitim ilerledikçe model kendi tahminlerini girdi olarak kullanır:
        inputs = önceki adımdaki model tahmini (self-feeding)

    Bu geçiş, öğretmen zorlaması olasılığı tf_prob ile kontrol edilir:
        tf_prob = max(min_tf, 1.0 - decay_rate * epoch)

    Neden önemli?
    - Saf öğretmen zorlaması: Test sırasında kendi hataları üzerine konuşlanamaz
    - Saf serbest bırakma: Erken eğitimde çok gürültülü
    - Zamanlanmış örnekleme: Kademeli geçiş, ikisinin avantajlarını birleştirir

    Türkçe dil modellemesinde önemi:
    - Türkçe'nin karmaşık ek yapısı nedeniyle hata birikimi riski yüksek
    - Zamanlanmış örnekleme bu riski azaltır
    - Özellikle uzun cümle üretiminde generalizasyonu iyileştirir
    """

    def compute_teacher_forcing_prob(
        self,
        epoch: int,
        scheduled_sampling_epochs: int,
        min_teacher_forcing: float,
        initial_teacher_forcing: float = 1.0,
    ) -> float:
        """
        Mevcut epoch için öğretmen zorlaması olasılığını hesaplar.

        Formül:
            decay_rate = (initial_tf - min_tf) / scheduled_sampling_epochs
            tf_prob = max(min_tf, initial_tf - decay_rate * epoch)

        Örnek (min_tf=0.1, epochs=50):
            Epoch 0  : tf_prob = 1.0  (tamamen öğretmen zorlaması)
            Epoch 25 : tf_prob = 0.55 (yarı yarıya)
            Epoch 50+: tf_prob = 0.1  (minimum, büyük ölçüde serbest bırakma)

        Args:
            epoch                  : Mevcut epoch numarası (0-indexed)
            scheduled_sampling_epochs: Bozunumun tamamlanacağı epoch sayısı
            min_teacher_forcing    : Minimum TF olasılığı (tabanı)
            initial_teacher_forcing: Başlangıç TF olasılığı (genellikle 1.0)

        Returns:
            tf_prob: [min_teacher_forcing, initial_teacher_forcing] aralığında float
        """
        if scheduled_sampling_epochs <= 0:
            return initial_teacher_forcing

        # Lineer bozunum oranı
        decay_rate = (initial_teacher_forcing - min_teacher_forcing) / scheduled_sampling_epochs

        # Kırpılmış lineer bozunum
        tf_prob = max(
            min_teacher_forcing,
            initial_teacher_forcing - decay_rate * epoch,
        )
        return tf_prob

    def apply_scheduled_sampling(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        tf_prob: float,
        device: torch.device,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Zamanlanmış örnekleme uygular: bazı tokenları model tahminleriyle değiştirir.

        Her pozisyon için bağımsız olarak karar verir:
        - Bernoulli(tf_prob) = 1 → gerçek token kullan (öğretmen zorlaması)
        - Bernoulli(tf_prob) = 0 → model tahmini kullan (serbest bırakma)

        Uygulama:
            1. Model ile kısa bir ileri geçiş yapılır (no_grad)
            2. Her pozisyon için Bernoulli örnekleme
            3. tf_prob=0 olan pozisyonlara model tahmini atanır

        Args:
            model   : PyTorch dil modeli
            inputs  : Orijinal giriş tokenleri [batch, seq_len]
            tf_prob : Öğretmen zorlaması olasılığı [0, 1]
            device  : Hesaplama cihazı
            temperature: Örnekleme sıcaklığı (1.0 = greedy argmax)

        Returns:
            Kısmen değiştirilmiş giriş tensörü [batch, seq_len]
        """
        # tf_prob=1.0 → tamamen öğretmen zorlaması, değişiklik yapma
        if tf_prob >= 1.0:
            return inputs

        # tf_prob=0.0 → tamamen serbest bırakma
        # (modelin tüm pozisyonlar için tahmin alınır)

        batch_size, seq_len = inputs.shape

        with torch.no_grad():
            # Model ile çıkış tahminleri al
            try:
                model_out = model(inputs)
                # Çıktı tuple olabilir (MoE/MoD), ilk elemanı al
                if isinstance(model_out, (tuple, list)):
                    logits = model_out[0]
                else:
                    logits = model_out

                # [batch, seq_len, vocab] → [batch, seq_len] token tahminleri
                if temperature == 1.0:
                    predicted_tokens = logits.argmax(dim=-1)
                else:
                    # Sıcaklık ölçekli örnekleme
                    scaled_logits = logits / max(temperature, 1e-8)
                    probs = F.softmax(scaled_logits, dim=-1)
                    batch_flat = batch_size * seq_len
                    predicted_tokens = torch.multinomial(
                        probs.reshape(batch_flat, -1), num_samples=1
                    ).reshape(batch_size, seq_len)

            except Exception as e:
                # Model ileri geçişi başarısız → öğretmen zorlamasını koru
                logger.warning(
                    f"Scheduled sampling için ileri geçiş başarısız: {e}. "
                    f"Öğretmen zorlaması korunuyor."
                )
                return inputs

            # Öğretmen zorlaması maskesi: True → gerçek token, False → model tahmini
            # Her pozisyon için bağımsız Bernoulli(tf_prob) örneklemesi
            tf_mask = torch.bernoulli(
                torch.full((batch_size, seq_len), tf_prob, device=device)
            ).bool()

            # Maske ile harmanla: tf_mask=True → inputs, tf_mask=False → predicted
            sampled_inputs = torch.where(tf_mask, inputs, predicted_tokens.to(inputs.device))

        return sampled_inputs


# ---------------------------------------------------------------------------
# Ana Eğitim Döngüsü
# ---------------------------------------------------------------------------

class TrainingLoop(ScheduledSamplingMixin):
    """
    Cevahir V3 Ana Eğitim Döngüsü
    ================================
    Epoch bazlı eğitim ve doğrulama döngülerini yönetir.

    Bu sınıf şu bileşenleri birleştirir:
        - BatchProcessor       : Batch verisi ön işleme
        - CompositeLossManager : Bileşik kayıp hesaplama
        - GradientManager      : Gradyan yönetimi
        - GradScaler           : AMP ile taşma koruması
        - ScheduledSamplingMixin: Öğretmen zorlaması bozunumu

    Eğitim Döngüsü (her epoch):
        1. model.train() modu
        2. Her batch için:
            a. Batch'i işle (inputs, targets)
            b. Zamanlanmış örnekleme uygula (aktifse)
            c. AMP context'te ileri geçiş
            d. Kayıp hesapla
            e. Geriye yayılım
            f. Gradyan birikimi (grad_accum_steps)
            g. Gradyan gürültüsü (aktifse)
            h. Gradyan kırpma
            i. Optimizer adımı
        3. Metrik hesapla ve döndür

    Kullanım:
        config = TrainingLoopConfig(use_amp=True, grad_accum_steps=4)
        loop = TrainingLoop(model, optimizer, loss_manager, config)

        train_metrics = loop.train_epoch(train_loader, epoch=5)
        val_metrics   = loop.validate_epoch(val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_manager: CompositeLossManager,
        config: TrainingLoopConfig,
        batch_processor: Optional[BatchProcessor] = None,
        gradient_manager: Optional[GradientManager] = None,
    ):
        """
        Args:
            model           : Eğitilen PyTorch dil modeli
            optimizer       : PyTorch optimizer (AdamW vb.)
            loss_manager    : CompositeLossManager nesnesi
            config          : TrainingLoopConfig konfigürasyon nesnesi
            batch_processor : BatchProcessor (None → otomatik oluşturulur)
            gradient_manager: GradientManager (None → otomatik oluşturulur)
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_manager = loss_manager
        self.config = config
        self.device = torch.device(config.device)

        # Batch işlemcisi
        self.batch_processor = batch_processor or BatchProcessor(
            device=config.device,
            pad_token_id=config.pad_token_id,
        )

        # Gradyan yöneticisi
        self.gradient_manager = gradient_manager or GradientManager(
            max_norm=config.max_grad_norm or 1.0,
            agc_lambda=config.agc_lambda,
            noise_eta=config.noise_eta,
        )

        # AMP GradScaler (yalnızca CUDA'da etkin)
        self._use_amp = (
            config.use_amp
            and torch.cuda.is_available()
            and "cuda" in config.device
        )
        # [FIX] Deprecated GradScaler() → torch.amp.GradScaler('cuda')
        self.scaler: Optional[_torch_amp.GradScaler] = (
            _torch_amp.GradScaler("cuda") if self._use_amp else None
        )

        # Global adım sayacı (gürültü bozunumu için)
        self.global_step: int = 0

        logger.info(
            f"TrainingLoop başlatıldı | "
            f"AMP={'ON' if self._use_amp else 'OFF'}, "
            f"grad_accum={config.grad_accum_steps}, "
            f"scheduled_sampling={'ON' if config.use_scheduled_sampling else 'OFF'}"
        )

    # ------------------------------------------------------------------
    # AMP Context Yöneticisi
    # ------------------------------------------------------------------

    @contextmanager
    def _amp_context(self) -> Generator:
        """
        AMP (Automatic Mixed Precision) bağlam yöneticisi.

        AMP aktifse float16/bfloat16, değilse normal float32 kullanılır.
        Bu bağlam yöneticisi ile ileri geçiş bellek verimliliği artar.
        """
        if self._use_amp:
            # [FIX] torch.cuda.amp.autocast deprecated → torch.amp.autocast("cuda")
            with torch.amp.autocast("cuda", dtype=self.config.amp_dtype):
                yield
        else:
            with nullcontext():
                yield

    # ------------------------------------------------------------------
    # Token Dağılım İstatistikleri
    # ------------------------------------------------------------------

    def _compute_token_distribution(
        self,
        targets: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Token dağılım istatistiklerini hesaplar.

        İzlenen Metrikler:
            - eos_ratio     : EOS token oranı (çok yüksekse model erken bitiyor)
            - pad_ratio     : Dolgu token oranı (veri doluluk oranı)
            - content_ratio : İçerik token oranı (= 1 - eos_ratio - pad_ratio)
            - unique_tokens : Kullanılan benzersiz token sayısı

        Amaç:
            - EOS oranı izleme: Model, yanıtları erkenden mi bitiriyor?
            - Yüksek EOS oranı → min_response_tokens artırılmalı

        Args:
            targets: Hedef token ID'leri [batch, seq_len]
            logits : Model çıktıları (opsiyonel, tahmin dağılımı için)

        Returns:
            Token dağılım istatistikleri sözlüğü
        """
        with torch.no_grad():
            cfg = self.config
            total = targets.numel()

            if total == 0:
                return {
                    "eos_ratio": 0.0,
                    "pad_ratio": 0.0,
                    "content_ratio": 0.0,
                    "unique_tokens": 0,
                }

            eos_count = (targets == cfg.eos_token_id).sum().item()
            pad_count = (targets == cfg.pad_token_id).sum().item()
            content_count = total - eos_count - pad_count

            # Benzersiz token sayısı (kelime dağarcığı kullanım çeşitliliği)
            unique_tokens = targets.unique().numel()

            stats = {
                "eos_ratio": eos_count / total,
                "pad_ratio": pad_count / total,
                "content_ratio": max(0.0, content_count / total),
                "unique_tokens": unique_tokens,
            }

            # Logit verilmişse tahmin dağılımını da hesapla
            if logits is not None:
                predicted = logits.argmax(dim=-1)
                pred_eos = (predicted == cfg.eos_token_id).sum().item()
                stats["pred_eos_ratio"] = pred_eos / total

        return stats

    # ------------------------------------------------------------------
    # Toplu Metrik Biriktiricisi
    # ------------------------------------------------------------------

    class _MetricAccumulator:
        """
        Epoch boyunca batch metriklerini biriktiren yardımcı sınıf.
        Weighted average hesabı için token sayısını da takip eder.
        """

        def __init__(self):
            self.reset()

        def reset(self):
            self.total_loss = 0.0
            self.total_ce = 0.0
            self.total_focal = 0.0
            self.total_entropy_reg = 0.0
            self.total_auxiliary = 0.0
            self.total_accuracy = 0.0
            self.total_entropy = 0.0
            self.total_grad_norm = 0.0
            self.total_tokens = 0
            self.n_batches = 0
            self.token_dist_sum: Dict = {}
            self.n_token_dist_samples = 0
            self.start_time = time.time()

        def update(
            self,
            loss_output: LossOutput,
            accuracy: float,
            entropy: float,
            grad_norm: float,
            n_tokens: int,
            token_dist: Optional[Dict] = None,
        ):
            self.total_loss += loss_output["total"].item()
            self.total_ce += loss_output["ce"]
            self.total_focal += loss_output["focal"]
            self.total_entropy_reg += loss_output["entropy_reg"]
            self.total_auxiliary += loss_output["auxiliary"]
            self.total_accuracy += accuracy
            self.total_entropy += entropy
            self.total_grad_norm += grad_norm
            self.total_tokens += n_tokens
            self.n_batches += 1

            if token_dist:
                for k, v in token_dist.items():
                    self.token_dist_sum[k] = self.token_dist_sum.get(k, 0.0) + v
                self.n_token_dist_samples += 1

        def compute(self) -> EpochMetrics:
            n = max(self.n_batches, 1)
            elapsed = max(time.time() - self.start_time, 1e-6)

            avg_ce = self.total_ce / n
            avg_token_dist = {
                k: v / max(self.n_token_dist_samples, 1)
                for k, v in self.token_dist_sum.items()
            }

            return EpochMetrics(
                loss=self.total_loss / n,
                accuracy=self.total_accuracy / n,
                perplexity=self._safe_exp(avg_ce),
                entropy=self.total_entropy / n,
                gradient_norm=self.total_grad_norm / n,
                tokens_per_sec=self.total_tokens / elapsed,
                loss_breakdown={
                    "ce": avg_ce,
                    "focal": self.total_focal / n,
                    "entropy_reg": self.total_entropy_reg / n,
                    "auxiliary": self.total_auxiliary / n,
                },
                token_dist=avg_token_dist,
            )

        @staticmethod
        def _safe_exp(x: float) -> float:
            try:
                return min(float("inf"), 2.718281828 ** min(x, 88.0))
            except (OverflowError, ValueError):
                return float("inf")

    # ------------------------------------------------------------------
    # Eğitim Epoch'u
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        data_loader: Any,
        epoch: int = 0,
    ) -> EpochMetrics:
        """
        Tek bir eğitim epoch'u çalıştırır.

        İşlem Adımları (her batch için):
            1. Batch'i (inputs, targets) çiftine dönüştür
            2. Zamanlanmış örnekleme uygula (tf_prob < 1.0 ise)
            3. AMP context'te model ileri geçişi
            4. Bileşik kayıp hesapla
            5. AMP scaler ile geriye yayılım
            6. Gradyan birikimi kontrolü (her grad_accum_steps'te adım at)
            7. Gradyan gürültüsü ekle (aktifse)
            8. Gradyan kırp (aktifse)
            9. Optimizer adımı + scaler güncelleme
            10. Her log_token_dist_every batch'te token dağılımı hesapla

        Args:
            data_loader: Eğitim veri yükleyicisi (iteratör)
            epoch      : Mevcut epoch numarası (zamanlanmış örnekleme için)

        Returns:
            EpochMetrics: Epoch metrikleri
        """
        self.model.train()
        accumulator = self._MetricAccumulator()
        cfg = self.config

        # Zamanlanmış örnekleme olasılığını hesapla
        tf_prob = 1.0
        if cfg.use_scheduled_sampling:
            tf_prob = self.compute_teacher_forcing_prob(
                epoch=epoch,
                scheduled_sampling_epochs=cfg.scheduled_sampling_epochs,
                min_teacher_forcing=cfg.min_teacher_forcing_prob,
                initial_teacher_forcing=cfg.initial_teacher_forcing,
            )
            if epoch % 5 == 0 or epoch < 3:
                logger.info(
                    f"Epoch {epoch}: Öğretmen zorlaması olasılığı = {tf_prob:.4f}"
                )

        # Gradyan birikimi için sayaç
        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, raw_batch in enumerate(data_loader):
            # ----------------------------------------------------------
            # 1. Batch işle
            # ----------------------------------------------------------
            try:
                inputs, targets = self.batch_processor.process(raw_batch)
            except Exception as e:
                logger.warning(f"Batch {batch_idx} işlenemedi: {e}. Atlanıyor.")
                continue

            n_tokens = inputs.numel()

            # ----------------------------------------------------------
            # 2. Zamanlanmış Örnekleme
            # ----------------------------------------------------------
            if cfg.use_scheduled_sampling and tf_prob < 1.0:
                inputs = self.apply_scheduled_sampling(
                    model=self.model,
                    inputs=inputs,
                    tf_prob=tf_prob,
                    device=self.device,
                )

            # ----------------------------------------------------------
            # 3. İleri Geçiş (AMP ile)
            # ----------------------------------------------------------
            with self._amp_context():
                try:
                    model_output = self.model(inputs)
                except Exception as e:
                    logger.error(f"İleri geçiş hatası (batch {batch_idx}): {e}")
                    continue

                # Model 3-demet döndürebilir: (logits, moe_loss, mod_loss)
                if isinstance(model_output, (tuple, list)):
                    logits = model_output[0]
                    # Yardımcı kayıpları topla
                    aux_losses = [
                        x for x in model_output[1:]
                        if isinstance(x, torch.Tensor)
                    ]
                    aux_loss = sum(aux_losses) if aux_losses else None
                else:
                    logits = model_output
                    aux_loss = None

                # --------------------------------------------------
                # 4. Kayıp Hesapla
                # --------------------------------------------------
                try:
                    loss_output: LossOutput = self.loss_manager.compute(
                        logits=logits,
                        targets=targets,
                        aux_loss=aux_loss,
                    )
                except Exception as e:
                    logger.error(f"Kayıp hesaplama hatası (batch {batch_idx}): {e}")
                    continue

                # Gradyan birikimi için normalize et
                normalized_loss = loss_output["total"] / cfg.grad_accum_steps

            # ----------------------------------------------------------
            # 5. Geriye Yayılım
            # ----------------------------------------------------------
            if self._use_amp and self.scaler is not None:
                self.scaler.scale(normalized_loss).backward()
            else:
                normalized_loss.backward()

            # [MEM-CRITICAL] Backward tamamlandı — logits ve normalized_loss serbest bırak.
            # logits: [B, T, V] float16 ≈ 3-4 GB; normalized_loss: skaler ama graph tutabilir.
            # Metrikler için detached kopyalar kaydediliyor, orjinaller artık gereksiz.
            # Bu DEL olmadan Python GC, yeni batch forward'ı başlamadan eski logits'i silmez
            # → eski + yeni logits aynı anda bellekte → OOM.
            _logits_for_metrics = logits.detach()  # Graph-free kopya (metrikler için)
            del logits, normalized_loss              # Computation graph ve logits serbest

            # ----------------------------------------------------------
            # 6. Gradyan Birikimi Kontrolü
            # ----------------------------------------------------------
            is_accumulation_step = (batch_idx + 1) % cfg.grad_accum_steps == 0
            is_last_batch = (batch_idx + 1 == len(data_loader)) if hasattr(data_loader, "__len__") else False

            if is_accumulation_step or is_last_batch:
                # Scaler'ı geri al (AMP için NaN/Inf kontrolü)
                if self._use_amp and self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                # Gradyan gürültüsü (Neelakantan et al. 2015)
                if cfg.use_gradient_noise:
                    self.gradient_manager.inject_gradient_noise(
                        model=self.model,
                        eta=cfg.noise_eta,
                        step=self.global_step + 1,
                    )

                # Gradyan kırpma
                grad_norm = 0.0
                if cfg.max_grad_norm is not None and cfg.max_grad_norm > 0:
                    if cfg.use_adaptive_clip:
                        # AGC: Brock et al. 2021
                        self.gradient_manager.adaptive_clip_gradients(
                            self.model, agc_lambda=cfg.agc_lambda
                        )
                        grad_norm = self.gradient_manager.calculate_gradient_norm(self.model)
                    else:
                        # Standart küresel kırpma
                        grad_norm = self.gradient_manager.clip_gradients(
                            self.model, max_norm=cfg.max_grad_norm
                        )
                else:
                    grad_norm = self.gradient_manager.calculate_gradient_norm(self.model)

                # Optimizer adımı
                if self._use_amp and self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # Gradyanları sıfırla
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                # [MEM] Fragmented serbest blokları geri al
                torch.cuda.empty_cache()

            else:
                # Birikim adımı değilse norm hesapla ama optimizer adımı atma
                grad_norm = self.gradient_manager.calculate_gradient_norm(self.model)

            # ----------------------------------------------------------
            # 7. Batch Metrikleri
            # ----------------------------------------------------------
            with torch.no_grad():
                accuracy = self.loss_manager.compute_accuracy(
                    logits=_logits_for_metrics,
                    targets=targets,
                    pad_id=cfg.pad_token_id,
                )
                entropy = compute_entropy(_logits_for_metrics)

            # Token dağılımı (her log_token_dist_every batch'te)
            token_dist = None
            if batch_idx % cfg.log_token_dist_every == 0:
                token_dist = self._compute_token_distribution(
                    targets=targets,
                    logits=_logits_for_metrics,
                )

            del _logits_for_metrics  # Metrikler hesaplandı, artık gereksiz

            accumulator.update(
                loss_output=loss_output,
                accuracy=accuracy,
                entropy=entropy,
                grad_norm=grad_norm,
                n_tokens=n_tokens,
                token_dist=token_dist,
            )

            # Ara loglama
            if batch_idx % cfg.log_every_n_batches == 0:
                logger.debug(
                    f"Epoch {epoch} | Batch {batch_idx} | "
                    f"Loss: {loss_output['total'].item():.4f} | "
                    f"Acc: {accuracy:.4f} | "
                    f"GradNorm: {grad_norm:.4f} | "
                    f"TF: {tf_prob:.3f}"
                )

        return accumulator.compute()

    # ------------------------------------------------------------------
    # Doğrulama Epoch'u
    # ------------------------------------------------------------------

    def validate_epoch(
        self,
        data_loader: Any,
    ) -> EpochMetrics:
        """
        Tek bir doğrulama epoch'u çalıştırır.

        Eğitim döngüsünden farkları:
            - model.eval() modu (Dropout/BatchNorm devre dışı)
            - torch.no_grad() (geriye yayılım yok, bellek tasarrufu)
            - Zamanlanmış örnekleme yok (gerçek hedefler kullanılır)
            - Gradyan gürültüsü/kırpma yok
            - Optimizer adımı yok

        Args:
            data_loader: Doğrulama veri yükleyicisi

        Returns:
            EpochMetrics: Doğrulama metrikleri
        """
        self.model.eval()
        accumulator = self._MetricAccumulator()
        cfg = self.config

        with torch.no_grad():
            for batch_idx, raw_batch in enumerate(data_loader):
                # --------------------------------------------------
                # Batch işle
                # --------------------------------------------------
                try:
                    inputs, targets = self.batch_processor.process(raw_batch)
                except Exception as e:
                    logger.warning(f"Doğrulama batch {batch_idx} işlenemedi: {e}. Atlanıyor.")
                    continue

                n_tokens = inputs.numel()

                # --------------------------------------------------
                # İleri Geçiş (AMP opsiyonel, geri yayılım yok)
                # --------------------------------------------------
                try:
                    with self._amp_context():
                        model_output = self.model(inputs)
                except Exception as e:
                    logger.error(f"Doğrulama ileri geçiş hatası (batch {batch_idx}): {e}")
                    continue

                # Çıktı ayrıştırma
                if isinstance(model_output, (tuple, list)):
                    logits = model_output[0]
                    aux_losses = [
                        x for x in model_output[1:]
                        if isinstance(x, torch.Tensor)
                    ]
                    aux_loss = sum(aux_losses) if aux_losses else None
                else:
                    logits = model_output
                    aux_loss = None

                # --------------------------------------------------
                # Kayıp Hesapla
                # --------------------------------------------------
                try:
                    loss_output: LossOutput = self.loss_manager.compute(
                        logits=logits,
                        targets=targets,
                        aux_loss=aux_loss,
                    )
                except Exception as e:
                    logger.error(f"Doğrulama kaybı hatası (batch {batch_idx}): {e}")
                    continue

                # Metrikler
                accuracy = self.loss_manager.compute_accuracy(
                    logits=logits,
                    targets=targets,
                    pad_id=cfg.pad_token_id,
                )
                entropy = compute_entropy(logits)

                # Token dağılımı
                token_dist = None
                if batch_idx % cfg.log_token_dist_every == 0:
                    token_dist = self._compute_token_distribution(
                        targets=targets,
                        logits=logits,
                    )

                accumulator.update(
                    loss_output=loss_output,
                    accuracy=accuracy,
                    entropy=entropy,
                    grad_norm=0.0,     # Doğrulamada gradyan yok
                    n_tokens=n_tokens,
                    token_dist=token_dist,
                )

                if batch_idx % cfg.log_every_n_batches == 0:
                    logger.debug(
                        f"Doğrulama | Batch {batch_idx} | "
                        f"Loss: {loss_output['total'].item():.4f} | "
                        f"Acc: {accuracy:.4f}"
                    )

        return accumulator.compute()

    # ------------------------------------------------------------------
    # Durum Yönetimi
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict:
        """
        TrainingLoop durumunu serileştirir (checkpoint için).

        Returns:
            global_step ve scaler durumunu içeren sözlük
        """
        state = {"global_step": self.global_step}
        if self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()
        return state

    def load_state_dict(self, state: Dict) -> None:
        """
        Checkpoint'ten TrainingLoop durumunu yükler.

        Args:
            state: state_dict() ile kaydedilmiş durum sözlüğü
        """
        self.global_step = state.get("global_step", 0)
        if self.scaler is not None and "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])

    def get_current_tf_prob(self, epoch: int) -> float:
        """
        Mevcut epoch için öğretmen zorlaması olasılığını döndürür.

        Args:
            epoch: Mevcut epoch numarası

        Returns:
            tf_prob: Öğretmen zorlaması olasılığı
        """
        if not self.config.use_scheduled_sampling:
            return 1.0
        return self.compute_teacher_forcing_prob(
            epoch=epoch,
            scheduled_sampling_epochs=self.config.scheduled_sampling_epochs,
            min_teacher_forcing=self.config.min_teacher_forcing_prob,
            initial_teacher_forcing=self.config.initial_teacher_forcing,
        )

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"TrainingLoop("
            f"device={cfg.device}, "
            f"AMP={'ON' if self._use_amp else 'OFF'}, "
            f"grad_accum={cfg.grad_accum_steps}, "
            f"scheduled_sampling={'ON' if cfg.use_scheduled_sampling else 'OFF'}, "
            f"global_step={self.global_step}"
            f")"
        )
