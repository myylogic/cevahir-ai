"""
Curriculum Learning Manager
============================
Referanslar:
    - Bengio et al. 2009, "Curriculum Learning" (ICML 2009)
      https://dl.acm.org/doi/10.1145/1553374.1553380

    - Elman 1993, "Learning and development in neural networks:
      The importance of starting small"
      https://doi.org/10.1016/0010-0277(93)90058-4

    - Bengio et al. 2015, "Scheduled Sampling for Sequence Prediction
      with Recurrent Neural Networks" (NeurIPS 2015)
      https://arxiv.org/abs/1506.03099

    - Platanios et al. 2019, "Competence-based Curriculum Learning for
      Neural Machine Translation" (NAACL 2019)
      https://arxiv.org/abs/1903.09848

Curriculum Learning Motivasyonu:
    İnsan öğrenmesinden esinlenen bu yaklaşım, modeli önce kolay örneklerle
    eğitir; ardından giderek daha zor örneklere geçer. Bu sayede:

    1. Training stabilitesi artar (erken aşamalarda büyük gradient'lerden kaçınılır)
    2. Convergence hızlanır (kolay örneklerin gradients'leri daha bilgilendiricidir)
    3. Final generalization genellikle daha iyi olur
    4. Catastrophic forgetting riski azalır

Desteklenen Stratejiler:

    LENGTH_BASED (Sequence uzunluğuna göre):
        - Kısa sequence'lardan başla, kademeli uzat
        - current_max_len = min_len + (max_len - min_len) * progress
        - Dil modelleri için yaygın yaklaşım

    LOSS_BASED (Loss değerine göre):
        - Düşük loss'lu örnekler (kolay) → yüksek loss'lu örnekler (zor)
        - Örnekler periyodik olarak yeniden sıralanır
        - Self-paced learning ile yakın ilişkili

    RANDOM:
        - Curriculum kullanılmaz (standart random sampling)
        - Baseline karşılaştırması için

Dynamic Sequence Length (Dinamik Sequence Uzunluğu):
    - Model, training süresince artan max_seq_length ile beslenir
    - Bellek verimliliği sağlar (erken aşamalarda kısa seq → küçük attention matris)
    - start_seq_len=128 → max_seq_len=512 lineer artış

Scheduled Sampling (Teacher Forcing Azaltma):
    - Bengio et al. 2015 yöntemi
    - Başlangıçta yüksek teacher forcing (model stable değilken)
    - Kademeli azaltma → model kendi tahminlerine daha fazla güvenir
    - tf_prob = max(min_tf, 1 - (epoch - ss_start) / (ss_end - ss_start) * (1 - min_tf))

Kullanım:
    config = CurriculumConfig(
        strategy=CurriculumStrategy.LENGTH_BASED,
        curriculum_epochs=20,
        min_seq_len=64,
        max_seq_len=512,
    )
    manager = CurriculumManager(config)

    for epoch in range(num_epochs):
        max_len = manager.get_current_max_len(epoch)
        filtered_dl = manager.filter_dataloader(dataloader, epoch)
        tf_prob = manager.get_teacher_forcing_prob(epoch)

        for batch in filtered_dl:
            # ... training loop ...
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset

logger = logging.getLogger(__name__)


# ===========================================================================
# Enumerasyon: Curriculum stratejileri
# ===========================================================================

class CurriculumStrategy(Enum):
    """Desteklenen curriculum learning stratejileri."""

    LENGTH_BASED = "length"
    """Kısa sequence'lardan uzuna doğru curriculum.
    Dil modeli training için en yaygın yaklaşım."""

    LOSS_BASED = "loss"
    """Kolay (düşük loss) örneklerden zor (yüksek loss) örneklere.
    Self-paced learning ile ilişkili."""

    RANDOM = "random"
    """Curriculum yok: standart rastgele örnekleme.
    Baseline karşılaştırması için."""


# ===========================================================================
# Konfigürasyon dataclass'ı
# ===========================================================================

@dataclass
class CurriculumConfig:
    """
    CurriculumManager için konfigürasyon parametreleri.

    Args:
        strategy: Curriculum stratejisi.
        curriculum_epochs: Curriculum'ın kaç epoch süreceği.
                          Bu epoch'tan sonra tüm örnekler kullanılır.
        min_seq_len: Başlangıç maksimum sequence uzunluğu (LENGTH_BASED).
        max_seq_len: Hedef (son) maksimum sequence uzunluğu (LENGTH_BASED).
        use_dynamic_seq_len: Dinamik sequence uzunluğu kullan.
        start_seq_len: Dinamik seq len başlangıç değeri.
        scheduled_sampling: Scheduled sampling (teacher forcing azaltma) kullan.
        min_teacher_forcing: En düşük teacher forcing olasılığı.
        ss_start_epoch: Scheduled sampling'in başlayacağı epoch.
        ss_end_epoch: Scheduled sampling'in tamamlanacağı epoch
                      (bu epochtan sonra min_teacher_forcing sabit kalır).
        loss_score_fn: LOSS_BASED strateji için örnek zorluk skoru fonksiyonu.
                       (örnek_index) -> float döndürmelidir.
        loss_update_interval: Loss skorlarının kaç epochta bir güncelleneceği.
    """

    # --- Genel parametreler ---
    strategy: CurriculumStrategy = CurriculumStrategy.LENGTH_BASED
    curriculum_epochs: int = 20

    # --- Uzunluk tabanlı parametreler ---
    min_seq_len: int = 64
    max_seq_len: int = 512

    # --- Dinamik sequence length ---
    use_dynamic_seq_len: bool = True
    start_seq_len: int = 128

    # --- Scheduled Sampling parametreleri ---
    scheduled_sampling: bool = True
    min_teacher_forcing: float = 0.3
    ss_start_epoch: int = 10
    ss_end_epoch: int = 60

    # --- Loss tabanlı parametreler ---
    loss_score_fn: Optional[Callable[[int], float]] = field(
        default=None,
        metadata={"help": "Örnek zorluk skoru fonksiyonu: idx -> float"},
    )
    loss_update_interval: int = 5

    def __post_init__(self) -> None:
        """Parametre doğrulaması."""
        if self.min_seq_len <= 0:
            raise ValueError(f"min_seq_len pozitif olmalıdır: {self.min_seq_len}")
        if self.max_seq_len < self.min_seq_len:
            raise ValueError(
                f"max_seq_len ({self.max_seq_len}) >= min_seq_len "
                f"({self.min_seq_len}) olmalıdır."
            )
        if self.curriculum_epochs < 0:
            raise ValueError(
                f"curriculum_epochs negatif olamaz: {self.curriculum_epochs}"
            )
        if not (0.0 <= self.min_teacher_forcing <= 1.0):
            raise ValueError(
                f"min_teacher_forcing [0, 1] arasında olmalıdır: {self.min_teacher_forcing}"
            )
        if self.ss_start_epoch >= self.ss_end_epoch:
            raise ValueError(
                f"ss_start_epoch ({self.ss_start_epoch}) < ss_end_epoch "
                f"({self.ss_end_epoch}) olmalıdır."
            )
        if self.strategy == CurriculumStrategy.LOSS_BASED and self.loss_score_fn is None:
            logger.warning(
                "LOSS_BASED strateji seçildi ancak loss_score_fn verilmedi. "
                "filter_dataloader çağrısında bu fonksiyon sağlanmalıdır."
            )


# ===========================================================================
# Ana CurriculumManager sınıfı
# ===========================================================================

class CurriculumManager:
    """
    Curriculum Learning yöneticisi.

    Eğitimi kolay örneklerden zora doğru organize eder.
    Sequence uzunluğu, loss skorları veya rastgele örnekleme stratejilerini
    destekler. Scheduled Sampling ile Teacher Forcing kademeli azaltmayı da yönetir.

    Args:
        config (CurriculumConfig): Curriculum konfigürasyonu.

    Referans:
        Bengio et al. 2009 (Curriculum Learning)
        Bengio et al. 2015 (Scheduled Sampling)
        Elman 1993 (Starting Small)
    """

    def __init__(self, config: CurriculumConfig) -> None:
        self.config = config

        # Loss skorları (LOSS_BASED strateji için)
        self._loss_scores: Optional[Dict[int, float]] = None
        self._last_loss_update_epoch: int = -1

        logger.info(
            "CurriculumManager başlatıldı: strategy=%s, curriculum_epochs=%d, "
            "seq_len=[%d, %d]",
            config.strategy.value,
            config.curriculum_epochs,
            config.min_seq_len,
            config.max_seq_len,
        )

    # ------------------------------------------------------------------
    # Sequence uzunluğu yönetimi
    # ------------------------------------------------------------------

    def get_current_max_len(self, epoch: int) -> int:
        """
        Verilen epoch için maksimum sequence uzunluğunu hesaplar.

        LENGTH_BASED veya use_dynamic_seq_len aktifse lineer artış:
            progress = epoch / curriculum_epochs  [0, 1]
            max_len = min_len + (max_len - min_len) * progress

        curriculum_epochs'tan sonra max_seq_len sabit kalır.

        Args:
            epoch (int): Mevcut epoch numarası (0-indexed).

        Returns:
            int: Geçerli epoch için maksimum sequence uzunluğu.
        """
        if epoch < 0:
            raise ValueError(f"epoch negatif olamaz: {epoch}")

        # Curriculum tamamlandıysa veya strateji ilgisizse max_seq_len döndür
        if (
            epoch >= self.config.curriculum_epochs
            or self.config.strategy == CurriculumStrategy.RANDOM
        ):
            return self.config.max_seq_len

        # Dinamik seq len: start_seq_len'den max_seq_len'e lineer artış
        if self.config.use_dynamic_seq_len:
            start = self.config.start_seq_len
        else:
            start = self.config.min_seq_len

        # Lineer interpolasyon
        progress = epoch / max(self.config.curriculum_epochs, 1)
        current_max = start + (self.config.max_seq_len - start) * progress
        current_max = int(math.ceil(current_max))

        # [min_seq_len, max_seq_len] aralığında tut
        current_max = max(self.config.min_seq_len, min(current_max, self.config.max_seq_len))

        logger.debug(
            "Epoch %d için max_seq_len=%d (progress=%.2f)",
            epoch, current_max, progress,
        )
        return current_max

    def should_include_sample(self, sample_len: int, epoch: int) -> bool:
        """
        Belirli uzunluktaki bir örneğin mevcut epoch'ta dahil edilip
        edilmeyeceğini belirler (LENGTH_BASED strateji).

        Args:
            sample_len (int): Örneğin sequence uzunluğu (token sayısı).
            epoch (int): Mevcut epoch numarası.

        Returns:
            bool: True → örnek dahil edilir, False → atlanır.

        Not:
            RANDOM stratejisinde her zaman True döndürür.
            curriculum_epochs sonrasında her zaman True döndürür.
        """
        if self.config.strategy == CurriculumStrategy.RANDOM:
            return True

        if epoch >= self.config.curriculum_epochs:
            return True

        if self.config.strategy != CurriculumStrategy.LENGTH_BASED:
            return True

        current_max = self.get_current_max_len(epoch)
        return sample_len <= current_max

    # ------------------------------------------------------------------
    # DataLoader filtreleme
    # ------------------------------------------------------------------

    def filter_dataloader(
        self,
        dataloader: DataLoader,
        epoch: int,
        sample_len_fn: Optional[Callable[[Any], int]] = None,
    ) -> DataLoader:
        """
        Curriculum stratejisine göre DataLoader'ı filtreler.

        LENGTH_BASED: Mevcut max_len'den uzun örnekleri filtreler.
        LOSS_BASED: Loss skorlarına göre sıralı Subset döndürür.
        RANDOM: Orijinal DataLoader döndürür (değişiklik yok).

        Args:
            dataloader (DataLoader): Filtrelenecek orijinal DataLoader.
            epoch (int): Mevcut epoch numarası.
            sample_len_fn (Callable, optional): Örnekten uzunluk çıkaran fonksiyon.
                Örnek: lambda sample: len(sample['input_ids'])
                LENGTH_BASED strateji için gerekli. Verilmezse uzunluk
                tahmini yapılamaz ve uyarı verilir.

        Returns:
            DataLoader: Filtrelenmiş (veya orijinal) DataLoader.

        Not:
            Döndürülen DataLoader orijinalin batch_size ve num_workers
            ayarlarını miras alır.
        """
        cfg = self.config

        # RANDOM: değişiklik yok
        if cfg.strategy == CurriculumStrategy.RANDOM:
            return dataloader

        # Curriculum tamamlandı: değişiklik yok
        if epoch >= cfg.curriculum_epochs:
            logger.debug(
                "Epoch %d >= curriculum_epochs %d: tüm örnekler kullanılıyor.",
                epoch, cfg.curriculum_epochs,
            )
            return dataloader

        dataset = dataloader.dataset

        # ---------------------------------------------------------------
        # LENGTH_BASED filtreleme
        # ---------------------------------------------------------------
        if cfg.strategy == CurriculumStrategy.LENGTH_BASED:
            current_max = self.get_current_max_len(epoch)

            if sample_len_fn is None:
                logger.warning(
                    "LENGTH_BASED strateji için sample_len_fn verilmedi. "
                    "Filtreleme yapılamıyor, orijinal DataLoader döndürülüyor."
                )
                return dataloader

            # Dahil edilecek örnek indekslerini bul
            included_indices: List[int] = []
            for idx in range(len(dataset)):  # type: ignore[arg-type]
                try:
                    sample = dataset[idx]
                    sample_len = sample_len_fn(sample)
                    if sample_len <= current_max:
                        included_indices.append(idx)
                except Exception as exc:
                    logger.warning(
                        "Örnek %d uzunluğu hesaplanamadı: %s. Dahil edilmiyor.",
                        idx, exc,
                    )

            if not included_indices:
                logger.warning(
                    "Epoch %d: Hiçbir örnek max_len=%d filtresini geçemedi. "
                    "Orijinal DataLoader döndürülüyor.",
                    epoch, current_max,
                )
                return dataloader

            subset = Subset(dataset, included_indices)
            filtered_dl = DataLoader(
                subset,
                batch_size=dataloader.batch_size,
                shuffle=True,
                num_workers=dataloader.num_workers,
                collate_fn=dataloader.collate_fn,
                pin_memory=dataloader.pin_memory,
                drop_last=dataloader.drop_last,
            )

            logger.info(
                "Epoch %d: LENGTH_BASED filtre uygulandı. "
                "%d / %d örnek dahil (max_len=%d).",
                epoch,
                len(included_indices),
                len(dataset),  # type: ignore[arg-type]
                current_max,
            )
            return filtered_dl

        # ---------------------------------------------------------------
        # LOSS_BASED filtreleme
        # ---------------------------------------------------------------
        elif cfg.strategy == CurriculumStrategy.LOSS_BASED:
            return self._filter_loss_based(dataloader, epoch)

        return dataloader

    def _filter_loss_based(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> DataLoader:
        """
        Loss tabanlı curriculum filtreleme (dahili yardımcı).

        Loss skorları periyodik olarak güncellenir (loss_update_interval).
        Her epoch'ta en kolay (düşük loss) örnekler seçilir; progress
        arttıkça daha zor örnekler eklenir.

        Args:
            dataloader (DataLoader): Orijinal DataLoader.
            epoch (int): Mevcut epoch.

        Returns:
            DataLoader: Loss sıralı Subset DataLoader.
        """
        cfg = self.config
        dataset = dataloader.dataset
        n_samples = len(dataset)  # type: ignore[arg-type]

        # Loss skorlarını güncelle (gerekirse)
        if (
            self._loss_scores is None
            or epoch - self._last_loss_update_epoch >= cfg.loss_update_interval
        ):
            if cfg.loss_score_fn is not None:
                logger.info(
                    "Epoch %d: Loss skorları güncelleniyor (%d örnek)...",
                    epoch, n_samples,
                )
                self._loss_scores = {
                    idx: cfg.loss_score_fn(idx) for idx in range(n_samples)
                }
                self._last_loss_update_epoch = epoch
            else:
                logger.warning(
                    "LOSS_BASED strateji için loss_score_fn verilmedi. "
                    "Orijinal DataLoader döndürülüyor."
                )
                return dataloader

        # Progress: curriculum'ın ne kadarı tamamlandı
        progress = min(1.0, epoch / max(cfg.curriculum_epochs, 1))

        # Dahil edilecek örnek sayısı (kolay→zor sıralamasında ilk %X)
        # Başlangıçta az sayıda (kolay) örnek, sonunda tümü
        n_include = max(1, int(n_samples * (0.3 + 0.7 * progress)))

        # Loss skoruna göre sırala (düşük loss = kolay)
        sorted_indices = sorted(
            self._loss_scores.keys(),
            key=lambda idx: self._loss_scores[idx],  # type: ignore[index]
        )
        included_indices = sorted_indices[:n_include]

        subset = Subset(dataset, included_indices)
        filtered_dl = DataLoader(
            subset,
            batch_size=dataloader.batch_size,
            shuffle=True,
            num_workers=dataloader.num_workers,
            collate_fn=dataloader.collate_fn,
            pin_memory=dataloader.pin_memory,
            drop_last=dataloader.drop_last,
        )

        logger.info(
            "Epoch %d: LOSS_BASED filtre uygulandı. "
            "%d / %d örnek dahil (progress=%.2f).",
            epoch, n_include, n_samples, progress,
        )
        return filtered_dl

    # ------------------------------------------------------------------
    # Teacher Forcing / Scheduled Sampling
    # ------------------------------------------------------------------

    def get_teacher_forcing_prob(self, epoch: int) -> float:
        """
        Mevcut epoch için teacher forcing olasılığını hesaplar.

        Scheduled Sampling (Bengio et al. 2015):
            - epoch < ss_start_epoch: tf_prob = 1.0 (tam teacher forcing)
            - ss_start_epoch <= epoch <= ss_end_epoch: lineer azalma
            - epoch > ss_end_epoch: tf_prob = min_teacher_forcing

        Teacher forcing azaltılmayan scheduled_sampling=False durumunda
        her zaman 1.0 döndürür.

        Lineer decay formülü:
            decay_length = ss_end_epoch - ss_start_epoch
            progress = (epoch - ss_start_epoch) / decay_length  [0, 1]
            tf_prob = 1.0 - progress * (1.0 - min_teacher_forcing)
                    = max(min_teacher_forcing, 1.0 - progress * (1 - min_tf))

        Args:
            epoch (int): Mevcut epoch numarası.

        Returns:
            float: Teacher forcing olasılığı [min_teacher_forcing, 1.0].
        """
        cfg = self.config

        # Scheduled sampling devre dışı
        if not cfg.scheduled_sampling:
            return 1.0

        # Scheduled sampling henüz başlamadı
        if epoch < cfg.ss_start_epoch:
            return 1.0

        # Scheduled sampling tamamlandı
        if epoch >= cfg.ss_end_epoch:
            return cfg.min_teacher_forcing

        # Lineer azalma aşaması
        decay_length = cfg.ss_end_epoch - cfg.ss_start_epoch
        progress = (epoch - cfg.ss_start_epoch) / max(decay_length, 1)

        # tf_prob: 1.0'dan min_teacher_forcing'e lineer düşüş
        tf_prob = 1.0 - progress * (1.0 - cfg.min_teacher_forcing)
        tf_prob = max(cfg.min_teacher_forcing, min(1.0, tf_prob))

        logger.debug(
            "Epoch %d: teacher_forcing_prob=%.4f (progress=%.2f)",
            epoch, tf_prob, progress,
        )
        return tf_prob

    # ------------------------------------------------------------------
    # Curriculum progress
    # ------------------------------------------------------------------

    def get_curriculum_progress(self, epoch: int) -> float:
        """
        Curriculum'ın tamamlanma oranını döndürür [0.0, 1.0].

        Args:
            epoch (int): Mevcut epoch.

        Returns:
            float: 0.0 = başlangıç, 1.0 = tamamlandı.
        """
        if self.config.curriculum_epochs <= 0:
            return 1.0
        return min(1.0, epoch / self.config.curriculum_epochs)

    def is_curriculum_complete(self, epoch: int) -> bool:
        """
        Curriculum aşamasının tamamlanıp tamamlanmadığını döndürür.

        Args:
            epoch (int): Mevcut epoch.

        Returns:
            bool: True → curriculum tamamlandı (tüm örnekler kullanılıyor).
        """
        return epoch >= self.config.curriculum_epochs

    # ------------------------------------------------------------------
    # Loss skoru güncelleme (LOSS_BASED için)
    # ------------------------------------------------------------------

    def update_loss_scores(self, scores: Dict[int, float], epoch: int) -> None:
        """
        Örnek loss skorlarını manuel olarak günceller (LOSS_BASED strateji).

        Training loop'ta her belirli periyotta bu metod çağrılabilir.

        Args:
            scores (Dict[int, float]): {örnek_index: loss_skoru} sözlüğü.
            epoch (int): Güncellemenin yapıldığı epoch.

        Örnek:
            # Her loss_update_interval epoch'ta bir:
            sample_losses = {
                idx: compute_sample_loss(model, dataset[idx])
                for idx in range(len(dataset))
            }
            curriculum_manager.update_loss_scores(sample_losses, current_epoch)
        """
        self._loss_scores = dict(scores)
        self._last_loss_update_epoch = epoch
        logger.info(
            "Loss skorları güncellendi: %d örnek, epoch=%d, "
            "min_loss=%.4f, max_loss=%.4f",
            len(scores),
            epoch,
            min(scores.values()) if scores else 0.0,
            max(scores.values()) if scores else 0.0,
        )

    # ------------------------------------------------------------------
    # Özet bilgisi
    # ------------------------------------------------------------------

    def get_status(self, epoch: int) -> Dict[str, Any]:
        """
        Mevcut curriculum durumunu özet olarak döndürür.

        Args:
            epoch (int): Mevcut epoch.

        Returns:
            Dict: Curriculum durumu özet sözlüğü.
        """
        return {
            "strategy": self.config.strategy.value,
            "epoch": epoch,
            "curriculum_complete": self.is_curriculum_complete(epoch),
            "curriculum_progress": f"{self.get_curriculum_progress(epoch):.2%}",
            "current_max_len": self.get_current_max_len(epoch),
            "teacher_forcing_prob": f"{self.get_teacher_forcing_prob(epoch):.4f}",
            "dynamic_seq_len": self.config.use_dynamic_seq_len,
            "scheduled_sampling": self.config.scheduled_sampling,
        }

    def log_status(self, epoch: int) -> None:
        """Mevcut curriculum durumunu log'a yazar."""
        status = self.get_status(epoch)
        logger.info(
            "CurriculumManager durumu [epoch=%d]: "
            "strateji=%s, tamamlandı=%s, progress=%s, "
            "max_len=%d, tf_prob=%s",
            epoch,
            status["strategy"],
            status["curriculum_complete"],
            status["curriculum_progress"],
            status["current_max_len"],
            status["teacher_forcing_prob"],
        )

    def __repr__(self) -> str:
        return (
            f"CurriculumManager("
            f"strategy={self.config.strategy.value}, "
            f"curriculum_epochs={self.config.curriculum_epochs}, "
            f"seq_len=[{self.config.min_seq_len}, {self.config.max_seq_len}], "
            f"scheduled_sampling={self.config.scheduled_sampling})"
        )
