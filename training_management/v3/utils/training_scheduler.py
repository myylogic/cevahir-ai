"""
Training Scheduler - Gelişmiş Çoklu Strateji LR Scheduler
==========================================================
Referanslar:
    - Loshchilov & Hutter 2016, "SGDR: Stochastic Gradient Descent with
      Warm Restarts" (Cosine Restarts)
      https://arxiv.org/abs/1608.03983

    - Smith & Touvron 2018, "Super-Convergence: Very Fast Training of
      Neural Networks Using Large Learning Rates" (One Cycle)
      https://arxiv.org/abs/1708.07120

    - Howard & Ruder 2018, "Universal Language Model Fine-Tuning for
      Text Classification" (LLRD - ULMFiT)
      https://arxiv.org/abs/1801.06146

    - Devlin et al. 2019, "BERT: Pre-training of Deep Bidirectional
      Transformers" (Linear Warmup + Decay)
      https://arxiv.org/abs/1810.04805

Scheduler Stratejileri:
    COSINE_RESTARTS:
        Cosine annealing ile periyodik restart. Her T_0 epochta bir
        LR yeniden başlar ve her restart'ta periyot T_mult ile büyür.
        Büyük modeller için önerilen.

    ONE_CYCLE:
        LR önce artar (pct_start), sonra agresif azalır.
        Super-convergence için tasarlanmış. Kısa training'lerde etkili.

    CONSTANT_WITH_WARMUP:
        Warmup sonrası sabit LR. Fine-tuning için uygundur.

    LINEAR_DECAY:
        Warmup sonrası lineer azalma. BERT tarzı training için.

    REDUCE_ON_PLATEAU:
        Validation metric kötüleşince LR düşürür.
        Convergence noktası belirsizse kullanılır.

LLRD (Layer-wise Learning Rate Decay):
    Derin ağlarda alt katmanlar (embedding, erken layer'lar) zaten
    iyi özellikler öğrenmiştir; bu katmanlar yüksek LR ile
    bozulabilir. LLRD her katmana farklı LR atar:
        LR_layer_k = base_lr * llrd_decay^(total_layers - k)
    En alt katman en düşük LR'ye sahipken üst katmanlar base_lr ile güncellenir.

Kullanım:
    config = SchedulerConfig(
        scheduler_type=SchedulerType.COSINE_RESTARTS,
        warmup_steps=1500,
        T_0=10,
        T_mult=2,
    )
    scheduler = TrainingScheduler(optimizer, config)

    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.step()
            scheduler.step_batch()   # Warmup adımları için

        val_loss = evaluate()
        scheduler.step_epoch(metric=val_loss)  # Epoch sonunda
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type

import torch
import torch.optim as optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    LambdaLR,
    OneCycleLR,
    ReduceLROnPlateau,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Enumerasyon: Scheduler türleri
# ===========================================================================

class SchedulerType(Enum):
    """Desteklenen LR scheduler stratejileri."""
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    """Validation metric'e göre LR'yi otomatik düşürür."""

    COSINE_RESTARTS = "cosine_restarts"
    """Cosine annealing + warm restarts (Loshchilov & Hutter 2016)."""

    ONE_CYCLE = "one_cycle"
    """One-cycle LR policy (Smith & Touvron 2018)."""

    CONSTANT_WITH_WARMUP = "constant_warmup"
    """Warmup sonrası sabit LR. Fine-tuning için."""

    LINEAR_DECAY = "linear_decay"
    """Warmup sonrası lineer azalma. BERT tarzı training için."""


# ===========================================================================
# Konfigürasyon dataclass'ı
# ===========================================================================

@dataclass
class SchedulerConfig:
    """
    TrainingScheduler için konfigürasyon parametreleri.

    Tüm scheduler türleri için ortak ve özel parametreler içerir.
    İhtiyaç duyulmayan parametreler görmezden gelinir.

    Args:
        scheduler_type: Kullanılacak scheduler stratejisi.
        warmup_steps: Linear warmup adım sayısı.
        warmup_start_factor: Warmup başlangıç LR katsayısı (base_lr * factor).
        total_steps: Toplam training adım sayısı (LINEAR_DECAY için gerekli).
        T_0: Cosine restarts ilk periyot epoch sayısı.
        T_mult: Cosine restarts periyot çarpanı.
        eta_min: Cosine restarts minimum LR.
        max_lr: OneCycle maksimum LR.
        pct_start: OneCycle'da LR artış oranı (toplam adımların yüzdesi).
        div_factor: OneCycle başlangıç LR böleni (max_lr / div_factor).
        final_div_factor: OneCycle son LR böleni (max_lr / final_div_factor).
        factor: ReduceLROnPlateau LR azaltma katsayısı.
        patience: ReduceLROnPlateau sabır epoch sayısı.
        threshold: ReduceLROnPlateau iyileşme eşiği.
        cooldown: ReduceLROnPlateau azaltma sonrası bekleme epoch'u.
        min_lr: ReduceLROnPlateau minimum LR.
        use_llrd: LLRD (Layer-wise LR Decay) kullan.
        llrd_decay: LLRD katmanlar arası decay katsayısı.
    """

    # --- Ortak parametreler ---
    scheduler_type: SchedulerType = SchedulerType.COSINE_RESTARTS
    warmup_steps: int = 1500
    warmup_start_factor: float = 0.1
    total_steps: int = 100_000  # LINEAR_DECAY için gerekli

    # --- Cosine Restarts (SGDR) parametreleri ---
    T_0: int = 10
    T_mult: int = 2
    eta_min: float = 1e-6

    # --- OneCycle parametreleri ---
    max_lr: float = 2e-4
    pct_start: float = 0.3
    div_factor: float = 25.0
    final_div_factor: float = 1e4

    # --- ReduceLROnPlateau parametreleri ---
    factor: float = 0.75
    patience: int = 15
    threshold: float = 0.005
    cooldown: int = 3
    min_lr: float = 1e-6

    # --- LLRD parametreleri ---
    use_llrd: bool = False
    llrd_decay: float = 0.9


# ===========================================================================
# Ana TrainingScheduler sınıfı
# ===========================================================================

class TrainingScheduler:
    """
    Çoklu scheduler stratejisi desteği ile gelişmiş LR yöneticisi.

    Linear warmup her zaman aktiftir (warmup_steps'e kadar).
    Warmup tamamlandıktan sonra seçilen scheduler stratejisi devreye girer.

    Warmup Mekanizması:
        Her batch sonrası step_batch() çağrısıyla warmup ilerler.
        Warmup tamamlanana kadar LR lineer artar:
            lr = base_lr * warmup_start_factor + (base_lr - base_lr * warmup_start_factor)
                 * (current_step / warmup_steps)

    LLRD Desteği:
        build_llrd_optimizer() static metodu ile LLRD'li optimizer oluşturulabilir.
        Her katman grubu farklı LR'ye sahip olur.

    Args:
        optimizer (torch.optim.Optimizer): Yönetilecek optimizer.
        config (SchedulerConfig): Scheduler konfigürasyonu.

    Referans:
        - Loshchilov & Hutter 2016 (Cosine Restarts)
        - Smith & Touvron 2018 (One Cycle)
        - Howard & Ruder 2018 (LLRD)
        - Devlin et al. 2019 (Linear Warmup)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: SchedulerConfig,
    ) -> None:
        self.optimizer = optimizer
        self.config = config

        # Warmup adım sayacı
        self._warmup_step: int = 0
        self._warmup_complete: bool = config.warmup_steps <= 0

        # Base LR değerlerini kaydet (warmup hesaplaması için)
        self._base_lrs: List[float] = [
            group["lr"] for group in optimizer.param_groups
        ]

        # Ana scheduler'ı oluştur (warmup sonrası devreye girer)
        self._scheduler = self._build_scheduler()

        # Epoch adım sayacı (scheduler.step_epoch için)
        self._epoch: int = 0

        logger.info(
            "TrainingScheduler başlatıldı: type=%s, warmup_steps=%d, "
            "base_lrs=%s",
            config.scheduler_type.value,
            config.warmup_steps,
            [f"{lr:.2e}" for lr in self._base_lrs],
        )

    # ------------------------------------------------------------------
    # Scheduler oluşturma
    # ------------------------------------------------------------------

    def _build_scheduler(self) -> Optional[Any]:
        """
        Konfigürasyona göre uygun PyTorch scheduler'ını oluşturur.

        Returns:
            Scheduler objesi veya None (CONSTANT_WITH_WARMUP için).
        """
        cfg = self.config
        optimizer = self.optimizer

        if cfg.scheduler_type == SchedulerType.COSINE_RESTARTS:
            # Cosine annealing with warm restarts (Loshchilov & Hutter 2016)
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=cfg.T_0,
                T_mult=cfg.T_mult,
                eta_min=cfg.eta_min,
            )
            logger.debug(
                "CosineAnnealingWarmRestarts oluşturuldu: T_0=%d, T_mult=%d, eta_min=%.2e",
                cfg.T_0, cfg.T_mult, cfg.eta_min,
            )
            return scheduler

        elif cfg.scheduler_type == SchedulerType.ONE_CYCLE:
            # OneCycleLR (Smith & Touvron 2018)
            # total_steps warmup dahil toplam adım sayısı
            scheduler = OneCycleLR(
                optimizer,
                max_lr=cfg.max_lr,
                total_steps=cfg.total_steps,
                pct_start=cfg.pct_start,
                div_factor=cfg.div_factor,
                final_div_factor=cfg.final_div_factor,
            )
            logger.debug(
                "OneCycleLR oluşturuldu: max_lr=%.2e, total_steps=%d, pct_start=%.2f",
                cfg.max_lr, cfg.total_steps, cfg.pct_start,
            )
            return scheduler

        elif cfg.scheduler_type == SchedulerType.LINEAR_DECAY:
            # Linear decay (BERT tarzı)
            # Warmup sonrası total_steps'e kadar lineer azalma
            total = max(cfg.total_steps - cfg.warmup_steps, 1)

            def _linear_decay_fn(current_step: int) -> float:
                """Warmup sonrası lineer decay lambda fonksiyonu."""
                if current_step < cfg.warmup_steps:
                    # Warmup aşamasında (TrainingScheduler warmup ile örtüşür)
                    return 1.0
                # Decay aşaması: 1.0'dan 0.0'a lineer azalma
                progress = (current_step - cfg.warmup_steps) / total
                return max(0.0, 1.0 - progress)

            scheduler = LambdaLR(optimizer, lr_lambda=_linear_decay_fn)
            logger.debug(
                "LinearDecay oluşturuldu: total_steps=%d, warmup_steps=%d",
                cfg.total_steps, cfg.warmup_steps,
            )
            return scheduler

        elif cfg.scheduler_type == SchedulerType.REDUCE_ON_PLATEAU:
            # ReduceLROnPlateau
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=cfg.factor,
                patience=cfg.patience,
                threshold=cfg.threshold,
                cooldown=cfg.cooldown,
                min_lr=cfg.min_lr,
            )
            logger.debug(
                "ReduceLROnPlateau oluşturuldu: factor=%.2f, patience=%d, min_lr=%.2e",
                cfg.factor, cfg.patience, cfg.min_lr,
            )
            return scheduler

        elif cfg.scheduler_type == SchedulerType.CONSTANT_WITH_WARMUP:
            # Warmup sonrası sabit LR (scheduler gerekmez)
            logger.debug("CONSTANT_WITH_WARMUP: warmup sonrası sabit LR.")
            return None

        else:
            raise ValueError(
                f"Bilinmeyen scheduler tipi: {cfg.scheduler_type}. "
                f"Desteklenenler: {[e.value for e in SchedulerType]}"
            )

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def _apply_warmup(self) -> None:
        """
        Linear warmup'ı uygular: LR'yi warmup_start_factor'dan 1.0'a lineer artırır.

        warmup_start_factor: Warmup başlangıcında base_lr'nin hangi kesrinden başlanır.
        Her step_batch() çağrısında bir adım ilerler.
        """
        progress = self._warmup_step / self.config.warmup_steps
        # LR = base_lr * (start_factor + (1 - start_factor) * progress)
        warmup_factor = self.config.warmup_start_factor + (
            1.0 - self.config.warmup_start_factor
        ) * progress

        for group, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
            group["lr"] = base_lr * warmup_factor

    # ------------------------------------------------------------------
    # Dışa açık adım metodları
    # ------------------------------------------------------------------

    def step_batch(self) -> None:
        """
        Her optimizer.step() sonrası çağrılmalıdır.

        Warmup tamamlanana kadar LR warmup adımını ilerletir.
        Warmup tamamlandıktan sonra batch-tabanlı scheduler'ları günceller
        (OneCycleLR, LinearDecay için gerekli).
        """
        if not self._warmup_complete:
            # Warmup devam ediyor
            self._warmup_step += 1
            self._apply_warmup()

            if self._warmup_step >= self.config.warmup_steps:
                # Warmup tamamlandı: base_lr'ye geri dön
                self._warmup_complete = True
                for group, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
                    group["lr"] = base_lr
                logger.info(
                    "Warmup tamamlandı (adım=%d). Ana scheduler aktif.",
                    self._warmup_step,
                )
        else:
            # Warmup sonrası: batch-tabanlı scheduler'lar (OneCycle, Linear)
            if self._scheduler is not None and isinstance(
                self._scheduler, (OneCycleLR, LambdaLR)
            ):
                self._scheduler.step()

    def step_epoch(self, metric: Optional[float] = None) -> None:
        """
        Her epoch sonrası çağrılmalıdır.

        Epoch-tabanlı scheduler'ları (CosineRestarts, ReduceOnPlateau) günceller.

        Args:
            metric (float, optional): ReduceLROnPlateau için validation metriği.
                                      Diğer scheduler'lar için görmezden gelinir.

        Raises:
            ValueError: ReduceLROnPlateau seçilmişse metric verilmemişse.
        """
        self._epoch += 1

        # Warmup tamamlanmamışsa epoch scheduler'ını çalıştırma
        if not self._warmup_complete:
            return

        if self._scheduler is None:
            # CONSTANT_WITH_WARMUP: epoch'ta scheduler güncelleme gerekmiyor
            return

        if isinstance(self._scheduler, ReduceLROnPlateau):
            if metric is None:
                raise ValueError(
                    "ReduceLROnPlateau için step_epoch(metric=...) ile "
                    "validation metriği verilmelidir."
                )
            self._scheduler.step(metric)
            logger.debug(
                "ReduceLROnPlateau adımı (epoch=%d, metric=%.6f)",
                self._epoch, metric,
            )

        elif isinstance(self._scheduler, CosineAnnealingWarmRestarts):
            # CosineAnnealingWarmRestarts epoch bazlı step
            self._scheduler.step()
            logger.debug(
                "CosineAnnealingWarmRestarts adımı (epoch=%d, lr=%s)",
                self._epoch,
                [f"{lr:.2e}" for lr in self.get_last_lr()],
            )

        # OneCycleLR ve LambdaLR zaten step_batch'ta güncelleniyor

    def get_last_lr(self) -> List[float]:
        """
        Mevcut LR değerlerini döndürür.

        Returns:
            List[float]: Her param group için güncel LR listesi.
        """
        return [group["lr"] for group in self.optimizer.param_groups]

    # ------------------------------------------------------------------
    # State dict yönetimi (checkpoint desteği)
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict:
        """
        Scheduler durumunu checkpoint için döndürür.

        Returns:
            Dict: Tam scheduler state sözlüğü.
        """
        state = {
            "warmup_step": self._warmup_step,
            "warmup_complete": self._warmup_complete,
            "epoch": self._epoch,
            "base_lrs": self._base_lrs,
            "config": {
                "scheduler_type": self.config.scheduler_type.value,
                "warmup_steps": self.config.warmup_steps,
                "warmup_start_factor": self.config.warmup_start_factor,
                "total_steps": self.config.total_steps,
                "T_0": self.config.T_0,
                "T_mult": self.config.T_mult,
                "eta_min": self.config.eta_min,
                "max_lr": self.config.max_lr,
                "pct_start": self.config.pct_start,
                "div_factor": self.config.div_factor,
                "final_div_factor": self.config.final_div_factor,
                "factor": self.config.factor,
                "patience": self.config.patience,
                "threshold": self.config.threshold,
                "cooldown": self.config.cooldown,
                "min_lr": self.config.min_lr,
                "use_llrd": self.config.use_llrd,
                "llrd_decay": self.config.llrd_decay,
            },
        }
        # Scheduler state
        if self._scheduler is not None:
            state["scheduler_state"] = self._scheduler.state_dict()
        return state

    def load_state_dict(self, state: Dict) -> None:
        """
        Checkpoint'ten scheduler durumunu yükler.

        Args:
            state (Dict): state_dict() ile kaydedilen sözlük.

        Raises:
            KeyError: Gerekli alanlar eksikse.
        """
        required = {"warmup_step", "warmup_complete", "epoch", "base_lrs"}
        missing = required - state.keys()
        if missing:
            raise KeyError(f"state_dict'te eksik alanlar: {missing}")

        self._warmup_step = state["warmup_step"]
        self._warmup_complete = state["warmup_complete"]
        self._epoch = state["epoch"]
        self._base_lrs = state["base_lrs"]

        if "scheduler_state" in state and self._scheduler is not None:
            self._scheduler.load_state_dict(state["scheduler_state"])

        logger.info(
            "TrainingScheduler state yüklendi: epoch=%d, warmup_complete=%s, "
            "current_lrs=%s",
            self._epoch,
            self._warmup_complete,
            [f"{lr:.2e}" for lr in self.get_last_lr()],
        )

    # ------------------------------------------------------------------
    # LLRD (Layer-wise Learning Rate Decay) static metodu
    # ------------------------------------------------------------------

    @staticmethod
    def build_llrd_optimizer(
        model: torch.nn.Module,
        base_lr: float,
        weight_decay: float = 0.01,
        llrd_decay: float = 0.9,
        num_layers: Optional[int] = None,
    ) -> AdamW:
        """
        LLRD (Layer-wise Learning Rate Decay) ile AdamW optimizer oluşturur.

        Transformer modellerde alt katmanlar (embedding, erken transformer layers)
        genellikle zaten iyi öğrenilmiş özelliklere sahiptir. Bu katmanları
        yüksek LR ile güncellemek bozulmaya yol açabilir.

        LLRD formülü:
            LR_embedding = base_lr * llrd_decay^num_layers   (en düşük)
            LR_layer_k   = base_lr * llrd_decay^(num_layers - k)
            LR_head      = base_lr                            (en yüksek)

        Model yapısı varsayımı (HuggingFace/standart Transformer):
            - model.embeddings veya model.embed_tokens
            - model.encoder.layer[i] veya model.layers[i]
            - model.classifier veya model.lm_head

        Args:
            model: LLRD uygulanacak model.
            base_lr: En üst katman için temel öğrenme hızı.
            weight_decay: AdamW weight decay. Varsayılan: 0.01.
            llrd_decay: Katmanlar arası LR decay katsayısı.
                        0.9 → her katmanda %10 azalma. Varsayılan: 0.9.
            num_layers: Transformer layer sayısı. None ise otomatik tespit.

        Returns:
            AdamW: Katman gruplarına göre farklı LR atanmış optimizer.

        Referans:
            Howard & Ruder 2018 (ULMFiT) - https://arxiv.org/abs/1801.06146
        """
        # Weight decay uygulanmayacak parametre türleri
        no_decay_names = {"bias", "LayerNorm.weight", "layer_norm.weight"}

        # Transformer katmanlarını bul
        # Yaygın isimlendirme: encoder.layer, layers, transformer.h
        layer_names_candidates = [
            "encoder.layer",
            "encoder.layers",
            "transformer.h",
            "layers",
            "model.layers",
        ]

        # Model katmanlarını tespit et
        layer_groups: List[List[tuple]] = []  # [(name, param), ...]
        embedding_params: List[tuple] = []    # Embedding parametreleri
        head_params: List[tuple] = []         # LM head / classifier parametreleri

        # Model parametrelerini katmanlara göre grupla
        all_named_params = list(model.named_parameters())
        layer_param_map: Dict[int, List[tuple]] = {}

        for name, param in all_named_params:
            if not param.requires_grad:
                continue

            # Katman indeksini bulmaya çalış (örn: encoder.layer.3.attention...)
            layer_idx = None
            for part in name.split("."):
                try:
                    layer_idx = int(part)
                    break
                except ValueError:
                    continue

            # Embedding ve head ayrımı
            is_embedding = any(
                keyword in name.lower()
                for keyword in ["embed", "position_encoding", "token_type"]
            )
            is_head = any(
                keyword in name.lower()
                for keyword in ["classifier", "lm_head", "output_proj", "head"]
            )

            if is_embedding and layer_idx is None:
                embedding_params.append((name, param))
            elif is_head and layer_idx is None:
                head_params.append((name, param))
            elif layer_idx is not None:
                if layer_idx not in layer_param_map:
                    layer_param_map[layer_idx] = []
                layer_param_map[layer_idx].append((name, param))
            else:
                # Gruplanamayan parametreler head'e ekle
                head_params.append((name, param))

        # Katman sayısını belirle
        if num_layers is None:
            if layer_param_map:
                num_layers = max(layer_param_map.keys()) + 1
            else:
                # Katman tespiti başarısız: tüm params aynı LR
                logger.warning(
                    "LLRD: Katman yapısı tespit edilemedi. "
                    "Tüm parametrelere base_lr=%.2e uygulanıyor.",
                    base_lr,
                )
                return AdamW(
                    [{"params": [p for _, p in all_named_params if p.requires_grad],
                      "lr": base_lr, "weight_decay": weight_decay}]
                )

        logger.info(
            "LLRD başlatıldı: num_layers=%d, base_lr=%.2e, llrd_decay=%.2f",
            num_layers, base_lr, llrd_decay,
        )

        # Param grupları oluştur
        optimizer_grouped_parameters: List[Dict] = []

        def _make_group(params_list: List[tuple], lr: float) -> None:
            """No-decay ve decay gruplarını optimizer listesine ekler."""
            decay_params = [
                p for name, p in params_list
                if not any(nd in name for nd in no_decay_names)
            ]
            no_decay_params = [
                p for name, p in params_list
                if any(nd in name for nd in no_decay_names)
            ]
            if decay_params:
                optimizer_grouped_parameters.append({
                    "params": decay_params,
                    "lr": lr,
                    "weight_decay": weight_decay,
                })
            if no_decay_params:
                optimizer_grouped_parameters.append({
                    "params": no_decay_params,
                    "lr": lr,
                    "weight_decay": 0.0,  # Bias ve LayerNorm'a weight decay uygulanmaz
                })

        # Embedding: en düşük LR
        if embedding_params:
            embed_lr = base_lr * (llrd_decay ** num_layers)
            _make_group(embedding_params, embed_lr)
            logger.debug("Embedding LR: %.2e", embed_lr)

        # Transformer katmanları: alt → üst artan LR
        for layer_idx in sorted(layer_param_map.keys()):
            # layer_idx=0: en alt → en düşük LR
            # layer_idx=num_layers-1: en üst → base_lr'ye yakın
            layer_lr = base_lr * (llrd_decay ** (num_layers - 1 - layer_idx))
            _make_group(layer_param_map[layer_idx], layer_lr)
            logger.debug("Layer %d LR: %.2e", layer_idx, layer_lr)

        # Head: base_lr (en yüksek)
        if head_params:
            _make_group(head_params, base_lr)
            logger.debug("Head LR: %.2e", base_lr)

        optimizer = AdamW(optimizer_grouped_parameters)
        logger.info(
            "LLRD AdamW oluşturuldu: %d param grubu, LR aralığı [%.2e, %.2e]",
            len(optimizer_grouped_parameters),
            min(g["lr"] for g in optimizer_grouped_parameters),
            max(g["lr"] for g in optimizer_grouped_parameters),
        )
        return optimizer

    # ------------------------------------------------------------------
    # Kullanışlı özellikler
    # ------------------------------------------------------------------

    @property
    def current_epoch(self) -> int:
        """Geçerli epoch numarasını döndürür."""
        return self._epoch

    @property
    def warmup_progress(self) -> float:
        """Warmup ilerleme oranını [0.0, 1.0] döndürür."""
        if self.config.warmup_steps <= 0:
            return 1.0
        return min(1.0, self._warmup_step / self.config.warmup_steps)

    @property
    def is_warmup_complete(self) -> bool:
        """Warmup tamamlanmışsa True döndürür."""
        return self._warmup_complete

    def __repr__(self) -> str:
        return (
            f"TrainingScheduler("
            f"type={self.config.scheduler_type.value}, "
            f"warmup_steps={self.config.warmup_steps}, "
            f"warmup_complete={self._warmup_complete}, "
            f"epoch={self._epoch}, "
            f"current_lrs={[f'{lr:.2e}' for lr in self.get_last_lr()]})"
        )
