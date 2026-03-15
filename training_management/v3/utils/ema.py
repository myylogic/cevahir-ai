"""
EMA - Exponential Moving Average for Model Weights
===================================================
Referans: Yazici et al. 2019, "The Unusual Effectiveness of Averaging in GAN Training"
          Polyak & Juditsky 1992, "Acceleration of Stochastic Approximation by Averaging"

Model ağırlıklarının üstel hareketli ortalamasını tutar.
Inference sırasında EMA ağırlıkları kullanmak genellikle daha iyi
generalization ve daha düşük variance sağlar.

Kullanım:
    ema = EMA(model, decay=0.9999, warmup_steps=100)
    for batch in dataloader:
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        ema.update()  # Her optimizer adımından sonra çağır

    # Inference:
    with ema.average_parameters():
        preds = model(test_x)  # EMA ağırlıklarıyla çalışır
"""

import copy
import logging
from contextlib import contextmanager
from typing import Dict, Generator, Iterable, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EMA:
    """
    Model ağırlıklarının Üstel Hareketli Ortalaması (Exponential Moving Average).

    Training ağırlıklarından bağımsız 'shadow weights' tutar.
    Inference'ta EMA ağırlıkları kullanmak:
    - Loss landscape'in daha düz bölgelerine karşılık gelir
    - Stochastic gradient noise etkisini azaltır
    - Genellikle daha iyi generalization sağlar

    Warmup Düzeltmesi:
        İlk adımlarda decay çok yüksekse shadow weights başlangıç
        modelinden uzaklaşabilir. Warmup ile decay kademeli artar:
        decay_t = min(decay, (1 + step) / (10 + step))

    Args:
        model (nn.Module): EMA uygulanacak model.
        decay (float): EMA decay faktörü. Büyük model / uzun training
                       için 0.9999 önerilir. Varsayılan: 0.9999.
        warmup_steps (int): Decay warmup adım sayısı. Varsayılan: 100.

    Referans:
        Yazici et al. 2019 - https://arxiv.org/abs/1806.04498
        Polyak & Juditsky 1992
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmup_steps: int = 100,
    ) -> None:
        # Decay değeri (0, 1) aralığında olmalı
        if not (0.0 < decay < 1.0):
            raise ValueError(
                f"decay değeri (0, 1) arasında olmalıdır, alınan: {decay}"
            )
        if warmup_steps < 0:
            raise ValueError(
                f"warmup_steps negatif olamaz, alınan: {warmup_steps}"
            )

        self.model = model
        self.decay = decay
        self.warmup_steps = warmup_steps

        # Adım sayacı (warmup düzeltmesi için)
        self._step: int = 0

        # Shadow params: EMA ağırlıkları (model params'tan bağımsız kopya)
        # Sadece float tensor'lar EMA'ya tabi tutulur (int, bool parametreler hariç)
        self.shadow_params: Dict[str, torch.Tensor] = {
            name: param.detach().clone().float()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        # Backup params: apply_shadow/restore döngüsü için orijinal ağırlıklar
        self._backup_params: Optional[Dict[str, torch.Tensor]] = None

        logger.info(
            "EMA başlatıldı: decay=%.6f, warmup_steps=%d, "
            "toplam shadow param sayısı=%d",
            decay,
            warmup_steps,
            len(self.shadow_params),
        )

    # ------------------------------------------------------------------
    # Dahili yardımcı metodlar
    # ------------------------------------------------------------------

    def _get_current_decay(self) -> float:
        """
        Warmup düzeltmesiyle anlık decay değerini hesaplar.

        İlk adımlarda decay düşük tutulur; warmup_steps sonrasında
        hedef decay'e yakınsar:
            decay_t = min(decay, (1 + step) / (10 + step))
        """
        # Warmup formülü: adım sayısı arttıkça decay yaklaşır hedef değerine
        warmup_decay = (1.0 + self._step) / (10.0 + self._step)
        return min(self.decay, warmup_decay)

    # ------------------------------------------------------------------
    # Ana metodlar
    # ------------------------------------------------------------------

    def update(self) -> None:
        """
        EMA shadow ağırlıklarını günceller.

        Her optimizer.step() çağrısından sonra çağrılmalıdır.
        shadow = decay_t * shadow + (1 - decay_t) * param

        Not: Gradient hesabı devre dışı bırakılır (no_grad).
        """
        self._step += 1
        decay_t = self._get_current_decay()

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if name not in self.shadow_params:
                    # Yeni parametre eklenmişse shadow'a ekle
                    logger.warning(
                        "EMA: '%s' parametresi shadow'da bulunamadı, ekleniyor.", name
                    )
                    self.shadow_params[name] = param.detach().clone().float()
                    continue

                shadow = self.shadow_params[name]
                # EMA güncelleme formülü
                shadow.mul_(decay_t).add_(
                    param.detach().float(), alpha=1.0 - decay_t
                )

    def apply_shadow(self) -> None:
        """
        Model parametrelerini EMA (shadow) ağırlıklarıyla değiştirir.

        Inference öncesinde çağrılır. restore() ile geri alınabilir.
        apply_shadow() sonrasında training'e devam edilmemelidir;
        bunun yerine average_parameters() context manager kullanın.

        Raises:
            RuntimeError: Zaten shadow uygulanmışsa (çift apply_shadow).
        """
        if self._backup_params is not None:
            raise RuntimeError(
                "apply_shadow() zaten çağrılmış. restore() ile geri alın."
            )

        # Orijinal ağırlıkları yedekle
        self._backup_params = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # Shadow ağırlıklarını modele yükle
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if name in self.shadow_params:
                    # shadow_params float, modelin orijinal dtype'ına çevir
                    param.data.copy_(
                        self.shadow_params[name].to(dtype=param.dtype)
                    )

    def restore(self) -> None:
        """
        EMA öncesi orijinal model ağırlıklarını geri yükler.

        apply_shadow() çağrısından sonra training'e devam etmek için
        çağrılmalıdır.

        Raises:
            RuntimeError: apply_shadow() çağrılmamışsa.
        """
        if self._backup_params is None:
            raise RuntimeError(
                "restore() çağrısından önce apply_shadow() çağrılmalıdır."
            )

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if name in self._backup_params:
                    param.data.copy_(self._backup_params[name])

        # Yedeği temizle
        self._backup_params = None

    @contextmanager
    def average_parameters(self) -> Generator[None, None, None]:
        """
        EMA ağırlıklarıyla model'i geçici olarak çalıştıran context manager.

        Kullanım:
            with ema.average_parameters():
                preds = model(test_inputs)

        Context içinde model EMA ağırlıklarını kullanır;
        context dışına çıkıldığında orijinal ağırlıklar otomatik geri yüklenir.

        Yields:
            None
        """
        self.apply_shadow()
        try:
            yield
        finally:
            self.restore()

    # ------------------------------------------------------------------
    # Checkpoint (kayıt/yükleme) desteği
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict:
        """
        EMA durumunu checkpoint için sözlük olarak döndürür.

        Returns:
            Dict: shadow_params, step, decay, warmup_steps içeren sözlük.
        """
        return {
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
            "step": self._step,
            # Her shadow param'ı CPU'ya taşı (checkpoint boyutunu optimize et)
            "shadow_params": {
                name: tensor.cpu()
                for name, tensor in self.shadow_params.items()
            },
        }

    def load_state_dict(self, state: Dict) -> None:
        """
        Checkpoint'ten EMA durumunu yükler.

        Args:
            state (Dict): state_dict() ile kaydedilen sözlük.

        Raises:
            KeyError: Gerekli alanlar state'de yoksa.
            ValueError: Decay veya warmup_steps uyumsuzsa.
        """
        required_keys = {"decay", "warmup_steps", "step", "shadow_params"}
        missing = required_keys - state.keys()
        if missing:
            raise KeyError(f"state_dict'te eksik alanlar: {missing}")

        # Hiper-parametre uyarıları
        if state["decay"] != self.decay:
            logger.warning(
                "EMA load_state_dict: decay uyumsuzluğu "
                "(checkpoint=%.6f, mevcut=%.6f). Checkpoint değeri kullanılıyor.",
                state["decay"],
                self.decay,
            )
        if state["warmup_steps"] != self.warmup_steps:
            logger.warning(
                "EMA load_state_dict: warmup_steps uyumsuzluğu "
                "(checkpoint=%d, mevcut=%d). Checkpoint değeri kullanılıyor.",
                state["warmup_steps"],
                self.warmup_steps,
            )

        self.decay = state["decay"]
        self.warmup_steps = state["warmup_steps"]
        self._step = state["step"]

        # Shadow params'ı uygun cihaza taşı
        device = next(self.model.parameters()).device
        self.shadow_params = {
            name: tensor.to(device=device, dtype=torch.float32)
            for name, tensor in state["shadow_params"].items()
        }

        logger.info(
            "EMA state yüklendi: step=%d, decay=%.6f, "
            "shadow param sayısı=%d",
            self._step,
            self.decay,
            len(self.shadow_params),
        )

    # ------------------------------------------------------------------
    # Kullanışlı özellikler
    # ------------------------------------------------------------------

    @property
    def num_updates(self) -> int:
        """Toplam EMA güncelleme adımı sayısını döndürür."""
        return self._step

    @property
    def current_decay(self) -> float:
        """Warmup düzeltmesiyle mevcut decay değerini döndürür."""
        return self._get_current_decay()

    def __repr__(self) -> str:
        return (
            f"EMA(decay={self.decay}, warmup_steps={self.warmup_steps}, "
            f"step={self._step}, current_decay={self.current_decay:.6f})"
        )
