# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: training_scheduler.py
Modül: training_management/v2/utils
Görev: Training Scheduler - Öğrenme oranını (LR) dinamik olarak ayarlamak için
       scheduler yönetimi. Birçok PyTorch scheduler türünü destekler (ReduceLROnPlateau,
       StepLR, ExponentialLR, CosineAnnealingLR, CosineAnnealingWarmRestarts,
       OneCycleLR, NoOp) ve opsiyonel lineer warmup uygular.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (scheduler yönetimi)
- Design Patterns: Strategy Pattern (farklı scheduler türleri için)
- Endüstri Standartları: Learning rate scheduling best practices

KULLANIM:
- Learning rate scheduling için
- Warmup uygulama için
- Gradient gate ile LR step atlama için
- Checkpoint uyumu için

BAĞIMLILIKLAR:
- torch.optim.lr_scheduler: PyTorch scheduler modülleri

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
)

from training_management.v2.utils.training_logger import TrainingLogger


#: An optimizer param group whose ``name`` is this value holds embedding weights. Groups are
#: tagged explicitly (see mark_embedding_param_groups); nothing is inferred from the size of a
#: group's learning rate, so a low fine-tuning LR does not turn a group into an embedding group.
EMBEDDING_GROUP_NAME = "embedding"


def is_embedding_group(group: Dict[str, Any]) -> bool:
    """True when an optimizer param group is explicitly tagged as an embedding group."""
    return group.get("name") == EMBEDDING_GROUP_NAME


def mark_embedding_param_groups(optimizer: torch.optim.Optimizer, model: torch.nn.Module) -> int:
    """Tag the param groups that hold only ``nn.Embedding`` weights of ``model``; return how many.

    A group that already has a ``name`` keeps it, and a group that mixes embedding weights
    with other parameters is not an embedding group.
    """
    embedding_ids = {id(module.weight) for module in model.modules() if isinstance(module, torch.nn.Embedding)}
    tagged = 0
    for group in optimizer.param_groups:
        params = list(group.get("params", []))
        if "name" not in group and params and all(id(p) in embedding_ids for p in params):
            group["name"] = EMBEDDING_GROUP_NAME
            tagged += 1
    return tagged


def _main_group_index(param_groups) -> int:
    """Index of the first param group that is not an embedding group (0 if all are)."""
    for index, group in enumerate(param_groups):
        if not is_embedding_group(group):
            return index
    return 0


class _LinearWarmupWrapper:
    """
    Lineer warmup + altta yatan gerçek scheduler (isteğe bağlı).
    warmup_steps boyunca LR, base_lr * warmup_start_factor'dan base_lr'a lineer artar.
    Açıkça "embedding" adı verilmiş gruplar, embedding_warmup_factor verilmişse
    base_lr * embedding_warmup_factor'dan başlar (1.0 = warmup yok).
    Sonrasında alttaki scheduler devreye girer.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        warmup_steps: int,
        base_scheduler: Optional[Any] = None,
        start_factor: float = 0.1,
        embedding_warmup_factor: Optional[float] = None,
        logger: Optional[TrainingLogger] = None,
    ) -> None:
        if warmup_steps <= 0:
            raise ValueError("warmup_steps > 0 olmalı.")
        self.optimizer = optimizer
        self.warmup_steps = int(warmup_steps)
        self.step_count = 0
        self.base_scheduler = base_scheduler
        self.start_factor = float(start_factor)
        # ✅ SOLID: Logger dependency injection (TrainingManager'dan geçirilir)
        if logger is None:
            # Fallback: Basit console logger (dosya logging yok)
            import logging
            self.logger = logging.getLogger("TrainingScheduler")
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter("%(message)s"))
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        else:
            self.logger = logger

        # Başlangıç LR'ları kaydet
        self._base_lrs = [pg["lr"] for pg in self.optimizer.param_groups]
        
        # Embedding grupları yalnız açık etiketle tanınır (param group "name" == "embedding",
        # bkz. mark_embedding_param_groups); LR büyüklüğünden tahmin yapılmaz, böylece 1e-5 ile
        # fine-tuning yapılan gruplar da normal warmup alır. embedding_warmup_factor=None ise
        # embedding grupları da start_factor ile ısınır.
        if embedding_warmup_factor is not None:
            embedding_warmup_factor = float(embedding_warmup_factor)
            if not math.isfinite(embedding_warmup_factor) or embedding_warmup_factor < 0:
                raise ValueError(f"embedding_warmup_factor must be a finite number >= 0, got {embedding_warmup_factor}")
        self.embedding_warmup_factor = embedding_warmup_factor
        
        # Warmup başlangıç LR'ları (etiketli embedding grupları için özel çarpan, diğerleri için normal)
        self._warmup_start_lrs = []
        for pg in self.optimizer.param_groups:
            factor = self.start_factor
            if embedding_warmup_factor is not None and is_embedding_group(pg):
                factor = embedding_warmup_factor
            self._warmup_start_lrs.append(pg["lr"] * factor)
        embedding_groups = sum(1 for pg in self.optimizer.param_groups if is_embedding_group(pg))
        if embedding_groups and embedding_warmup_factor is not None:
            self.logger.log_info(
                f"[Warmup] {embedding_groups} embedding grubu: start factor {embedding_warmup_factor:.2f} "
                f"(diğer gruplar {self.start_factor:.2f})"
            )
        
        # param gruplarını warmup başlangıcına ayarla
        for pg, lr in zip(self.optimizer.param_groups, self._warmup_start_lrs):
            pg["lr"] = float(lr)

        self.logger.log_info(
            f"[Warmup] Başlatıldı: steps={self.warmup_steps}, start_factor={self.start_factor:.4f}"
        )

    def get_last_lr(self) -> float:
        # Ana LR: açıkça etiketlenmiş bir embedding grubu olmayan ilk grup
        groups = self.optimizer.param_groups
        return float(groups[_main_group_index(groups)]["lr"])

    def step(self, *args, **kwargs) -> None:
        """
        Warmup evresinde lineer artır; warmup bitince base_scheduler.step(...) çalıştır.
        ReduceLROnPlateau gibi metrik tabanlı scheduler’a metric'i transparan geçiririz.
        """
        if self.step_count < self.warmup_steps:
            # lineer artış
            t = (self.step_count + 1) / float(self.warmup_steps)
            for pg, start_lr, base_lr in zip(self.optimizer.param_groups, self._warmup_start_lrs, self._base_lrs):
                pg["lr"] = float(start_lr + t * (base_lr - start_lr))
            self.step_count += 1
            self.logger.log_debug(f"[Warmup] step={self.step_count}/{self.warmup_steps}, lr={self.get_last_lr():.8f}")
            return

        # Warmup bitti → base scheduler
        if self.base_scheduler is not None:
            return self._delegate_step_to_base(**kwargs)

    def _delegate_step_to_base(self, **kwargs) -> None:
        bs = self.base_scheduler
        if bs is None:
            return
        if isinstance(bs, ReduceLROnPlateau):
            metric = kwargs.get("metric", None)
            if metric is None:
                raise ValueError("Warmup sonrası ReduceLROnPlateau için metric verilmelidir.")
            bs.step(metric)
        else:
            bs.step()

    def state_dict(self) -> Dict[str, Any]:
        d = {
            "warmup_steps": self.warmup_steps,
            "step_count": self.step_count,
            "start_factor": self.start_factor,
            "base_lrs": self._base_lrs,
            "warmup_start_lrs": self._warmup_start_lrs,
            "has_base": self.base_scheduler is not None,
        }
        if self.base_scheduler is not None and hasattr(self.base_scheduler, "state_dict"):
            d["base_state"] = self.base_scheduler.state_dict()
        return d

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.warmup_steps = int(state.get("warmup_steps", self.warmup_steps))
        self.step_count = int(state.get("step_count", 0))
        self.start_factor = float(state.get("start_factor", self.start_factor))
        self._base_lrs = list(state.get("base_lrs", self._base_lrs))
        self._warmup_start_lrs = list(state.get("warmup_start_lrs", self._warmup_start_lrs))
        if state.get("has_base") and self.base_scheduler is not None and "base_state" in state:
            self.base_scheduler.load_state_dict(state["base_state"])


class TrainingScheduler:
    """
    Scheduler sarmalayıcısı: farklı PyTorch lr_scheduler'larını tutarlı bir arayüzle yönetir.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_type: str = "ReduceLROnPlateau",
        *,
        logger: Optional[TrainingLogger] = None,
        warmup_steps: int = 0,
        warmup_start_factor: float = 0.1,
        embedding_warmup_factor: Optional[float] = None,  # None: embedding grupları da start_factor ile ısınır
        **kwargs: Any,
    ) -> None:
        """
        Args:
            optimizer: PyTorch optimizer
            scheduler_type: Scheduler adı
            logger: TrainingLogger (opsiyonel)
            warmup_steps: >0 ise lineer warmup etkin
            warmup_start_factor: warmup başlangıç çarpanı (0-1)
            embedding_warmup_factor: "embedding" adlı param gruplarının warmup başlangıç çarpanı
                (1.0 = warmup yok); None ise bu gruplar da warmup_start_factor kullanır
            **kwargs: ilgili scheduler parametreleri
        """
        # ✅ SOLID: Logger dependency injection (TrainingManager'dan geçirilir)
        if logger is None:
            # Fallback: Basit console logger (dosya logging yok)
            import logging
            self.logger = logging.getLogger("TrainingScheduler")
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter("%(message)s"))
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        else:
            self.logger = logger
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type

        nested = kwargs.pop("scheduler_kwargs", {})
        kwargs = {**nested, **kwargs}
        self.step_per_batch = scheduler_type.strip().lower() in {"onecyclelr", "onecycle"}
        if self.step_per_batch and warmup_steps:
            raise ValueError("OneCycleLR includes its own warmup; set warmup_steps=0")
        base_scheduler = self._initialize_scheduler(optimizer, scheduler_type, **kwargs)

        # Opsiyonel warmup wrapper
        if warmup_steps and warmup_steps > 0:
            wrapper = _LinearWarmupWrapper(
                optimizer,
                warmup_steps=warmup_steps,
                base_scheduler=base_scheduler,
                start_factor=warmup_start_factor,
                embedding_warmup_factor=embedding_warmup_factor,
                logger=self.logger,
            )
            self.scheduler = wrapper
            self.logger.log_info(
                f"{scheduler_type} için lineer warmup etkin: steps={warmup_steps}, start_factor={warmup_start_factor}"
            )
        else:
            self.scheduler = base_scheduler

        # debug
        try:
            params = getattr(self.scheduler, "__dict__", {})
            self.logger.log_debug(f"Scheduler parametreleri: {params}")
        except Exception:
            pass

        self.logger.log_info(f"{scheduler_type} öğrenme oranı planlayıcısı başarıyla başlatıldı.")

    # ---------------------------------------------------------------- init helpers

    def _initialize_scheduler(self, optimizer: torch.optim.Optimizer, scheduler_type: str, **kwargs: Any):
        st = scheduler_type.strip().lower()

        if st in ("reducelronplateau", "reduce_lr_on_plateau", "reduce_on_plateau", "plateau", "rop"):
            return ReduceLROnPlateau(
                optimizer=optimizer,
                mode=kwargs.get("mode", "min"),
                factor=kwargs.get("factor", 0.5),
                patience=kwargs.get("patience", 8),
                threshold=kwargs.get("threshold", 1e-4),
                cooldown=kwargs.get("cooldown", 3),
                min_lr=kwargs.get("min_lr", 5e-6),
                eps=kwargs.get("eps", 1e-8),
            )

        if st in ("steplr", "step"):
            return StepLR(
                optimizer=optimizer,
                step_size=kwargs.get("step_size", 30),
                gamma=kwargs.get("gamma", 0.1),
            )

        if st in ("exponentiallr", "exp", "exponential"):
            return ExponentialLR(
                optimizer=optimizer,
                gamma=kwargs.get("gamma", 0.9),
            )

        if st in ("cosineannealinglr", "cosine", "cosineannealing"):
            return CosineAnnealingLR(
                optimizer=optimizer,
                T_max=kwargs.get("T_max", 50),
                eta_min=kwargs.get("eta_min", 0.0),
            )

        if st in ("cosineannealingwarmrestarts", "cosinewarm", "cawr"):
            return CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=kwargs.get("T_0", 50),
                T_mult=kwargs.get("T_mult", 1),
                eta_min=kwargs.get("eta_min", 0.0),
            )

        if st in ("onecyclelr", "onecycle"):
            # OneCycleLR total_steps ya da (epochs*steps_per_epoch) gerektirir
            args: Dict[str, Any] = {
                "max_lr": kwargs.get("max_lr", 1e-3),
                "pct_start": kwargs.get("pct_start", 0.3),
                "anneal_strategy": kwargs.get("anneal_strategy", "cos"),
                "div_factor": kwargs.get("div_factor", 25.0),
                "final_div_factor": kwargs.get("final_div_factor", 1e4),
                "three_phase": kwargs.get("three_phase", False),
            }
            if "total_steps" in kwargs:
                args["total_steps"] = int(kwargs["total_steps"])
            else:
                # Kullanıcı epochs ve steps_per_epoch veriyorsa bunları kullan
                if "epochs" in kwargs and "steps_per_epoch" in kwargs:
                    args["epochs"] = int(kwargs["epochs"])
                    args["steps_per_epoch"] = int(kwargs["steps_per_epoch"])
                else:
                    raise ValueError(
                        "OneCycleLR için 'total_steps' ya da ('epochs' ve 'steps_per_epoch') verilmelidir."
                    )
            return OneCycleLR(optimizer=optimizer, **args)

        if st in ("noop", "none", "constant"):
            # no-op scheduler
            class _NoOp:
                def __init__(self, optimizer):
                    self.optimizer = optimizer

                def step(self, *a, **k):
                    pass

                def state_dict(self):
                    return {}

                def load_state_dict(self, d):
                    pass

                def get_last_lr(self):
                    return [self.optimizer.param_groups[0]["lr"]]

            return _NoOp(optimizer)

        raise ValueError(f"Bilinmeyen scheduler türü: {scheduler_type}")

    def step_batch(self):
        warming_up = isinstance(self.scheduler, _LinearWarmupWrapper) and self.scheduler.step_count < self.scheduler.warmup_steps
        if warming_up or self.step_per_batch:
            self.step()

    def step_epoch(self, metric=None):
        warming_up = isinstance(self.scheduler, _LinearWarmupWrapper) and self.scheduler.step_count < self.scheduler.warmup_steps
        if not warming_up and not self.step_per_batch:
            self.step(metric=metric)

    # --------------------------------------------------------------------- stepping

    def step(
        self,
        metric: Optional[float] = None,
        *,
        gradient_norm: Optional[float] = None,
        gradient_gate: Optional[float] = None,
    ) -> None:
        """
        Öğrenme oranını günceller.

        Args:
            metric: ReduceLROnPlateau (veya warmup sonrası buna delegasyon) için gerekli metrik (ör. val_loss).
            gradient_norm: (opsiyonel) hesapladığınız gradient normu.
            gradient_gate: (opsiyonel) gradient_norm < gradient_gate ise LR güncellemesi atlanır.
        """
        self.logger.log_debug("Scheduler step() çağrıldı.")

        # Gradient kapısı
        if gradient_gate is not None and gradient_norm is not None:
            try:
                if float(gradient_norm) < float(gradient_gate):
                    self.logger.log_warning(
                        f"Gradient Norm düşük ({gradient_norm:.6f} < {float(gradient_gate):.6f}). LR step atlandı."
                    )
                    return
            except Exception:
                # Tip/float dönüşümünde sorun olursa görmezden gel
                pass

        # Warmup wrapper ise delegasyon
        if isinstance(self.scheduler, _LinearWarmupWrapper):
            # ReduceLROnPlateau delegasyonu durumunda metric gerekli olabilir.
            self.scheduler.step(metric=metric)
            step_count = self.scheduler.step_count
            warmup_steps = self.scheduler.warmup_steps
            lr = self.get_last_lr()
            # Progress bar ile karışmaması için anlamlı noktalarda logla (her batch değil)
            if step_count == warmup_steps:
                self.logger.log_info(f"\n[LR] Warmup bitti → {lr:.2e} (sonraki adımlar: plateau/epoch)")
            elif step_count % 50 == 0:
                self.logger.log_info(f"\n[LR] Warmup step {step_count}/{warmup_steps} → {lr:.2e}")
            return

        # Base scheduler davranışı
        if isinstance(self.scheduler, ReduceLROnPlateau):
            if metric is None:
                raise ValueError("ReduceLROnPlateau için 'metric' zorunludur.")
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

        # Yeni satırla yaz ki progress bar ile aynı satıra karışmasın
        self.logger.log_info(f"\n[LR] Scheduler güncellendi → {self.get_last_lr():.2e}")

    # ------------------------------------------------------------------------ utils

    def get_last_lr(self) -> float:
        """
        Mevcut öğrenme oranını döndürür (normal parametreler için LR, embedding değil).
        """
        try:
            # Ana LR: açıkça etiketlenmiş bir embedding grubu olmayan ilk grup
            groups = self.optimizer.param_groups
            lr = float(groups[_main_group_index(groups)]["lr"])
        except Exception as e:
            self.logger.log_error(f"Öğrenme oranı alınırken hata oluştu: {e}")
            raise
        self.logger.log_debug(f"Son öğrenme oranı: {lr:.8f}")
        return lr

    # Checkpoint uyumu
    def state_dict(self) -> Dict[str, Any]:
        if hasattr(self.scheduler, "state_dict"):
            sd = self.scheduler.state_dict()
        else:
            sd = {}
        return {
            "scheduler_type": self.scheduler_type,
            "scheduler_state": sd,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        # Tür uyumu sorumluluğu çağıranda; burada sadece iç state yüklenir.
        try:
            sched_state = state.get("scheduler_state", {})
            if hasattr(self.scheduler, "load_state_dict"):
                self.scheduler.load_state_dict(sched_state)
            self.logger.log_info("Scheduler state başarıyla yüklendi.")
        except Exception as e:
            self.logger.log_error(f"Scheduler state yüklenemedi: {e}")


def _scaled_lr(value: Any, factor: float) -> Any:
    if isinstance(value, torch.Tensor):
        return value.mul_(factor)
    return float(value) * factor


def apply_continuation_learning_rate(optimizer: torch.optim.Optimizer, target_lr: float, scheduler: Optional[Any] = None) -> float:
    """Make ``target_lr`` the main learning rate of a resumed run; return the factor applied.

    Call after the optimizer and scheduler state were restored: that state carries the
    checkpointed run's learning rate. Every param group is multiplied by one factor,
    ``target_lr / reference``, so per-group ratios (e.g. ``embedding_lr_scale``) survive. The
    reference is the largest group LR, or the largest warmup target while warmup is still in
    progress, so that the remaining ramp ends at ``target_lr``. Warmup anchors, ``initial_lr``
    and the base scheduler's ``base_lrs``/``_last_lr`` are scaled by the same factor, so a later
    scheduler step continues from the new LR instead of returning to the old one. Absolute
    floors (``min_lr``, ``eta_min``) and the plateau history (best metric, patience) are kept.
    """
    target = float(target_lr)
    if not math.isfinite(target) or target <= 0:
        raise ValueError(f"continuation_learning_rate must be a positive finite number, got {target_lr!r}")
    inner = scheduler.scheduler if isinstance(scheduler, TrainingScheduler) else scheduler
    wrapper = inner if isinstance(inner, _LinearWarmupWrapper) else None
    base = wrapper.base_scheduler if wrapper is not None else inner
    if isinstance(base, OneCycleLR):
        raise ValueError("OneCycleLR defines its own learning-rate curve; continuation_learning_rate cannot override it")
    groups = optimizer.param_groups
    if wrapper is not None and wrapper.step_count < wrapper.warmup_steps:
        reference = max(float(lr) for lr in wrapper._base_lrs)
    else:
        reference = max(float(group["lr"]) for group in groups)
    if reference <= 0:
        raise ValueError("Cannot rescale a zero learning rate to continuation_learning_rate")
    factor = target / reference
    for group in groups:
        for key in ("lr", "initial_lr"):
            if key in group:
                group[key] = _scaled_lr(group[key], factor)
    if wrapper is not None:
        wrapper._base_lrs = [float(lr) * factor for lr in wrapper._base_lrs]
        wrapper._warmup_start_lrs = [float(lr) * factor for lr in wrapper._warmup_start_lrs]
    if base is not None and hasattr(base, "base_lrs"):
        base.base_lrs = [float(lr) * factor for lr in base.base_lrs]
    if base is not None and hasattr(base, "_last_lr"):
        base._last_lr = [float(group["lr"]) for group in groups]
    return factor


def main_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """Learning rate of the main param group: the first group not tagged as an embedding group.

    This is the LR that TrainingScheduler.get_last_lr() reports and that warmup ramps. It is not
    the largest group LR: an embedding group kept out of warmup can sit above it.
    """
    groups = optimizer.param_groups
    return float(groups[_main_group_index(groups)]["lr"])


def find_plateau_scheduler(scheduler: Any) -> ReduceLROnPlateau | None:
    """The ReduceLROnPlateau behind a TrainingScheduler and/or warmup wrapper, or None."""
    inner = scheduler.scheduler if isinstance(scheduler, TrainingScheduler) else scheduler
    if isinstance(inner, _LinearWarmupWrapper):
        inner = inner.base_scheduler
    return inner if isinstance(inner, ReduceLROnPlateau) else None
