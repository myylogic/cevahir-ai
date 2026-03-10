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


class _LinearWarmupWrapper:
    """
    Lineer warmup + altta yatan gerçek scheduler (isteğe bağlı).
    warmup_steps boyunca LR, base_lr * warmup_start_factor'dan base_lr'a lineer artar.
    Sonrasında alttaki scheduler devreye girer.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        warmup_steps: int,
        base_scheduler: Optional[Any] = None,
        start_factor: float = 0.1,
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
        
        # [OK] GRADIENT FIX: Embedding için özel warmup start_factor
        # Embedding base LR zaten düşük (0.00001), warmup'ta daha da düşürmemek için
        # Embedding için start_factor=1.0 (warmup yok) - embedding LR sabit kalır
        embedding_warmup_factor = float(getattr(self, "embedding_warmup_factor", 1.0))
        
        # Warmup başlangıç LR'ları (embedding için özel, diğerleri için normal)
        self._warmup_start_lrs = []
        for i, pg in enumerate(self.optimizer.param_groups):
            base_lr = pg["lr"]
            
            # Embedding tespiti: Base LR çok düşükse (<0.00002) embedding grubudur
            # Embedding LR = 0.00001 (base_lr × 0.05), normal LR = 0.0002
            is_embedding = (base_lr < 0.00002)  # Threshold: 2e-5
            
            # Embedding parametreleri için özel warmup (warmup yok, sabit LR)
            if is_embedding:
                warmup_lr = base_lr * embedding_warmup_factor  # 1.0 = warmup yok, sabit LR
                if i == 0:  # İlk grup için log
                    self.logger.log_info(
                        f"[Warmup] Embedding warmup factor: {embedding_warmup_factor:.2f} "
                        f"(embedding LR sabit: {warmup_lr:.2e}, base_lr={base_lr:.2e})"
                    )
            else:
                # Diğer parametreler için normal warmup
                warmup_lr = base_lr * self.start_factor
            self._warmup_start_lrs.append(warmup_lr)
        
        # param gruplarını warmup başlangıcına ayarla
        for pg, lr in zip(self.optimizer.param_groups, self._warmup_start_lrs):
            pg["lr"] = float(lr)

        self.logger.log_info(
            f"[Warmup] Başlatıldı: steps={self.warmup_steps}, start_factor={self.start_factor:.4f}"
        )

    def get_last_lr(self) -> float:
        # [OK] GRADIENT FIX: Normal parametreler için LR döndür (embedding değil)
        # Embedding LR sabit (0.00001), diğer parametreler için LR warmup'tan geçiyor
        # İkinci parametre grubu normal params (ilk grup embedding olabilir)
        if len(self.optimizer.param_groups) > 1:
            # Normal parametreler için LR (warmup'tan geçen)
            return float(self.optimizer.param_groups[1]["lr"])
        else:
            # Sadece bir grup varsa onu kullan
            return float(self.optimizer.param_groups[0]["lr"])

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
        embedding_warmup_factor: float = 1.0,  # [OK] Embedding için warmup factor (1.0 = warmup yok)
        **kwargs: Any,
    ) -> None:
        """
        Args:
            optimizer: PyTorch optimizer
            scheduler_type: Scheduler adı
            logger: TrainingLogger (opsiyonel)
            warmup_steps: >0 ise lineer warmup etkin
            warmup_start_factor: warmup başlangıç çarpanı (0-1)
            embedding_warmup_factor: Embedding için warmup factor (1.0 = warmup yok, sabit LR)
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

        base_scheduler = self._initialize_scheduler(optimizer, scheduler_type, **kwargs)

        # Opsiyonel warmup wrapper
        if warmup_steps and warmup_steps > 0:
            wrapper = _LinearWarmupWrapper(
                optimizer,
                warmup_steps=warmup_steps,
                base_scheduler=base_scheduler,
                start_factor=warmup_start_factor,
                logger=self.logger,
            )
            # [OK] GRADIENT FIX: Embedding warmup factor'ü wrapper'a aktar
            wrapper.embedding_warmup_factor = float(embedding_warmup_factor)
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
            # [OK] GRADIENT FIX: Normal parametreler için LR döndür (embedding değil)
            # Embedding LR sabit (0.00001), diğer parametreler için LR warmup'tan geçiyor
            # İkinci parametre grubu normal params (ilk grup embedding olabilir)
            if len(self.optimizer.param_groups) > 1:
                # Normal parametreler için LR (warmup'tan geçen)
                lr = float(self.optimizer.param_groups[1]["lr"])
            else:
                # Sadece bir grup varsa onu kullan
                lr = float(self.optimizer.param_groups[0]["lr"])
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
