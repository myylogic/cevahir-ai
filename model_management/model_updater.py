# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: model_updater.py
Modül: model_management
Görev: Model Updater - Model parametrelerini ve bileşenlerini güncellemek için
       kullanılan modül. Model parametre güncelleme, optimizer güncelleme,
       scheduler güncelleme, learning rate güncelleme ve selective parameter
       güncelleme işlemlerini yapar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (model güncelleme)
- Design Patterns: Updater Pattern (model güncelleme)
- Endüstri Standartları: Model updating best practices

KULLANIM:
- Model parametrelerini güncellemek için
- Optimizer güncellemek için
- Scheduler güncellemek için
- Learning rate güncelleme için

BAĞIMLILIKLAR:
- torch: PyTorch işlemleri
- torch.optim: Optimizer modülleri
- torch.optim.lr_scheduler: Scheduler modülleri

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations

import fnmatch
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    StepLR,
    MultiStepLR,
    CosineAnnealingLR,
    OneCycleLR,
    LRScheduler,
)

# Modül-özel logger (root logger'ı yeniden konfig etmeyelim)
updater_logger = logging.getLogger("model_updater")
if not updater_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    handler.setFormatter(formatter)
    updater_logger.addHandler(handler)
    updater_logger.setLevel(logging.INFO)


# ------------------------------ Yardımcılar ------------------------------ #

def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    return x if isinstance(x, (list, tuple)) else [x]


def _validate_type(name: str, val: Any, types: Tuple[type, ...]) -> None:
    if not isinstance(val, types):
        raise TypeError(f"'{name}' tipi {types} olmalı, gelen: {type(val)}")


def _validate_range(name: str, val: float, minv: Optional[float] = None, maxv: Optional[float] = None) -> None:
    if minv is not None and val < minv:
        raise ValueError(f"'{name}' en az {minv} olmalı, gelen: {val}")
    if maxv is not None and val > maxv:
        raise ValueError(f"'{name}' en fazla {maxv} olmalı, gelen: {val}")


def _match_parameter_names(named_parameters: Iterable[Tuple[str, nn.Parameter]],
                           patterns: Sequence[str]) -> List[Tuple[str, nn.Parameter]]:
    patterns = [p.strip() for p in patterns if p and p.strip()]
    if not patterns:
        return []
    matched: List[Tuple[str, nn.Parameter]] = []
    for name, param in named_parameters:
        for pat in patterns:
            # hem glob hem regex destekleyelim: /r:.../ -> regex
            if pat.startswith("r:"):
                if re.search(pat[2:], name, re.IGNORECASE):
                    matched.append((name, param))
                    break
            else:
                # [OK] Case-insensitive glob matching
                if fnmatch.fnmatch(name.lower(), pat.lower()):
                    matched.append((name, param))
                    break
                # [OK] Substring matching (backward compatibility)
                # "embed" pattern'i "language_embedding" veya "dil_katmani" ile eşleşmeli
                if pat.lower() in name.lower():
                    matched.append((name, param))
                    break
    return matched


def _filter_param_groups_after_freeze(optimizer: Optimizer) -> None:
    """
    Dondurma/çözme sonrası param_groups içinden artık train edilmeyecek paramları çıkarır.
    Yeni paramları otomatik eklemeyiz (karmaşık olabilir); bunun yerine gerekmiyorsa no-op.
    """
    for group in optimizer.param_groups:
        params_before = len(group["params"])
        group["params"] = [p for p in group["params"] if p.requires_grad]
        removed = params_before - len(group["params"])
        if removed > 0:
            updater_logger.info(f"Optimizer param_group'undan {removed} adet dondurulmuş parametre çıkarıldı.")


@dataclass
class UpdateReport:
    model_updates: List[str]
    optimizer_updates: List[str]
    scheduler_updates: List[str]

    def as_dict(self) -> Dict[str, List[str]]:
        return {
            "model": self.model_updates,
            "optimizer": self.optimizer_updates,
            "scheduler": self.scheduler_updates,
        }


# ------------------------------ Ana Sınıf ------------------------------ #

class ModelUpdater:
    """
    ModelUpdater:
    Model, optimizer ve scheduler yapılandırmalarını güvenli, denetlenebilir ve esnek biçimde günceller.

    Desteklenen başlıca özellikler:
      - Model üzerinde setattr ile basit alan güncelleme
      - Katman dondurma/çözme (glob veya regex pattern'larıyla)
      - Cihaz taşıma (device)
      - Optimizer öğrenme oranı, weight_decay, betas, eps, momentum gibi yaygın alanlar
      - Param group güncellemeleri ve dondurma sonrası temizlik
      - ReduceLROnPlateau / StepLR / MultiStepLR / CosineAnnealingLR / OneCycleLR için güvenli güncellemeler
      - Dry-run: Uygulamadan önce ne olacağını raporla
    """

    # -------------------- MODEL -------------------- #
    @staticmethod
    def update_model(
        model: nn.Module,
        update_params: Mapping[str, Any],
        *,
        dry_run: bool = False,
    ) -> UpdateReport:
        """
        Modelin belirli parametrelerini günceller.

        update_params destekleri (örnek şema):
        {
          "setattr": {"dropout_p": 0.2, "some_flag": True},
          "freeze": ["encoder.*", "r:^backbone\\.layers\\.(0|1)\\."],
          "unfreeze": ["head.*"],
          "device": "cuda:0"
        }
        """
        report = UpdateReport(model_updates=[], optimizer_updates=[], scheduler_updates=[])

        try:
            updater_logger.info("Model güncelleme işlemi başlatılıyor...")

            # 1) setattr
            for attr, value in _as_list(update_params.get("setattr", {})).items() if isinstance(update_params.get("setattr"), dict) else []:
                if not hasattr(model, attr):
                    updater_logger.warning(f"Modelde '{attr}' alanı yok; atlanıyor.")
                    continue
                if dry_run:
                    report.model_updates.append(f"(dry-run) setattr: {attr} = {value!r}")
                else:
                    setattr(model, attr, value)
                    report.model_updates.append(f"setattr: {attr} = {value!r}")

            # 2) freeze / unfreeze (pattern destekli)
            def _apply_requires_grad(patterns: Sequence[str], flag: bool, label: str) -> None:
                matches = _match_parameter_names(model.named_parameters(), patterns)
                if not matches:
                    updater_logger.info(f"{label}: eşleşme bulunamadı.")
                    return
                for name, p in matches:
                    if dry_run:
                        report.model_updates.append(f"(dry-run) {label}: {name} -> requires_grad={flag}")
                    else:
                        p.requires_grad = flag
                        report.model_updates.append(f"{label}: {name} -> requires_grad={flag}")

            freeze_patterns = _as_list(update_params.get("freeze"))
            if freeze_patterns:
                _apply_requires_grad(freeze_patterns, False, "freeze")

            unfreeze_patterns = _as_list(update_params.get("unfreeze"))
            if unfreeze_patterns:
                _apply_requires_grad(unfreeze_patterns, True, "unfreeze")

            # 3) device taşıma
            if "device" in update_params:
                device = update_params["device"]
                _validate_type("device", device, (str, torch.device))
                if dry_run:
                    report.model_updates.append(f"(dry-run) to(device={device})")
                else:
                    model.to(device)
                    report.model_updates.append(f"to(device={device})")

            updater_logger.info("Model parametre güncellemesi tamamlandı.")
            return report

        except Exception as e:
            updater_logger.error(f"Model güncellenirken hata oluştu: {str(e)}", exc_info=True)
            raise RuntimeError("Model parametre güncelleme işlemi başarısız oldu.") from e

    # -------------------- OPTIMIZER -------------------- #
    @staticmethod
    def update_optimizer(
        optimizer: Optimizer,
        update_params: Mapping[str, Any],
        *,
        filter_frozen_params: bool = True,
        dry_run: bool = False,
    ) -> UpdateReport:
        """
        Optimizer parametrelerini günceller.

        Desteklenen anahtarlar:
          - learning_rate / lr (float)
          - weight_decay (float)
          - betas (tuple/list of two floats)  [Adam/AdamW]
          - eps (float)                        [Adam/AdamW]
          - momentum, dampening, nesterov      [SGD]
          - maximize (bool)                    [PyTorch 2+ optim]
          - foreach / fused / capturable ...   (mevcutsa güncellenir)
        """
        report = UpdateReport(model_updates=[], optimizer_updates=[], scheduler_updates=[])

        try:
            updater_logger.info("Optimizer güncelleme işlemi başlatılıyor...")

            # normalized alias
            lr = update_params.get("learning_rate", update_params.get("lr", None))
            if lr is not None:
                _validate_type("learning_rate", lr, (int, float))
                _validate_range("learning_rate", float(lr), 0.0)
                for group in optimizer.param_groups:
                    if dry_run:
                        report.optimizer_updates.append(f"(dry-run) lr: {group.get('lr')} -> {lr}")
                    else:
                        group["lr"] = float(lr)
                        report.optimizer_updates.append(f"lr: -> {lr}")

            if "weight_decay" in update_params:
                wd = float(update_params["weight_decay"])
                _validate_range("weight_decay", wd, 0.0)
                for group in optimizer.param_groups:
                    if dry_run:
                        report.optimizer_updates.append(f"(dry-run) weight_decay: {group.get('weight_decay')} -> {wd}")
                    else:
                        group["weight_decay"] = wd
                        report.optimizer_updates.append(f"weight_decay -> {wd}")

            if "betas" in update_params:
                betas = tuple(update_params["betas"])
                if len(betas) != 2:
                    raise ValueError("betas iki elemanlı olmalı (beta1, beta2).")
                for group in optimizer.param_groups:
                    if "betas" in group:
                        if dry_run:
                            report.optimizer_updates.append(f"(dry-run) betas: {group.get('betas')} -> {betas}")
                        else:
                            group["betas"] = betas
                            report.optimizer_updates.append(f"betas -> {betas}")

            if "eps" in update_params:
                eps = float(update_params["eps"])
                _validate_range("eps", eps, 0.0)
                for group in optimizer.param_groups:
                    if "eps" in group:
                        if dry_run:
                            report.optimizer_updates.append(f"(dry-run) eps: {group.get('eps')} -> {eps}")
                        else:
                            group["eps"] = eps
                            report.optimizer_updates.append(f"eps -> {eps}")

            # SGD-özel
            for key in ("momentum", "dampening", "nesterov"):
                if key in update_params:
                    val = update_params[key]
                    for group in optimizer.param_groups:
                        if key in group:
                            if dry_run:
                                report.optimizer_updates.append(f"(dry-run) {key}: {group.get(key)} -> {val}")
                            else:
                                group[key] = val
                                report.optimizer_updates.append(f"{key} -> {val}")

            # Diğer mevcut anahtarlar: foreach, fused, capturable, maximize vb.
            for key in ("foreach", "fused", "capturable", "maximize"):
                if key in update_params:
                    val = update_params[key]
                    for group in optimizer.param_groups:
                        if key in group:
                            if dry_run:
                                report.optimizer_updates.append(f"(dry-run) {key}: {group.get(key)} -> {val}")
                            else:
                                group[key] = val
                                report.optimizer_updates.append(f"{key} -> {val}")

            # Dondurma sonrası param_group filtreleme
            if filter_frozen_params and not dry_run:
                _filter_param_groups_after_freeze(optimizer)

            updater_logger.info("Optimizer parametre güncellemesi tamamlandı.")
            return report

        except KeyError as ke:
            updater_logger.error(f"Parametre anahtarı hatalı: {str(ke)}", exc_info=True)
            raise ValueError(f"Geçersiz parametre anahtarı: {str(ke)}") from ke
        except Exception as e:
            updater_logger.error(f"Optimizer güncellenirken hata oluştu: {str(e)}", exc_info=True)
            raise RuntimeError("Optimizer güncelleme işlemi başarısız oldu.") from e

    # -------------------- SCHEDULER -------------------- #
    @staticmethod
    def update_scheduler(
        scheduler: LRScheduler,
        update_params: Mapping[str, Any],
        *,
        dry_run: bool = False,
    ) -> UpdateReport:
        """
        Scheduler parametrelerini günceller (uygun alanlar için).

        ReduceLROnPlateau: factor, patience, threshold, cooldown, min_lr, threshold_mode
        StepLR: step_size, gamma
        MultiStepLR: milestones, gamma
        CosineAnnealingLR: T_max, eta_min
        OneCycleLR: max_lr, total_steps, epochs, steps_per_epoch, pct_start, anneal_strategy, div_factor, final_div_factor
        """
        report = UpdateReport(model_updates=[], optimizer_updates=[], scheduler_updates=[])

        try:
            updater_logger.info("Scheduler güncelleme işlemi başlatılıyor...")

            def _set_attr_if_present(obj: Any, key: str, val: Any) -> None:
                if hasattr(obj, key):
                    if dry_run:
                        report.scheduler_updates.append(f"(dry-run) {key}: {getattr(obj, key)} -> {val}")
                    else:
                        setattr(obj, key, val)
                        report.scheduler_updates.append(f"{key} -> {val}")
                else:
                    updater_logger.warning(f"{key} parametresi scheduler'da bulunamadı. Güncelleme atlandı.")

            if isinstance(scheduler, ReduceLROnPlateau):
                for k in ("factor", "patience", "threshold", "cooldown", "min_lrs", "threshold_mode", "verbose"):
                    if k in update_params:
                        _set_attr_if_present(scheduler, k, update_params[k])

            elif isinstance(scheduler, StepLR):
                for k in ("step_size", "gamma", "last_epoch", "verbose"):
                    if k in update_params:
                        _set_attr_if_present(scheduler, k, update_params[k])

            elif isinstance(scheduler, MultiStepLR):
                for k in ("milestones", "gamma", "last_epoch", "verbose"):
                    if k in update_params:
                        _set_attr_if_present(scheduler, k, update_params[k])

            elif isinstance(scheduler, CosineAnnealingLR):
                for k in ("T_max", "eta_min", "last_epoch", "verbose"):
                    if k in update_params:
                        _set_attr_if_present(scheduler, k, update_params[k])

            elif isinstance(scheduler, OneCycleLR):
                # OneCycleLR çoğu parametreyi init sırasında ister; güncelleme sınırlı olabilir.
                for k in ("max_lr", "pct_start", "anneal_strategy", "div_factor", "final_div_factor", "three_phase", "verbose"):
                    if k in update_params:
                        _set_attr_if_present(scheduler, k, update_params[k])
                # steps_per_epoch/epochs değişimi genelde yeniden yaratmayı gerektirir.
                for k in ("steps_per_epoch", "epochs", "total_steps"):
                    if k in update_params:
                        updater_logger.warning(f"{k} güncellemesi OneCycleLR için yeniden oluşturma gerektirebilir.")

            else:
                # Genel fallback: objede olanları set et
                for k, v in update_params.items():
                    _set_attr_if_present(scheduler, k, v)

            # Öğrenme oranı güncellemesi scheduler.optimizer üzerinden yapılabilir
            if hasattr(scheduler, "optimizer") and ("learning_rate" in update_params or "lr" in update_params):
                lr = update_params.get("learning_rate", update_params.get("lr"))
                ModelUpdater.update_learning_rate(scheduler.optimizer, lr, dry_run=dry_run)
                report.scheduler_updates.append(f"optimizer.lr -> {lr}")

            updater_logger.info("Scheduler parametre güncellemesi tamamlandı.")
            return report

        except Exception as e:
            updater_logger.error(f"Scheduler güncellenirken hata oluştu: {str(e)}", exc_info=True)
            raise RuntimeError("Scheduler güncelleme işlemi başarısız oldu.") from e

    # -------------------- LR kısayol -------------------- #
    @staticmethod
    def update_learning_rate(optimizer: Optimizer, new_lr: float, *, dry_run: bool = False) -> UpdateReport:
        """
        Optimizer'ın öğrenme oranını (learning rate) günceller.
        """
        report = UpdateReport(model_updates=[], optimizer_updates=[], scheduler_updates=[])
        try:
            _validate_type("learning_rate", new_lr, (int, float))
            _validate_range("learning_rate", float(new_lr), 0.0)
            updater_logger.info(f"Learning rate güncelleme işlemi başlatılıyor... Yeni learning rate: {new_lr}")
            for group in optimizer.param_groups:
                if dry_run:
                    report.optimizer_updates.append(f"(dry-run) lr: {group.get('lr')} -> {new_lr}")
                else:
                    group["lr"] = float(new_lr)
                    report.optimizer_updates.append(f"lr -> {new_lr}")
            updater_logger.info("Learning rate güncellemesi tamamlandı.")
            return report
        except Exception as e:
            updater_logger.error(f"Learning rate güncellenirken hata oluştu: {str(e)}", exc_info=True)
            raise RuntimeError("Learning rate güncelleme işlemi başarısız oldu.") from e

    # -------------------- Toplu API -------------------- #
    @staticmethod
    def bulk_update(
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        model: Optional[nn.Module] = None,
        update_params: Optional[Mapping[str, Any]] = None,
        *,
        dry_run: bool = False,
        filter_frozen_params: bool = True,
    ) -> UpdateReport:
        """
        Model, optimizer ve scheduler için toplu güncelleme işlemi yapar.

        update_params şeması (esnek):
        {
          "model": { ... ModelUpdater.update_model ... },
          "optimizer": { ... ModelUpdater.update_optimizer ... },
          "scheduler": { ... ModelUpdater.update_scheduler ... }
        }
        """
        if update_params is None:
            update_params = {}

        report = UpdateReport(model_updates=[], optimizer_updates=[], scheduler_updates=[])

        try:
            updater_logger.info("Toplu güncelleme işlemi başlatılıyor...")

            if model and "model" in update_params:
                r = ModelUpdater.update_model(model, update_params["model"], dry_run=dry_run)
                report.model_updates.extend(r.model_updates)

            if optimizer and "optimizer" in update_params:
                r = ModelUpdater.update_optimizer(
                    optimizer,
                    update_params["optimizer"],
                    filter_frozen_params=filter_frozen_params,
                    dry_run=dry_run,
                )
                report.optimizer_updates.extend(r.optimizer_updates)

            if scheduler and "scheduler" in update_params:
                r = ModelUpdater.update_scheduler(scheduler, update_params["scheduler"], dry_run=dry_run)
                report.scheduler_updates.extend(r.scheduler_updates)

            updater_logger.info("Toplu güncelleme işlemi tamamlandı.")
            return report

        except Exception as e:
            updater_logger.error(f"Toplu güncelleme sırasında hata oluştu: {str(e)}", exc_info=True)
            raise RuntimeError("Toplu güncelleme işlemi başarısız oldu.") from e
