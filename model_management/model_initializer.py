# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: model_initializer.py
Modül: model_management
Görev: Model Initializer - Model, optimizer, loss fonksiyonu ve scheduler başlatma
       işlemleri. Model instance oluşturma, optimizer ve scheduler başlatma,
       device yönetimi ve config doğrulama işlemlerini yapar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (model başlatma)
- Design Patterns: Initializer Pattern (model başlatma)
- Endüstri Standartları: Model initialization best practices

KULLANIM:
- Model instance oluşturmak için
- Optimizer başlatmak için
- Scheduler başlatmak için
- Device yönetimi için

BAĞIMLILIKLAR:
- torch: PyTorch işlemleri
- torch.nn: Neural network modülleri
- torch.optim: Optimizer modülleri

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.optim as optim

# Bu modül logunu ayrı tutalım; root logger'ı yeniden yapılandırmayalım.
initializer_logger = logging.getLogger("model_initializer")
if not initializer_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    handler.setFormatter(formatter)
    initializer_logger.addHandler(handler)
    initializer_logger.setLevel(logging.INFO)


# ----------------------------- Yardımcı Fonksiyonlar ----------------------------- #

def _resolve_device(config: Dict[str, Any]) -> torch.device:
    """config['device'] var ise kullan; yoksa cuda/mps/cpu sırasıyla seç."""
    dev = str(config.get("device", "")).strip().lower()
    # Eğer açıkça "cpu" belirtilmişse, CPU kullan (testler için önemli!)
    if dev == "cpu":
        return torch.device("cpu")
    if dev in {"cuda", "gpu"} and torch.cuda.is_available():
        return torch.device("cuda")
    if dev == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if not dev:
        # Otomatik seçim (config'te device yoksa)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    # kullanıcı farklı bir şey verdiyse düşelim cpu'ya
    initializer_logger.warning(f"Tanınmayan device='{dev}', CPU kullanılacak.")
    return torch.device("cpu")


def _apply_seed(config: Dict[str, Any]) -> None:
    """Deterministik davranış istenirse seed uygula (opsiyonel)."""
    seed = config.get("seed")
    if seed is None:
        return
    try:
        import random
        import numpy as np  # type: ignore
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        if config.get("deterministic", False):
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
        initializer_logger.info(f"Deterministik seed uygulandı: {seed}")
    except Exception as e:
        initializer_logger.warning(f"Seed uygulanırken hata: {e}")


def _with_aliases(config: Dict[str, Any], aliases: Dict[str, str]) -> Dict[str, Any]:
    """
    'learning_rate' -> 'lr' gibi alias eşlemeleri uygular.
    Hedef anahtar zaten varsa override etmez.
    """
    out = dict(config)
    for src, dst in aliases.items():
        if src in config and dst not in out:
            out[dst] = config[src]
    return out


def _filter_kwargs_for_ctor(model_class: Type[nn.Module], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Model ctor imzasına göre güvenli argüman süzme."""
    try:
        sig = inspect.signature(model_class.__init__)
        allowed = set(p.name for p in sig.parameters.values() if p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD))
        # __init__(self, ...) → 'self' dışındakiler
        allowed.discard("self")
        return {k: v for k, v in cfg.items() if k in allowed}
    except (TypeError, ValueError):
        # imza okunamazsa konservatif davran: hiçbir şey geçme
        initializer_logger.warning("Model imzası okunamadı; ctor'a konfig parametreleri geçirilmeyecek.")
        return {}


def _split_decay_params(model: nn.Module, no_decay_keywords: Iterable[str]) -> List[Dict[str, Any]]:
    """
    Weight decay'i bias/LayerNorm gibi paramlardan hariç tutmak için param grupları oluşturur.
    """
    no_decay_keywords = list(no_decay_keywords)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k.lower() in n.lower() for k in no_decay_keywords):
            no_decay.append(p)
        else:
            decay.append(p)
    groups: List[Dict[str, Any]] = []
    if decay:
        groups.append({"params": decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


# ----------------------------- Ana Sınıf ----------------------------- #

class ModelInitializer:
    """
    ModelInitializer:
    Neural network modeli, optimizer, kayıp fonksiyonu ve scheduler başlatımı için yardımcı sınıf.
    """

    # ------------------------- MODEL ------------------------- #
    @staticmethod
    def build_model(
        model_class: Type[nn.Module],
        config: Dict[str, Any],
        *,
        extra_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        compile_model: Optional[bool] = None,
    ) -> nn.Module:
        """
        Modeli güvenli şekilde inşa eder, cihaza taşır ve opsiyonel olarak torch.compile eder.

        Args:
            model_class: Oluşturulacak model tipi.
            config: Konfig.
            extra_kwargs: Ek ctor argümanları (opsiyonel).
            device: Zorla cihaz (opsiyonel).
            compile_model: Torch 2.x ile derleme bayrağı (opsiyonel; yoksa config['torch_compile']).

        Returns:
            nn.Module
        """
        _apply_seed(config)

        dev = device or _resolve_device(config)

        # alias: learning_rate -> lr gibi (bazı modeller ctor'da lr bekleyebilir)
        cfg_for_ctor = _with_aliases(
            config,
            aliases={"learning_rate": "lr", "n_heads": "num_heads", "drop_rate": "dropout"},
        )
        if extra_kwargs:
            cfg_for_ctor.update(extra_kwargs)

        # ctor imzasına uygunları geçir
        ctor_kwargs = _filter_kwargs_for_ctor(model_class, cfg_for_ctor)

        # Bazı modeller vocab_size'ı kesin ister → uyarı verelim
        if "vocab_size" not in ctor_kwargs and "vocab_size" in config:
            ctor_kwargs["vocab_size"] = config["vocab_size"]
        
        # CevahirNeuralNetwork için eksik parametreleri ekle (default değerlerle)
        if model_class.__name__ == "CevahirNeuralNetwork":
            required_params = {
                "learning_rate": config.get("learning_rate", 1e-3),
                "dropout": config.get("dropout", 0.1),
                "embed_dim": config.get("embed_dim", config.get("d_model", 512)),
                "seq_proj_dim": config.get("seq_proj_dim", config.get("d_model", 512)),
                "num_heads": config.get("num_heads", config.get("n_heads", 8)),
            }
            for param_name, default_value in required_params.items():
                if param_name not in ctor_kwargs:
                    ctor_kwargs[param_name] = default_value
                    initializer_logger.warning(
                        f"CevahirNeuralNetwork için {param_name} bulunamadı, "
                        f"default değer kullanılıyor: {default_value}"
                    )

        try:
            initializer_logger.info(f"Model oluşturuluyor: {model_class.__name__} (device={dev})")
            model = model_class(**ctor_kwargs)
            model = model.to(dev)

            # Torch 2.0+ derleme (opsiyonel)
            if compile_model is None:
                compile_model = bool(config.get("torch_compile", False))
            if compile_model and hasattr(torch, "compile"):
                compile_kwargs = {
                    "mode": config.get("torch_compile_mode", "default"),
                    "fullgraph": bool(config.get("torch_compile_fullgraph", False)),
                    "dynamic": bool(config.get("torch_compile_dynamic", False)),
                }
                try:
                    model = torch.compile(model, **compile_kwargs)  # type: ignore[attr-defined]
                    initializer_logger.info(f"Model torch.compile ile derlendi: {compile_kwargs}")
                except Exception as ce:
                    initializer_logger.warning(f"torch.compile başarısız, derlenmemiş model kullanılacak: {ce}")

            initializer_logger.info(f"Model hazır: {type(model).__name__}")
            return model

        except Exception as e:
            initializer_logger.error(f"Model başlatılırken hata: {e}", exc_info=True)
            raise RuntimeError("Model başlatılamadı.") from e

    # ------------------------- OPTIMIZER ------------------------- #
    @staticmethod
    def initialize_optimizer(
        model: nn.Module,
        config: Dict[str, Any],
    ) -> optim.Optimizer:
        """
        Seçilebilir optimizer kurar ve param grupları ile no-decay uygular.

        Desteklenenler: adamw, adam, sgd, radam, rmsprop (isimler küçük/büyük harf duyar değil)
        """
        try:
            opt_name = str(config.get("optimizer", "adamw")).lower()
            lr = float(config.get("learning_rate", 1e-3))
            weight_decay = float(config.get("weight_decay", 0.0))
            betas = tuple(config.get("betas", (0.9, 0.999)))
            eps = float(config.get("eps", 1e-8))
            momentum = float(config.get("momentum", 0.9))
            nesterov = bool(config.get("nesterov", False))

            # bias/LayerNorm/BatchNorm için wd=0 param grupları
            no_decay_keywords = config.get(
                "no_weight_decay_keywords",
                ["bias", "layernorm", "ln", "batchnorm", "bn", "norm.weight", "norm"],
            )
            
            # Embedding LR: config'te embedding_lr_scale verilir (train.py: 1.0 = ana LR ile aynı, endüstri standardı).
            # Eski 0.1 değeri EOS/nadir token öğrenimini zayıflatıyordu; varsayılan 1.0 (scale yok).
            embedding_lr_scale = float(config.get("embedding_lr_scale", 1.0))  # Default: 1.0 (scale yok)
            
            # Parameter groups: Embedding, No-decay, Decay
            embedding_params = []
            decay_params = []
            no_decay_params = []
            
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                # Embedding layer parametreleri
                if "embedding" in n.lower() and "embedding.weight" in n.lower():
                    embedding_params.append(p)
                # No-decay parametreleri (bias, norm, etc.)
                elif any(k.lower() in n.lower() for k in no_decay_keywords):
                    no_decay_params.append(p)
                # Normal decay parametreleri
                else:
                    decay_params.append(p)
            
            # Parameter groups oluştur
            param_groups = []
            if embedding_params:
                param_groups.append({
                    "params": embedding_params,
                    "lr": lr * embedding_lr_scale,  # [OK] Embedding için düşük LR
                    "weight_decay": weight_decay,
                })
                initializer_logger.info(
                    "Embedding layer LR: %s (base_lr=%s × embedding_lr_scale=%s)",
                    f"{lr * embedding_lr_scale:.2e}",
                    f"{lr:.2e}",
                    embedding_lr_scale,
                )
            if decay_params:
                param_groups.append({
                    "params": decay_params,
                    "lr": lr,  # [OK] Normal LR
                    "weight_decay": weight_decay,
                })
            if no_decay_params:
                param_groups.append({
                    "params": no_decay_params,
                    "lr": lr,  # [OK] Normal LR
                    "weight_decay": 0.0,
                })
            
            # Eğer embedding parametresi yoksa eski yöntemi kullan (geriye dönük uyumluluk)
            if not embedding_params:
                param_groups = _split_decay_params(model, no_decay_keywords)
                for g in param_groups:
                    g.setdefault("weight_decay", weight_decay)
                    g.setdefault("lr", lr)  # [OK] Eski yöntem için de LR ekle

            initializer_logger.info(f"Optimizer başlatılıyor: {opt_name} (lr={lr}, wd={weight_decay})")

            if opt_name == "adamw":
                # PyTorch 2.2+ 'fused' opsiyonunu sessizce destekleyelim
                fused = bool(config.get("fused", False)) and hasattr(optim, "AdamW")
                kwargs = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
                if fused:
                    kwargs["fused"] = True  # type: ignore[assignment]
                optimizer = optim.AdamW(param_groups, **kwargs)
            elif opt_name == "adam":
                optimizer = optim.Adam(param_groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
            elif opt_name == "radam":
                optimizer = optim.RAdam(param_groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
            elif opt_name == "rmsprop":
                optimizer = optim.RMSprop(param_groups, lr=lr, eps=eps, weight_decay=weight_decay, momentum=momentum)
            elif opt_name == "sgd":
                optimizer = optim.SGD(
                    param_groups, lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay
                )
            else:
                raise ValueError(f"Desteklenmeyen optimizer: {opt_name}")

            initializer_logger.info(f"Optimizer hazır: {optimizer.__class__.__name__}")
            return optimizer

        except Exception as e:
            initializer_logger.error(f"Optimizer başlatılırken hata oluştu: {e}", exc_info=True)
            raise RuntimeError("Optimizer başlatılamadı.") from e

    # ------------------------- CRITERION ------------------------- #
    @staticmethod
    def initialize_criterion(config: Dict[str, Any]) -> nn.Module:
        """
        Kayıp fonksiyonunu kurar.
        criterion türleri: 'cross_entropy' (varsayılan), 'bce_with_logits', 'mse', 'smooth_l1'
        """
        try:
            name = str(config.get("criterion", "cross_entropy")).lower()
            label_smoothing = float(config.get("label_smoothing", 0.0))
            reduction = str(config.get("reduction", "mean"))
            ignore_index = int(config.get("ignore_index", -100))

            if name in {"cross_entropy", "ce"}:
                # class_weights: list/tensor olabilir
                class_weights = config.get("class_weights")
                weight = None
                if class_weights is not None:
                    weight = torch.as_tensor(class_weights, dtype=torch.float32)
                criterion = nn.CrossEntropyLoss(
                    weight=weight,
                    label_smoothing=label_smoothing,
                    ignore_index=ignore_index,
                    reduction=reduction,
                )
                # PAD token'ları loss'tan çıkar (ignore_index=0). Eğitim başında doğrula.
                initializer_logger.info(
                    f"CrossEntropyLoss ignore_index={ignore_index} (0=PAD ignore, -100=hepsini say)"
                )
            elif name in {"bce_with_logits", "bce"}:
                pos_weight = config.get("pos_weight")
                pos_weight_tensor = torch.as_tensor(pos_weight, dtype=torch.float32) if pos_weight is not None else None
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor, reduction=reduction)
            elif name == "mse":
                criterion = nn.MSELoss(reduction=reduction)
            elif name in {"smooth_l1", "huber"}:
                beta = float(config.get("smooth_l1_beta", 1.0))
                criterion = nn.SmoothL1Loss(beta=beta, reduction=reduction)
            else:
                raise ValueError(f"Desteklenmeyen criterion: {name}")

            initializer_logger.info(f"Loss fonksiyonu hazır: {criterion.__class__.__name__}")
            return criterion

        except Exception as e:
            initializer_logger.error(f"Loss fonksiyonu başlatılırken hata oluştu: {e}", exc_info=True)
            raise RuntimeError("Loss fonksiyonu başlatılamadı.") from e

    # ------------------------- SCHEDULER ------------------------- #
    @staticmethod
    def initialize_scheduler(
        optimizer: optim.Optimizer,
        config: Dict[str, Any],
    ) -> Optional[optim.lr_scheduler.LRScheduler]:
        """
        LR scheduler kurar.
        Desteklenenler:
            - reduce_on_plateau (varsayılan)
            - cosine, cosine_warm_restarts
            - step, exponential
            - onecycle (epoch ve steps_per_epoch ister)
        """
        try:
            sched = config.get("scheduler")  # dict olabilir
            sched_type = (sched or {}).get("type") if isinstance(sched, dict) else None
            if sched_type is None:
                # [OK] Backward compatibility: hem "scheduler" hem "scheduler_type" kontrol et
                sched_type = config.get("scheduler_type")
                if sched_type is None and isinstance(sched, str):
                    sched_type = sched
                if sched_type is None:
                    sched_type = "reduce_on_plateau"
                sched_type = str(sched_type).lower()

            initializer_logger.info(f"Scheduler başlatılıyor: {sched_type}")

            if sched_type in {"reduce_on_plateau", "plateau", "rop"}:
                factor = float(config.get("lr_decay_factor", 0.1))
                patience = int(config.get("lr_decay_patience", 10))
                threshold = float(config.get("lr_threshold", 1e-4))
                min_lr = float(config.get("lr_min", 5e-6))  # [OK] YENİ: Minimum LR desteği
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=factor, patience=patience, threshold=threshold, min_lr=min_lr
                )
            elif sched_type in {"cosine", "cosineannealing"}:
                T_max = int(config.get("cosine_T_max", 10))
                eta_min = float(config.get("cosine_eta_min", 0.0))
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
            elif sched_type in {"cosine_warm_restarts", "cosinewarmrestarts", "cawr"}:
                T_0 = int(config.get("cawr_T_0", 10))
                T_mult = int(config.get("cawr_T_mult", 1))
                eta_min = float(config.get("cawr_eta_min", 0.0))
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
                )
            elif sched_type in {"step", "steplr"}:
                # Config'te scheduler_step_size veya step_size olabilir
                step_size = int(config.get("scheduler_step_size", config.get("step_size", 10)))
                # Config'te scheduler_gamma veya gamma olabilir
                gamma = float(config.get("scheduler_gamma", config.get("gamma", 0.1)))
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            elif sched_type in {"exponential", "explr"}:
                gamma = float(config.get("gamma", 0.95))
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
            elif sched_type in {"onecycle", "onecyclelr"}:
                max_lr = float(config.get("onecycle_max_lr", config.get("learning_rate", 1e-3)))
                steps_per_epoch = int(config.get("steps_per_epoch", 0))
                epochs = int(config.get("epochs", 0))
                if steps_per_epoch <= 0 or epochs <= 0:
                    raise ValueError("OneCycleLR için 'steps_per_epoch' ve 'epochs' pozitif olmalıdır.")
                pct_start = float(config.get("onecycle_pct_start", 0.3))
                div_factor = float(config.get("onecycle_div_factor", 25.0))
                final_div_factor = float(config.get("onecycle_final_div_factor", 1e4))
                anneal_strategy = str(config.get("onecycle_anneal", "cos")).lower()
                scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=max_lr,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    pct_start=pct_start,
                    div_factor=div_factor,
                    final_div_factor=final_div_factor,
                    anneal_strategy=anneal_strategy,
                )
            elif sched_type in {"none", "off", "disable"}:
                scheduler = None
            else:
                raise ValueError(f"Desteklenmeyen scheduler türü: {sched_type}")

            if scheduler is not None:
                initializer_logger.info(f"Scheduler hazır: {scheduler.__class__.__name__}")
            else:
                initializer_logger.info("Scheduler devre dışı.")
            return scheduler

        except Exception as e:
            initializer_logger.error(f"Scheduler başlatılırken hata oluştu: {e}", exc_info=True)
            raise RuntimeError("Scheduler başlatılamadı.") from e

    # ------------------------- Toplu Yardımcılar ------------------------- #
    @staticmethod
    def build_training_components(
        model: nn.Module,
        config: Dict[str, Any],
    ) -> Tuple[optim.Optimizer, nn.Module, Optional[optim.lr_scheduler.LRScheduler]]:
        """
        Tek çağrıda optimizer, criterion ve scheduler döndürür.
        """
        optimizer = ModelInitializer.initialize_optimizer(model, config)
        criterion = ModelInitializer.initialize_criterion(config)
        scheduler = ModelInitializer.initialize_scheduler(optimizer, config)
        return optimizer, criterion, scheduler
