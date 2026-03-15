# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: config_manager_v3.py
Modül: training_system/v3/core
Görev: Config Manager V3 - Tüm V3 parametrelerini TrainingManager'a geçirir.
       V2 ConfigManager sadece ~20 parametre geçiriyordu. V3 55+ parametre geçirir.
       V3 parametreler: entropy_coeff, focal loss, curriculum, EMA, SAM, SWA,
       Lookahead, LLRD, scheduled sampling, gradient health, token dist, inference probe.

MİMARİ:
- SOLID: Single Responsibility (config hazırlama ve validasyon)
- Design Patterns: Adapter Pattern (base config → V3 TrainingManager config)
- Endüstri Standartları: MLOps config management

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.

================================================================================
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ConfigManagerV3:
    """
    Config Manager V3 — Eksiksiz parametre aktarımı.

    V2 farkı:
    - V2: ~20 temel parametre
    - V3: 55+ parametre (entropy, focal, curriculum, EMA, SAM, SWA, LLRD, ...)
    - V3 validasyon: Parametre tipi ve aralık kontrolü
    - V3 data config: GPU batching parametreleri
    """

    def __init__(self, logger_instance: Optional[Any] = None):
        self.logger = logger_instance or logging.getLogger(self.__class__.__name__)

    def prepare_training_config(
        self,
        base_config: Dict[str, Any],
        tokenizer_core,
        device: str,
    ) -> Dict[str, Any]:
        """
        V3 TrainingManager için eksiksiz config hazırla.

        Args:
            base_config: train.py'deki TRAIN_CONFIG
            tokenizer_core: TokenizerCore instance
            device: "cuda" veya "cpu"

        Returns:
            V3 TrainingManager config dictionary (55+ parametre)
        """
        # Tokenizer'dan özel token ID'leri al
        special_ids = tokenizer_core._special_ids()
        pad_token_id = special_ids.get("<PAD>", 0)
        bos_token_id = special_ids.get("<BOS>", 1)
        eos_token_id = special_ids.get("<EOS>", 2)
        unk_token_id = special_ids.get("<UNK>", 3)
        vocab_size = tokenizer_core.get_vocab_size()

        # ─────────────────────────────────────────────
        # 1. TEMEL EĞİTİM PARAMETRELERİ (V2 uyumlu)
        # ─────────────────────────────────────────────
        config = {
            # Model
            "vocab_size": vocab_size,
            "pad_token_id": pad_token_id,
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id,
            "unk_token_id": unk_token_id,

            # Eğitim döngüsü
            "epochs": int(base_config.get("epochs", 10)),
            "batch_size": int(base_config.get("batch_size", 8)),
            "seq_len": int(base_config.get("max_seq_length", 512)),
            "device": device,
            "max_grad_norm": float(base_config.get("max_grad_norm", 1.0)),
            "grad_accum_steps": int(base_config.get("grad_accum_steps", 1)),
            "use_amp": bool(base_config.get("use_amp", True)),
            "early_stopping_patience": int(base_config.get("early_stopping_patience", 3)),

            # Checkpoint
            "checkpoint_dir": str(base_config.get("checkpoint_dir", "./checkpoints")),
            "max_checkpoints": int(base_config.get("max_checkpoints", 5)),
            "save_every_n_epochs": int(base_config.get("save_every_n_epochs", 10)),

            # TensorBoard
            "enable_tensorboard": bool(
                base_config.get("enable_tensorboard", base_config.get("use_tensorboard", True))
            ),
            "tensorboard_log_dir": str(base_config.get("tensorboard_log_dir", "./runs")),

            # İzleme
            "track_memory": bool(base_config.get("track_memory", True)),
            "track_performance": bool(base_config.get("track_performance", True)),
            "calculate_advanced_metrics": bool(base_config.get("calculate_advanced_metrics", False)),
            "enable_training_analytics": bool(base_config.get("enable_training_analytics", False)),
            "gradient_explosion_threshold": float(base_config.get("gradient_explosion_threshold", 10.0)),

            # Scheduler
            "scheduler_type": str(base_config.get("scheduler_type", "ReduceLROnPlateau")),
            "scheduler_kwargs": dict(base_config.get("scheduler_kwargs", {})),

            # Warmup
            "warmup_steps": int(base_config.get("warmup_steps", 0)),
            "warmup_start_factor": float(base_config.get("warmup_start_factor", 0.1)),
            "embedding_warmup_factor": float(base_config.get("embedding_warmup_factor", 1.0)),

            # Data
            "train_val_split": float(base_config.get("train_val_split", 0.8)),
            "split_seed": int(base_config.get("split_seed", 42)),
        }

        # ─────────────────────────────────────────────
        # 2. LOSS FONKSİYONU V3
        # ─────────────────────────────────────────────
        config.update({
            "label_smoothing": float(base_config.get("label_smoothing", 0.1)),
            "eos_token_weight": float(base_config.get("eos_token_weight", 1.0)),
            "entropy_coeff": float(base_config.get("entropy_coeff", 0.0)),
            "use_focal_loss": bool(base_config.get("use_focal_loss", False)),
            "focal_gamma": float(base_config.get("focal_gamma", 2.0)),
            "aux_loss_weight": float(base_config.get("aux_loss_weight", 0.01)),
        })

        # ─────────────────────────────────────────────
        # 3. OPTİMİZER V3
        # ─────────────────────────────────────────────
        config.update({
            # SAM (Sharpness-Aware Minimization)
            "use_sam": bool(base_config.get("use_sam", False)),
            "sam_rho": float(base_config.get("sam_rho", 0.05)),
            "sam_adaptive": bool(base_config.get("sam_adaptive", False)),

            # Lookahead
            "use_lookahead": bool(base_config.get("use_lookahead", False)),
            "lookahead_k": int(base_config.get("lookahead_k", 5)),
            "lookahead_alpha": float(base_config.get("lookahead_alpha", 0.5)),

            # AGC (Adaptive Gradient Clipping — Brock et al. 2021)
            "use_agc": bool(base_config.get("use_agc", False)),
            "agc_clip_factor": float(base_config.get("agc_clip_factor", 0.01)),
            "agc_eps": float(base_config.get("agc_eps", 1e-3)),

            # Gradient Noise (Neelakantan et al. 2015)
            "use_gradient_noise": bool(base_config.get("use_gradient_noise", False)),
            "gradient_noise_eta": float(base_config.get("gradient_noise_eta", 0.3)),
            "gradient_noise_gamma": float(base_config.get("gradient_noise_gamma", 0.55)),
        })

        # ─────────────────────────────────────────────
        # 4. EMA / SWA
        # ─────────────────────────────────────────────
        config.update({
            # EMA (Exponential Moving Average)
            "use_ema": bool(base_config.get("use_ema", True)),
            "ema_decay": float(base_config.get("ema_decay", 0.999)),
            "ema_update_after_step": int(base_config.get("ema_update_after_step", 100)),
            "ema_update_every": int(base_config.get("ema_update_every", 10)),

            # SWA (Stochastic Weight Averaging — Izmailov et al. 2018)
            "use_swa": bool(base_config.get("use_swa", False)),
            "swa_start_epoch": int(base_config.get("swa_start_epoch", 80)),
            "swa_lr": float(base_config.get("swa_lr", 1e-5)),
            "swa_anneal_epochs": int(base_config.get("swa_anneal_epochs", 10)),
            "swa_anneal_strategy": str(base_config.get("swa_anneal_strategy", "cos")),
        })

        # ─────────────────────────────────────────────
        # 5. LEARNING RATE SCHEDULE V3
        # ─────────────────────────────────────────────
        config.update({
            # LLRD (Layer-wise Learning Rate Decay)
            "use_llrd": bool(base_config.get("use_llrd", False)),
            "llrd_decay_factor": float(base_config.get("llrd_decay_factor", 0.9)),

            # Cosine Annealing with Restarts (SGDR — Loshchilov & Hutter 2016)
            "use_cosine_restarts": bool(base_config.get("use_cosine_restarts", False)),
            "cosine_restart_period": int(base_config.get("cosine_restart_period", 10)),
            "cosine_restart_factor": float(base_config.get("cosine_restart_factor", 1.0)),
        })

        # ─────────────────────────────────────────────
        # 6. SCHEDULED SAMPLING (Bengio et al. 2015)
        # ─────────────────────────────────────────────
        config.update({
            "use_scheduled_sampling": bool(base_config.get("use_scheduled_sampling", True)),
            "ss_start_epoch": int(base_config.get("ss_start_epoch", 10)),
            "ss_decay_rate": float(base_config.get("ss_decay_rate", 0.05)),
            "min_teacher_forcing": float(base_config.get("min_teacher_forcing", 0.3)),
        })

        # ─────────────────────────────────────────────
        # 7. CURRICULUM LEARNING
        # ─────────────────────────────────────────────
        config.update({
            "use_curriculum": bool(base_config.get("use_curriculum", False)),
            "curriculum_strategy": str(base_config.get("curriculum_strategy", "length_based")),
            "curriculum_max_len_start": int(base_config.get("curriculum_max_len_start", 64)),
            "curriculum_warmup_epochs": int(base_config.get("curriculum_warmup_epochs", 20)),
        })

        # ─────────────────────────────────────────────
        # 8. GÜVENLİK VE NaN KURTARMA
        # ─────────────────────────────────────────────
        config.update({
            "nan_tolerance": int(base_config.get("nan_tolerance", 3)),
            "nan_lr_reduction": float(base_config.get("nan_lr_reduction", 0.5)),
            "spike_n_sigma": float(base_config.get("spike_n_sigma", 3.0)),
            "spike_window_size": int(base_config.get("spike_window_size", 20)),
            "spike_lr_reduction": float(base_config.get("spike_lr_reduction", 0.8)),
        })

        # ─────────────────────────────────────────────
        # 9. MONİTÖRİNG V3
        # ─────────────────────────────────────────────
        config.update({
            "inference_probe_interval": int(base_config.get("inference_probe_interval", 5)),
            "log_gradient_health": bool(base_config.get("log_gradient_health", True)),
            "log_token_dist": bool(base_config.get("log_token_dist", True)),
        })

        # ─────────────────────────────────────────────
        # 10. GPU BATCHING V3
        # ─────────────────────────────────────────────
        config.update({
            "use_bucket_batching": bool(base_config.get("use_bucket_batching", True)),
            "num_buckets": int(base_config.get("num_buckets", 32)),
            "use_dynamic_padding": bool(base_config.get("use_dynamic_padding", True)),
            "data_loader_num_workers": int(base_config.get("data_loader_num_workers", 0)),
            "data_loader_pin_memory": bool(base_config.get("data_loader_pin_memory", True)),
            "prefetch_factor": int(base_config.get("prefetch_factor", 2)),
            "persistent_workers": bool(base_config.get("persistent_workers", True)),
        })

        # ─────────────────────────────────────────────
        # 11. CACHE V3
        # ─────────────────────────────────────────────
        config.update({
            "cache_dir": str(base_config.get("cache_dir", ".cache/preprocessed_data")),
            "enable_data_cache": bool(base_config.get("enable_data_cache", True)),
            "cache_strict_mode": bool(base_config.get("cache_strict_mode", True)),
            "cache_verify_integrity": bool(base_config.get("cache_verify_integrity", True)),
        })

        # ─────────────────────────────────────────────
        # Validasyon
        # ─────────────────────────────────────────────
        self._validate(config)

        self.logger.info(
            f"[ConfigV3] Config hazırlandı: {len(config)} parametre "
            f"(vocab_size={vocab_size}, device={device})"
        )

        return config

    def _validate(self, config: Dict[str, Any]) -> None:
        """Kritik parametre validasyonu."""
        errors = []

        # Temel aralık kontrolü
        if config["label_smoothing"] < 0 or config["label_smoothing"] > 0.5:
            errors.append(f"label_smoothing={config['label_smoothing']} dışında [0, 0.5]")

        if config["entropy_coeff"] < 0 or config["entropy_coeff"] > 1.0:
            errors.append(f"entropy_coeff={config['entropy_coeff']} dışında [0, 1.0]")

        if config["ema_decay"] <= 0 or config["ema_decay"] >= 1.0:
            errors.append(f"ema_decay={config['ema_decay']} dışında (0, 1)")

        if config["batch_size"] <= 0:
            errors.append(f"batch_size={config['batch_size']} <= 0")

        if config["epochs"] <= 0:
            errors.append(f"epochs={config['epochs']} <= 0")

        if errors:
            self.logger.error(f"[ConfigV3] Validasyon hataları:\n" + "\n".join(f"  - {e}" for e in errors))
            raise ValueError(f"Config validasyon hatası: {'; '.join(errors)}")

        self.logger.debug(f"[ConfigV3] Validasyon geçildi")
