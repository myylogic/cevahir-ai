# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: config_manager.py
Modül: training_system/v2/core
Görev: Config Manager - V2 TrainingManager Config Hazırlama. V2 TrainingManager
       için config hazırlama ve adaptasyonu. Base config'i V2 TrainingManager
       formatına dönüştürür.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (config hazırlama)
- Design Patterns: Adapter Pattern (config adaptasyonu)
- Endüstri Standartları: Config management best practices

KULLANIM:
- V2 TrainingManager için config hazırlamak için
- Base config'i adapte etmek için
- Config normalizasyonu için

BAĞIMLILIKLAR:
- TrainingManager: V2 eğitim yönetimi

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from typing import Dict, Any, Optional


class ConfigManager:
    """V2 TrainingManager için config hazırlayan manager"""
    
    def __init__(self, logger: Optional[Any] = None):
        """
        Args:
            logger: Logger instance (opsiyonel)
        """
        self.logger = logger
    
    def prepare_training_config(
        self,
        base_config: Dict[str, Any],
        tokenizer_core,
        device: str
    ) -> Dict[str, Any]:
        """
        V2 TrainingManager için config hazırla.
        
        Args:
            base_config: Base config dictionary
            tokenizer_core: TokenizerCore instance
            device: Device string ("cuda" or "cpu")
            
        Returns:
            V2 TrainingManager config dictionary
        """
        vocab_size = tokenizer_core.get_vocab_size()
        # PAD/BOS/EOS/UNK token ID'lerini _special_ids() ile al (TrainingLoop advanced_metrics ile uyum)
        special_ids = tokenizer_core._special_ids()
        pad_token_id = special_ids.get("<PAD>", 0)
        bos_token_id = special_ids.get("<BOS>", 1)
        eos_token_id = special_ids.get("<EOS>", 2)
        unk_token_id = special_ids.get("<UNK>", 3)
        
        config = {
            "vocab_size": vocab_size,
            "epochs": int(base_config.get("epochs", 10)),
            "batch_size": int(base_config.get("batch_size", 8)),
            "seq_len": int(base_config.get("max_seq_length", 512)),
            "device": device,
            "max_grad_norm": float(base_config.get("max_grad_norm", 1.0)),
            "grad_accum_steps": int(base_config.get("grad_accum_steps", 1)),
            "use_amp": bool(base_config.get("use_amp", True)),
            "early_stopping_patience": int(base_config.get("early_stopping_patience", 3)),
            "checkpoint_dir": str(base_config.get("checkpoint_dir", "./checkpoints")),
            "enable_tensorboard": bool(base_config.get("enable_tensorboard", base_config.get("use_tensorboard", True))),
            "tensorboard_log_dir": str(base_config.get("tensorboard_log_dir", "./runs")),
            "track_memory": bool(base_config.get("track_memory", True)),
            "track_performance": bool(base_config.get("track_performance", True)),
            "scheduler_type": str(base_config.get("scheduler_type", "ReduceLROnPlateau")),
            "scheduler_kwargs": dict(base_config.get("scheduler_kwargs", {})),
            "pad_token_id": pad_token_id,
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id,
            "unk_token_id": unk_token_id,
            "warmup_steps": int(base_config.get("warmup_steps", 0)),
            "warmup_start_factor": float(base_config.get("warmup_start_factor", 0.1)),
            "embedding_warmup_factor": float(base_config.get("embedding_warmup_factor", 1.0)),
            "enable_training_analytics": bool(base_config.get("enable_training_analytics", False)),
            "max_checkpoints": int(base_config.get("max_checkpoints", 5)),
            "gradient_explosion_threshold": float(base_config.get("gradient_explosion_threshold", 10.0)),
            "calculate_advanced_metrics": bool(base_config.get("calculate_advanced_metrics", False)),
        }
        
        if self.logger:
            self.logger.debug(f"V2 TrainingManager config hazırlandı: {len(config)} parametre")
        
        return config

