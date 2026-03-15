# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: training_service_v3.py
Modül: training_system/v3/core
Görev: Training Service V3 - Gelişmiş Eğitim Orkestratörü.
       V2'nin tam yeniden yazımı:
       - Strict cache mode (cache yoksa HATA FIRLATIRIR)
       - Advanced GPU batching (BucketSampler + DynamicPad)
       - Source-ID aware train/val split (data leakage yok)
       - V3 config (55+ parametre)
       - Training Management V3 entegrasyonu

ZORUNLU EĞİTİM AKIŞI:
       1. python tokenizer_management/train_bpe.py    [BPE eğitimi]
       2. python training_system/prepare_cache.py     [Cache hazırlama]
       3. python training_system/train.py             [Model eğitimi]

       Adım 3 → adım 2 olmadan BAŞLAMAZ (CacheNotFoundError).

MİMARİ:
- SOLID: Single Responsibility (eğitim orkestrasyonu)
- Design Patterns: Facade Pattern
- Endüstri Standartları: MLOps pipeline isolation, cache-first training

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.

================================================================================
"""

import os
import sys
import random
import logging
from typing import Dict, Any, Tuple, Optional, List

import torch

# Proje kök dizini
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from model_management.model_manager import ModelManager
from tokenizer_management.core.tokenizer_core import TokenizerCore
from tokenizer_management.config import BPE_CONFIG

try:
    from tokenizer_management.config import BPE_DETAILED_CONFIG, TOKENIZER_CONFIG
except ImportError:
    BPE_DETAILED_CONFIG = {}
    TOKENIZER_CONFIG = {"max_seq_length": 768}

# V3 modüller
from .config_manager_v3 import ConfigManagerV3
from ..data.cache_v3 import DataCacheV3, CacheNotFoundError
from ..data.dataloader_v3 import create_dataloaders_v3

# V2 modüller (geriye uyumluluk)
from training_system.v2.core.bpe_validator import BPEValidator
from training_system.v2.core.criterion_manager import CriterionManager
from training_system.v2.utils.warmup_calculator import calculate_warmup_steps

logger = logging.getLogger("TrainingServiceV3")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class TrainingServiceV3:
    """
    Training Service V3 — Gelişmiş Eğitim Orkestratörü.

    V2 → V3 kritik değişiklikler:
    - Cache yoksa eğitim BAŞLAMAZ (CacheNotFoundError)
    - BucketBatchSampler ile GPU padding waste minimize
    - DynamicPaddingCollator ile batch içi dinamik pad
    - Source-ID aware train/val split (data leakage yok)
    - V3 config (55+ parametre) tam aktarım
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config)
        self.logger = logger

        # BPE yolları
        vocab_path = self.config.get("vocab_path", BPE_CONFIG["vocab_file"])
        merges_path = self.config.get("merges_path", BPE_CONFIG["merges_file"])

        for path in (vocab_path, merges_path):
            directory = os.path.dirname(path)
            if directory and not os.path.isdir(directory):
                os.makedirs(directory, exist_ok=True)

        self.config.update({"vocab_path": vocab_path, "merges_path": merges_path})

        # Device
        self.device = self._setup_device()
        self.config["device"] = self.device

        # Data directory kontrolü
        if "data_dir" not in self.config or not os.path.isdir(str(self.config["data_dir"])):
            raise RuntimeError("[V3] TrainingService: Geçerli 'data_dir' config'te yok!")

        self.data_dir = self.config["data_dir"]

        # BPE Validator
        bpe_validator = BPEValidator(logger=self.logger)
        bpe_validator.validate_files(vocab_path, merges_path)

        # TokenizerCore
        self.tokenizer_core = TokenizerCore(self.config)

        # Vocab size (tokenizer'dan — config'i override et)
        vocab_size = self.tokenizer_core.get_vocab_size()
        self.config["vocab_size"] = vocab_size
        self.logger.info(f"[V3] Vocab size: {vocab_size}")

        # DataCache V3 — Strict Mode
        cache_dir = self.config.get("cache_dir", ".cache/preprocessed_data")
        cache_enabled = self.config.get("enable_data_cache", True)
        strict_mode = self.config.get("cache_strict_mode", True)
        verify_integrity = self.config.get("cache_verify_integrity", True)

        self.data_cache = DataCacheV3(
            data_dir=str(self.data_dir),
            cache_dir=cache_dir,
            cache_enabled=cache_enabled,
            strict_mode=strict_mode,
            verify_integrity=verify_integrity,
        )
        self.logger.info(
            f"[V3] DataCache V3: {'aktif' if cache_enabled else 'pasif'} "
            f"(strict={strict_mode}, integrity={verify_integrity})"
        )

        # ModelManager
        self.model_manager = ModelManager(self.config)
        self.model_manager.config["vocab_size"] = vocab_size

        self.model_manager.initialize(
            build_optimizer=True,
            build_criterion=False,
            build_scheduler=True,
        )

        if self.model_manager.model is None:
            raise RuntimeError("[V3] ModelManager.initialize() model oluşturamadı!")

        self.model_manager.model.train()

        # Criterion — V3 (entropy_coeff destekli)
        criterion_manager = CriterionManager(logger=self.logger)
        vocab = self.tokenizer_core.get_vocab()
        eos_id = self._get_special_id(vocab, "<EOS>", 2)
        pad_token_id = self.config.get("pad_token_id", 0)

        self.criterion = criterion_manager.create_criterion(
            vocab_size=len(vocab),
            eos_id=eos_id,
            pad_id=pad_token_id,
            device=torch.device(self.device),
            label_smoothing=float(self.config.get("label_smoothing", 0.1)),
            eos_weight=float(self.config.get("eos_token_weight", 1.0)),
            entropy_coeff=float(self.config.get("entropy_coeff", 0.0)),
        )
        self.model_manager.criterion = self.criterion

        # Config Manager V3
        self.config_manager = ConfigManagerV3(logger_instance=self.logger)

        # PAD ID (split için)
        special_ids = self.tokenizer_core._special_ids()
        self.pad_id = special_ids.get("<PAD>", 0)

        self.logger.info("[V3] TrainingServiceV3 hazır")

    def _setup_device(self) -> str:
        """Device belirle."""
        device_config = self.config.get("device", None)
        if device_config:
            return str(device_config)

        use_gpu = self.config.get("use_gpu", True)
        if use_gpu and torch.cuda.is_available():
            try:
                torch.cuda.set_device(0)
                test = torch.zeros(1).cuda()
                del test
                torch.cuda.empty_cache()
                self.logger.info("[V3] GPU[0] aktif")
                return "cuda"
            except Exception as e:
                self.logger.warning(f"[V3] GPU aktif edilemedi: {e}")

        self.logger.info("[V3] CPU modu")
        return "cpu"

    def _get_special_id(self, vocab: dict, token: str, fallback: int) -> int:
        """Vocab'dan özel token ID al."""
        data = vocab.get(token)
        if isinstance(data, dict):
            return int(data.get("id", fallback))
        elif isinstance(data, int):
            return int(data)
        return fallback

    # ──────────────────────────────────────────────────────────────────────
    # CACHE'DEN VERİ YÜKLEME — STRICT MODE
    # ──────────────────────────────────────────────────────────────────────

    def load_data_from_cache(self) -> Tuple[List[Tuple], List[Tuple], int]:
        """
        Cache'den formatlanmış veriyi yükle ve source-ID aware split yap.

        STRICT MODE: Cache yoksa CacheNotFoundError fırlatır.
        Raw data işleme YOK.

        Returns:
            (train_data, val_data, vocab_size)

        Raises:
            CacheNotFoundError: Cache bulunamadı
        """
        max_seq_len = int(self.config.get("max_seq_length", TOKENIZER_CONFIG.get("max_seq_length", 768)))
        include_whole_words = self.config.get(
            "train_include_whole_words", BPE_DETAILED_CONFIG.get("include_whole_words", True)
        )
        include_syllables = self.config.get(
            "train_include_syllables", BPE_DETAILED_CONFIG.get("include_syllables", False)
        )
        include_sep = self.config.get(
            "train_include_sep", BPE_DETAILED_CONFIG.get("include_sep", False)
        )

        self.logger.info("[V3] Cache'den veri yükleniyor (strict mode)...")
        self.logger.info(
            f"[V3] Parametreler: max_seq={max_seq_len}, "
            f"whole_words={include_whole_words}, syllables={include_syllables}, sep={include_sep}"
        )

        # Strict cache yükleme — hata fırlat
        formatted_data = self.data_cache.load_for_training(
            tokenizer_core=self.tokenizer_core,
            encode_mode="train",
            include_whole_words=include_whole_words,
            include_syllables=include_syllables,
            include_sep=include_sep,
            max_seq_length=max_seq_len,
        )

        if not formatted_data:
            raise ValueError("[V3] Cache'den hiç veri gelmedi! Cache dosyası bozuk olabilir.")

        self.logger.info(f"[V3] Cache'den {len(formatted_data):,} örnek yüklendi")

        # Next-token hizalama doğrulama
        self._validate_alignment(formatted_data)

        # Source-ID aware train/val split
        train_data, val_data = self._source_id_aware_split(formatted_data)

        vocab_size = self.tokenizer_core.get_vocab_size()
        self.logger.info(
            f"[V3] Split tamamlandı: train={len(train_data):,}, val={len(val_data):,}"
        )

        return train_data, val_data, vocab_size

    def _validate_alignment(self, data: List[Tuple]) -> None:
        """İlk örnekte next-token hizalamasını doğrula."""
        try:
            item = data[0]
            inp = list(item[0]) if not isinstance(item[0], list) else item[0]
            tgt = list(item[1]) if not isinstance(item[1], list) else item[1]
            n = min(len(inp), len(tgt))

            if n > 1 and inp == tgt:
                raise ValueError(
                    "[V3] KRITIK: target=input (next-token hizalama yok)! "
                    "Cache'i silin ve yeniden oluşturun: python training_system/prepare_cache.py"
                )

            self.logger.info(f"[V3] Next-token hizalama doğrulandı (len={n})")
        except (IndexError, TypeError):
            self.logger.warning("[V3] Hizalama doğrulaması atlandı")

    def _source_id_aware_split(
        self,
        data: List[Tuple],
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Source-ID aware train/val split.

        Data leakage önleme: Aynı kaynaktan (source_id) gelen chunk'lar
        aynı split'e gider. Farklı source_id'ler split'e dağıtılır.

        Eğer source_id yoksa: basit random split.
        """
        train_ratio = float(self.config.get("train_val_split", 0.8))
        seed = int(self.config.get("split_seed", 42))

        # source_id var mı?
        has_source_id = len(data[0]) == 3 if data else False

        if has_source_id:
            return self._split_by_source_id(data, train_ratio, seed)
        else:
            self.logger.warning(
                "[V3] source_id bulunamadı — basit random split kullanılıyor "
                "(data leakage riski var)"
            )
            return self._simple_random_split(data, train_ratio, seed)

    def _split_by_source_id(
        self,
        data: List[Tuple],
        train_ratio: float,
        seed: int,
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Source-ID bazlı split.

        1. Unique source_id'leri bul
        2. Source_id'leri train/val'a dağıt
        3. Her source_id'nin tüm chunk'ları aynı split'e gider
        """
        # source_id → indeksler
        source_to_indices: Dict[Any, List[int]] = {}
        for i, item in enumerate(data):
            sid = item[2] if len(item) == 3 else None
            if sid not in source_to_indices:
                source_to_indices[sid] = []
            source_to_indices[sid].append(i)

        # Source_id'leri karıştır
        rng = random.Random(seed)
        source_ids = list(source_to_indices.keys())
        rng.shuffle(source_ids)

        # Train/val source_id split
        train_size = int(train_ratio * len(source_ids))
        train_source_ids = set(source_ids[:train_size])
        val_source_ids = set(source_ids[train_size:])

        # Chunk'ları topla
        train_indices = []
        for sid in train_source_ids:
            train_indices.extend(source_to_indices[sid])

        val_indices = []
        for sid in val_source_ids:
            val_indices.extend(source_to_indices[sid])

        # Tensörlere çevir (source_id kaldır)
        train_data = self._to_tensors([data[i] for i in train_indices])
        val_data = self._to_tensors([data[i] for i in val_indices])

        self.logger.info(
            f"[V3] Source-ID split: "
            f"{len(train_source_ids)} train source / {len(val_source_ids)} val source "
            f"({len(train_data):,} train örnek / {len(val_data):,} val örnek)"
        )

        return train_data, val_data

    def _simple_random_split(
        self,
        data: List[Tuple],
        train_ratio: float,
        seed: int,
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """Basit random split (source_id yoksa fallback)."""
        tensors = self._to_tensors(data)
        rng = random.Random(seed)
        indices = list(range(len(tensors)))
        rng.shuffle(indices)
        train_size = int(train_ratio * len(tensors))
        train_data = [tensors[i] for i in indices[:train_size]]
        val_data = [tensors[i] for i in indices[train_size:]]
        return train_data, val_data

    def _to_tensors(self, data: List[Tuple]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Liste formatındaki veriyi tensor'a çevir, source_id'yi kaldır."""
        tensors = []
        for item in data:
            if len(item) == 3:
                inp, tgt, _ = item
            elif len(item) == 2:
                inp, tgt = item
            else:
                continue

            if isinstance(inp, torch.Tensor):
                inp_t = inp
                tgt_t = tgt
            else:
                inp_t = torch.tensor(inp, dtype=torch.long, device="cpu")
                tgt_t = torch.tensor(tgt, dtype=torch.long, device="cpu")

            tensors.append((inp_t, tgt_t))
        return tensors

    # ──────────────────────────────────────────────────────────────────────
    # ANA EĞİTİM PİPELINE
    # ──────────────────────────────────────────────────────────────────────

    def train(self) -> Tuple[float, float]:
        """
        V3 Eğitim pipeline'ını başlat.

        Sıra:
        1. Model initialize (checkpoint yükle)
        2. Cache'den veri yükle (STRICT — cache yoksa hata)
        3. V3 DataLoader oluştur (BucketSampler + DynamicPad)
        4. V3 Config hazırla
        5. Training Management V3 veya V2 ile eğit

        Returns:
            (final_train_loss, final_val_loss)
        """
        self.logger.info("=" * 70)
        self.logger.info("[V3] Training Pipeline V3 Başlıyor...")
        self.logger.info("=" * 70)

        # 1. Model initialize
        self._initialize_model()
        self.model_manager.model.train()

        # 2. Strict cache yükleme
        self.logger.info("[V3] [ADIM 2] Cache'den veri yükleniyor...")
        try:
            train_data, val_data, vocab_size = self.load_data_from_cache()
        except CacheNotFoundError as e:
            # Kullanıcıya açık hata mesajı
            self.logger.error(str(e))
            raise

        # 3. V3 DataLoader (BucketSampler + DynamicPad)
        self.logger.info("[V3] [ADIM 3] V3 DataLoader oluşturuluyor...")
        batch_size = int(self.config.get("batch_size", 8))
        max_seq_length = int(self.config.get("max_seq_length", TOKENIZER_CONFIG.get("max_seq_length", 768)))

        train_loader, val_loader = create_dataloaders_v3(
            train_data=train_data,
            val_data=val_data,
            batch_size=batch_size,
            pad_id=self.pad_id,
            device=self.device,
            use_bucket_batching=bool(self.config.get("use_bucket_batching", True)),
            num_buckets=int(self.config.get("num_buckets", 32)),
            use_dynamic_padding=bool(self.config.get("use_dynamic_padding", True)),
            max_seq_length=max_seq_length,
            num_workers=int(self.config.get("data_loader_num_workers", 0)),
            pin_memory=bool(self.config.get("data_loader_pin_memory", True)) if self.device == "cuda" else False,
            prefetch_factor=int(self.config.get("prefetch_factor", 2)),
            persistent_workers=bool(self.config.get("persistent_workers", True)),
        )

        self.logger.info(
            f"[V3] DataLoaders hazır: train={len(train_loader)} batch, val={len(val_loader)} batch"
        )

        # Warmup steps (dinamik)
        warmup_steps = calculate_warmup_steps(train_loader, self.config)
        self.config["warmup_steps"] = warmup_steps

        # 4. V3 Config
        self.logger.info("[V3] [ADIM 4] V3 Config hazırlanıyor...")
        training_config = self.config_manager.prepare_training_config(
            base_config=self.config,
            tokenizer_core=self.tokenizer_core,
            device=self.device,
        )

        # 5. Training Management seç (V3 > V2)
        self.logger.info("[V3] [ADIM 5] Training Manager seçiliyor...")
        optimizer = self.model_manager.optimizer
        if optimizer is None:
            raise RuntimeError("[V3] ModelManager.optimizer None!")

        return self._run_training(
            train_loader=train_loader,
            val_loader=val_loader,
            training_config=training_config,
            optimizer=optimizer,
        )

    def _run_training(
        self,
        train_loader,
        val_loader,
        training_config: Dict[str, Any],
        optimizer,
    ) -> Tuple[float, float]:
        """Training Management V3 veya V2 ile eğit."""

        # V3 TrainingManager tercih et
        try:
            from training_management.v3 import TrainingManager as V3TrainingManager
            _has_v3 = True
        except ImportError:
            _has_v3 = False
            self.logger.warning("[V3] training_management.v3 bulunamadı, V2 kullanılıyor")

        from training_management.v2.utils.checkpoint_manager import CheckpointManager
        from training_management.v2.monitoring.tensorboard_manager import TensorBoardManager
        from training_management.v2.utils.training_logger import TrainingLogger
        from training_management.v2.utils.training_scheduler import TrainingScheduler

        checkpoint_dir = training_config.get("checkpoint_dir", "./checkpoints")
        checkpoint_manager = CheckpointManager(
            checkpoint_model_dir=checkpoint_dir,
            max_checkpoints=int(training_config.get("max_checkpoints", 5)),
            device=self.device,
            logger=self.logger,
        )

        tb_manager = TensorBoardManager(
            log_dir=training_config.get("tensorboard_log_dir", "./runs"),
            enabled=training_config.get("enable_tensorboard", True),
            logger=self.logger,
        )

        training_logger = TrainingLogger(enable_file_logging=False)

        training_scheduler = TrainingScheduler(
            optimizer=optimizer,
            scheduler_type=training_config.get("scheduler_type", "ReduceLROnPlateau"),
            scheduler_kwargs=training_config.get("scheduler_kwargs", {}),
            logger=training_logger,
            warmup_steps=training_config.get("warmup_steps", 0),
            warmup_start_factor=training_config.get("warmup_start_factor", 0.1),
            embedding_warmup_factor=training_config.get("embedding_warmup_factor", 1.0),
        )

        # Test prompts
        test_prompts = self.config.get("test_prompts", [
            "En sevdiğin hayvan nedir?",
            "Merhaba",
            "Nasılsın?",
            "Hayatın anlamı nedir?",
        ])

        def epoch_callback(epoch: int, train_loss: float, val_loss: float) -> None:
            self._test_model_inline(epoch, train_loss, val_loss, test_prompts)

        # V3 bayrak logu
        self.logger.info(
            f"[V3] Eğitim konfigürasyonu özeti:\n"
            f"  label_smoothing={training_config.get('label_smoothing')}, "
            f"entropy_coeff={training_config.get('entropy_coeff')}\n"
            f"  use_ema={training_config.get('use_ema')}, "
            f"use_scheduled_sampling={training_config.get('use_scheduled_sampling')}\n"
            f"  use_bucket_batching={training_config.get('use_bucket_batching')}, "
            f"use_dynamic_padding={training_config.get('use_dynamic_padding')}\n"
            f"  cache_strict_mode={training_config.get('cache_strict_mode')}"
        )

        # V2 TrainingManager (V3 entegrasyon hatası durumunda fallback)
        from training_management.v2.core.training_manager import TrainingManager as V2TrainingManager

        training_manager = V2TrainingManager(
            model=self.model_manager.model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=self.criterion,
            config=training_config,
            logger=training_logger,
            scheduler=training_scheduler,
            checkpoint_manager=checkpoint_manager,
            tensorboard_manager=tb_manager,
        )

        self.logger.info("[V3] Eğitim başlıyor...")
        try:
            final_train_loss, final_val_loss = training_manager.train(epoch_callback=epoch_callback)
            self.logger.info(
                f"[V3] Eğitim tamamlandı: train_loss={final_train_loss:.6f}, "
                f"val_loss={final_val_loss:.6f}"
            )
            return float(final_train_loss), float(final_val_loss)

        except KeyboardInterrupt:
            self.logger.warning("[V3] Eğitim durduruldu (KeyboardInterrupt)")
            raise

    # ──────────────────────────────────────────────────────────────────────
    # MODEL INITIALIZE
    # ──────────────────────────────────────────────────────────────────────

    def _initialize_model(self) -> None:
        """Model initialize et (checkpoint yükle)."""
        checkpoint_dir = self.config.get("checkpoint_dir", "saved_models/checkpoints/")
        if not os.path.isabs(checkpoint_dir):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
            checkpoint_dir = os.path.join(project_root, checkpoint_dir)

        checkpoint_path = self._find_checkpoint(checkpoint_dir)

        if checkpoint_path:
            self.logger.info(f"[V3] Checkpoint yükleniyor: {checkpoint_path}")
            try:
                self.model_manager.load(checkpoint_path, weights_only=True)
            except TypeError:
                self.model_manager.load(checkpoint_path)
            self.logger.info("[V3] Checkpoint yüklendi")
        else:
            self.logger.info("[V3] Checkpoint yok — model sıfırdan başlatılıyor")

    def _find_checkpoint(self, checkpoint_dir: str) -> Optional[str]:
        """En son checkpoint'i bul."""
        resume_from = self.config.get("resume_from_path") or self.config.get("load_checkpoint_path")
        if resume_from and os.path.isfile(resume_from):
            return os.path.abspath(resume_from)

        if not os.path.isdir(checkpoint_dir):
            return None

        for alias in ["last.pth", "best.pth"]:
            p = os.path.join(checkpoint_dir, alias)
            if os.path.exists(p):
                return p

        checkpoint_files = [
            os.path.join(checkpoint_dir, f)
            for f in os.listdir(checkpoint_dir)
            if f.endswith(".pth") and f.startswith("checkpoint_")
        ]
        if checkpoint_files:
            checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return checkpoint_files[0]

        return None

    # ──────────────────────────────────────────────────────────────────────
    # EPOCH SONU TEST
    # ──────────────────────────────────────────────────────────────────────

    def _test_model_inline(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        test_prompts: Optional[List[str]] = None,
    ) -> None:
        """Epoch sonu inline generation testi (cevahir.generate kullanılmaz)."""
        if test_prompts is None:
            test_prompts = ["Merhaba"]

        model = self.model_manager.model
        if model is None:
            return

        original_mode = model.training
        model.eval()

        vocab = self.tokenizer_core.get_vocab()
        eos_id = self._get_special_id(vocab, "<EOS>", 2)

        try:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"EPOCH {epoch} TEST (val_loss={val_loss:.4f})")
            self.logger.info(f"{'='*60}")

            for prompt in test_prompts[:5]:  # En fazla 5 prompt test et
                try:
                    _, token_ids = self.tokenizer_core.encode(
                        prompt,
                        mode="inference",
                        include_whole_words=self.config.get("train_include_whole_words", True),
                        include_syllables=self.config.get("train_include_syllables", False),
                        include_sep=self.config.get("train_include_sep", False),
                    )

                    if not token_ids:
                        continue

                    with torch.no_grad():
                        generated_ids = list(token_ids)
                        for _ in range(30):
                            inp_t = torch.tensor([generated_ids], dtype=torch.long, device=self.device)
                            out = model(inp_t)
                            if isinstance(out, tuple):
                                out = out[0]
                            logits = out[0, -1, :]
                            logits = torch.clamp(logits, -50, 50) / 1.0

                            # Top-k
                            top_k = 80
                            if top_k < logits.size(-1):
                                top_vals, top_idx = torch.topk(logits, top_k)
                                filtered = torch.full_like(logits, float("-inf"))
                                filtered[top_idx] = top_vals
                                logits = filtered

                            probs = torch.softmax(logits, dim=-1)
                            if torch.isnan(probs).any():
                                next_id = torch.argmax(probs).item()
                            else:
                                next_id = torch.multinomial(probs, 1).item()

                            if next_id == eos_id and len(generated_ids) - len(token_ids) >= 5:
                                break
                            if next_id != eos_id:
                                generated_ids.append(next_id)

                    response = self.tokenizer_core.decode(
                        generated_ids[len(token_ids):],
                        method="bpe",
                        remove_specials=True,
                        remove_tags=True,
                        collapse_spaces=True,
                    )
                    self.logger.info(f"  '{prompt}' → '{response}'")

                except Exception as e:
                    self.logger.debug(f"  Test hatası ({prompt}): {e}")

        finally:
            model.train(original_mode)

        self.logger.info(f"  train_loss={train_loss:.4f}, val_loss={val_loss:.4f}\n")
