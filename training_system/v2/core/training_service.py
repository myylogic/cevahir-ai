# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: training_service.py
Modül: training_system/v2/core
Görev: Training Service V2 - Modüler Eğitim Orkestratörü. Eğitim sürecini koordine
       etmek (Facade Pattern). Her sorumluluk ayrı modülde (SOLID principles).
       BPEValidator, CriterionManager, DataPreparator, ConfigManager ile
       eğitim sürecini yönetir.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (eğitim orkestrasyonu), Facade Pattern
- Design Patterns: Facade Pattern (eğitim sürecini koordine eder)
- Endüstri Standartları: PyTorch training orchestration

KULLANIM:
- Model eğitimi başlatmak için
- Eğitim sürecini koordine etmek için
- TrainingService instance oluşturup eğitimi başlatmak için

GENERATION (iki ayri yol - izole test icin):
- Epoch sonu test: _test_model_after_epoch() icinde INLINE generation (cevahir.generate
  cagrilmaz). Tokenizer + model dogrudan kullanilir; EOS'ta hemen durur.
- Canli inference: model/cevahir.py Cevahir.generate() kullanilir; min_new_tokens ile
  erken EOS yok sayilabilir. Iki surec ayri test edilebilir.

BAĞIMLILIKLAR:
- BPEValidator: BPE dosya validasyonu
- CriterionManager: Loss function yönetimi
- ConfigManager: Config yönetimi
- ModelManager: Model yönetimi
- TrainingManager: Eğitim yönetimi
- DataCache: Cache yönetimi (cache-first data preparation)

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

import os
import sys
import torch
import logging
import hashlib
from typing import Dict, Any, Tuple, Optional, List

# Proje dizinini sys.path'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from model_management.model_manager import ModelManager
from training_management.v2.core.training_manager import TrainingManager as V2TrainingManager
from tokenizer_management.core.tokenizer_core import TokenizerCore, TokenizerCoreError
from tokenizer_management.config import BPE_CONFIG

# Config import (cache key uyumluluğu için)
try:
    from tokenizer_management.config import BPE_DETAILED_CONFIG, TOKENIZER_CONFIG
except ImportError:
    # Fallback
    BPE_DETAILED_CONFIG = {}
    TOKENIZER_CONFIG = {"max_seq_length": 768}

# V2 modüller
from .bpe_validator import BPEValidator
from .criterion_manager import CriterionManager
from .config_manager import ConfigManager

# Utils
from ..utils.data_loader_wrapper import create_dataloaders

# DataCache (training_system/data_cache.py'den)
from training_system.data_cache import DataCache

# Log yapılandırması
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
service_logger = logging.getLogger("TrainingServiceV2")

MODEL_SAVE_PATH = os.path.join("saved_models", "cevahir_model.pth")
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)


class TrainingService:
    """
    Training Service V2 - Modüler Eğitim Orkestratörü
    
    Sorumluluklar:
    - Eğitim pipeline'ını koordine etmek
    - Model, optimizer, scheduler, criterion yönetimi
    - Cache-first data preparation
    - V2 TrainingManager entegrasyonu
    - BPE dosya validasyonu
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        TrainingService V2'yi başlat.
        
        Args:
            config: Config dictionary
        """
        self.config = dict(config)
        self.logger = service_logger
        
        # BPE yolları
        vocab_path = self.config.get("vocab_path", BPE_CONFIG["vocab_file"])
        merges_path = self.config.get("merges_path", BPE_CONFIG["merges_file"])
        
        # BPE dizinlerini oluştur
        for path in (vocab_path, merges_path):
            directory = os.path.dirname(path)
            if directory and not os.path.isdir(directory):
                os.makedirs(directory, exist_ok=True)
                self.logger.info(f"Created directory for BPE file: {directory}")
        
        self.config.update({"vocab_path": vocab_path, "merges_path": merges_path})
        
        # Device belirleme
        self.device = self._setup_device()
        self.config["device"] = self.device
        
        # Data directory kontrolü
        if "data_dir" not in self.config or not os.path.isdir(str(self.config["data_dir"])):
            raise RuntimeError("TrainingService requires a valid 'data_dir' in config.")
        
        self.data_dir = self.config["data_dir"]
        
        # BPE Validator
        bpe_validator = BPEValidator(logger=self.logger)
        bpe_validator.validate_files(vocab_path, merges_path)
        
        # TokenizerCore
        self.tokenizer_core = TokenizerCore(self.config)
        
        # [OK] KRİTİK: TokenizerCore'dan vocab_size al ve config'e ekle (ModelManager'dan önce!)
        # Config'te zaten vocab_size varsa (70000 gibi), override et!
        vocab_size = self.tokenizer_core.get_vocab_size()
        old_vocab_size = self.config.get("vocab_size", "YOK")
        self.config["vocab_size"] = vocab_size  # Açıkça override et
        self.logger.info(f"[OK] Vocab size config'e eklendi: {vocab_size} (önceki: {old_vocab_size})")
        self.logger.info(f"[OK] Config vocab_size kontrolü: {self.config.get('vocab_size')}")
        
        # DataCache
        cache_dir = self.config.get("cache_dir", ".cache/preprocessed_data")
        cache_enabled = self.config.get("enable_data_cache", True)
        self.data_cache = DataCache(
            data_dir=str(self.data_dir),
            cache_dir=cache_dir,
            cache_enabled=cache_enabled
        )
        self.logger.info(f"Data cache: {'[OK] Aktif' if cache_enabled else ' Pasif'} ({cache_dir})")
        
        # ModelManager (artık config'te vocab_size var)
        # [OK] KRİTİK: Config'teki vocab_size'ı tekrar kontrol et
        final_vocab_size = self.config.get("vocab_size")
        self.logger.info(f"[OK] ModelManager'a geçirilecek vocab_size: {final_vocab_size}")
        if final_vocab_size != vocab_size:
            self.logger.warning(f" Vocab size uyumsuz! Config: {final_vocab_size}, Tokenizer: {vocab_size}")
            self.config["vocab_size"] = vocab_size  # Tekrar override et
            self.logger.info(f"[OK] Vocab size tekrar override edildi: {vocab_size}")
        
        self.model_manager = ModelManager(self.config)
        
        # [OK] KRİTİK: ModelManager'ın config'ini açıkça güncelle (initialize() öncesi)
        mm_vocab_size = self.model_manager.config.get("vocab_size")
        self.logger.info(f"[OK] ModelManager.config vocab_size (önce): {mm_vocab_size}")
        if mm_vocab_size != vocab_size:
            self.model_manager.config["vocab_size"] = vocab_size  # Açıkça override et
            self.logger.info(f"[OK] ModelManager.config vocab_size güncellendi: {vocab_size}")
        
        # Model'i initialize et (optimizer ve scheduler da oluşturulacak)
        # initialize() metodu model, optimizer, criterion, scheduler'ı oluşturur
        self.model_manager.initialize(
            build_optimizer=True,
            build_criterion=False,  # Criterion'ı biz CriterionManager ile oluşturuyoruz
            build_scheduler=True
        )
        
        # [OK] KRİTİK: Model oluşturulduktan sonra vocab_size kontrolü
        if self.model_manager.model is not None:
            # Model'in embedding layer'ından vocab_size'ı kontrol et
            if hasattr(self.model_manager.model, 'embedding') and hasattr(self.model_manager.model.embedding, 'num_embeddings'):
                model_vocab_size = self.model_manager.model.embedding.num_embeddings
                self.logger.info(f"[OK] Model embedding vocab_size: {model_vocab_size}")
                if model_vocab_size != vocab_size:
                    self.logger.error(f" HATA: Model vocab_size ({model_vocab_size}) != Tokenizer vocab_size ({vocab_size})!")
        
        if self.model_manager.model is None:
            raise RuntimeError("ModelManager.initialize() model oluşturamadı!")
        
        self.model_manager.model.train()  # type: ignore
        
        # CriterionManager - EOS weight ve label smoothing (config'ten; varsayılanlar EOS öğrenimi için güvenli)
        criterion_manager = CriterionManager(logger=self.logger)
        vocab = self.tokenizer_core.get_vocab()
        vocab_size = len(vocab)
        
        # EOS ID bulma
        eos_id = None
        eos_data = vocab.get("<EOS>")
        if isinstance(eos_data, dict):
            eos_id = eos_data.get("id")
        elif isinstance(eos_data, int):
            eos_id = eos_data
        else:
            eos_id = 3  # Fallback
        
        pad_token_id = self.config.get("pad_token_id", 0)
        device_torch = torch.device(self.device)
        
        # eos_weight=1.0 gerekli; 0.1 EOS gradient'ini 10x zayıflatıyor, EOS prob 0 kalıyor
        eos_weight = self.config.get("eos_token_weight", 1.0)
        label_smoothing = self.config.get("label_smoothing", 0.1)
        self.criterion = criterion_manager.create_criterion(
            vocab_size=vocab_size,
            eos_id=eos_id,
            pad_id=pad_token_id,
            device=device_torch,
            label_smoothing=label_smoothing,
            eos_weight=eos_weight
        )
        
        # ModelManager'a criterion'ı ver
        self.model_manager.criterion = self.criterion
        
        # Modüller
        self.bpe_validator = bpe_validator
        self.criterion_manager = criterion_manager
        self.config_manager = ConfigManager(logger=self.logger)
        
        # pad_id için instance variable (overlap kontrolü için)
        self.pad_id = None
    
    def _setup_device(self) -> str:
        """Device'ı belirle (GPU/CPU)"""
        device_config = self.config.get("device", None)
        if device_config:
            return str(device_config)
        
        use_gpu = self.config.get("use_gpu", True)
        if use_gpu and torch.cuda.is_available():
            try:
                torch.cuda.set_device(0)
                self.logger.info("[OK] GPU[0] aktif edildi")
                # Test
                test_tensor = torch.zeros(1).cuda()
                self.logger.info(f"[OK] GPU test başarılı - device: {test_tensor.device}")
                del test_tensor
                torch.cuda.empty_cache()
                return "cuda"
            except Exception as e:
                self.logger.warning(f" GPU aktif edilemedi: {e}")
        
        self.logger.info(" CPU modunda çalışılacak")
        return "cpu"
    
    def train(self) -> Tuple[float, float]:
        """
        Eğitimi başlat.
        
        Returns:
            Tuple of (final_train_loss, final_val_loss)
        """
        self.logger.info("=" * 60)
        self.logger.info("Training Pipeline Starting...")
        self.logger.info("=" * 60)
        
        # Model'i initialize et (checkpoint varsa yükle)
        self._initialize_model()
        
        # Eğitim moduna geç
        if self.model_manager.model is None:
            raise RuntimeError("Model is not initialized.")
        self.model_manager.model.train()  # type: ignore
        
        # Data preparation (cache-first, autoregressive formatting)
        self.logger.info("Data preparation başlıyor...")
        train_data, val_data, vocab_size = self.prepare_from_cache(
            data_cache=self.data_cache,
            tokenizer_core=self.tokenizer_core,
            config=self.config
        )
        
        # Minimal DataLoader wrapper
        batch_size = int(self.config.get("batch_size", 8))
        train_loader, val_loader = create_dataloaders(
            train_data=train_data,
            val_data=val_data,
            batch_size=batch_size,
            device=self.device,
            num_workers=self.config.get("data_loader_num_workers", 0),
            pin_memory=self.config.get("data_loader_pin_memory", True) if self.device == "cuda" else False
        )
        
        self.logger.info(
            f"[OK] DataLoaders hazır: train_batches={len(train_loader)}, val_batches={len(val_loader)}"
        )
        
        #  YENİ: Warmup steps dinamik hesaplama (train_loader hazır olduktan sonra)
        from ..utils.warmup_calculator import calculate_warmup_steps
        warmup_steps = calculate_warmup_steps(train_loader, self.config)
        self.config["warmup_steps"] = warmup_steps  # Config'e ekle
        self.logger.info(
            f"[Warmup] Dinamik hesaplama: warmup_steps={warmup_steps} "
            f"(batches_per_epoch={len(train_loader)}, grad_accum={self.config.get('grad_accum_steps', 1)}, "
            f"warmup_epochs={self.config.get('warmup_epochs', 1)})"
        )
        
        # V2 TrainingManager config
        training_config = self.config_manager.prepare_training_config(
            base_config=self.config,
            tokenizer_core=self.tokenizer_core,
            device=self.device
        )
        
        # Optimizer ve Scheduler (ModelManager.initialize() ile oluşturuldu)
        optimizer = self.model_manager.optimizer
        scheduler = self.model_manager.scheduler
        
        if optimizer is None:
            raise RuntimeError("ModelManager.optimizer None! initialize() çağrıldı mı?")
        if scheduler is None:
            self.logger.warning("ModelManager.scheduler None - TrainingScheduler ile oluşturulacak")
        
        # CheckpointManager (V2 TrainingManager için)
        from training_management.v2.utils.checkpoint_manager import CheckpointManager
        checkpoint_dir = training_config.get("checkpoint_dir", "./checkpoints")
        checkpoint_manager = CheckpointManager(
            checkpoint_model_dir=checkpoint_dir,
            max_checkpoints=int(training_config.get("max_checkpoints", 5)),
            device=self.device,
            logger=self.logger
        )
        
        # TensorBoardManager (V2 TrainingManager için)
        from training_management.v2.monitoring.tensorboard_manager import TensorBoardManager
        tb_log_dir = training_config.get("tensorboard_log_dir", "./runs")
        tb_enabled = training_config.get("enable_tensorboard", True)
        tensorboard_manager = TensorBoardManager(log_dir=tb_log_dir, enabled=tb_enabled, logger=self.logger)
        
        # Logger (V2 TrainingManager için)
        from training_management.v2.utils.training_logger import TrainingLogger
        training_logger = TrainingLogger(enable_file_logging=False)  # File logging kapalı
        
        # Scheduler (V2 TrainingManager için)
        from training_management.v2.utils.training_scheduler import TrainingScheduler
        scheduler_type = training_config.get("scheduler_type", "ReduceLROnPlateau")
        scheduler_kwargs = training_config.get("scheduler_kwargs", {})
        # Warmup parametreleri (dinamik hesaplanan warmup_steps kullan)
        warmup_steps = training_config.get("warmup_steps", self.config.get("warmup_steps", 0))
        warmup_start_factor = training_config.get("warmup_start_factor", self.config.get("warmup_start_factor", 0.1))
        embedding_warmup_factor = training_config.get("embedding_warmup_factor", self.config.get("embedding_warmup_factor", 1.0))  # [OK] GRADIENT FIX
        training_scheduler = TrainingScheduler(
            optimizer=optimizer,
            scheduler_type=scheduler_type,
            scheduler_kwargs=scheduler_kwargs,
            logger=training_logger,  # [OK] TrainingLogger instance'ı geçir
            warmup_steps=warmup_steps,
            warmup_start_factor=warmup_start_factor,
            embedding_warmup_factor=embedding_warmup_factor,  # [OK] GRADIENT FIX: Embedding warmup yok
        )
        
        # [DEBUG] Model instance kontrolü (checkpoint kaydetme öncesi)
        model_to_pass = self.model_manager.model
        if model_to_pass is not None:
            model_state_dict = model_to_pass.state_dict()
            model_keys = list(model_state_dict.keys())
            model_type = type(model_to_pass).__name__
            is_simple_model = (
                len(model_keys) == 3 and 
                all(k in model_keys for k in ["embed.weight", "proj.weight", "proj.bias"])
            )
            self.logger.info("=" * 60)
            self.logger.info("[CHECKPOINT DEBUG] TrainingManager'a geçirilecek model kontrolü:")
            self.logger.info(f"  Model Type: {model_type}")
            self.logger.info(f"  State Dict Keys: {len(model_keys)}")
            self.logger.info(f"  İlk 10 Key: {model_keys[:10]}")
            self.logger.info(f"  SimpleModel mi? {is_simple_model}")
            if is_simple_model:
                self.logger.error("   UYARI: SimpleModel instance'ı geçiriliyor!")
            else:
                self.logger.info("   CevahirNeuralNetwork instance'ı geçiriliyor")
            self.logger.info("=" * 60)
        
        # V2 TrainingManager oluştur
        training_manager = V2TrainingManager(
            model=model_to_pass,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=self.criterion,  # [OK] EOS weight 0.1 ve label smoothing 0.1 içeriyor
            config=training_config,
            logger=training_logger,
            scheduler=training_scheduler,
            checkpoint_manager=checkpoint_manager,
            tensorboard_manager=tensorboard_manager,
        )
        
        # Test prompts (her epoch sonunda test için) - 19 çeşitli soru
        test_prompts = self.config.get("test_prompts", [
            # Eğitim verisinden (ezberlenmiş olmalı)
            "En sevdiğin hayvan nedir?",
            "Mutluluk nedir?",
            "Aşk nedir?",
            
            # Basit sorular (genelleme testi)
            "Merhaba",
            "Nasılsın?",
            "Adın ne?",
            
            # Orta zorluk
            "2+2 kaç eder?",
            "Türkiye'nin başkenti neresi?",
            "Bugün hava nasıl?",
            
            # Zor sorular (soyut düşünme)
            "Evren nasıl oluştu?",
            "Bilinç nedir?",
            "Zaman nedir?",
            
            # Çok kısa (tek kelime beklentisi)
            "Evet mi hayır mı?",
            "Hangi renk?",
            "Ne zaman?",
            
            # Felsefi (eğitim verisine yakın)
            "Hayatın anlamı nedir?",
            "Ölüm nedir?",
            "İnsan ne demektir?",
        ])
        
        # Epoch callback: Her epoch sonunda test generation yap
        def epoch_callback(epoch: int, train_loss: float, val_loss: float) -> None:
            """Her epoch sonunda model'i test et"""
            self._test_model_after_epoch(epoch, train_loss, val_loss, test_prompts)
        
        # Eğitimi başlat
        self.logger.info("=" * 60)
        self.logger.info("V2 TrainingManager ile eğitim başlıyor...")
        self.logger.info("=" * 60)
        
        try:
            final_train_loss, final_val_loss = training_manager.train(epoch_callback=epoch_callback)
            
            self.logger.info("=" * 60)
            self.logger.info("Eğitim tamamlandı!")
            self.logger.info(f"Final Train Loss: {final_train_loss:.6f}")
            self.logger.info(f"Final Val Loss: {final_val_loss:.6f}")
            self.logger.info("=" * 60)
            
            return float(final_train_loss), float(final_val_loss)
        
        except KeyboardInterrupt:
            self.logger.warning(" Eğitim kullanıcı tarafından durduruldu (KeyboardInterrupt).")
            self.logger.info(" Son checkpoint zaten kaydedilmiş olmalı (best model için).")
            # TrainingManager her epoch sonunda best model'i kaydediyor, 
            # bu yüzden son epoch'un checkpoint'i zaten kaydedilmiş olacak
            raise
    
    def _test_model_after_epoch(
        self, 
        epoch: int, 
        train_loss: float, 
        val_loss: float, 
        test_prompts: Optional[list] = None
    ) -> float:
        """Her epoch sonunda modeli test et - basit inference testi.
        NOT: Burada cevahir.generate() CAGIRILMAZ; izole bir generation dongusu kullanilir.
        Boylece (1) training_service generation bug'lari ayri test edilir,
        (2) cevahir.generate() ayri test edilebilir. Iki surec birbirinden bagimsizdir.
        Returns:
            Epoch boyunca görülen maksimum EOS olasılığı (takip için; yoksa 0.0).
        """
        if test_prompts is None:
            test_prompts = ["selam beni anlayabiliyor musun"]
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"EPOCH {epoch} SONU TEST (inline generation - cevahir.generate kullanilmaz)")
        self.logger.info(f"{'='*60}")
        
        # Model'i inference moduna al
        if self.model_manager.model is None:
            self.logger.warning(f"Epoch {epoch} test: Model henüz initialize edilmemiş")
            return 0.0

        original_mode = self.model_manager.model.training
        self.model_manager.model.eval()

        epoch_max_eos_prob = 0.0
        try:
            for prompt in test_prompts:
                try:
                    # Her prompt öncesi KV cache temizle (önceki prompt cevabının sonraki cevaba kaymaması için)
                    if hasattr(self.model_manager, "clear_kv_cache"):
                        self.model_manager.clear_kv_cache()
                    # Tokenize
                    _, token_ids = self.tokenizer_core.encode(
                        prompt,
                        mode="inference",
                        include_whole_words=self.config.get("train_include_whole_words", True),
                        include_syllables=self.config.get("train_include_syllables", False),
                        include_sep=self.config.get("train_include_sep", False),
                    )
                    
                    if not token_ids:
                        self.logger.warning(f"Epoch {epoch} test: Prompt tokenize edilemedi: '{prompt}'")
                        continue
                    
                    # Basit autoregressive generation
                    with torch.no_grad():
                        generated_ids = list(token_ids)
                        max_new_tokens = 30  # Daha uzun cevaplar için artırıldı
                        
                        # EOS token ID ve vocab bilgisi (tek seferlik / ilk prompt'ta detay log)
                        vocab = self.tokenizer_core.get_vocab()
                        vocab_size = len(vocab)
                        eos_id = None
                        if isinstance(vocab.get("<EOS>"), dict):
                            eos_id = vocab["<EOS>"].get("id")
                        elif isinstance(vocab.get("<EOS>"), int):
                            eos_id = vocab["<EOS>"]
                        pad_id = self.config.get("pad_token_id", 0)
                        is_first_prompt = (prompt == test_prompts[0])
                        if is_first_prompt:
                            self.logger.info(f"  [GEN-DEBUG] vocab_size={vocab_size}, eos_id={eos_id}, pad_id={pad_id}")
                            self.logger.info(f"  [GEN-DEBUG] prompt token_ids (first 15): {token_ids[:15]}")
                        
                        tokens_generated = 0
                        eos_generated = False
                        eos_prob_history = []  # EOS probability tracking
                        first_step_argmax_id = None  # ilk adimda argmax (sampling olmadan) hangi token
                        
                        # [CRITICAL FIX] Minimum length constraint
                        min_new_tokens = 5  # At least 5 tokens before allowing EOS
                        
                        # [IMPROVED] Sampling parameters
                        top_k = 100  # FIXED: 40 → 100 (more diversity)
                        temperature = 1.0  # FIXED: 0.8 → 1.0 (less peaked distribution)
                        repetition_penalty = 1.5  # FIXED: 1.2 → 1.5 (stronger penalty, E3 showed 33% repetition!)
                        repetition_window = 15  # FIXED: 10 → 15 (look back further)
                        
                        for step in range(max_new_tokens):
                            input_tensor = torch.tensor([generated_ids], dtype=torch.long, device=self.device)
                            output = self.model_manager.model(input_tensor)
                            if isinstance(output, tuple):
                                output = output[0]
                            
                            next_logits = output[0, -1, :]
                            next_logits = torch.clamp(next_logits, min=-50.0, max=50.0)
                            
                            # [NEW] Repetition penalty - penalize recently generated tokens
                            if repetition_penalty > 1.0 and len(generated_ids) > 0:
                                recent_tokens = generated_ids[-repetition_window:]  # Last N tokens
                                for token_id in set(recent_tokens):
                                    if token_id < next_logits.size(-1):
                                        next_logits[token_id] /= repetition_penalty
                            
                            # [IMPROVED] Temperature
                            if temperature > 0:
                                next_logits = next_logits / temperature
                            
                            # [NEW] Top-k filtering (reduces EOS bias)
                            if top_k > 0 and top_k < next_logits.size(-1):
                                top_k_values, top_k_indices = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                                filtered_logits = torch.full_like(next_logits, float('-inf'))
                                filtered_logits[top_k_indices] = top_k_values
                                next_logits = filtered_logits
                            
                            probs = torch.softmax(next_logits, dim=-1)
                            
                            # EOS probability tracking
                            if eos_id is not None:
                                eos_prob = probs[eos_id].item()
                                eos_prob_history.append(eos_prob)
                            
                            # Ilk adimda argmax ID'yi kaydet (sampling olmadan ne uretirdi - debug)
                            if step == 0 and is_first_prompt:
                                first_step_argmax_id = torch.argmax(probs).item()
                            
                            # Sample
                            if torch.isnan(probs).any() or torch.isinf(probs).any():
                                predicted_id = torch.argmax(probs).item()
                            else:
                                predicted_id = torch.multinomial(probs, 1).item()
                            
                            # [CRITICAL FIX] EOS kontrolü - minimum length constraint
                            if eos_id is not None and predicted_id == eos_id:
                                if tokens_generated >= min_new_tokens:
                                    eos_generated = True
                                    break
                                else:
                                    # EOS too early - ignore and continue generation
                                    continue
                            
                            generated_ids.append(predicted_id)
                            tokens_generated += 1
                        
                        # Decode
                        prompt_length = len(token_ids)
                        predicted_token_ids = generated_ids[prompt_length:]
                        
                        if len(predicted_token_ids) == 0:
                            response = ""
                        else:
                            response = self.tokenizer_core.decode(
                                predicted_token_ids,
                                method="bpe",
                                remove_specials=True,
                                remove_tags=True,
                                collapse_spaces=True,
                                lowercase=False,
                            )
                        
                        # Log
                        self.logger.info(f"  Prompt: '{prompt}'")
                        self.logger.info(f"  Response: '{response}'")
                        self.logger.info(f"  Tokens: {len(predicted_token_ids)}")
                        self.logger.info(f"  EOS generated: {eos_generated}")
                        if eos_prob_history:
                            avg_eos_prob = sum(eos_prob_history) / len(eos_prob_history)
                            max_eos_prob = max(eos_prob_history)
                            epoch_max_eos_prob = max(epoch_max_eos_prob, max_eos_prob)
                            self.logger.info(f"  EOS prob (avg/max): {avg_eos_prob:.4f} / {max_eos_prob:.4f}")
                        # Generation debug: uretilen token ID'leri (ilk 10) - ? veya tekrarlari anlamak icin
                        self.logger.info(f"  [GEN-DEBUG] generated token_ids (first 10): {predicted_token_ids[:10]}")
                        if is_first_prompt and first_step_argmax_id is not None:
                            self.logger.info(f"  [GEN-DEBUG] first step argmax (no sampling) token_id: {first_step_argmax_id}")
                        if "?" in response or (len(set(predicted_token_ids)) <= 2 and len(predicted_token_ids) > 5):
                            self.logger.info(f"  [GEN-DEBUG] all generated token_ids: {predicted_token_ids}")
                        self.logger.info("")
                
                except Exception as e:
                    self.logger.warning(f"Epoch {epoch} test: Prompt '{prompt}' için hata: {e}")
        
        finally:
            # Model'i training moduna geri al
            self.model_manager.model.train(original_mode)

        self.logger.info(f"  [Epoch {epoch} özet] train_loss={train_loss:.4f} val_loss={val_loss:.4f} eos_max_prob={epoch_max_eos_prob:.4f}")
        self.logger.info(f"{'='*60}\n")
        return float(epoch_max_eos_prob)

    def _initialize_model(self) -> None:
        """Model'i initialize et (checkpoint varsa yükle)"""
        # Checkpoint dizinini config'ten al (V2 TrainingManager ile aynı)
        # Önce config'ten al, yoksa train.py'deki default değeri kullan
        checkpoint_dir = self.config.get("checkpoint_dir", "saved_models/checkpoints/")
        if not os.path.isabs(checkpoint_dir):
            # Relative path ise, proje root'a göre çöz
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
            checkpoint_dir = os.path.join(project_root, checkpoint_dir)
        
        # Normalize path (trailing slash'i kaldır)
        checkpoint_dir = checkpoint_dir.rstrip("/").rstrip("\\")
        
        # Alternatif dizinleri de kontrol et (checkpoints/checkpoint uyumsuzluğu için)
        alt_checkpoint_dir = checkpoint_dir.replace("checkpoints", "checkpoint")
        if alt_checkpoint_dir == checkpoint_dir:  # Değişmediyse, tekil versiyonunu dene
            alt_checkpoint_dir = checkpoint_dir.replace("checkpoint", "checkpoints")
        
        # Kontrol edilecek dizinler listesi
        checkpoint_dirs_to_check = [checkpoint_dir]
        if alt_checkpoint_dir != checkpoint_dir:
            checkpoint_dirs_to_check.append(alt_checkpoint_dir)
        
        old_model_path = MODEL_SAVE_PATH
        
        checkpoint_path = None
        checkpoint_source = None
        
        # [Devam eğitimi] Config'te açık yol varsa önce onu dene
        resume_from = self.config.get("resume_from_path") or self.config.get("load_checkpoint_path")
        if resume_from and os.path.isfile(resume_from):
            checkpoint_path = os.path.abspath(resume_from)
            checkpoint_source = f"config (resume_from_path): {os.path.basename(checkpoint_path)}"
        
        # Her dizinde checkpoint ara (resume_from ile bulunamadıysa)
        if not checkpoint_path:
            for chk_dir in checkpoint_dirs_to_check:
                if not os.path.isdir(chk_dir):
                    continue
                # CheckpointManager'ın alias dosyalarını kontrol et
                last_checkpoint = os.path.join(chk_dir, "last.pth")
                best_checkpoint = os.path.join(chk_dir, "best.pth")
                # Öncelik sırası: last.pth > best.pth > en son checkpoint
                if os.path.exists(last_checkpoint):
                    checkpoint_path = last_checkpoint
                    checkpoint_source = f"last.pth ({os.path.basename(chk_dir)})"
                    break
                elif os.path.exists(best_checkpoint):
                    checkpoint_path = best_checkpoint
                    checkpoint_source = f"best.pth ({os.path.basename(chk_dir)})"
                    break
                else:
                    checkpoint_files = [
                        os.path.join(chk_dir, f)
                        for f in os.listdir(chk_dir)
                        if f.endswith('.pth') and f.startswith('checkpoint_')
                    ]
                    if checkpoint_files:
                        checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                        checkpoint_path = checkpoint_files[0]
                        checkpoint_source = f"{os.path.basename(checkpoint_path)} ({os.path.basename(chk_dir)})"
                        break
        
        # Hiçbir checkpoint bulunamadıysa, eski MODEL_SAVE_PATH'i kontrol et (resume_from da yoksa)
        if not checkpoint_path and os.path.exists(old_model_path):
            checkpoint_path = old_model_path
            checkpoint_source = "cevahir_model.pth"
        
        try:
            if checkpoint_path:
                self.logger.info(f"[OK] Checkpoint bulundu: {checkpoint_source}")
                self.logger.info(f"   Yükleniyor: {checkpoint_path}")
                try:
                    self.model_manager.load(checkpoint_path, weights_only=True)  # type: ignore
                except TypeError:
                    self.model_manager.load(checkpoint_path)  # type: ignore
                self.logger.info("[OK] Model checkpoint'ten yüklendi")
            else:
                self.logger.info(" Checkpoint bulunamadı, model yeni başlatılacak")
                searched_dirs = checkpoint_dirs_to_check + [os.path.dirname(old_model_path)]
                self.logger.info(f"   Aranan dizinler: {', '.join(searched_dirs)}")
        except Exception as e:
            self.logger.warning(f" Model yükleme hatası (devam ediliyor): {e}")
    
    def prepare_from_cache(
        self,
        data_cache,
        tokenizer_core,
        config: Dict[str, Any]
    ) -> Tuple[List[Tuple], List[Tuple], int]:
        """
        Cache'den formatlanmış veriyi yükle ve train/val split yap.
        
        VERİ AKIŞI (cache -> eğitim):
        - get_or_process() cache'den (inp_list, tgt_list) veya (inp, tgt, source_id) döner.
        - Cache hazır ise (prepare_cache.py ile BOS/EOS/PAD eklenmiş): inp = [BOS, c0, ..., cK, PAD...],
          tgt = [c0, ..., cK, EOS, PAD...]. Pozisyon hizası: Input[i] -> bir sonraki token = Target[i];
          yani Target[K] = EOS (cK'dan sonra EOS öğretilir). Loss'ta PAD maskelenir, EOS dahil tüm
          içerik pozisyonları kullanılır. Tokenizer Manager bu veriyi üretmez; cache'den gelen veri
          doğrudan tensor'a çevrilip DataLoader'a verilir, ek formatlama yapılmaz.
        - Cache yoksa: process_func() = load_training_data() çağrılır. format_func=None ise ham veri
          (inp, inp) saklanır → YANLIŞ: target[t]=input[t] olur, next-token hedefi kaybolur.
          Bu yüzden format_func ile BOS/EOS ve hizalama (target[t]=input[t+1]) uygulanmalı.
        
        Args:
            data_cache: DataCache instance
            tokenizer_core: TokenizerCore instance
            config: Config dictionary
            
        Returns:
            Tuple of (train_data, val_data, vocab_size)
        """
        # Config'ten parametreleri al
        max_seq_len = int(config.get("max_seq_length", TOKENIZER_CONFIG.get("max_seq_length", 768)))
        include_whole_words = config.get("train_include_whole_words", BPE_DETAILED_CONFIG.get("include_whole_words", True))
        include_syllables = config.get("train_include_syllables", BPE_DETAILED_CONFIG.get("include_syllables", False))
        include_sep = config.get("train_include_sep", BPE_DETAILED_CONFIG.get("include_sep", False))
        
        # Cache'den yükle (cache yoksa process edilir ve cache'e kaydedilir)
        def process_data():
            """Cache yoksa veriyi işle"""
            return tokenizer_core.load_training_data(
                encode_mode="train",
                include_whole_words=include_whole_words,
                include_syllables=include_syllables,
                include_sep=include_sep,
            )
        
        # KRİTİK: Cache ilk kez oluşturulurken (cache miss) veri formatlanmazsa (inp, inp) saklanır;
        # bu da target[t]=input[t] demek → model "sonraki token" yerine "mevcut token"ı öğrenir, eğitim bozulur.
        # Bu yüzden format_func ile BOS/EOS ve next-token hizalaması (target[t]=input[t+1]) zorunlu.
        special_ids = tokenizer_core._special_ids()
        BOS_ID = special_ids.get("<BOS>", 1)
        EOS_ID = special_ids.get("<EOS>", 2)
        PAD_ID = special_ids.get("<PAD>", 0)
        
        def format_data_for_autoregressive(raw_data: list) -> list:
            """Ham (inp, tgt) veya (inp, tgt, source_id) → BOS/EOS/PAD ile target[t]=input[t+1] hizalı veri."""
            out = []
            for item in raw_data:
                if len(item) == 3:
                    inp_ids, tgt_ids, source_id = item
                elif len(item) == 2:
                    inp_ids, tgt_ids = item
                    source_id = None
                else:
                    continue
                inp_ids = list(inp_ids)
                tgt_ids = list(tgt_ids)
                # Next-token: seq_in = [BOS, t1, ..., tN], seq_tgt = [t1, ..., tN, EOS]
                seq_in = [BOS_ID] + inp_ids
                seq_tgt = tgt_ids + [EOS_ID]
                # Truncate
                if len(seq_in) > max_seq_len:
                    seq_in = seq_in[:max_seq_len]
                if len(seq_tgt) > max_seq_len:
                    seq_tgt = seq_tgt[:max_seq_len - 1] + [EOS_ID]
                # Aynı uzunluk
                current_len = max(len(seq_in), len(seq_tgt))
                if len(seq_in) < current_len:
                    seq_in = seq_in + [seq_in[-1]] * (current_len - len(seq_in)) if seq_in else [BOS_ID] * current_len
                if len(seq_tgt) < current_len:
                    seq_tgt = seq_tgt + [PAD_ID] * (current_len - len(seq_tgt)) if seq_tgt[-1] == EOS_ID else seq_tgt + [EOS_ID] + [PAD_ID] * (current_len - len(seq_tgt) - 1)
                # Pad to max_seq_len
                if len(seq_in) < max_seq_len:
                    seq_in += [PAD_ID] * (max_seq_len - len(seq_in))
                if len(seq_tgt) < max_seq_len:
                    seq_tgt += [PAD_ID] * (max_seq_len - len(seq_tgt))
                if source_id is not None:
                    out.append((seq_in, seq_tgt, source_id))
                else:
                    out.append((seq_in, seq_tgt))
            return out
        
        if self.logger:
            self.logger.info(" [1] Loading formatted data from cache...")
        
        # Vocab size
        vocab = tokenizer_core.get_vocab()
        vocab_size = len(vocab)
        
        # Cache'den yükle; cache yoksa format_data_for_autoregressive ile doğru hizalama uygulanır
        formatted_data, from_cache = data_cache.get_or_process(
            tokenizer_core=tokenizer_core,
            encode_mode="train",
            include_whole_words=include_whole_words,
            include_syllables=include_syllables,
            include_sep=include_sep,
            max_seq_length=max_seq_len,
            process_func=process_data,
            alignment_format="autoregressive_v2",
            format_data=True,
            format_func=format_data_for_autoregressive,
        )
        
        if from_cache and self.logger:
            self.logger.info("[OK] Veri cache'den yüklendi (formatlanmış, hazır)")
        
        if not formatted_data:
            raise ValueError("Cache'den hiç veri gelmedi!")
        
        if self.logger:
            self.logger.info(f" [2] {len(formatted_data)} examples loaded.")
        
        # KRİTİK: İlk örnekte next-token hizalamasını doğrula (target[t]=input[t+1])
        try:
            item0 = formatted_data[0]
            inp_list = item0[0] if isinstance(item0[0], (list, tuple)) else list(item0[0])
            tgt_list = item0[1] if isinstance(item0[1], (list, tuple)) else list(item0[1])
            n = min(len(inp_list), len(tgt_list))
            misaligned = 0
            for i in range(n - 1):
                if tgt_list[i] != inp_list[i + 1]:
                    misaligned += 1
            if n > 1 and misaligned == n - 1 and inp_list == tgt_list:
                self.logger.error(
                    "[CRITICAL] Veri hizalaması YANLIŞ: target=input (next-token yok). "
                    "Eski cache kullanılıyor olabilir. Cache'i silip eğitimi yeniden başlatın: "
                    "cache klasörünü temizleyin veya prepare_cache.py çalıştırın."
                )
                raise ValueError(
                    "Data alignment bug: target equals input. Clear cache and restart training."
                )
            if self.logger and n > 1:
                self.logger.info(f" [OK] İlk örnek next-token hizalama doğrulandı (uzunluk={n}).")
        except (IndexError, TypeError) as e:
            if self.logger:
                self.logger.warning(f" [Veri doğrulama atlandı: {e}]")
        
        # Cache'den gelen veri List formatında, Tensor'a çevir ve source_id kaldır
        formatted_tensors = []
        for item in formatted_data:
            if len(item) == 3:
                inp_list, tgt_list, _ = item  # source_id'yi kaldır
            elif len(item) == 2:
                inp_list, tgt_list = item
            else:
                if self.logger:
                    self.logger.warning(f"[!] Geçersiz örnek formatı atlandı: len={len(item)}")
                continue
            
            inp_tensor = torch.tensor(inp_list, dtype=torch.long, device="cpu")
            tgt_tensor = torch.tensor(tgt_list, dtype=torch.long, device="cpu")
            formatted_tensors.append((inp_tensor, tgt_tensor))
        
        # Basit train/val split
        import random
        split_seed = config.get("split_seed", 42)
        train_ratio = config.get("train_val_split", 0.8)
        
        random.seed(split_seed)
        indices = list(range(len(formatted_tensors)))
        random.shuffle(indices)
        
        train_size = int(train_ratio * len(formatted_tensors))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_data = [formatted_tensors[i] for i in train_indices]
        val_data = [formatted_tensors[i] for i in val_indices]
        
        if self.logger:
            self.logger.info(
                f" [3] Data prepared → Total: {len(formatted_tensors)}, "
                f"Train: {len(train_data)}, Val: {len(val_data)}"
            )
        
        return train_data, val_data, vocab_size

