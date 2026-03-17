# -*- coding: utf-8 -*-

"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================
Dosya: train.py
Modül: training_system
Görev: Cevahir eğitim giriş noktası - Model eğitimi için ana script. Global logging,
       ortam bilgisi, config normalizasyonu (gradient_clip -> max_grad_norm),
       dizin oluşturma, TrainingService ile eğitimi başlatma ve koşu özeti yazma
       işlemlerini yönetir.



MİMARİ:
- SOLID Prensipleri: Single Responsibility (eğitim giriş noktası)
- Design Patterns: Script Pattern (standalone training script)
- Endüstri Standartları: PyTorch training workflow



KULLANIM:
- Model eğitimi başlatmak için
- TrainingService ile eğitim sürecini yönetmek için
- Standalone script olarak çalıştırılır



BAĞIMLILIKLAR:
- TrainingService: Eğitim servisi
- ModelManager: Model yönetimi
- TokenizerCore: Tokenization işlemleri
- Config modülleri: Yapılandırma yönetimi

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
import json
import logging
from pprint import pformat
from pathlib import Path
from typing import Any, Dict
import torch

# [V6 OOM FIX] Fragmentation önleme: Ayrılmış ama kullanılmayan bellek büyüdüğünde
# PyTorch cUDA allocator'ı non-contiguous segment'lere izin ver.
# expandable_segments=True → büyük contiguous blok bulunamadığında allocator parçalı alır.
# Bu ayar torch.cuda başlamadan önce set edilmeli.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")



# Proje kök dizinini sys.path'e ekle
# train.py training_system/ içinde, proje root'a ulaşmak için bir üst dizine çık
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)



from training_system.v2.core.training_service import TrainingService  # noqa: E402

# model_management exception'ları — OOM gibi hatalar için kullanıcıya açıklayıcı log
try:
    from model_management.exceptions import OOMRecoveryError, CevahirModelError
    _MM_EXCEPTIONS_AVAILABLE = True
except ImportError:
    OOMRecoveryError = None   # type: ignore
    CevahirModelError = None  # type: ignore
    _MM_EXCEPTIONS_AVAILABLE = False

# V3 Training System — varsa kullan, yoksa V2'ye geri dön
try:
    from training_system.v3 import TrainingServiceV3
    _TRAINING_SYSTEM_V3_AVAILABLE = True
except ImportError:
    TrainingServiceV3 = None  # type: ignore
    _TRAINING_SYSTEM_V3_AVAILABLE = False

# V3 TrainingManager — varsa kullan, yoksa V2'ye geri dön
try:
    from training_management.v3 import TrainingManager as V3TrainingManager
    _V3_AVAILABLE = True
except ImportError:
    V3TrainingManager = None
    _V3_AVAILABLE = False
# Config'ten BPE ve tokenizer ayarlarını al

try:
    from tokenizer_management.config import (
        get_bpe_detailed_config,
        TOKENIZER_CONFIG,
        BPE_CONFIG
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    # train_logger henüz tanımlanmadı, logging kullan
    logging.getLogger("TrainScript").warning("tokenizer_management.config import edilemedi, default değerler kullanılacak")



# =========================
# Global Logging
# =========================

LOG_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
# Root handler seviyelerini INFO'ya düşür
logging.getLogger().setLevel(logging.INFO)
for h in logging.getLogger().handlers:
    h.setLevel(logging.INFO)

train_logger = logging.getLogger("TrainScript")
#  Modül seviyesi loglar kaldırıldı (import sırasında çalışmaması için)
# Loglar main() fonksiyonu içinde yazılacak
# =========================
# Yardımcılar
# =========================

def set_seed(seed: int = 42) -> None:
    try:
        import random
        import numpy as np



        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Daha deterministik CUDA (gerekirse kapatılabilir)
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False     # type: ignore[attr-defined]
    except Exception as e:
        logging.getLogger("Seed").warning(f"Seed ayarlanamadı: {e}")



def ensure_dirs(*paths: str) -> None:
    for p in paths:
        if not p:
            continue
        pp = Path(p)
        target = pp.parent if pp.suffix else pp
        target.mkdir(parents=True, exist_ok=True)



def log_env_info() -> None:
    try:
        train_logger.info(f"PyTorch version: {torch.__version__}")

        

        # Colab için GPU kontrolü - daha agresif kontrol
        cuda_available = torch.cuda.is_available()
        train_logger.info(f"torch.cuda.is_available(): {cuda_available}")

        

        if cuda_available:
            n = torch.cuda.device_count()
            train_logger.info(f"CUDA available: True | device_count={n}")
            for i in range(n):
                name = torch.cuda.get_device_name(i)
                cc = torch.cuda.get_device_capability(i)
                train_logger.info(f"GPU[{i}] {name} | CC={cc}")
            try:
                free, total = torch.cuda.mem_get_info()  # type: ignore[attr-defined]
                train_logger.info(f"GPU memory: free={free/1e9:.2f} GB / total={total/1e9:.2f} GB")

                

                # GPU'yu aktif et - Colab için önemli
                torch.cuda.set_device(0)
                train_logger.info(f"[OK] GPU[0] aktif edildi")

            
                # Test tensörü ile GPU'yu kontrol et
                test_tensor = torch.zeros(1).cuda()
                train_logger.info(f"[OK] GPU test başarılı - tensor device: {test_tensor.device}")
                del test_tensor
                torch.cuda.empty_cache()
            except Exception as e:
                train_logger.warning(f" GPU kullanımında sorun: {e}")
        else:
            train_logger.info(" CUDA available: False (CPU modunda çalışılacak)")
    except Exception as e:
        train_logger.warning(f" Ortam bilgisi alınamadı: {e}")

def load_tokenizer_config() -> Dict[str, Any]:
    """
    Config'ten BPE ve tokenizer ayarlarını yükle.
    Returns:
        Dict: BPE ve tokenizer ayarları
    """
    if not CONFIG_AVAILABLE:
        return {}
    
    try:
        # BPE detaylı config
        bpe_config = get_bpe_detailed_config()
        
        # Tokenizer config

        tokenizer_config = TOKENIZER_CONFIG.copy()
        
        # BPE config (eski, uyumluluk için)
        bpe_legacy = BPE_CONFIG.copy()
        
        # Birleştir (öncelik: bpe_config > tokenizer_config > bpe_legacy)
        config = {
            # Tokenizer ayarları (TOKENIZER_CONFIG'ten)
            "max_seq_length": tokenizer_config.get("max_seq_length", 512),
            "vocab_size": tokenizer_config.get("vocab_size", None),  # Dinamik, TokenizerCore belirler
            
            # BPE dosya yolları (bpe_config'ten - vocab_file/merges_file veya vocab_path/merges_path)
            # Önce vocab_path/merges_path'e bak, yoksa vocab_file/merges_file'e bak
            "vocab_path": (
                bpe_config.get("vocab_path") or 
                bpe_config.get("vocab_file") or 
                bpe_legacy.get("vocab_file") or 
                "data/vocab_lib/vocab.json"
            ),
            "merges_path": (
                bpe_config.get("merges_path") or 
                bpe_config.get("merges_file") or 
                bpe_legacy.get("merges_file") or 
                "data/merges_lib/merges.txt"
            ),
            
            # BPE rebuild (training-specific, default False)
            "bpe_rebuild": False,  # Vocab/merges mevcut, rebuild yok
            
            # BPE merge operations (bpe_config'ten - target_merges veya merge_operations)
            "bpe_max_merges": bpe_config.get("target_merges", bpe_config.get("merge_operations", bpe_legacy.get("merge_operations", 50000))),
            "bpe_min_frequency": bpe_config.get("min_frequency", bpe_legacy.get("min_frequency", 2)),
            "bpe_max_iter": bpe_config.get("max_iter", bpe_legacy.get("max_iter", 50000)),
            
            # BPE tokenization ayarları (bpe_config'ten)
            "bpe_include_syllables": bpe_config.get("include_syllables", False),
            "bpe_include_whole_words": bpe_config.get("include_whole_words", True),
            "bpe_include_sep": bpe_config.get("include_sep", False),
            
            # Eğitim tokenizasyonu (BPE training ile uyumlu olmalı - bpe_config'ten)
            "train_include_whole_words": bpe_config.get("include_whole_words", True),
            "train_include_syllables": bpe_config.get("include_syllables", False),
            "train_include_sep": bpe_config.get("include_sep", False),
            # GPU tokenizer desteği (bpe_config'ten)
            "use_gpu": bpe_config.get("use_gpu", True),
            "tokenizer_batch_size": bpe_config.get("batch_size", bpe_legacy.get("batch_size", 128)),
        }
        return config
    except Exception as e:
        # train_logger henüz tanımlanmadı, logging kullan
        logging.getLogger("TrainScript").warning(f"Config yüklenirken hata: {e}, default değerler kullanılacak")
        return {}
def normalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Eğitim/Logger/TB alanlarını tek yerde normalize et."""
    out = dict(cfg)
    # Config'ten BPE ve tokenizer ayarlarını yükle
    tokenizer_config = load_tokenizer_config()
    if tokenizer_config:
        # Config'ten gelen değerlerle override et
        # BPE ve tokenizer ayarları HER ZAMAN config'ten alınır (hardcoded override yok)
        bpe_tokenizer_keys = [
            "vocab_path", "merges_path", "bpe_rebuild",
            "bpe_max_merges", "bpe_min_frequency", "bpe_max_iter",
            "bpe_include_syllables", "bpe_include_whole_words", "bpe_include_sep",
            "train_include_whole_words", "train_include_syllables", "train_include_sep",
            "max_seq_length", "use_gpu", "tokenizer_batch_size", "vocab_size"
        ]

        for key, value in tokenizer_config.items():
            # BPE ve tokenizer ayarları her zaman config'ten alınır
            if key in bpe_tokenizer_keys:
                out[key] = value
            elif key not in out:  # Diğer ayarlar sadece belirtilmemişse config'ten al
                out[key] = value
    # gradient_clip -> max_grad_norm eşlemesi
    if "gradient_clip" in out and "max_grad_norm" not in out:
        out["max_grad_norm"] = out.get("gradient_clip", 1.0)

    # Scheduler kwargs hazırlama (V2 TrainingService için)
    if "scheduler_kwargs" not in out:
        scheduler_kwargs = {}
        # ReduceLROnPlateau parametreleri
        if out.get("lr_decay_factor"):
            scheduler_kwargs["factor"] = out.get("lr_decay_factor")
        if out.get("lr_decay_patience"):
            scheduler_kwargs["patience"] = out.get("lr_decay_patience")
        if out.get("lr_threshold"):
            scheduler_kwargs["threshold"] = out.get("lr_threshold")
        if out.get("lr_min"):
            scheduler_kwargs["min_lr"] = out.get("lr_min")
        if scheduler_kwargs:
            out["scheduler_kwargs"] = scheduler_kwargs
    # TrainingManager batch logları & TB adım logları
    out.setdefault("log_batches_to_console", True)
    out.setdefault("tb_log_train_step", True)
    out.setdefault("tb_log_val_step", True)  # A100'de açılabilir - validation step'leri logla
    out.setdefault("tb_log_graph_from_tm", False)  # istersen True yapabilirsin
    # TensorBoard varsayılanları
    out.setdefault("use_tensorboard", True)
    out.setdefault("tb_log_every_n", 20)
    out.setdefault("tb_log_histograms", False)
    out.setdefault("tb_log_attention_image", True)
    # Güvenli cihaz & matmul hassasiyeti
    # Colab için GPU'yu zorla - CUDA varsa kesinlikle kullan
    if torch.cuda.is_available():
        out.setdefault("device", "cuda")
        # GPU'yu aktif et
        try:
            torch.cuda.set_device(0)
            train_logger.info("[OK] GPU[0] aktif edildi")
        except Exception:
            pass
    else:
        out.setdefault("device", "cpu")
        train_logger.warning(" GPU kullanılamıyor, CPU modunda çalışılacak")
    # [OK] V5: GQA, Sliding Window, YaRN varsayılanları (geriye dönük uyumluluk)
    out.setdefault("num_kv_heads", None)       # None = standart MHA (GQA devre dışı)
    out.setdefault("sliding_window", None)     # None = full attention
    out.setdefault("rope_scaling_type", "none")  # standart RoPE
    out.setdefault("rope_scaling_factor", 1.0)   # ölçekleme yok
    # GPU tokenizer desteği (config'ten alındı - load_tokenizer_config() içinde)
    # NOT: use_gpu zaten load_tokenizer_config() içinde config'ten alındı
    # batch_size model eğitimi için (tokenizer_batch_size değil)
    out.setdefault("batch_size", 16)  # GPU için optimize batch size (model training batch size)
    try:
        # PyTorch 2.x ile matmul hassasiyeti (AMP ile iyi çalışır)
        torch.set_float32_matmul_precision("high")  # type: ignore[attr-defined]
    except Exception:
        pass
    return out
def log_run_summary(svc: TrainingService) -> None:
    tb_dir = str(TRAIN_CONFIG.get("tb_log_dir", "runs/cevahir_training"))
    meta_dir = os.path.join(tb_dir, "meta")
    ensure_dirs(tb_dir, meta_dir, TRAIN_CONFIG.get("model_save_path", ""), TRAIN_CONFIG.get("checkpoint_dir", ""))
    mm = svc.model_manager
    summary = {
        "device": str(mm.device),
        "model": type(mm.model).__name__ if mm.model is not None else None,
        "optimizer": type(mm.optimizer).__name__ if mm.optimizer is not None else None,
        "criterion": type(mm.criterion).__name__ if mm.criterion is not None else None,
        "scheduler": type(mm.scheduler).__name__ if mm.scheduler is not None else None,
        "config_keys": {
            # Çok büyük config’leri tamamen yazmayalım; önemli anahtarları özetleyelim
            "epochs": mm.config.get("epochs"),
            "batch_size": mm.config.get("batch_size"),
            "learning_rate": mm.config.get("learning_rate"),
            "max_seq_length": mm.config.get("max_seq_length"),
            "embed_dim": mm.config.get("embed_dim"),
            "seq_proj_dim": mm.config.get("seq_proj_dim"),
            "num_heads": mm.config.get("num_heads"),
            "max_grad_norm": mm.config.get("max_grad_norm"),
            "use_tensorboard": mm.config.get("use_tensorboard"),
        },
    }
    print("\n=== RUN SUMMARY ===\n" + pformat(summary, width=120))
    with open(os.path.join(meta_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
# =========================
# Konfigürasyon
# =========================
TRAIN_CONFIG: Dict[str, Any] = {
    # === Yol Ayarları ===
    "data_dir": "education",  # Proje kök dizininden çalıştığımız için
    "checkpoint_dir": "saved_models/checkpoints/",
    "training_history_path": "saved_models/training_history.json",
    "model_save_path": "saved_models/cevahir_model.pth",
    # === Eğitim Parametreleri ===
    "epochs": 100,#100 için artırıldı - daha fazla epoch ile daha iyi öğrenme
    # [V6 OOM FIX] batch_size 32 → 16: Dense causal mask fix sonrası 16 güvenli,
    # efektif batch = 16×8 = 128 (önceki 32×4=128 ile aynı) — grad_accum artırıldı
    "batch_size": 64,  # 32 → 16: OOM güvenlik marjı (H100 94 GB ama fragmentation riski)
    "grad_accum_steps": 8,  # 4 → 8: efektif batch = 16×8 = 128 (aynı kalır)
    "learning_rate": 0.0002,  #  DÜZELTME: 0.0001 → 0.0002 (2x artış, model daha hızlı öğrenir)
    "continuation_learning_rate": None,  # Devam eğitimi: set edilirse (örn. 1e-4) checkpoint varken LR bu değere çekilir
    "weight_decay": 0.01,  # NEW: L2 regularization (standard for Adam optimizer)
    # ==========================================================================
    # Optimizer Seçimi
    # ==========================================================================
    # "adamw"      → Standart AdamW (varsayılan, her ortamda çalışır)
    # "adamw8bit"  → bitsandbytes 8-bit AdamW — optimizer m/v durumlarını uint8
    #                olarak saklar → ~8 GB VRAM tasarrufu (Dettmers et al. 2022)
    #                Gereksinim: pip install bitsandbytes
    #                bitsandbytes yoksa otomatik standart AdamW'a fallback yapar.
    # "adam"       → Standart Adam (weight_decay L2, AdamW gibi decoupled değil)
    # "radam"      → RAdam — ısınma aşamasında varyans düzeltmesi (Liu et al. 2019)
    # "sgd"        → SGD + momentum (Nesterov opsiyonel)
    # ==========================================================================
    "optimizer": "adamw8bit",  # 8-bit AdamW → ~8 GB VRAM tasarrufu (bitsandbytes gerekli)
    "pad_token_id": 0,  # [OK] SABİTLEME: PAD ID'yi sezgiye bırakma, tokenizer TOKEN_MAPPING (<PAD>=0)
    "ignore_index": 0,  #  KRİTİK: Loss hesaplamasında PAD token'larını ignore et (PAD_ID=0)
    # [CRITICAL FIX] EOS Token Weight - Degenerate Generation Fix
    # BEFORE: eos_token_weight=10.0 → Model learned to generate EOS immediately (empty responses)
    # AFTER:  eos_token_weight=1.0  → Standard weight, no aggressive bias
    # Evidence: E18-E19 showed 91-96% EOS prob, all responses empty/gibberish
    "label_smoothing": 0.1,  # V3: Label Smoothing (Szegedy et al. 2016) — entropy collapse önleme
    "eos_token_weight": 1.0,  # FIXED: 10.0 → 1.0 (removes aggressive EOS bias, allows natural generation)
    # [OK] YENİ: Learning Rate Scheduler Ayarları (daha dengeli)
    #  KRİTİK DÜZELTME: LR scheduler çok agresif, model öğrenemiyor!
    "scheduler_type": "reduce_on_plateau",  # "reduce_on_plateau" | "plateau" | "rop" | "cosine" | "step" | "onecycle" | "none"
    "lr_decay_factor": 0.75,  #  DÜZELTME: 0.7 → 0.75 (daha yavaş düşüş, LR daha uzun süre yüksek kalır)
    "lr_decay_patience": 15,  #  DÜZELTME: 8 → 15 (2x sabır, scheduler daha geç devreye girer)
    "lr_threshold": 0.005,  #  DÜZELTME: 0.01 → 0.005 (daha hassas, küçük değişimlerde LR düşürmesin)
    "lr_min": 1e-6,  #  YENİ: Minimum LR (çok düşmesin)
    # [OK] YENİ: Warmup Ayarları (stabil başlangıç için)
    #  DİNAMİK: warmup_steps train_loader hazır olduktan sonra hesaplanacak (training_service.py'de)
    "warmup_epochs": 1,  # İlk N epoch warmup (dinamik hesaplama için)
    "warmup_steps": 1500,  # Fallback değer (dinamik hesaplama yapılamazsa)
    "warmup_start_factor": 0.1,  #  ENDÜSTRİ STANDARDI: 0.4 → 0.1 (LR'nin %10'undan başla)
    "early_stopping_patience": 10,  # Daha fazla sabır
    "gradient_clip": 1.0,  #  KRİTİK DÜZELTME: 5.0 → 1.0 (gradient explosion önleme, MaxGrad=1228.51 çok yüksek!)
    # Embedding LR scale: 1.0 = embedding ana model ile aynı LR (endüstri standardı; ayrı scale yok).
    "embedding_lr_scale": 1.0,
    # [OK] GRADIENT FIX: Embedding için warmup yok (sabit LR)
    # Embedding base LR zaten düşük (0.00001), warmup'ta daha da düşürmemek için
    # embedding_warmup_factor=1.0 → Embedding LR warmup'tan geçmez, sabit kalır
    "embedding_warmup_factor": 1.0,  # 1.0 = warmup yok (embedding LR sabit: 0.00001)
    # === GPU Tokenizer Desteği ===
    # NOT: use_gpu ve tokenizer_batch_size normalize_config() içinde BPE_DETAILED_CONFIG'ten alınacak
    # Hardcoded değerler YOK - config'ten alınır!
    # === DataLoader Optimizasyonu V3 ===
    # num_workers: Linux/Colab'ta 4 worker veriyi GPU'ya paralel prefetch eder (RAM → GPU bant genişliği kullanımı).
    # Windows'ta spawn tabanlı multiprocessing pickle sorunlarına yol açabileceğinden otomatik 0 seçilir.
    # 179 GB RAM ile num_workers=4 + pin_memory=True → GPU'nun asla veri beklememesi sağlanır.
    "data_loader_num_workers": 0 if sys.platform.startswith("win") else 4,
    "data_loader_pin_memory": True,  # GPU transfer hızlandırma — page-locked bellek → DMA transfer (CUDA için)
    # --- GPU Batching V3 (BucketSampler + DynamicPad) ---
    "use_bucket_batching": True,    # BucketBatchSampler: seq uzunluğuna göre gruplama → padding waste azalır
    "num_buckets": 32,              # Bucket sayısı (daha fazla = daha az padding, daha az randomness)
    "use_dynamic_padding": True,    # DynamicPaddingCollator: batch içi max uzunluğa pad (global pad yok)
    "prefetch_factor": 2,           # Worker başına prefetch batch (num_workers>0 ise aktif)
    "persistent_workers": True,     # Worker'ları epoch'lar arası canlı tut (num_workers>0 ise)
    # --- Cache V3 (Strict Mode) ---
    "cache_strict_mode": True,      # Cache yoksa eğitim BAŞLAMAZ (CacheNotFoundError fırlatır)
    "cache_verify_integrity": True, # Cache SHA-256 checksum doğrulama
    # === Logging ===
    "enable_file_logging": False,  #  DOSYA LOGGING KAPALI: Windows dosya kilidi sorununu önlemek için
    # === TensorBoard ===
    "use_tensorboard": True,  # Eğitimi izlemek için açık
    "tb_log_dir": "runs/cevahir_training",
    "tb_log_every_n": 20,
    "tb_log_attention_image": True,
    "tb_log_histograms": True,  # A100'de açılabilir - memory yeterli
    "skip_tb_graph_logging": False,  # A100'de açılabilir
    "skip_tb_data_dashboard": False,  # A100'de açılabilir
    # === TrainingManager Özellikleri ===
    "use_amp": True,  # Mixed precision training - A100'de 2x hız artışı sağlar
    "enable_advanced_metrics": False,  #  KAPALI: Advanced metrics RAM'i patlatıyor (validation sırasında OOM)
    # NOT: Advanced metrics tüm predictions/targets'ı RAM'de topluyor, 319 batch için çok fazla memory kullanıyor
    "enable_memory_tracking": True,  # Memory usage tracking
    "enable_performance_tracking": True,  # Performance metrics (batch time, tokens/sec)
    "enable_visualization": True,  # Training visualizations
    # === Model Yapısı - V-2 Mimarisi (Endüstri Standardı) ===
    # NOT: max_seq_length normalize_config() içinde TOKENIZER_CONFIG'ten alınacak
    # [KAPASİTE] 256→384: daha iyi sohbet için genişlik artırıldı (tie_weights için seq_proj_dim=embed_dim)
    "embed_dim": 512,  # Embedding dimension (256: hızlı test, 384: daha iyi kapasite, 512: Colab’ta bellek izle)
    "seq_proj_dim": 512,  # tie_weights için embed_dim ile aynı olmalı
    "num_heads": 8,  # head_dim = 384/6 = 64 (endüstri standardı 64–128)
    "dropout": 0.15,  # Dropout rate (0.15 optimal - overfitting önleme ve genelleme dengesi, GPT-2/3 standardı: 0.1-0.2)
    "attention_type": "multi_head",
    "normalization_type": "layer_norm",
    # [OK] V-2 PARAMETRELERİ (Endüstri Standardı: GPT-2/3/4, BERT, T5)
    # [KAPASİTE] 6→8 layer: derinlik artışı, sohbet kalitesi için
    "num_layers": 8,  # Transformer encoder layer sayısı (6: hafif, 8: önerilen, 12: daha ağır)
    "ffn_dim": None,  # None ise otomatik: seq_proj_dim * 4 (endüstri standardı)
    "pre_norm": True,  # Pre-norm (GPT-2/3/4) veya Post-norm (BERT) - True önerilir
    "causal_mask": True,  # Autoregressive training için causal masking (GPT standardı)
    # [OK] V-3 OPTİMİZASYONLARI (Endüstri Standardı: GPT-3+, Claude, Gemini)
    "use_flash_attention": True,  # Flash Attention 2.0 (HIZLANDIRMA: Attention işlemlerini 2-3x hızlandırır)
    "pe_mode": "rope",  # Positional encoding: "sinusoidal" | "learned" | "rope" (V4 default: rope)
    "use_gradient_checkpointing": True,  # Memory-efficient training (V4 default: True)
    "tie_weights": True,  # Input embedding ve output layer weight sharing (V4 default: True, embed_dim == seq_proj_dim gerekli)
    # [OK] V-4 İYİLEŞTİRMELERİ (Endüstri Standardı: GPT-3+/4, LLaMA, PaLM, Claude, Gemini)
    "use_rmsnorm": True,  # RMSNorm kullan (V4 default: True, LayerNorm yerine)
    "use_swiglu": True,  # SwiGLU activation kullan (V4 default: True, GELU yerine)
    "use_kv_cache": False,  # KV Cache: training'de False (gereksiz buffer allocation önlenir); inference'ta True yap
    "max_cache_len": 2048,  # Maximum cache length (V4 default: 2048)
    "use_advanced_checkpointing": False,  # Advanced checkpointing (opsiyonel, V4 default: False)
    "checkpointing_strategy": "selective",  # Checkpointing strategy (V4 default: "selective")
    "quantization_type": "none",  # Quantization type (V4 default: "none")
    # ==========================================================================
    # MoE (Mixture of Experts) - V4 Mimari Özelliği
    # ==========================================================================
    # MoE, her FFN bloğunu N adet "expert" alt ağa böler ve her token için
    # sadece top-k expert aktive edilir. GPT-4, Mixtral, Switch Transformer standardı.
    #
    # ⚠️  MoE'yu ETKİNLEŞTİRMEK İÇİN (sonraki eğitimde):
    #   "use_moe": True   → MoE aktif (mevcut checkpoint ile başlat YA DA sıfırdan eğit)
    #   "num_experts": 8  → 8 expert (VRAM'e göre 4-16 arası ayarla)
    #   "moe_top_k": 2    → Her token için 2 expert seçilir (Mixtral standardı)
    #
    # 💡 MoE gerektirdiği VRAM: num_experts × FFN parametreleri
    #    512 embed, 8 expert, ffn_dim=2048 → ~16M ek parametre
    #    Başlamak için: num_experts=4, moe_top_k=1 (daha az VRAM)
    #
    # ⚠️  MoE ile eğitime başlarken mevcut checkpoint'ten yüklemek uyumsuzluk
    #    yaratabilir. Yeni eğitim başlatmak en güvenli yoldur.
    # ==========================================================================
    "use_moe": False,  # [OOM FIX] False: MoE devre dışı — aktifleştirmek için True yap (üstteki notu oku)
    # OOM AÇIKLAMASI: use_moe=True iken 8 expert × ffn_dim=2048 → anlık aktivasyon VRAM'i
    # ~8x büyüyor (her expert tüm batch için forward çalışıyor). 512 embed, 8 expert:
    # 8 × (512→2048→512) × batch_size × seq_len → OOM. num_experts=4, moe_top_k=1 ile başla.
    "num_experts": 8,  # Expert sayısı (MoE aktifse, VRAM'e göre 4-16 arası)
    "moe_top_k": 2,   # Her token için seçilecek expert sayısı (Mixtral standardı: 2)

    # ==========================================================================
    # V5 MİMARİ YENİLİKLERİ — GQA + Sliding Window + YaRN RoPE
    # ==========================================================================
    #
    # ── GQA (Grouped Query Attention) ──────────────────────────────────────────
    # KV head sayısını azaltarak cache boyutu ve inference hızı iyileştirilir.
    # num_kv_heads = None  → standart MHA (geriye dönük uyumluluk, tüm Q=K=V)
    # num_kv_heads = 1     → MQA: maksimum hız, minimal cache
    # num_kv_heads = 2     → GQA: num_heads=8'de %75 KV cache azalması  ← ÖNERİLEN
    # num_kv_heads = 4     → GQA: %50 azalma, kalite/hız dengesi
    #
    # 🔄 Mevcut checkpoint'ten yüklemek için:
    #    num_kv_heads=None (MHA) ile eğitilmiş model, num_kv_heads=2 ile
    #    checkpoint uyumsuzdur → yeni eğitim başlatılmalıdır.
    # ==========================================================================
    "num_kv_heads": 2,     # GQA: num_heads=8'de %75 KV cache azalması (LLaMA-2/3 standardı)

    # ── Sliding Window Attention ───────────────────────────────────────────────
    # Her token yalnızca önceki N token'a attend eder (long-context verimlilik).
    # None  → full attention (default, kısa context için yeterli)
    # 512   → hafif yerellik (hızlı, kısa metinler için)
    # 2048  → Gemma standardı (uzun context, dengeli)
    # 4096  → Mistral-7B standardı (en uzun context desteği)
    #
    # ⚠️  Sliding window + causal mask birlikte çalışır (causal mask üst üste gelir).
    # ==========================================================================
    "sliding_window": 512,  # None→full | 512 | 2048→Gemma | 4096→Mistral

    # ── YaRN RoPE Context Uzatma ───────────────────────────────────────────────
    # Standart RoPE 2048 token ötesinde zayıflar; YaRN bu sınırı kaldırır.
    # rope_scaling_type = "none"   → standart RoPE, 2048 token
    # rope_scaling_type = "yarn"   → YaRN NTK-by-parts (LLaMA-3.1 standardı) ← ÖNERİLEN
    # rope_scaling_type = "linear" → Position Interpolation (basit, daha az etkin)
    #
    # rope_scaling_factor: hedef_context / orijinal_context
    #   4096 / 2048 = 2.0   → 2x uzatma
    #   8192 / 2048 = 4.0   → 4x uzatma (LLaMA-3.1 short context standardı)
    #
    # ⚠️  YaRN, fine-tuning (az adım) ile bile etkin; pretrain gerektirmez.
    # ==========================================================================
    "rope_scaling_type": "yarn",   # "none" | "yarn" (önerilir) | "linear"
    "rope_scaling_factor": 2.0,    # 1.0=devre dışı | 2.0=2x | 4.0=4x uzatma

    # ==========================================================================
    # V3 EĞİTİM SİSTEMİ — İleri Seviye Bileşenler
    # ==========================================================================

    # --- Kayıp Fonksiyonu (CompositeLossManager) ---
    # label_smoothing yukarıda tanımlı (0.1)
    "entropy_coeff": 0.01,          # Entropy regularization (Pereyra et al. 2017) — overconfidence cezası
    "use_focal_loss": False,         # Focal Loss (Lin et al. 2017) — imbalanced token sınıfları için
    "focal_gamma": 2.0,              # Focal Loss gamma (2.0 önerilir; yüksek=kolay tokenlara daha az ağırlık)
    "aux_loss_weight": 0.01,         # MoE/MoD auxiliary loss ağırlığı (MoE aktifse devreye girer)

    # --- Exposure Bias: Scheduled Sampling (Bengio et al. 2015) ---
    "use_scheduled_sampling": True,  # Teacher forcing → kendi tahminleri geçişi
    "ss_start_epoch": 10,            # Kaçıncı epoch'ta scheduled sampling başlasın
    "ss_decay_rate": 0.05,           # Her epoch teacher forcing oranı bu kadar düşer
    "min_teacher_forcing": 0.3,      # Teacher forcing oranı bu değerin altına düşmez

    # --- EMA Ağırlıkları (Yazici et al. 2019) ---
    "use_ema": True,                 # Exponential Moving Average ağırlıkları aktif
    "ema_decay": 0.999,              # EMA bozunum faktörü (0.999 önerilen)

    # --- SAM Optimizer (Foret et al. 2021) ---
    "use_sam": False,                # SAM Optimizer — her adım 2x ileri geçiş (daha yavaş, daha iyi genelleme)
    "sam_rho": 0.05,                 # SAM pertürbation büyüklüğü (0.05 önerilen)

    # --- Lookahead Optimizer (Zhang et al. 2019) ---
    "use_lookahead": False,          # Lookahead — slow/fast ağırlık çifti
    "lookahead_k": 5,                # Her k adımda slow weights güncellenir
    "lookahead_alpha": 0.5,          # Slow weights interpolasyon faktörü

    # --- SWA: Stochastic Weight Averaging (Izmailov et al. 2018) ---
    "use_swa": False,                # SWA aktif (son epoch'larda ağırlık ortalaması alır)
    "swa_start_epoch": 80,           # SWA kaçıncı epoch'tan itibaren başlasın
    "swa_lr": 1e-5,                  # SWA learning rate (sabit, düşük)

    # --- LLRD: Layer-wise Learning Rate Decay ---
    "use_llrd": False,               # LLRD aktif (her katmana farklı LR)
    "llrd_decay_factor": 0.9,        # Her katman bir öncekinin decay_factor katı LR alır

    # --- Curriculum Learning (Bengio et al. 2009) ---
    "use_curriculum": False,         # Curriculum Learning — kolay→zor örnek sıralama
    "curriculum_strategy": "length_based",  # "length_based" | "loss_based" | "random"
    "curriculum_max_len_start": 64,  # Başlangıçta max sequence uzunluğu
    "curriculum_warmup_epochs": 20,  # Bu kadar epoch sonra tam veri seti (max_seq_length)

    # --- Güvenlik: NaN Kurtarma ---
    "nan_tolerance": 3,              # Art arda bu kadar NaN sonrası checkpoint'e geri dön
    "nan_lr_reduction": 0.5,         # NaN sonrası LR bu faktörle azaltılır

    # --- Güvenlik: Loss Spike Detection ---
    "spike_n_sigma": 3.0,            # N-sigma eşiğinin üstündeki kayıp artışı spike sayılır
    "spike_window_size": 20,         # Referans pencere büyüklüğü (kaç batch)
    "spike_lr_reduction": 0.8,       # Spike sonrası LR azaltma faktörü

    # --- İzleme ---
    "inference_probe_interval": 5,   # Kaç epoch'ta bir çıkarım kalite testi yapılır
    "log_gradient_health": True,     # Her epoch gradyan sağlık raporu logla
    "log_token_dist": True,          # Token dağılım monitörünü logla
    "save_every_n_epochs": 10,       # Periyodik checkpoint kaydetme sıklığı

    # === Vocab Boyutu (DİNAMİK) ===
    # NOT: vocab_size normalize_config() içinde config'ten alınacak
    # TokenizerCore tarafından otomatik belirlenecek
    # === Cihaz ve Diğer ===
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,

    # === Training Pipeline & Monitoring ===
    "validate_pipeline_before_training": False,  # Eğitim öncesi validation (opsiyonel)
    "enable_training_monitoring": True,  # Training monitoring (epoch sonu analizi)
    "monitoring_window_size": 5,  # Son N epoch'u takip et

    # === BPE ve Tokenizer Ayarları ===
    # NOT: Bu ayarlar normalize_config() içinde tokenizer_management/config.py'den otomatik yüklenecek
    # Hardcoded değerler YOK - tüm değerler config'ten alınır!
    #
    # Config'ten yüklenecekler (normalize_config() içinde load_tokenizer_config() ile):
    # - vocab_path, merges_path (BPE_DETAILED_CONFIG'ten)
    # - bpe_max_merges, bpe_min_frequency, bpe_max_iter (BPE_DETAILED_CONFIG'ten)
    # - bpe_include_syllables, bpe_include_whole_words, bpe_include_sep (BPE_DETAILED_CONFIG'ten)
    # - train_include_whole_words, train_include_syllables, train_include_sep (BPE_DETAILED_CONFIG'ten)
    # - max_seq_length (TOKENIZER_CONFIG'ten)
    # - use_gpu, tokenizer_batch_size (BPE_DETAILED_CONFIG'ten)
    #
    # Fallback değerler load_tokenizer_config() içinde tanımlı (config yüklenemezse)
}

# =========================
# main
# =========================
def main() -> None:
    try:
        # [MEM] CUDA bellek fragment azaltma — CUDA init ÖNCESINDE ayarlanmalı.
        # expandable_segments:True → PyTorch, OS'tan ihtiyaç kadar küçük bloklar ister,
        # büyük "havuz" yerine esnek segment kullanır → fragmentation %30-60 azalır.
        # Not: log_env_info() CUDA'yı init eder, bu satır ondan önce gelmeli.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        #  Loglar burada yazılacak (modül seviyesinde değil)
        train_logger.info("Eğitim başlatılıyor...")
        train_logger.info("Eğitim süreci başlatılıyor...")
        # Ortam bilgisini yaz
        log_env_info()
        #  Seed: Genel random seed (PyTorch, NumPy, Python random için)
        # NOT: Bu seed train/val split için kullanılmaz (data_preparator'da ayrı split_seed var)
        GLOBAL_SEED = int(TRAIN_CONFIG.get("seed", 42))
        set_seed(GLOBAL_SEED)
        train_logger.info(f"[Seed] Global seed ayarlandı: {GLOBAL_SEED} (train/val split için değil)")
        # Dizinleri garanti altına al
        ensure_dirs(
            TRAIN_CONFIG.get("tb_log_dir", "runs/cevahir_training"),
            TRAIN_CONFIG.get("model_save_path", "saved_models/cevahir_model.pth"),
            TRAIN_CONFIG.get("checkpoint_dir", "saved_models/checkpoints/"),
            TRAIN_CONFIG.get("training_history_path", "saved_models/training_history.json"),
            TRAIN_CONFIG.get("vocab_path", "data/vocab_lib/vocab.json"),
            TRAIN_CONFIG.get("merges_path", "data/merges_lib/merges.txt"),
        )
        # Config normalizasyonu
        effective_cfg = normalize_config(TRAIN_CONFIG)
        # Devam eğitimi: checkpoint varsa ve continuation_learning_rate set edilmişse LR'yi düşür
        continuation_lr = effective_cfg.get("continuation_learning_rate")
        if continuation_lr is not None:
            root = Path(project_root)
            chk_dir = effective_cfg.get("checkpoint_dir", "saved_models/checkpoints/")
            model_path = effective_cfg.get("model_save_path", "saved_models/cevahir_model.pth")
            checkpoint_exists = (
                (root / model_path).exists()
                or (root / chk_dir.rstrip("/").rstrip("\\") / "last.pth").exists()
                or (root / chk_dir.rstrip("/").rstrip("\\") / "best.pth").exists()
            )
            if checkpoint_exists:
                effective_cfg["learning_rate"] = float(continuation_lr)
                train_logger.info(f"[Devam eğitimi] Checkpoint bulundu, LR override: learning_rate={continuation_lr}")
        # Etkin kısa özet
        train_logger.info(
            "EFFECTIVE CONFIG (özet):\n" + pformat(
                {
                    k: effective_cfg[k]
                    for k in (
                        "device", "epochs", "batch_size", "learning_rate",
                        "max_seq_length", "embed_dim", "seq_proj_dim",
                        "num_heads", "max_grad_norm", "use_tensorboard",
                        "tb_log_dir", "tb_log_every_n",
                        # V-2 parametreleri
                        "num_layers", "ffn_dim", "pre_norm", "causal_mask",
                        # V-3 optimizasyonları
                        "use_flash_attention", "pe_mode", "use_gradient_checkpointing", "tie_weights",
                        # V-4 iyileştirmeleri
                        "use_rmsnorm", "use_swiglu", "use_kv_cache", "max_cache_len",
                        "use_advanced_checkpointing", "checkpointing_strategy", "quantization_type",
                        "use_moe", "num_experts", "moe_top_k",
                        # V-5 yenilikleri (GQA, Sliding Window, YaRN)
                        "num_kv_heads", "sliding_window",
                        "rope_scaling_type", "rope_scaling_factor",
                        # BPE ve tokenizer ayarları (config'ten yüklendi)
                        "vocab_path", "merges_path", "bpe_rebuild",
                        "bpe_max_merges", "bpe_min_frequency", "bpe_max_iter",
                        "bpe_include_syllables", "bpe_include_whole_words", "bpe_include_sep",
                        "train_include_whole_words", "train_include_syllables", "train_include_sep",
                        "use_gpu", "tokenizer_batch_size",
                    )
                    if k in effective_cfg
                },
                width=120
            )
        )
        # Config kaynağı bilgisi
        if CONFIG_AVAILABLE:
            train_logger.info("[OK] BPE ve tokenizer ayarları tokenizer_management/config.py'den yüklendi")
        else:
            train_logger.warning("  Config yüklenemedi, fallback değerler kullanılıyor")
        # Eğitim servisi seçimi — V3 > V2 öncelik sırası
        if _TRAINING_SYSTEM_V3_AVAILABLE:
            train_logger.info(
                "[V3] Training System V3 kullanılıyor (strict cache, GPU batching V3, config V3)..."
            )
            train_logger.info(
                "[V3] Zorunlu adımlar:\n"
                "  1. python tokenizer_management/train_bpe.py\n"
                "  2. python training_system/prepare_cache.py  ← ZORUNLU\n"
                "  3. python training_system/train.py          ← ŞU AN\n"
                "  Cache bulunamazsa CacheNotFoundError fırlatılır."
            )

            train_logger.info("Eğitim servisi V3 başlatılıyor...")
            training_service_v3 = TrainingServiceV3({
                **effective_cfg,
                "model_module": "src.neural_network",
                "model_class":  "CevahirNeuralNetwork",
                "use_v3_training": True,
            })

            # Eğitim V3
            train_logger.info("Eğitim V3 başlıyor...")
            train_loss, val_loss = training_service_v3.train()
            train_logger.info(f"[V3] Eğitim tamamlandı. Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        else:
            # V2 fallback
            if _V3_AVAILABLE:
                train_logger.info("[V2+V3] TrainingSystem V3 yok, training_management.v3 ile V2 TrainingService...")
            else:
                train_logger.warning("[V2] V3 modülleri import edilemedi, V2 TrainingService kullanılıyor.")

            train_logger.info("Eğitim servisi V2 başlatılıyor...")
            training_service = TrainingService({
                **effective_cfg,
                "model_module": "src.neural_network",
                "model_class":  "CevahirNeuralNetwork",
                "use_v3_training": _V3_AVAILABLE,
            })

            # Koşu özeti (V2)
            log_run_summary(training_service)

            # Eğitim V2
            train_logger.info("Eğitim V2 başlıyor...")
            train_loss, val_loss = training_service.train()
            train_logger.info(f"[V2] Eğitim tamamlandı. Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        print("\nTensorBoard’ı açmak için:\n  tensorboard --logdir runs/cevahir_training --port 6006\n")
    except KeyboardInterrupt:
        train_logger.warning("Eğitim kullanıcı tarafından durduruldu (KeyboardInterrupt).")
    except torch.cuda.OutOfMemoryError as oom:
        # Eğitim döngüsü dışında (örn. model build sırasında) CUDA OOM
        train_logger.error(
            f"[OOM] CUDA Out-of-Memory — model derleme veya ilk forward sırasında. "
            f"batch_size veya max_seq_length küçültün. Hata: {oom}",
            exc_info=False,
        )
    except Exception as e:
        # Typed model_management hatalarına ek açıklama ekle
        if _MM_EXCEPTIONS_AVAILABLE and CevahirModelError and isinstance(e, CevahirModelError):
            train_logger.error(
                f"[ModelError] {type(e).__name__}: {e}",
                exc_info=True,
            )
        else:
            train_logger.error(f"Eğitim sırasında hata oluştu: {str(e)}", exc_info=True)



if __name__ == "__main__":
    main()

