"""
Checkpoint yükleme sorununu debug etmek için detaylı analiz script'i
"""
import torch
import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any

# Root dizini path'e ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_tokenizer_config_from_train_py() -> Dict[str, Any]:
    """training_system/train.py'deki load_tokenizer_config fonksiyonunu simüle et"""
    try:
        from tokenizer_management.config import (
            get_bpe_detailed_config,
            TOKENIZER_CONFIG,
            BPE_CONFIG
        )
        bpe_config = get_bpe_detailed_config()
        tokenizer_config = TOKENIZER_CONFIG.copy()
        bpe_legacy = BPE_CONFIG.copy()

        config = {
            "max_seq_length": tokenizer_config.get("max_seq_length", 512),
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
            "bpe_rebuild": False,
            "bpe_max_merges": bpe_config.get("target_merges", bpe_config.get("merge_operations", bpe_legacy.get("merge_operations", 50000))),
            "bpe_min_frequency": bpe_config.get("min_frequency", bpe_legacy.get("min_frequency", 2)),
            "bpe_max_iter": bpe_config.get("max_iter", bpe_legacy.get("max_iter", 60000)),
            "bpe_include_syllables": bpe_config.get("include_syllables", False),
            "bpe_include_whole_words": bpe_config.get("include_whole_words", True),
            "bpe_include_sep": bpe_config.get("include_sep", False),
            "train_include_whole_words": bpe_config.get("train_include_whole_words", True),
            "train_include_syllables": bpe_config.get("train_include_syllables", False),
            "train_include_sep": bpe_config.get("train_include_sep", False),
            "use_gpu": torch.cuda.is_available(),
            "tokenizer_batch_size": tokenizer_config.get("batch_size", 32),
            "vocab_size": tokenizer_config.get("vocab_size", None),
        }
        return config
    except ImportError:
        logger.error("tokenizer_management.config import edilemedi. Lütfen path'i kontrol edin.")
        return {}


def load_config():
    """Config dosyasını yükle - test script'indeki yöntemi kullan"""
    try:
        from training_system.train import TRAIN_CONFIG
        config = TRAIN_CONFIG.copy()
        tokenizer_config_from_train = load_tokenizer_config_from_train_py()
        config.update(tokenizer_config_from_train)
        return config
    except Exception as e:
        logger.warning(f"TRAIN_CONFIG yüklenemedi, alternatif yöntem deneniyor: {e}")
        return {}


def debug_checkpoint_loading():
    """Checkpoint yükleme sorununu detaylı analiz et"""
    
    # 1. Config yükle
    config = load_config()
    if not config:
        logger.error("❌ Config yüklenemedi")
        return
    logger.info("✅ Config yüklendi")
    
    # 2. Checkpoint yükle
    checkpoint_path = "tests/test_checkpoints/best.pth"
    if not Path(checkpoint_path).exists():
        logger.error(f"❌ Checkpoint bulunamadı: {checkpoint_path}")
        return
    
    logger.info(f"📂 Checkpoint yükleniyor: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # 3. Checkpoint state dict keys'lerini al
    model_state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", None))
    if model_state_dict is None:
        logger.error("❌ Checkpoint'te model_state_dict bulunamadı")
        return
    
    checkpoint_keys = set(model_state_dict.keys())
    logger.info(f"📊 Checkpoint'te {len(checkpoint_keys)} key var")
    logger.info(f"   İlk 10 key: {list(checkpoint_keys)[:10]}")
    
    # 4. Model oluştur
    logger.info("\n🔨 Model oluşturuluyor...")
    try:
        from model_management.model_manager import ModelManager
        from tokenizer_management.core.tokenizer_core import TokenizerCore
        
        # TokenizerCore oluştur
        tokenizer_config_for_core = config.copy()
        if "tokenizer" in config and isinstance(config["tokenizer"], dict):
            tokenizer_config_for_core.update(config["tokenizer"])
        
        if "vocab_path" not in tokenizer_config_for_core:
            tokenizer_config_for_core["vocab_path"] = config.get("vocab_path", "data/vocab_lib/vocab.json")
        if "merges_path" not in tokenizer_config_for_core:
            tokenizer_config_for_core["merges_path"] = config.get("merges_path", "data/merges_lib/merges.txt")
        
        tokenizer_core = TokenizerCore(tokenizer_config_for_core)
        vocab_size = tokenizer_core.get_vocab_size()
        config["vocab_size"] = vocab_size
        
        model_manager = ModelManager(config)
        model_manager.config["vocab_size"] = vocab_size
        model_manager.initialize()
        
        logger.info("✅ Model oluşturuldu")
    except Exception as e:
        logger.error(f"❌ Model oluşturulamadı: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. Model state dict keys'lerini al
    model = model_manager.model
    model_state_dict_before = model.state_dict()
    model_keys = set(model_state_dict_before.keys())
    logger.info(f"📊 Model'de {len(model_keys)} key var")
    logger.info(f"   İlk 10 key: {list(model_keys)[:10]}")
    
    # 6. Keys karşılaştırması
    logger.info("\n🔍 KEYS KARŞILAŞTIRMASI:")
    common_keys = checkpoint_keys & model_keys
    missing_keys = model_keys - checkpoint_keys
    unexpected_keys = checkpoint_keys - model_keys
    
    logger.info(f"✅ Ortak keys: {len(common_keys)}")
    logger.info(f"❌ Model'de var, checkpoint'te yok: {len(missing_keys)}")
    logger.info(f"⚠️ Checkpoint'te var, model'de yok: {len(unexpected_keys)}")
    
    if missing_keys:
        logger.info(f"\n❌ Model'de var, checkpoint'te yok (ilk 20):")
        for key in list(missing_keys)[:20]:
            logger.info(f"   - {key}")
    
    if unexpected_keys:
        logger.info(f"\n⚠️ Checkpoint'te var, model'de yok (ilk 20):")
        for key in list(unexpected_keys)[:20]:
            logger.info(f"   - {key}")
    
    # 7. İlk ortak key'in değerlerini karşılaştır
    if common_keys:
        first_common_key = list(common_keys)[0]
        checkpoint_value = model_state_dict[first_common_key]
        model_value_before = model_state_dict_before[first_common_key]
        
        logger.info(f"\n📊 İlk ortak key: {first_common_key}")
        logger.info(f"   Checkpoint shape: {checkpoint_value.shape}")
        logger.info(f"   Model shape (önce): {model_value_before.shape}")
        logger.info(f"   Checkpoint mean: {checkpoint_value.float().mean().item():.6f}")
        logger.info(f"   Model mean (önce): {model_value_before.float().mean().item():.6f}")
        logger.info(f"   Eşit mi? {torch.equal(checkpoint_value, model_value_before)}")
        logger.info(f"   Yaklaşık eşit mi? {torch.allclose(checkpoint_value.float(), model_value_before.float(), atol=1e-5)}")
    
    # 8. load_state_dict() çağrısını simüle et
    logger.info("\n🔄 load_state_dict() çağrılıyor (strict=False)...")
    try:
        missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
        logger.info(f"✅ load_state_dict() başarılı")
        logger.info(f"   Missing keys: {len(missing)}")
        logger.info(f"   Unexpected keys: {len(unexpected)}")
        
        if missing:
            logger.info(f"   Missing keys (ilk 20): {list(missing)[:20]}")
        if unexpected:
            logger.info(f"   Unexpected keys (ilk 20): {list(unexpected)[:20]}")
    except Exception as e:
        logger.error(f"❌ load_state_dict() başarısız: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 9. Yükleme sonrası kontrol
    model_state_dict_after = model.state_dict()
    if common_keys and first_common_key:
        model_value_after = model_state_dict_after[first_common_key]
        logger.info(f"\n📊 Yükleme sonrası kontrol:")
        logger.info(f"   Model mean (sonra): {model_value_after.float().mean().item():.6f}")
        logger.info(f"   Checkpoint ile eşit mi? {torch.equal(checkpoint_value, model_value_after)}")
        logger.info(f"   Yaklaşık eşit mi? {torch.allclose(checkpoint_value.float(), model_value_after.float(), atol=1e-5)}")
        logger.info(f"   Önceki ile eşit mi? {torch.equal(model_value_before, model_value_after)}")
    
    # 10. İlk parametre kontrolü (test script'indeki gibi)
    first_param_before = next(iter(model_manager.model.parameters()))
    first_param_before_copy = first_param_before.clone().detach()
    param_mean_before = first_param_before_copy.mean().item()
    param_std_before = first_param_before_copy.std().item()
    
    logger.info(f"\n📊 İlk parametre kontrolü (yükleme öncesi):")
    logger.info(f"   Mean: {param_mean_before:.6f}")
    logger.info(f"   Std: {param_std_before:.6f}")
    
    # Yükleme yapılmış, şimdi kontrol et
    first_param_after = next(iter(model_manager.model.parameters()))
    first_param_after_copy = first_param_after.clone().detach()
    param_mean_after = first_param_after_copy.mean().item()
    param_std_after = first_param_after_copy.std().item()
    
    logger.info(f"\n📊 İlk parametre kontrolü (yükleme sonrası):")
    logger.info(f"   Mean: {param_mean_after:.6f}")
    logger.info(f"   Std: {param_std_after:.6f}")
    logger.info(f"   Değişti mi? {not torch.equal(first_param_before_copy, first_param_after_copy)}")
    logger.info(f"   Fark: {abs(param_mean_before - param_mean_after):.10f}")


if __name__ == "__main__":
    debug_checkpoint_loading()

