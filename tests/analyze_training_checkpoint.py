"""
FAZE 1: Gerçek Eğitim Checkpoint Analizi

Bu script, eğitim sırasında kaydedilen best.pth checkpoint'ini detaylı analiz eder:
- Hangi model'den geldiği
- Neden sadece 3 key var
- Key'lerin hangi modülden geldiği
- Checkpoint metadata'sı
"""

import torch
import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Root dizini path'e ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_checkpoint_structure(checkpoint_path: str) -> Dict[str, Any]:
    """Checkpoint yapısını analiz et"""
    logger.info("=" * 80)
    logger.info("CHECKPOINT YAPISI ANALİZİ")
    logger.info("=" * 80)
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Checkpoint keys
    checkpoint_keys = list(checkpoint.keys())
    logger.info(f"\n📊 Checkpoint Top-Level Keys: {len(checkpoint_keys)}")
    for key in checkpoint_keys:
        value_type = type(checkpoint[key]).__name__
        if isinstance(checkpoint[key], dict):
            value_info = f"dict with {len(checkpoint[key])} keys"
        elif isinstance(checkpoint[key], torch.Tensor):
            value_info = f"Tensor {list(checkpoint[key].shape)}"
        else:
            value_info = str(checkpoint[key])[:100] if checkpoint[key] else "None"
        logger.info(f"  - {key}: {value_type} ({value_info})")
    
    return checkpoint


def analyze_model_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Model state dict'i analiz et"""
    logger.info("\n" + "=" * 80)
    logger.info("MODEL STATE DICT ANALİZİ")
    logger.info("=" * 80)
    
    # Model state dict'i al
    model_state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", {}))
    
    if not model_state_dict:
        logger.error("❌ Checkpoint'te model_state_dict veya state_dict bulunamadı!")
        return {}
    
    # Key'leri al
    keys = list(model_state_dict.keys())
    logger.info(f"\n📊 Model State Dict Keys: {len(keys)}")
    logger.info(f"Keys: {keys}")
    
    # Key'leri kategorize et
    key_categories = {
        "embedding": [],
        "projection": [],
        "layers": [],
        "output": [],
        "other": []
    }
    
    for key in keys:
        key_lower = key.lower()
        if "embed" in key_lower:
            key_categories["embedding"].append(key)
        elif "proj" in key_lower:
            key_categories["projection"].append(key)
        elif "layer" in key_lower or key.startswith("layers"):
            key_categories["layers"].append(key)
        elif "output" in key_lower:
            key_categories["output"].append(key)
        else:
            key_categories["other"].append(key)
    
    logger.info("\n📂 Key Kategorileri:")
    for category, category_keys in key_categories.items():
        if category_keys:
            logger.info(f"  {category.upper()}: {len(category_keys)} keys")
            for key in category_keys:
                tensor = model_state_dict[key]
                shape = list(tensor.shape) if hasattr(tensor, 'shape') else "N/A"
                logger.info(f"    - {key}: {shape}")
    
    # Tensor shape analizi
    logger.info("\n📏 Tensor Shape Analizi:")
    for key in keys:
        tensor = model_state_dict[key]
        if hasattr(tensor, 'shape'):
            shape = list(tensor.shape)
            numel = tensor.numel()
            logger.info(f"  {key}: shape={shape}, numel={numel}")
    
    return {
        "keys": keys,
        "num_keys": len(keys),
        "key_categories": key_categories,
        "model_state_dict": model_state_dict
    }


def analyze_checkpoint_metadata(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Checkpoint metadata'sını analiz et"""
    logger.info("\n" + "=" * 80)
    logger.info("CHECKPOINT METADATA ANALİZİ")
    logger.info("=" * 80)
    
    metadata = {}
    
    # Epoch
    epoch = checkpoint.get("epoch", None)
    if epoch is not None:
        logger.info(f"📅 Epoch: {epoch}")
        metadata["epoch"] = epoch
    
    # Config
    config = checkpoint.get("config", None)
    if config:
        logger.info(f"⚙️ Config: {type(config).__name__}")
        if isinstance(config, dict):
            logger.info(f"  Config keys: {list(config.keys())[:10]}...")  # İlk 10 key
            metadata["config"] = config
        else:
            metadata["config"] = str(config)
    
    # Training history
    training_history = checkpoint.get("training_history", None)
    if training_history:
        logger.info(f"📊 Training History: {type(training_history).__name__}")
        if isinstance(training_history, dict):
            logger.info(f"  History keys: {list(training_history.keys())}")
            metadata["training_history"] = training_history
    
    # Metric
    metric = checkpoint.get("metric", None)
    if metric is not None:
        logger.info(f"📈 Metric: {metric}")
        metadata["metric"] = metric
    
    # Saved at
    saved_at = checkpoint.get("saved_at", None)
    if saved_at:
        logger.info(f"🕐 Saved at: {saved_at}")
        metadata["saved_at"] = saved_at
    
    # Optimizer state dict
    optimizer_state_dict = checkpoint.get("optimizer_state_dict", checkpoint.get("optimizer_state", None))
    if optimizer_state_dict:
        logger.info(f"⚡ Optimizer State Dict: {type(optimizer_state_dict).__name__}")
        if isinstance(optimizer_state_dict, dict):
            logger.info(f"  Optimizer state keys: {list(optimizer_state_dict.keys())[:10]}...")
        metadata["has_optimizer"] = True
    else:
        metadata["has_optimizer"] = False
    
    # Scheduler state dict
    scheduler_state_dict = checkpoint.get("scheduler_state_dict", checkpoint.get("scheduler_state", None))
    if scheduler_state_dict:
        logger.info(f"📉 Scheduler State Dict: {type(scheduler_state_dict).__name__}")
        metadata["has_scheduler"] = True
    else:
        metadata["has_scheduler"] = False
    
    return metadata


def identify_model_type(model_state_dict: Dict[str, Any]) -> str:
    """State dict'ten model tipini tespit et"""
    logger.info("\n" + "=" * 80)
    logger.info("MODEL TİPİ TESPİTİ")
    logger.info("=" * 80)
    
    keys = list(model_state_dict.keys())
    num_keys = len(keys)
    
    # Key pattern analizi
    patterns = {
        "dil_katmani": any("dil_katmani" in k for k in keys),
        "layers": any(k.startswith("layers.") for k in keys),
        "embed": any("embed" in k.lower() for k in keys),
        "proj": any("proj" in k.lower() for k in keys),
        "output": any("output" in k.lower() for k in keys),
        "transformer": any("transformer" in k.lower() for k in keys),
    }
    
    logger.info(f"📊 Key Patterns:")
    for pattern, found in patterns.items():
        status = "✅" if found else "❌"
        logger.info(f"  {status} {pattern}: {found}")
    
    # Model tipi tahmini
    if num_keys == 3 and all(k in keys for k in ["embed.weight", "proj.weight", "proj.bias"]):
        logger.info("\n🔍 TAHMİN: Çok basit bir embedding + projection modeli")
        logger.info("  Bu, CevahirNeuralNetwork'dan GELMİYOR!")
        logger.info("  Muhtemelen eski bir model veya test modeli")
        return "simple_embedding_projection"
    
    elif patterns["dil_katmani"] and patterns["layers"]:
        logger.info("\n🔍 TAHMİN: CevahirNeuralNetwork (Modern mimari)")
        logger.info(f"  Keys: {num_keys} (beklenen: 200+)")
        return "cevahir_neural_network"
    
    elif patterns["layers"] and num_keys > 100:
        logger.info("\n🔍 TAHMİN: Transformer-based model (genel)")
        return "transformer_model"
    
    elif patterns["embed"] and patterns["proj"]:
        logger.info("\n🔍 TAHMİN: Embedding + Projection modeli (basit)")
        return "embedding_projection"
    
    else:
        logger.info("\n🔍 TAHMİN: Bilinmeyen model tipi")
        return "unknown"


def compare_with_current_model(model_state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Checkpoint'i mevcut model mimarisi ile karşılaştır"""
    logger.info("\n" + "=" * 80)
    logger.info("MEVCUT MODEL MİMARİSİ İLE KARŞILAŞTIRMA")
    logger.info("=" * 80)
    
    try:
        from model_management.model_manager import ModelManager
        from tokenizer_management.core.tokenizer_core import TokenizerCore
        from training_system.train import TRAIN_CONFIG
        
        # Config yükle
        config = TRAIN_CONFIG.copy()
        
        # TokenizerCore oluştur
        tokenizer_config = config.copy()
        if "vocab_path" not in tokenizer_config:
            tokenizer_config["vocab_path"] = config.get("vocab_path", "data/vocab_lib/vocab.json")
        if "merges_path" not in tokenizer_config:
            tokenizer_config["merges_path"] = config.get("merges_path", "data/merges_lib/merges.txt")
        
        tokenizer_core = TokenizerCore(tokenizer_config)
        vocab_size = tokenizer_core.get_vocab_size()
        config["vocab_size"] = vocab_size
        
        # ModelManager oluştur
        model_manager = ModelManager(config)
        model_manager.config["vocab_size"] = vocab_size
        model_manager.initialize()
        
        # Model state dict
        current_model_state_dict = model_manager.model.state_dict()
        current_keys = set(current_model_state_dict.keys())
        checkpoint_keys = set(model_state_dict.keys())
        
        logger.info(f"\n📊 Karşılaştırma:")
        logger.info(f"  Checkpoint keys: {len(checkpoint_keys)}")
        logger.info(f"  Current model keys: {len(current_keys)}")
        
        common_keys = checkpoint_keys & current_keys
        missing_in_checkpoint = current_keys - checkpoint_keys
        unexpected_in_checkpoint = checkpoint_keys - current_keys
        
        logger.info(f"  ✅ Ortak keys: {len(common_keys)}")
        logger.info(f"  ❌ Model'de var, checkpoint'te yok: {len(missing_in_checkpoint)}")
        logger.info(f"  ⚠️ Checkpoint'te var, model'de yok: {len(unexpected_in_checkpoint)}")
        
        if unexpected_in_checkpoint:
            logger.info(f"\n⚠️ Checkpoint'te beklenmeyen keys (ilk 10):")
            for key in list(unexpected_in_checkpoint)[:10]:
                logger.info(f"    - {key}")
        
        if missing_in_checkpoint and len(missing_in_checkpoint) > 0:
            logger.info(f"\n❌ Checkpoint'te eksik keys (ilk 20):")
            for key in list(missing_in_checkpoint)[:20]:
                logger.info(f"    - {key}")
        
        return {
            "common_keys": len(common_keys),
            "missing_in_checkpoint": len(missing_in_checkpoint),
            "unexpected_in_checkpoint": len(unexpected_in_checkpoint),
            "keys_match": len(common_keys) == len(current_keys) == len(checkpoint_keys)
        }
        
    except Exception as e:
        logger.error(f"❌ Mevcut model ile karşılaştırma hatası: {e}", exc_info=True)
        return {}


def main():
    """Ana analiz fonksiyonu"""
    logger.info("=" * 80)
    logger.info("FAZE 1: GERÇEK EĞİTİM CHECKPOINT ANALİZİ")
    logger.info("=" * 80)
    logger.info("")
    
    # Checkpoint path
    checkpoint_path = "tests/test_checkpoints/best.pth"
    
    if not Path(checkpoint_path).exists():
        logger.error(f"❌ Checkpoint bulunamadı: {checkpoint_path}")
        return
    
    logger.info(f"📂 Checkpoint: {checkpoint_path}")
    logger.info("")
    
    # 1. Checkpoint yapısını analiz et
    checkpoint = analyze_checkpoint_structure(checkpoint_path)
    
    # 2. Model state dict'i analiz et
    model_analysis = analyze_model_state_dict(checkpoint)
    if not model_analysis:
        logger.error("❌ Model state dict analizi başarısız!")
        return
    
    # 3. Checkpoint metadata'sını analiz et
    metadata = analyze_checkpoint_metadata(checkpoint)
    
    # 4. Model tipini tespit et
    model_type = identify_model_type(model_analysis["model_state_dict"])
    
    # 5. Mevcut model ile karşılaştır
    comparison = compare_with_current_model(model_analysis["model_state_dict"])
    
    # Özet
    logger.info("\n" + "=" * 80)
    logger.info("ANALİZ ÖZETİ")
    logger.info("=" * 80)
    logger.info(f"📊 Checkpoint Keys: {model_analysis['num_keys']}")
    logger.info(f"🔍 Model Tipi: {model_type}")
    logger.info(f"📅 Epoch: {metadata.get('epoch', 'N/A')}")
    logger.info(f"📈 Metric: {metadata.get('metric', 'N/A')}")
    logger.info(f"✅ Keys Eşleşmesi: {comparison.get('keys_match', False)}")
    
    if model_analysis['num_keys'] == 3:
        logger.info("\n⚠️ KRİTİK BULGU: Checkpoint sadece 3 key içeriyor!")
        logger.info("  Bu checkpoint, CevahirNeuralNetwork'dan GELMİYOR!")
        logger.info("  Muhtemelen eski bir model veya test modeli kaydedilmiş.")
    elif comparison.get('keys_match', False):
        logger.info("\n✅ Checkpoint, mevcut model mimarisi ile eşleşiyor!")
    else:
        logger.info("\n⚠️ Checkpoint, mevcut model mimarisi ile eşleşmiyor!")
        logger.info(f"  Eksik keys: {comparison.get('missing_in_checkpoint', 0)}")
        logger.info(f"  Beklenmeyen keys: {comparison.get('unexpected_in_checkpoint', 0)}")
    
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

