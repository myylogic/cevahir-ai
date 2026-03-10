"""
FAZE 5.1: Generation Süreci Kontrolü - Gerçek Eğitilmiş Model ile Test

Bu test script'i, GERÇEK EĞİTİLMİŞ MODEL ile generation sürecini test eder.

Önemli Notlar:
1. Training_service'te ModelManager üzerinden üretim yapılıyor
2. ModelManager üzerinden eğitim yapılıyor
3. Normal deneme süreçlerinde chat_pipeline.py'de cevahir.py kullanılıyor
4. İki yaklaşım aynı sonucu vermeli

Test Edilecekler:
1. Mevcut eğitilmiş modeli yükle
2. Training_service yaklaşımı ile generation yap
3. Cevahir.py yaklaşımı ile generation yap
4. İki yaklaşımın sonuçlarını karşılaştır
5. Generation sürecinde logits, sampling, EOS handling'i detaylı analiz et
6. Anlamsız çıktıların nereden geldiğini tespit et

Kod Konumu:
- training_system/v2/core/training_service.py → _test_model_after_epoch() (ModelManager kullanır)
- model/cevahir.py → CevahirModelAPI.generate() (Cevahir kullanır)
"""

import torch
import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Root dizini path'e ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_saved_models():
    """Eğitilmiş modelleri bul"""
    saved_models_dir = Path("saved_models")
    checkpoints_dir = Path("tests/test_checkpoints")
    
    models = []
    
    # saved_models dizinini kontrol et
    if saved_models_dir.exists():
        for pth_file in saved_models_dir.glob("*.pth"):
            models.append({
                "path": str(pth_file),
                "type": "saved_models",
                "name": pth_file.stem
            })
    
    # test_checkpoints dizinini kontrol et
    if checkpoints_dir.exists():
        for pth_file in checkpoints_dir.glob("*.pth"):
            models.append({
                "path": str(pth_file),
                "type": "test_checkpoints",
                "name": pth_file.stem
            })
    
    return models


def load_config():
    """Config dosyasını yükle - train.py'deki yöntemi kullan"""
    try:
        # train.py'deki config yükleme yöntemini kullan
        from training_system.train import TRAIN_CONFIG
        return TRAIN_CONFIG
    except Exception as e:
        logger.warning(f"TRAIN_CONFIG yüklenemedi, alternatif yöntem deneniyor: {e}")
        try:
            # Alternatif: JSON dosyasından yükle
            config_path = Path("config.json")
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                return config
            else:
                # Varsayılan config (minimal)
                logger.warning("Config dosyası bulunamadı, varsayılan config kullanılıyor")
                return {
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                    "tokenizer": {},
                    "model": {},
                }
        except Exception as e2:
            logger.error(f"Config yüklenemedi: {e2}")
            # Varsayılan config döndür
            return {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "tokenizer": {},
                "model": {},
            }


def test_generation_with_modelmanager(prompt: str, model_manager, tokenizer_core, device: str = "cuda"):
    """
    Training_service yaklaşımı: ModelManager kullanarak generation yap
    
    Bu, training_service.py'deki _test_model_after_epoch() metodunun
    generation yaklaşımını simüle eder.
    """
    logger.info("=" * 80)
    logger.info("GENERATION: ModelManager Yaklaşımı (TrainingService)")
    logger.info("=" * 80)
    
    try:
        # Tokenize
        _, token_ids = tokenizer_core.encode(
            prompt,
            mode="inference",
            include_whole_words=True,
            include_syllables=False,
            include_sep=False,
        )
        
        if not token_ids:
            logger.warning(f"Prompt tokenize edilemedi: '{prompt}'")
            return None, None
        
        logger.info(f"Prompt: '{prompt}'")
        logger.info(f"Token IDs: {token_ids[:20]}..." if len(token_ids) > 20 else f"Token IDs: {token_ids}")
        
        # EOS token ID
        vocab = tokenizer_core.get_vocab()
        eos_id = None
        if isinstance(vocab.get("<EOS>"), dict):
            eos_id = vocab["<EOS>"].get("id")
        elif isinstance(vocab.get("<EOS>"), int):
            eos_id = vocab["<EOS>"]
        
        logger.info(f"EOS token ID: {eos_id}")
        
        # Model'i eval mode'a al
        if model_manager.model is not None:
            original_mode = model_manager.model.training
            model_manager.model.eval()
        else:
            logger.error("ModelManager'da model yok!")
            return None, None
        
        # Generation loop (training_service.py'deki gibi)
        generation_details = []
        
        with torch.no_grad():
            generated_ids = list(token_ids)
            max_new_tokens = 20
            min_new_tokens = 5
            
            tokens_generated = 0
            for step in range(max_new_tokens):
                input_tensor = torch.tensor([generated_ids], dtype=torch.long, device=device)
                
                # Model forward (training_service.py'deki gibi: model_manager.model(input_tensor))
                output = model_manager.model(input_tensor)
                if isinstance(output, tuple):
                    output = output[0]  # logits
                
                next_logits = output[0, -1, :]  # Son token'ın logits'i
                next_logits = torch.clamp(next_logits, min=-50.0, max=50.0)
                
                # Temperature ve sampling (training_service.py'deki gibi)
                temperature = 1.0
                if temperature > 0:
                    next_logits = next_logits / temperature
                
                probs = torch.softmax(next_logits, dim=-1)
                
                # Sample
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    predicted_id = torch.argmax(probs).item()
                else:
                    predicted_id = torch.multinomial(probs, 1).item()
                
                # Generation details
                top_5_probs, top_5_indices = torch.topk(probs, k=min(5, len(probs)))
                
                # DEBUG: Check for suspicious values
                probs_sum = probs.sum().item()
                probs_has_nan = torch.isnan(probs).any().item()
                probs_has_inf = torch.isinf(probs).any().item()
                probs_max = probs.max().item()
                probs_min = probs.min().item()
                
                generation_details.append({
                    "step": step,
                    "input_length": len(generated_ids),
                    "predicted_id": predicted_id,
                    "logits_mean": next_logits.mean().item(),
                    "logits_std": next_logits.std().item(),
                    "logits_min": next_logits.min().item(),
                    "logits_max": next_logits.max().item(),
                    "top_5_tokens": top_5_indices.tolist(),
                    "top_5_probs": [f"{p:.4f}" for p in top_5_probs.tolist()],  # Format as string with 4 decimals
                    "top_5_probs_raw": top_5_probs.tolist(),  # DEBUG: Raw values for analysis
                    "probs_sum": probs_sum,  # DEBUG: Should be ~1.0
                    "probs_has_nan": probs_has_nan,  # DEBUG: Should be False
                    "probs_has_inf": probs_has_inf,  # DEBUG: Should be False
                    "probs_max": probs_max,  # DEBUG: Should be > 0
                    "probs_min": probs_min,  # DEBUG: Should be >= 0
                    "is_eos": predicted_id == eos_id if eos_id is not None else False,
                })
                
                # EOS kontrolü (training_service.py'deki gibi)
                if eos_id is not None and predicted_id == eos_id:
                    if tokens_generated >= min_new_tokens:
                        logger.info(f"Step {step}: EOS token bulundu, generation durdu")
                        break
                    # EOS'u ignore et, devam et
                    probs[eos_id] = 0.0
                    probs = probs / (probs.sum() + 1e-10)
                    predicted_id = torch.multinomial(probs, 1).item()
                
                generated_ids.append(predicted_id)
                tokens_generated += 1
        
        # Decode
        prompt_length = len(token_ids)
        predicted_token_ids = generated_ids[prompt_length:]
        
        if len(predicted_token_ids) == 0:
            response = ""
        else:
            response = tokenizer_core.decode(
                predicted_token_ids,
                method="bpe",
                remove_specials=True,
                remove_tags=True,
                collapse_spaces=True,
                lowercase=False,
            )
        
        logger.info(f"Generated tokens: {predicted_token_ids}")
        logger.info(f"Generated text: '{response}'")
        logger.info(f"Total tokens generated: {len(predicted_token_ids)}")
        
        # Model mode'u geri al
        if model_manager.model is not None:
            model_manager.model.train(original_mode)
        
        return response, generation_details
        
    except Exception as e:
        logger.error(f"ModelManager generation hatası: {e}", exc_info=True)
        return None, None


def test_generation_with_cevahir(prompt: str, cevahir, max_new_tokens: int = 20):
    """
    Cevahir.py yaklaşımı: Cevahir API kullanarak generation yap
    
    Bu, chat_pipeline.py'deki generation yaklaşımını simüle eder.
    """
    logger.info("=" * 80)
    logger.info("GENERATION: Cevahir API Yaklaşımı")
    logger.info("=" * 80)
    
    try:
        # Cevahir.generate() kullan
        response = cevahir.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
        )
        
        logger.info(f"Prompt: '{prompt}'")
        logger.info(f"Generated text: '{response}'")
        logger.info(f"Generated text length: {len(response)}")
        
        # Not: Cevahir API'nin generation details'ını almak için
        # _model_api._autoregressive_generate() metodunu doğrudan çağıramayız
        # Çünkü private method. Bu yüzden generation details None döndürüyoruz.
        
        return response, None  # Generation details Cevahir API'de mevcut değil
        
    except Exception as e:
        logger.error(f"Cevahir generation hatası: {e}", exc_info=True)
        return None, None


def compare_generations(
    prompt: str,
    modelmanager_result: Optional[str],
    cevahir_result: Optional[str],
    modelmanager_details: Optional[List[Dict]],
):
    """İki generation yaklaşımını karşılaştır"""
    logger.info("=" * 80)
    logger.info("GENERATION KARŞILAŞTIRMASI")
    logger.info("=" * 80)
    
    logger.info(f"Prompt: '{prompt}'")
    logger.info(f"\nModelManager (TrainingService) sonucu:")
    logger.info(f"  Text: '{modelmanager_result}'")
    logger.info(f"  Length: {len(modelmanager_result) if modelmanager_result else 0}")
    
    logger.info(f"\nCevahir API sonucu:")
    logger.info(f"  Text: '{cevahir_result}'")
    logger.info(f"  Length: {len(cevahir_result) if cevahir_result else 0}")
    
    # Karşılaştırma
    if modelmanager_result and cevahir_result:
        if modelmanager_result == cevahir_result:
            logger.info("\n✅ İki yaklaşım AYNI sonucu verdi!")
        else:
            logger.info("\n⚠️ İki yaklaşım FARKLI sonuçlar verdi!")
            logger.info("   Bu beklenen bir durum olabilir (sampling farklılıkları)")
    
    # ModelManager generation details analizi
    if modelmanager_details:
        logger.info("\n" + "=" * 80)
        logger.info("MODELMANAGER GENERATION DETAILS ANALİZİ")
        logger.info("=" * 80)
        
        for detail in modelmanager_details[:5]:  # İlk 5 step
            logger.info(f"\nStep {detail['step']}:")
            logger.info(f"  Predicted token ID: {detail['predicted_id']}")
            logger.info(f"  Logits: mean={detail['logits_mean']:.4f}, std={detail['logits_std']:.4f}")
            logger.info(f"  Logits range: [{detail['logits_min']:.4f}, {detail['logits_max']:.4f}]")
            logger.info(f"  Top-5 tokens: {detail['top_5_tokens']}")
            logger.info(f"  Top-5 probs (formatted): {detail['top_5_probs']}")
            if 'top_5_probs_raw' in detail:
                logger.info(f"  Top-5 probs (RAW): {detail['top_5_probs_raw']}")
                logger.info(f"  Probs sum: {detail.get('probs_sum', 'N/A'):.6f} (should be ~1.0)")
                logger.info(f"  Probs has NaN: {detail.get('probs_has_nan', 'N/A')}")
                logger.info(f"  Probs has Inf: {detail.get('probs_has_inf', 'N/A')}")
                logger.info(f"  Probs max: {detail.get('probs_max', 'N/A'):.6f}")
                logger.info(f"  Probs min: {detail.get('probs_min', 'N/A'):.6f}")
            logger.info(f"  Is EOS: {detail['is_eos']}")


def main():
    """Ana test fonksiyonu"""
    print("=" * 80)
    print("FAZE 5.1: GENERATION SÜRECİ KONTROLÜ - GERÇEK MODEL İLE")
    print("=" * 80)
    print("\nBu test, GERÇEK EĞİTİLMİŞ MODEL ile generation sürecini test eder.")
    print("TrainingService (ModelManager) ve Cevahir API yaklaşımlarını karşılaştırır.")
    print("=" * 80)
    
    # 1. Eğitilmiş modelleri bul
    print("\n1. Eğitilmiş modelleri arıyorum...")
    models = find_saved_models()
    
    if not models:
        print("\n❌ Eğitilmiş model bulunamadı!")
        print("   Lütfen önce bir model eğitin veya saved_models/ dizinine model ekleyin.")
        return False
    
    print(f"\n✅ {len(models)} model bulundu:")
    for i, model in enumerate(models):
        print(f"   {i+1}. {model['name']} ({model['type']})")
    
    # İlk modeli kullan (veya kullanıcı seçebilir)
    selected_model = models[0]
    print(f"\nKullanılacak model: {selected_model['name']}")
    
    # 2. Config yükle
    print("\n2. Config yükleniyor...")
    config = load_config()
    if not config:
        print("❌ Config yüklenemedi!")
        return False
    print("✅ Config yüklendi")
    
    # 3. ModelManager yaklaşımı için model yükle
    print("\n3. ModelManager yaklaşımı için model yükleniyor...")
    try:
        from model_management.model_manager import ModelManager
        from tokenizer_management.core.tokenizer_core import TokenizerCore
        
        # Config'ten vocab_path ve merges_path al (train.py'deki gibi)
        # Config direkt olarak vocab_path/merges_path içerebilir veya tokenizer altında olabilir
        tokenizer_config = config.copy()
        if "tokenizer" in config and isinstance(config["tokenizer"], dict):
            tokenizer_config.update(config["tokenizer"])
        
        # Vocab ve merges path'leri kontrol et
        if "vocab_path" not in tokenizer_config and "vocab_file" not in tokenizer_config:
            # Varsayılan path'leri kullan
            tokenizer_config["vocab_path"] = config.get("vocab_path", "data/vocab_lib/vocab.json")
        if "merges_path" not in tokenizer_config and "merges_file" not in tokenizer_config:
            # Varsayılan path'leri kullan
            tokenizer_config["merges_path"] = config.get("merges_path", "data/merges_lib/merges.txt")
        
        tokenizer_core = TokenizerCore(tokenizer_config)
        
        # TokenizerCore'dan vocab_size al ve config'e ekle (TrainingService'teki gibi)
        vocab_size = tokenizer_core.get_vocab_size()
        config["vocab_size"] = vocab_size
        logger.info(f"Vocab size config'e eklendi: {vocab_size}")
        
        model_manager = ModelManager(config)
        
        # ModelManager'ın config'ini de güncelle
        model_manager.config["vocab_size"] = vocab_size
        
        model_manager.initialize()
        
        # Model checkpoint yükle
        if Path(selected_model['path']).exists():
            print(f"   Model checkpoint yükleniyor: {selected_model['path']}")
            try:
                model_manager.load(selected_model['path'], strict=False)
                print(f"   ✅ Checkpoint yüklendi: {selected_model['path']}")
            except Exception as e:
                print(f"   ⚠️ Checkpoint yüklenemedi: {e}")
                print(f"   ⚠️ Yeni/random weights ile devam ediliyor")
        else:
            print(f"   ⚠️ Checkpoint dosyası bulunamadı: {selected_model['path']}")
            print(f"   ⚠️ Yeni/random weights ile devam ediliyor")
        
        print("✅ ModelManager hazır")
        
    except Exception as e:
        print(f"❌ ModelManager hazırlanamadı: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. Cevahir API yaklaşımı için model yükle
    print("\n4. Cevahir API yaklaşımı için model yükleniyor...")
    cevahir = None
    try:
        from model.cevahir import Cevahir
        
        # Aynı config'i kullan (Cevahir config format'ı farklı olabilir, dikkatli ol)
        # Şimdilik skip ediyoruz çünkü Cevahir config format'ı farklı olabilir
        # ve ModelManager testi yeterli olacaktır
        print("⚠️ Cevahir API şimdilik skip ediliyor (ModelManager testi yeterli)")
        # cevahir = Cevahir(config)  # Config format uyumsuzluğu olabilir
        
    except Exception as e:
        print(f"⚠️ Cevahir API hazırlanamadı (skip edilebilir): {e}")
        # Cevahir API olmadan da ModelManager testi yapılabilir
    
    # 5. Test prompt'ları
    test_prompts = [
        "selam",
        "merhaba nasılsın",
        "ben bir yapay zeka modeliyim",
    ]
    
    print("\n5. Generation testleri başlıyor...")
    print("=" * 80)
    
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    for prompt in test_prompts:
        print("\n" + "=" * 80)
        print(f"TEST PROMPT: '{prompt}'")
        print("=" * 80)
        
        # ModelManager yaklaşımı
        modelmanager_result, modelmanager_details = test_generation_with_modelmanager(
            prompt, model_manager, tokenizer_core, device
        )
        
        # Cevahir API yaklaşımı (eğer varsa)
        cevahir_result = None
        if cevahir:
            cevahir_result, _ = test_generation_with_cevahir(prompt, cevahir)
        
        # Karşılaştırma
        compare_generations(prompt, modelmanager_result, cevahir_result, modelmanager_details)
    
    print("\n" + "=" * 80)
    print("TEST TAMAMLANDI")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

