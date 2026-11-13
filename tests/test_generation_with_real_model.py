"""
FAZE 5.1: Generation Süreci Kontrolü - Gerçek Eğitilmiş Model ile Test

Bu test modülü, GERÇEK EĞİTİLMİŞ MODEL ile generation sürecini test eder.

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

Not: Eğitilmiş checkpoint'ler (saved_models/*.pth, tests/test_checkpoints/*.pth) repo'da
tutulmaz; bulunamazsa testler atlanır (skip). Cevahir API karşılaştırması orijinal script'te
de devre dışıydı (config format uyumsuzluğu); karşılaştırma yalnızca ModelManager sonucunu loglar.
"""

import copy
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import torch

# Root dizini path'e ekle
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)

# Eğitilmiş modellerin arandığı dizinler (CWD'den bağımsız, repo köküne göre)
CHECKPOINT_DIRS = [
    ("saved_models", REPO_ROOT / "saved_models"),
    ("test_checkpoints", REPO_ROOT / "tests" / "test_checkpoints"),
]

TEST_PROMPTS = [
    "selam",
    "merhaba nasılsın",
    "ben bir yapay zeka modeliyim",
]

MAX_NEW_TOKENS = 20
MIN_NEW_TOKENS = 5


def find_saved_models() -> List[Dict[str, str]]:
    """Eğitilmiş modelleri bul"""
    models = []
    for model_type, directory in CHECKPOINT_DIRS:
        if directory.is_dir():
            for pth_file in sorted(directory.glob("*.pth")):
                models.append({
                    "path": str(pth_file),
                    "type": model_type,
                    "name": pth_file.stem,
                })
    return models


def load_config() -> Dict[str, Any]:
    """Config dosyasını yükle - train.py'deki yöntemi kullan"""
    try:
        # train.py'deki config yükleme yöntemini kullan (modül düzeyindeki dict değişmesin)
        from training_system.train import TRAIN_CONFIG
        return copy.deepcopy(TRAIN_CONFIG)
    except Exception as e:
        logger.warning(f"TRAIN_CONFIG yüklenemedi, alternatif yöntem deneniyor: {e}")
        try:
            # Alternatif: JSON dosyasından yükle
            config_path = REPO_ROOT / "config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            # Varsayılan config (minimal)
            logger.warning("Config dosyası bulunamadı, varsayılan config kullanılıyor")
        except Exception as e2:
            logger.error(f"Config yüklenemedi: {e2}")
        return {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "tokenizer": {},
            "model": {},
        }


def run_generation_with_modelmanager(prompt: str, model_manager, tokenizer_core, device: Optional[str] = None):
    """
    Training_service yaklaşımı: ModelManager kullanarak generation yap

    Bu, training_service.py'deki _test_model_after_epoch() metodunun
    generation yaklaşımını simüle eder. Hatalar yutulmaz, çağırana iletilir.
    """
    logger.info("=" * 80)
    logger.info("GENERATION: ModelManager Yaklaşımı (TrainingService)")
    logger.info("=" * 80)

    if device is None:
        device = str(model_manager.device)

    # Tokenize
    _, token_ids = tokenizer_core.encode(
        prompt,
        mode="inference",
        include_whole_words=True,
        include_syllables=False,
        include_sep=False,
    )
    assert token_ids, f"Prompt tokenize edilemedi: '{prompt}'"

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
    assert model_manager.model is not None, "ModelManager'da model yok!"
    original_mode = model_manager.model.training
    model_manager.model.eval()

    # Generation loop (training_service.py'deki gibi)
    generation_details = []
    try:
        with torch.no_grad():
            generated_ids = list(token_ids)
            tokens_generated = 0
            for step in range(MAX_NEW_TOKENS):
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
                generation_details.append({
                    "step": step,
                    "input_length": len(generated_ids),
                    "predicted_id": predicted_id,
                    "logits_mean": next_logits.mean().item(),
                    "logits_std": next_logits.std().item(),
                    "logits_min": next_logits.min().item(),
                    "logits_max": next_logits.max().item(),
                    "top_5_tokens": top_5_indices.tolist(),
                    "top_5_probs": [f"{p:.4f}" for p in top_5_probs.tolist()],  # 4 ondalıklı metin
                    "top_5_probs_raw": top_5_probs.tolist(),
                    "probs_sum": probs.sum().item(),  # ~1.0 olmalı
                    "probs_has_nan": torch.isnan(probs).any().item(),  # False olmalı
                    "probs_has_inf": torch.isinf(probs).any().item(),  # False olmalı
                    "probs_max": probs.max().item(),  # > 0 olmalı
                    "probs_min": probs.min().item(),  # >= 0 olmalı
                    "is_eos": predicted_id == eos_id if eos_id is not None else False,
                })

                # EOS kontrolü (training_service.py'deki gibi)
                if eos_id is not None and predicted_id == eos_id:
                    if tokens_generated >= MIN_NEW_TOKENS:
                        logger.info(f"Step {step}: EOS token bulundu, generation durdu")
                        break
                    # EOS'u ignore et, devam et
                    probs[eos_id] = 0.0
                    probs = probs / (probs.sum() + 1e-10)
                    predicted_id = torch.multinomial(probs, 1).item()

                generated_ids.append(predicted_id)
                tokens_generated += 1
    finally:
        # Model mode'u geri al
        model_manager.model.train(original_mode)

    # Decode
    predicted_token_ids = generated_ids[len(token_ids):]
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

    return response, generation_details


def run_generation_with_cevahir(prompt: str, cevahir, max_new_tokens: int = MAX_NEW_TOKENS):
    """
    Cevahir.py yaklaşımı: Cevahir API kullanarak generation yap

    Bu, chat_pipeline.py'deki generation yaklaşımını simüle eder.
    Not: Orijinal script'te olduğu gibi şu an hiçbir test tarafından çağrılmıyor.
    """
    logger.info("=" * 80)
    logger.info("GENERATION: Cevahir API Yaklaşımı")
    logger.info("=" * 80)

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
    return response, None


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
            logger.info(f"  Top-5 probs (RAW): {detail['top_5_probs_raw']}")
            logger.info(f"  Probs sum: {detail['probs_sum']:.6f} (should be ~1.0)")
            logger.info(f"  Probs has NaN: {detail['probs_has_nan']}")
            logger.info(f"  Probs has Inf: {detail['probs_has_inf']}")
            logger.info(f"  Probs max: {detail['probs_max']:.6f}")
            logger.info(f"  Probs min: {detail['probs_min']:.6f}")
            logger.info(f"  Is EOS: {detail['is_eos']}")


@pytest.fixture(scope="module")
def selected_model() -> Dict[str, str]:
    """İlk eğitilmiş modeli seç; yoksa modüldeki testleri atla."""
    models = find_saved_models()
    if not models:
        pytest.skip(
            "Eğitilmiş model checkpoint'i yok: saved_models/*.pth veya tests/test_checkpoints/*.pth "
            "bulunamadı (checkpoint'ler repo'da tutulmaz; önce training_system/train.py ile model eğitin)"
        )
    for i, model in enumerate(models):
        logger.info(f"{i + 1}. {model['name']} ({model['type']})")
    logger.info(f"Kullanılacak model: {models[0]['name']}")
    return models[0]


@pytest.fixture(scope="module")
def train_config(selected_model) -> Dict[str, Any]:
    """train.py config'i (checkpoint yoksa hiç yüklenmez)."""
    return load_config()


@pytest.fixture(scope="module")
def tokenizer_core(train_config):
    """Config'teki vocab/merges ile TokenizerCore (train.py'deki gibi)."""
    from tokenizer_management.core.tokenizer_core import TokenizerCore

    # Config direkt olarak vocab_path/merges_path içerebilir veya tokenizer altında olabilir
    tokenizer_config = dict(train_config)
    if isinstance(train_config.get("tokenizer"), dict):
        tokenizer_config.update(train_config["tokenizer"])
    # Inference için veri yükleyici gerekmez; göreli data_dir CWD'ye bağımlıdır
    tokenizer_config.pop("data_dir", None)

    # Vocab ve merges path'leri yoksa repo'daki varsayılanları kullan (mutlak yol)
    if "vocab_path" not in tokenizer_config and "vocab_file" not in tokenizer_config:
        tokenizer_config["vocab_path"] = str(REPO_ROOT / "data" / "vocab_lib" / "vocab.json")
    if "merges_path" not in tokenizer_config and "merges_file" not in tokenizer_config:
        tokenizer_config["merges_path"] = str(REPO_ROOT / "data" / "merges_lib" / "merges.txt")

    return TokenizerCore(tokenizer_config)


@pytest.fixture(scope="module")
def model_manager(selected_model, train_config, tokenizer_core):
    """Eğitilmiş checkpoint'i yüklenmiş ModelManager (TrainingService'teki gibi)."""
    from model_management.model_manager import ModelManager

    # TokenizerCore'dan vocab_size al ve config'e ekle (TrainingService'teki gibi)
    model_config = dict(train_config)
    model_config["vocab_size"] = tokenizer_core.get_vocab_size()
    # TRAIN_CONFIG TensorBoard'u açar ve göreli runs/ yolunu kullanır; testler repo ağacına yazmamalı.
    model_config["use_tensorboard"] = False
    model_config = {k: v for k, v in model_config.items() if not k.startswith("tb_")}

    # Checkpoint sözleşmesi tokenizer kimliğini doğrular: tokenizer verilmezse yükleme
    # "Checkpoint requires a tokenizer to verify token identity" ile reddedilir.
    manager = ModelManager(model_config, tokenizer=tokenizer_core)
    # Yalnızca inference: optimizer/scheduler kurulmaz ve yüklenmez (Cevahir._init_model gibi)
    manager.initialize(build_optimizer=False, build_criterion=False, build_scheduler=False)

    # Eski script yükleme hatasında rastgele ağırlıklarla devam ediyordu; bu, gerçek modeli
    # test etme amacını boşa çıkarır. Checkpoint varsa yüklenebilmeli.
    try:
        manager.load(selected_model["path"], strict=False, weights_only=True)
    except Exception as exc:
        pytest.fail(
            f"Eğitilmiş checkpoint yüklenemedi: {selected_model['path']}: {exc} "
            f"(neden: {exc.__cause__!r})"
        )
    return manager


@pytest.mark.parametrize("prompt", TEST_PROMPTS)
def test_generation_with_modelmanager(prompt, model_manager, tokenizer_core):
    """Training_service yaklaşımı (ModelManager) ile gerçek eğitilmiş modelden generation."""
    response, details = run_generation_with_modelmanager(prompt, model_manager, tokenizer_core)

    # Cevahir API yaklaşımı orijinal script'te de devre dışıydı (ModelManager testi yeterli)
    compare_generations(prompt, response, None, details)

    assert isinstance(response, str), f"Generation metin döndürmedi: {response!r}"
    assert details, "Generation hiçbir adım üretmedi"

    vocab_size = tokenizer_core.get_vocab_size()
    for detail in details:
        step_info = (
            f"prompt='{prompt}', step={detail['step']}, predicted_id={detail['predicted_id']}, "
            f"top_5={detail['top_5_tokens']} {detail['top_5_probs']}, response='{response}'"
        )
        assert not detail["probs_has_nan"], f"Olasılıklar NaN içeriyor ({step_info})"
        assert not detail["probs_has_inf"], f"Olasılıklar Inf içeriyor ({step_info})"
        assert abs(detail["probs_sum"] - 1.0) < 0.01, \
            f"Olasılık toplamı ~1.0 olmalı: {detail['probs_sum']:.6f} ({step_info})"
        assert detail["probs_min"] >= 0.0, f"Negatif olasılık: {detail['probs_min']} ({step_info})"
        assert detail["probs_max"] > 0.0, f"Tüm olasılıklar sıfır ({step_info})"
        assert 0 <= detail["predicted_id"] < vocab_size, f"Geçersiz token ID ({step_info})"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
