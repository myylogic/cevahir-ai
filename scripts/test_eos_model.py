# -*- coding: utf-8 -*-
"""
Mevcut modelin EOS (End of Sequence) token'ı öğrenip öğrenmediğini test eder.
Cache kullanmaz; sadece yüklü checkpoint ile birkaç kısa prompt üzerinde
son pozisyonda EOS olasılığını ve argmax tahminini raporlar.

Kullanım (proje kökünden):
  python scripts/test_eos_model.py
  python scripts/test_eos_model.py --path saved_models/cevahir_model.pth
"""
import os
import sys
import argparse
import logging

# Proje kökü
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _get_special_id(vocab: dict, name: str) -> int:
    v = vocab.get(name)
    if isinstance(v, dict):
        return int(v.get("id", -1))
    if isinstance(v, int):
        return v
    return -1


def main():
    parser = argparse.ArgumentParser(description="Mevcut modelde EOS öğrenimini test et")
    parser.add_argument(
        "--path",
        default=None,
        help="Checkpoint yolu (yoksa saved_models/cevahir_model.pth veya son checkpoint)"
    )
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    load_path = args.path
    if not load_path:
        checkpoint_dir = os.path.join(BASE, "saved_models", "checkpoints")
        if os.path.isdir(checkpoint_dir):
            import glob
            checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
            if checkpoints:
                load_path = max(checkpoints, key=lambda p: int(p.split("_")[-1].replace(".pth", "")))
        if not load_path or not os.path.exists(load_path or ""):
            load_path = os.path.join(BASE, "saved_models", "cevahir_model.pth")

    if not os.path.exists(load_path):
        logger.error("Model dosyası bulunamadı: %s", load_path)
        sys.exit(1)

    logger.info("Model: %s", load_path)
    logger.info("Device: %s", device)

    # Tokenizer: BOS/EOS id ve encode
    from tokenizer_management.core.tokenizer_core import TokenizerCore
    vocab_path = os.path.join(BASE, "data", "vocab_lib", "vocab.json")
    merges_path = os.path.join(BASE, "data", "merges_lib", "merges.txt")
    if not os.path.exists(vocab_path):
        logger.error("Vocab bulunamadı: %s", vocab_path)
        sys.exit(1)

    tokenizer = TokenizerCore({"vocab_path": vocab_path, "merges_path": merges_path})
    tokenizer.finalize_vocab()
    vocab = tokenizer.get_vocab()
    BOS_ID = _get_special_id(vocab, "<BOS>")
    EOS_ID = _get_special_id(vocab, "<EOS>")
    if EOS_ID < 0:
        logger.error("Vocab'da <EOS> id bulunamadı")
        sys.exit(1)
    logger.info("BOS_ID=%s, EOS_ID=%s", BOS_ID, EOS_ID)

    # Model: ChatPipeline ile aynı config (train.py ile uyumlu)
    from model.cevahir import Cevahir, CevahirConfig

    model_cfg = {
        "vocab_size": len(vocab),
        "embed_dim": 256,
        "seq_proj_dim": 256,
        "num_heads": 4,
        "num_layers": 6,
        "ffn_dim": None,
        "pre_norm": True,
        "causal_mask": True,
        "tie_weights": True,
        "use_rmsnorm": True,
        "use_swiglu": True,
        "use_kv_cache": True,
        "max_cache_len": 2048,
        "pe_mode": "rope",
    }
    cevahir_config = CevahirConfig(
        device=device,
        tokenizer={"vocab_path": vocab_path, "merges_path": merges_path},
        load_model_path=load_path,
        model=model_cfg,
    )
    cevahir = Cevahir(cevahir_config)

    # Test promptları (cümle sonunda model EOS üretmeli)
    test_prompts = ["Merhaba", "Selam", "Nasılsın", "İyiyim", "Teşekkürler"]

    def encode_text(text: str):
        _, ids = tokenizer.encode(text, mode="inference", add_special_tokens=False)
        ids = ids if isinstance(ids, list) else (ids.tolist() if hasattr(ids, "tolist") else list(ids))
        # Eğitim formatına benzer: Input'ta BOS başta
        return [BOS_ID] + ids

    results = []
    with torch.no_grad():
        for prompt in test_prompts:
            input_ids = encode_text(prompt)
            if not input_ids:
                results.append((prompt, 0.0, False, -1, "encode boş"))
                continue
            try:
                # Cevahir unified API üzerinden forward; inference=True → eval modu + no_grad
                logits = cevahir.forward(input_ids, inference=True, return_aux=False)
            except Exception as e:
                results.append((prompt, 0.0, False, -1, str(e)))
                continue
            # Son pozisyon: bir sonraki token (EOS beklenir)
            last_logits = logits[0, -1, :].float()
            probs = F.softmax(last_logits, dim=-1)
            eos_prob = probs[EOS_ID].item()
            pred_id = last_logits.argmax().item()
            results.append((prompt, eos_prob, pred_id == EOS_ID, pred_id, None))

    # Rapor
    logger.info("")
    logger.info("=" * 60)
    logger.info("EOS TEST SONUÇLARI (mevcut model)")
    logger.info("=" * 60)
    logger.info("%-14s | EOS olasılığı | Tahmin EOS? | Tahmin id", "Prompt")
    logger.info("-" * 60)
    for prompt, eos_prob, is_eos, pred_id, err in results:
        if err:
            logger.info("%-14s | HATA: %s", prompt, err)
        else:
            logger.info("%-14s | %12.6f | %10s | %s", prompt, eos_prob, "Evet" if is_eos else "Hayır", pred_id)
    logger.info("-" * 60)
    n_ok = sum(1 for r in results if r[4] is None and r[2])
    n_total = sum(1 for r in results if r[4] is None)
    avg_eos = sum(r[1] for r in results if r[4] is None) / n_total if n_total else 0
    logger.info("Özet: EOS doğru tahmin %d/%d, ortalama EOS olasılığı %.6f", n_ok, n_total, avg_eos)
    logger.info("=" * 60)
    if n_total and avg_eos < 1e-4 and n_ok == 0:
        logger.info("Model EOS'u pratikte öğrenmemiş (EOS prob ~0). Eğitim verisi/hyperparametre kontrolü önerilir.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
