"""
Training Loop Gradient Flow Analizi

Bu test, training loop sırasında gradient flow'un düzgün olup olmadığını kontrol eder.
Eğitim config'inin (training_system/train.py TRAIN_CONFIG) tam boyutlu modeliyle birkaç
sentetik batch eğitilir; her parametrenin gradient almış olması ve hiçbir parametrede
vanishing (ortalama norm < 1e-6) ya da exploding (ortalama norm > 1000) gradient
olmaması beklenir.
"""
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List
from collections import defaultdict

import pytest
import torch
import torch.nn as nn

# Root directory'yi path'e ekle (dosya doğrudan çalıştırıldığında da import'lar çalışsın)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


def register_gradient_hooks(model: nn.Module) -> Dict[str, List[float]]:
    """Model'e gradient hook'ları kaydet ve gradient norm'larını topla"""
    gradient_norms = defaultdict(list)
    
    def make_hook(name: str):
        def hook(grad):
            if grad is not None:
                grad_norm = grad.norm().item()
                gradient_norms[name].append(grad_norm)
                if grad_norm > 1000:
                    print(f"⚠️ Gradient exploding: {name} = {grad_norm:.6f}")
                elif grad_norm < 1e-6:
                    print(f"⚠️ Gradient vanishing: {name} = {grad_norm:.6f}")
            return grad
        return hook
    
    # Her parametre için hook kaydet
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(make_hook(name))
    
    return gradient_norms


def analyze_gradient_flow(model: nn.Module, gradient_norms: Dict[str, List[float]]) -> Dict[str, any]:
    """Gradient flow analizi yap"""
    results = {
        "total_parameters": 0,
        "parameters_with_gradients": 0,
        "parameters_without_gradients": 0,
        "vanishing_gradients": [],  # norm < 1e-6
        "exploding_gradients": [],  # norm > 1000
        "healthy_gradients": [],
        "layer_stats": {}
    }
    
    for name, norms in gradient_norms.items():
        if not norms:
            results["parameters_without_gradients"] += 1
            continue
        
        results["parameters_with_gradients"] += 1
        avg_norm = sum(norms) / len(norms)
        max_norm = max(norms)
        min_norm = min(norms)
        
        # Kategori belirle
        if avg_norm < 1e-6:
            results["vanishing_gradients"].append({
                "name": name,
                "avg_norm": avg_norm,
                "max_norm": max_norm,
                "min_norm": min_norm
            })
        elif avg_norm > 1000:
            results["exploding_gradients"].append({
                "name": name,
                "avg_norm": avg_norm,
                "max_norm": max_norm,
                "min_norm": min_norm
            })
        else:
            results["healthy_gradients"].append({
                "name": name,
                "avg_norm": avg_norm,
                "max_norm": max_norm,
                "min_norm": min_norm
            })
        
        # Layer bazında istatistikler
        layer_name = name.split('.')[0]  # İlk kısım (örn: "dil_katmani", "layers")
        if layer_name not in results["layer_stats"]:
            results["layer_stats"][layer_name] = {
                "count": 0,
                "avg_norms": []
            }
        results["layer_stats"][layer_name]["count"] += 1
        results["layer_stats"][layer_name]["avg_norms"].append(avg_norm)
    
    # Toplam parametre sayısı ve gradient almayan parametreler. Hook yalnız gradient alan
    # parametrelerde tetiklendiğinden yukarıdaki boş-liste sayımı hep 0 kalıyordu; modelden bul.
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    results["total_parameters"] = len(trainable)
    without = [name for name in trainable if not gradient_norms.get(name)]
    results["parameters_without_gradients"] = len(without)
    results["parameters_without_gradients_names"] = without
    
    # Layer bazında ortalama norm'ları hesapla
    for layer_name in results["layer_stats"]:
        avg_norms = results["layer_stats"][layer_name]["avg_norms"]
        results["layer_stats"][layer_name]["mean_norm"] = sum(avg_norms) / len(avg_norms) if avg_norms else 0
        results["layer_stats"][layer_name]["max_norm"] = max(avg_norms) if avg_norms else 0
        results["layer_stats"][layer_name]["min_norm"] = min(avg_norms) if avg_norms else 0
    
    return results


def _format_items(items: List[Dict[str, Any]], limit: int = 10) -> str:
    shown = ", ".join(f"{i['name']} (avg={i['avg_norm']:.3e}, max={i['max_norm']:.3e})" for i in items[:limit])
    more = f" ... ve {len(items) - limit} tane daha" if len(items) > limit else ""
    return shown + more


@pytest.fixture(autouse=True)
def _seeded_rng():
    """Sentetik veri ve dropout deterministik olsun; global RNG durumu test sonrası geri yüklenir."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        yield


def test_gradient_flow_training_loop():
    """Training loop'ta gradient flow testi"""
    from model_management.model_manager import ModelManager
    from tokenizer_management.config import BPE_CONFIG
    from tokenizer_management.core.tokenizer_core import TokenizerCore
    from training_system.train import TRAIN_CONFIG

    config = TRAIN_CONFIG.copy()

    # TokenizerCore oluştur. Göreli "data/vocab_lib/..." yolları çalışma dizinine göre
    # çözülüyordu ve dosya yoksa BPEManager boş vocab/merges dosyaları OLUŞTURUR;
    # repo'daki dosyaların mutlak yollarını kullan ve tokenizer'ı salt-okunur aç.
    tokenizer_config = config.copy()
    tokenizer_config.setdefault("vocab_path", BPE_CONFIG["vocab_file"])
    tokenizer_config.setdefault("merges_path", BPE_CONFIG["merges_file"])
    tokenizer_config["read_only"] = True
    for key in ("vocab_path", "merges_path"):
        # Bu dosyalar depoda izleniyor; yoksa checkout bozuktur ve test sessizce atlanmamalı.
        assert Path(tokenizer_config[key]).is_file(), f"BPE dosyası bulunamadı: {tokenizer_config[key]}"
    # data_dir göreli ("education"); test başka bir çalışma dizininden de çalışabilsin.
    tokenizer_config["data_dir"] = str(ROOT / config["data_dir"])

    tokenizer_core = TokenizerCore(tokenizer_config)
    vocab_size = tokenizer_core.get_vocab_size()
    config["vocab_size"] = vocab_size

    # Sentetik veride gerçek özel token ID'leri kullanılsın (eski kod BOS=0 ve EOS=1
    # yazıyordu; tokenizer'da 0=<PAD>, 1=<UNK>).
    special_ids = tokenizer_core._special_ids()
    bos_id, eos_id = special_ids["<BOS>"], special_ids["<EOS>"]

    # ModelManager oluştur
    model_manager = ModelManager(config)
    model_manager.config["vocab_size"] = vocab_size
    model_manager.initialize()

    model = model_manager.model
    optimizer = model_manager.optimizer
    criterion = model_manager.criterion
    logger.info("Model hazır: %s, toplam parametre: %s",
                type(model).__name__, f"{sum(p.numel() for p in model.parameters()):,}")

    # Gradient hook'ları kaydet
    gradient_norms = register_gradient_hooks(model)

    # Basit bir training loop oluştur
    batch_size = 4
    seq_len = 32
    num_batches = 3  # İlk 3 batch'i test et

    for batch_idx in range(num_batches):
        # Synthetic input oluştur
        # Input: [BOS, t1, t2, ..., tN]
        # Target: [t1, t2, ..., tN, EOS]
        input_ids = torch.randint(1, vocab_size - 1, (batch_size, seq_len + 1), device=model_manager.device)
        input_ids[:, 0] = bos_id
        target_ids = torch.cat(
            [input_ids[:, 1:], torch.full((batch_size, 1), eos_id, dtype=torch.long, device=model_manager.device)],
            dim=1,
        )

        # Forward pass
        optimizer.zero_grad()
        output = model(input_ids)

        # Output tuple ise (output, attn_weights, ...) sadece output'u al
        logits = output[0] if isinstance(output, tuple) else output

        # Logits shape: [batch_size, seq_len, vocab_size] → düzleştir
        if logits.dim() == 3:
            logits = logits.reshape(-1, logits.size(-1))
        targets = target_ids.reshape(-1)

        # Loss hesapla
        loss = criterion(logits, targets)
        assert math.isfinite(loss.item()), f"Batch {batch_idx + 1}: loss sonlu değil: {loss.item()}"

        # Backward pass
        loss.backward()

        # Gradient norm kontrolü
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        total_grad_norm = math.sqrt(sum(g.norm().item() ** 2 for g in grads))
        assert math.isfinite(total_grad_norm), (
            f"Batch {batch_idx + 1}: toplam gradient norm sonlu değil: {total_grad_norm}"
        )
        logger.info("Batch %d/%d: loss=%.6f, toplam gradient norm=%.6f, gradient alan parametre=%d",
                    batch_idx + 1, num_batches, loss.item(), total_grad_norm, len(grads))

        # Optimizer step
        optimizer.step()

    # Gradient flow analizi
    results = analyze_gradient_flow(model, gradient_norms)
    for layer_name, stats in sorted(results["layer_stats"].items()):
        logger.info("%s: parametre=%d, ortalama norm=%.6e, min=%.6e, max=%.6e",
                    layer_name, stats["count"], stats["mean_norm"], stats["min_norm"], stats["max_norm"])

    # Sonuç değerlendirmesi
    # Hiç gradient almayan eğitilebilir parametre, vanishing gradient'in en uç hâlidir
    # (frozen katmanlar requires_grad=False olduğundan bu sayıma girmez).
    assert results["parameters_without_gradients"] == 0, (
        f"{results['parameters_without_gradients']}/{results['total_parameters']} eğitilebilir parametre "
        f"gradient almadı: {results['parameters_without_gradients_names'][:10]}"
    )
    assert not results["vanishing_gradients"], (
        "VANISHING GRADIENT SORUNU: gradient'ler çok küçük, model öğrenemiyor olabilir "
        "(residual connection'ları / normalization'ı kontrol et): "
        + _format_items(results["vanishing_gradients"])
    )
    assert not results["exploding_gradients"], (
        "EXPLODING GRADIENT SORUNU: gradient'ler çok büyük, training unstable "
        "(gradient clipping / learning rate kontrol et): "
        + _format_items(results["exploding_gradients"])
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
