"""
Training Loop Gradient Flow Analizi

Bu test, training loop sırasında gradient flow'un düzgün olup olmadığını kontrol eder.
"""
import torch
import torch.nn as nn
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# Root directory'yi path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    
    # Toplam parametre sayısı
    results["total_parameters"] = sum(1 for p in model.parameters() if p.requires_grad)
    
    # Layer bazında ortalama norm'ları hesapla
    for layer_name in results["layer_stats"]:
        avg_norms = results["layer_stats"][layer_name]["avg_norms"]
        results["layer_stats"][layer_name]["mean_norm"] = sum(avg_norms) / len(avg_norms) if avg_norms else 0
        results["layer_stats"][layer_name]["max_norm"] = max(avg_norms) if avg_norms else 0
        results["layer_stats"][layer_name]["min_norm"] = min(avg_norms) if avg_norms else 0
    
    return results


def test_gradient_flow_training_loop():
    """Training loop'ta gradient flow testi"""
    print("=" * 80)
    print("TRAINING LOOP GRADIENT FLOW ANALİZİ")
    print("=" * 80)
    
    try:
        from model_management.model_manager import ModelManager
        from tokenizer_management.core.tokenizer_core import TokenizerCore
        from training_system.train import TRAIN_CONFIG
        from torch.utils.data import DataLoader, TensorDataset
        import torch.nn.functional as F
        
        config = TRAIN_CONFIG.copy()
        
        # TokenizerCore oluştur
        tokenizer_config = config.copy()
        if "vocab_path" not in tokenizer_config:
            tokenizer_config["vocab_path"] = "data/vocab_lib/vocab.json"
        if "merges_path" not in tokenizer_config:
            tokenizer_config["merges_path"] = "data/merges_lib/merges.txt"
        
        tokenizer_core = TokenizerCore(tokenizer_config)
        vocab_size = tokenizer_core.get_vocab_size()
        config["vocab_size"] = vocab_size
        
        # ModelManager oluştur
        model_manager = ModelManager(config)
        model_manager.config["vocab_size"] = vocab_size
        model_manager.initialize()
        
        model = model_manager.model
        optimizer = model_manager.optimizer
        criterion = model_manager.criterion
        
        print(f"\n✅ Model hazır: {type(model).__name__}")
        print(f"   Toplam parametre sayısı: {sum(p.numel() for p in model.parameters()):,}")
        
        # Gradient hook'ları kaydet
        gradient_norms = register_gradient_hooks(model)
        print(f"\n✅ Gradient hook'ları kaydedildi")
        
        # Basit bir training loop oluştur
        batch_size = 4
        seq_len = 32
        
        # Synthetic data oluştur (BOS, tokens, EOS formatında)
        num_batches = 3  # İlk 3 batch'i test et
        
        print(f"\n{'=' * 80}")
        print("TRAINING LOOP BAŞLIYOR")
        print(f"{'=' * 80}\n")
        
        for batch_idx in range(num_batches):
            # Synthetic input oluştur
            # Input: [BOS, t1, t2, ..., tN]
            # Target: [t1, t2, ..., tN, EOS]
            input_ids = torch.randint(1, vocab_size - 1, (batch_size, seq_len + 1), device=model_manager.device)
            input_ids[:, 0] = 0  # BOS token
            target_ids = input_ids[:, 1:].clone()  # Shift target
            target_ids = torch.cat([target_ids, torch.zeros(batch_size, 1, dtype=torch.long, device=model_manager.device)], dim=1)  # EOS ekle
            target_ids[:, -1] = 1  # EOS token (vocab_size'e göre ayarla gerekirse)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(input_ids)
            
            # Output tuple ise (output, attn_weights, ...) sadece output'u al
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            
            # Logits shape: [batch_size, seq_len, vocab_size]
            # Target shape: [batch_size, seq_len]
            if logits.dim() == 3:
                logits = logits.view(-1, logits.size(-1))
                targets = target_ids.view(-1)
            else:
                targets = target_ids.view(-1)
            
            # Loss hesapla
            loss = criterion(logits, targets)
            
            print(f"Batch {batch_idx + 1}/{num_batches}:")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  Logits mean: {logits.mean().item():.6f}, std: {logits.std().item():.6f}")
            
            # Backward pass
            loss.backward()
            
            # Gradient norm kontrolü
            total_grad_norm = 0
            param_count = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_grad_norm = param.grad.norm().item()
                    total_grad_norm += param_grad_norm ** 2
                    param_count += 1
            
            total_grad_norm = total_grad_norm ** 0.5
            print(f"  Total gradient norm: {total_grad_norm:.6f}")
            print(f"  Parameters with gradients: {param_count}/{sum(1 for p in model.parameters() if p.requires_grad)}")
            
            # Optimizer step
            optimizer.step()
            print()
        
        # Gradient flow analizi
        print(f"\n{'=' * 80}")
        print("GRADIENT FLOW ANALİZİ")
        print(f"{'=' * 80}\n")
        
        results = analyze_gradient_flow(model, gradient_norms)
        
        print(f"Toplam Parametre Sayısı: {results['total_parameters']}")
        print(f"Gradient Olan Parametreler: {results['parameters_with_gradients']}")
        print(f"Gradient Olmayan Parametreler: {results['parameters_without_gradients']}")
        print()
        
        print(f"✅ Sağlıklı Gradient'ler: {len(results['healthy_gradients'])}")
        print(f"❌ Vanishing Gradient'ler: {len(results['vanishing_gradients'])}")
        print(f"❌ Exploding Gradient'ler: {len(results['exploding_gradients'])}")
        print()
        
        # Vanishing gradient'leri göster
        if results['vanishing_gradients']:
            print("⚠️ VANISHING GRADIENT'LER:")
            for item in results['vanishing_gradients'][:10]:  # İlk 10'unu göster
                print(f"  - {item['name']}: avg={item['avg_norm']:.6e}, max={item['max_norm']:.6e}")
            if len(results['vanishing_gradients']) > 10:
                print(f"  ... ve {len(results['vanishing_gradients']) - 10} tane daha")
            print()
        
        # Exploding gradient'leri göster
        if results['exploding_gradients']:
            print("⚠️ EXPLODING GRADIENT'LER:")
            for item in results['exploding_gradients'][:10]:  # İlk 10'unu göster
                print(f"  - {item['name']}: avg={item['avg_norm']:.6e}, max={item['max_norm']:.6e}")
            if len(results['exploding_gradients']) > 10:
                print(f"  ... ve {len(results['exploding_gradients']) - 10} tane daha")
            print()
        
        # Layer bazında istatistikler
        print("LAYER BAZINDA GRADIENT NORM İSTATİSTİKLERİ:")
        print("-" * 80)
        for layer_name, stats in sorted(results['layer_stats'].items()):
            print(f"{layer_name}:")
            print(f"  Parametre sayısı: {stats['count']}")
            print(f"  Ortalama norm: {stats['mean_norm']:.6e}")
            print(f"  Min norm: {stats['min_norm']:.6e}")
            print(f"  Max norm: {stats['max_norm']:.6e}")
            print()
        
        # Sonuç değerlendirmesi
        print(f"{'=' * 80}")
        print("SONUÇ DEĞERLENDİRMESİ")
        print(f"{'=' * 80}\n")
        
        has_issues = False
        
        if results['vanishing_gradients']:
            print("❌ VANISHING GRADIENT SORUNU TESPİT EDİLDİ!")
            print("   → Gradient'ler çok küçük, model öğrenemiyor olabilir")
            print("   → Çözüm: Residual connection'ları kontrol et, normalization'ı düzelt")
            has_issues = True
        
        if results['exploding_gradients']:
            print("❌ EXPLODING GRADIENT SORUNU TESPİT EDİLDİ!")
            print("   → Gradient'ler çok büyük, training unstable")
            print("   → Çözüm: Gradient clipping ekle, learning rate düşür")
            has_issues = True
        
        if results['parameters_without_gradients'] > 0:
            print(f"⚠️ {results['parameters_without_gradients']} parametre gradient almamış")
            print("   → Bu normal olabilir (örn: frozen layers)")
        
        if not has_issues:
            print("✅ Gradient flow sağlıklı görünüyor!")
            print("   → Gradient norm'ları makul aralıkta")
            print("   → Sorun başka bir yerde olabilir (weight update, loss calculation, vb.)")
        
        return results
        
    except Exception as e:
        print(f"❌ Hata: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    results = test_gradient_flow_training_loop()
    
    if results:
        print("\n✅ Test tamamlandı!")
    else:
        print("\n❌ Test başarısız!")

