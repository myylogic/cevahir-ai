"""
Checkpoint'teki weights'lerin detaylı analizi
"""
import torch
import sys
from pathlib import Path
from collections import defaultdict

# Root directory'yi path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

def analyze_checkpoint_weights(checkpoint_path: str):
    """Checkpoint'teki weights'lerin detaylı analizi"""
    print("=" * 80)
    print("CHECKPOINT WEIGHTS DETAYLI ANALİZ")
    print("=" * 80)
    
    # Checkpoint yükle
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    
    if state_dict is None:
        print("❌ Checkpoint'te model_state_dict bulunamadı!")
        return
    
    print(f"\n✅ Checkpoint'te toplam {len(state_dict)} parametre grubu bulundu\n")
    
    # Kategorilere göre grupla
    categories = defaultdict(list)
    
    for key, param in state_dict.items():
        mean = param.mean().item()
        std = param.std().item()
        min_val = param.min().item()
        max_val = param.max().item()
        numel = param.numel()
        
        # Kategori belirle
        if "embedding" in key.lower() or "embed" in key.lower():
            category = "EMBEDDING"
        elif "norm" in key.lower() or "scale" in key.lower():
            category = "NORMALIZATION"
        elif "attn" in key.lower() or "attention" in key.lower():
            category = "ATTENTION"
        elif "ffn" in key.lower() or "feedforward" in key.lower():
            category = "FEEDFORWARD"
        elif "proj" in key.lower() or "projection" in key.lower():
            category = "PROJECTION"
        elif "output" in key.lower() or "out" in key.lower():
            category = "OUTPUT"
        else:
            category = "OTHER"
        
        categories[category].append({
            "key": key,
            "mean": mean,
            "std": std,
            "min": min_val,
            "max": max_val,
            "numel": numel,
            "shape": list(param.shape)
        })
    
    # Her kategori için özet
    print("=" * 80)
    print("KATEGORİ BAZINDA ÖZET")
    print("=" * 80)
    
    for category in sorted(categories.keys()):
        params = categories[category]
        print(f"\n📁 {category} ({len(params)} parametre grubu):")
        print("-" * 80)
        
        # İstatistikler
        means = [p["mean"] for p in params]
        stds = [p["std"] for p in params]
        mins = [p["min"] for p in params]
        maxs = [p["max"] for p in params]
        
        print(f"  Mean: {sum(means)/len(means):.6f} (min={min(means):.6f}, max={max(means):.6f})")
        print(f"  Std:  {sum(stds)/len(stds):.6f} (min={min(stds):.6f}, max={max(stds):.6f})")
        print(f"  Range: [{min(mins):.6f}, {max(maxs):.6f}]")
        
        # İlk 3 örnek
        print(f"\n  İlk 3 örnek:")
        for i, p in enumerate(params[:3]):
            print(f"    {i+1}. {p['key']}")
            print(f"       Shape: {p['shape']}, Mean: {p['mean']:.6f}, Std: {p['std']:.6f}")
    
    # Random weights analizi
    print("\n" + "=" * 80)
    print("RANDOM WEIGHTS ANALİZİ")
    print("=" * 80)
    
    all_means = [p["mean"] for params in categories.values() for p in params]
    all_stds = [p["std"] for params in categories.values() for p in params]
    
    overall_mean = sum(all_means) / len(all_means)
    overall_std = sum(all_stds) / len(all_stds)
    
    # Random weights kriteri
    is_random_like = abs(overall_mean) < 0.001 and overall_std < 0.15
    
    print(f"\nGenel İstatistikler:")
    print(f"  Mean (tüm parametreler): {overall_mean:.6f}")
    print(f"  Std (tüm parametreler): {overall_std:.6f}")
    print(f"\nRandom Weights Kriteri:")
    print(f"  |mean| < 0.001: {abs(overall_mean) < 0.001} ({abs(overall_mean):.6f})")
    print(f"  std < 0.15: {overall_std < 0.15} ({overall_std:.6f})")
    print(f"\n{'❌ RANDOM WEIGHTS GİBİ GÖRÜNÜYOR!' if is_random_like else '✅ EĞİTİLMİŞ WEIGHTS GİBİ GÖRÜNÜYOR'}")
    
    # Eğitilmiş model göstergeleri
    print("\n" + "=" * 80)
    print("EĞİTİLMİŞ MODEL GÖSTERGELERİ")
    print("=" * 80)
    
    # Eğitilmiş bir model'de beklenen aralıklar
    expected_mean_range = (-0.1, 0.1)  # Genellikle 0'a yakın ama 0 değil
    expected_std_range = (0.05, 2.0)   # Genellikle 0.1-1.0 arası
    
    mean_in_range = expected_mean_range[0] <= overall_mean <= expected_mean_range[1]
    std_in_range = expected_std_range[0] <= overall_std <= expected_std_range[1]
    
    print(f"\nBeklenen Aralıklar (Eğitilmiş Model):")
    print(f"  Mean: [{expected_mean_range[0]:.3f}, {expected_mean_range[1]:.3f}]")
    print(f"  Std:  [{expected_std_range[0]:.3f}, {expected_std_range[1]:.3f}]")
    print(f"\nMevcut Değerler:")
    print(f"  Mean: {overall_mean:.6f} {'✅' if mean_in_range else '❌'}")
    print(f"  Std:  {overall_std:.6f} {'✅' if std_in_range else '❌'}")
    
    if mean_in_range and std_in_range:
        print("\n✅ Checkpoint'teki weights EĞİTİLMİŞ model'e benziyor!")
    else:
        print("\n❌ Checkpoint'teki weights RANDOM model'e benziyor!")
        print("   → Model eğitilmemiş olabilir veya checkpoint yanlış kaydedilmiş olabilir")


def compare_with_model(checkpoint_path: str):
    """Checkpoint'teki weights ile model'in weights'lerini karşılaştır"""
    print("\n" + "=" * 80)
    print("CHECKPOINT vs MODEL KARŞILAŞTIRMA")
    print("=" * 80)
    
    try:
        from model_management.model_manager import ModelManager
        from tokenizer_management.core.tokenizer_core import TokenizerCore
        from training_system.train import TRAIN_CONFIG
        
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
        
        # ModelManager oluştur (yüklemeden önce)
        model_manager = ModelManager(config)
        model_manager.config["vocab_size"] = vocab_size
        model_manager.initialize()
        
        # Model'in ilk parametresi (yüklemeden önce)
        model_param_before = next(iter(model_manager.model.parameters()))
        model_mean_before = model_param_before.mean().item()
        model_std_before = model_param_before.std().item()
        
        print(f"\nModel (Yüklemeden Önce):")
        print(f"  İlk parametre mean: {model_mean_before:.6f}")
        print(f"  İlk parametre std:  {model_std_before:.6f}")
        
        # Checkpoint yükle
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
        
        if state_dict is None:
            print("❌ Checkpoint'te model_state_dict bulunamadı!")
            return
        
        # Checkpoint'teki ilk parametre
        checkpoint_first_key = next(iter(state_dict.keys()))
        checkpoint_first_param = state_dict[checkpoint_first_key]
        checkpoint_mean = checkpoint_first_param.mean().item()
        checkpoint_std = checkpoint_first_param.std().item()
        
        print(f"\nCheckpoint:")
        print(f"  İlk parametre key: {checkpoint_first_key}")
        print(f"  İlk parametre mean: {checkpoint_mean:.6f}")
        print(f"  İlk parametre std:  {checkpoint_std:.6f}")
        
        # Model'e yükle
        model_manager.load(checkpoint_path, strict=False)
        
        # Model'in ilk parametresi (yüklemeden sonra)
        model_param_after = next(iter(model_manager.model.parameters()))
        model_mean_after = model_param_after.mean().item()
        model_std_after = model_param_after.std().item()
        
        print(f"\nModel (Yüklemeden Sonra):")
        print(f"  İlk parametre mean: {model_mean_after:.6f}")
        print(f"  İlk parametre std:  {model_std_after:.6f}")
        
        # Karşılaştırma
        print(f"\nKarşılaştırma:")
        print(f"  Model before vs after mean farkı: {abs(model_mean_before - model_mean_after):.6f}")
        print(f"  Model before vs after std farkı:  {abs(model_std_before - model_std_after):.6f}")
        print(f"  Checkpoint vs Model after mean farkı: {abs(checkpoint_mean - model_mean_after):.6f}")
        print(f"  Checkpoint vs Model after std farkı:  {abs(checkpoint_std - model_std_after):.6f}")
        
        # Keys karşılaştırması
        model_keys = set(model_manager.model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())
        
        missing = model_keys - checkpoint_keys
        unexpected = checkpoint_keys - model_keys
        common = model_keys & checkpoint_keys
        
        print(f"\nKeys Karşılaştırması:")
        print(f"  Model keys: {len(model_keys)}")
        print(f"  Checkpoint keys: {len(checkpoint_keys)}")
        print(f"  Ortak keys: {len(common)}")
        print(f"  Model'de var, checkpoint'te yok: {len(missing)}")
        print(f"  Checkpoint'te var, model'de yok: {len(unexpected)}")
        
        if missing:
            print(f"\n  ⚠️ Missing keys (ilk 10):")
            for key in list(missing)[:10]:
                print(f"    - {key}")
        
        if unexpected:
            print(f"\n  ⚠️ Unexpected keys (ilk 10):")
            for key in list(unexpected)[:10]:
                print(f"    - {key}")
        
        # Weights yüklendi mi?
        weights_loaded = abs(model_mean_before - model_mean_after) > 1e-6
        print(f"\nWeights Yüklendi mi?")
        print(f"  Mean değişti: {weights_loaded} (fark: {abs(model_mean_before - model_mean_after):.6f})")
        
        if weights_loaded:
            print("  ✅ Weights yüklendi!")
        else:
            print("  ❌ Weights YÜKLENMEDİ! (Değişiklik yok)")
    
    except Exception as e:
        print(f"❌ Hata: {e}", exc_info=True)


def main():
    """Ana fonksiyon"""
    import sys
    
    # Checkpoint path'i belirle
    checkpoint_path = None
    
    # Komut satırı argümanı var mı?
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        # Varsayılan: saved_models/checkpoints/best.pth
        checkpoint_path = "saved_models/checkpoints/best.pth"
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint bulunamadı: {checkpoint_path}")
        return
    
    print(f"Checkpoint: {checkpoint_path}\n")
    
    # 1. Checkpoint weights analizi
    analyze_checkpoint_weights(checkpoint_path)
    
    # 2. Model ile karşılaştırma
    compare_with_model(checkpoint_path)


if __name__ == "__main__":
    main()

