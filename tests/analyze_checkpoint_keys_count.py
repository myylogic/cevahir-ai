#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checkpoint Keys Count Analizi
Colab'dan indirilen checkpoint'teki model_state_dict'in kaç key içerdiğini kontrol eder.
"""

import os
import sys
import torch

# Root dizini ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def analyze_checkpoint(checkpoint_path: str):
    """Checkpoint'teki model_state_dict'in key sayısını analiz eder."""
    print(f"\n{'='*60}")
    print(f"CHECKPOINT KEYS ANALİZİ")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}\n")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint bulunamadı: {checkpoint_path}")
        return
    
    # Checkpoint yükle
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Format bilgisi
    print("Checkpoint Format:")
    print(f"  Top-level keys: {list(checkpoint.keys())}")
    print(f"  Top-level key sayısı: {len(checkpoint.keys())}")
    print()
    
    # Model state dict kontrolü
    if "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]
        model_keys = list(model_state_dict.keys())
        
        print(f"Model State Dict:")
        print(f"  Key sayısı: {len(model_keys)}")
        print(f"  İlk 10 key: {model_keys[:10]}")
        print()
        
        # SimpleModel kontrolü
        is_simple_model = (
            len(model_keys) == 3 and
            all(k in model_keys for k in ["embed.weight", "proj.weight", "proj.bias"])
        )
        
        # CevahirNeuralNetwork kontrolü
        is_cevahir = len(model_keys) > 200 and "dil_katmani" in model_keys[0]
        
        print("Model Tipi Tespiti:")
        if is_simple_model:
            print(f"  ❌ SimpleModel tespit edildi! (3 key)")
            print(f"     Bu checkpoint YANLIŞ model'den kaydedilmiş!")
        elif is_cevahir:
            print(f"  ✅ CevahirNeuralNetwork tespit edildi! ({len(model_keys)} key)")
        else:
            print(f"  ⚠️ Bilinmeyen model tipi ({len(model_keys)} key)")
        
        # Epoch bilgisi
        if "epoch" in checkpoint:
            print(f"\nEpoch: {checkpoint['epoch']}")
        
        # Metric bilgisi
        if "metric" in checkpoint:
            print(f"Metric: {checkpoint['metric']}")
            
    else:
        print("❌ 'model_state_dict' checkpoint'te bulunamadı!")

if __name__ == "__main__":
    # Test edilecek checkpoint'ler
    checkpoints = [
        "tests/test_checkpoints/best.pth",
        "saved_models/checkpoints/best.pth",
        "saved_models/checkpoints/checkpoint_epoch_0001.pth",
        "saved_models/checkpoints/checkpoint_epoch_0002.pth",
    ]
    
    for ckpt_path in checkpoints:
        if os.path.exists(ckpt_path):
            analyze_checkpoint(ckpt_path)
            print()

