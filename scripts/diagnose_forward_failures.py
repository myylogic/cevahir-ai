# -*- coding: utf-8 -*-
"""
Tanı: Batch bağımsızlığı ve causal invariance'ın hangi aşamada bozulduğunu bulur.
- Aynı sequence iki batch slot'ta verildiğinde çıktılar nerede farklılaşıyor?
- Pozisyon 1 logitleri, pozisyon 2+ değişince nerede etkileniyor?
"""
import os
import sys
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def get_minimal_model():
    from src.neural_network import CevahirNeuralNetwork
    cfg = {
        "learning_rate": 1e-4, "dropout": 0.1, "vocab_size": 2000, "embed_dim": 64,
        "seq_proj_dim": 64, "num_heads": 2, "attention_type": "multi_head",
        "normalization_type": "layer_norm", "device": "cpu", "log_level": 40,
        "num_layers": 2, "ffn_dim": 256, "pre_norm": True, "causal_mask": True,
        "use_flash_attention": False, "pe_mode": "rope", "use_gradient_checkpointing": False,
        "tie_weights": True, "use_rmsnorm": True, "use_swiglu": True,
        "use_kv_cache": False, "max_cache_len": 512, "use_advanced_checkpointing": False,
        "checkpointing_strategy": "selective", "quantization_type": "none",
        "use_moe": False, "num_experts": 2, "moe_top_k": 1,
    }
    model = CevahirNeuralNetwork(**cfg)
    model.eval()
    return model

def diagnose_batch_independence():
    """Aynı sequence [s,s] verildiğinde hangi aşamada row0 != row1 oluyor?"""
    print("=" * 60)
    print("TANI 1: BATCH BAĞIMSIZLIK – Aynı sequence iki slot'ta")
    print("=" * 60)
    torch.manual_seed(42)
    model = get_minimal_model()
    V = model.embedding.num_embeddings
    s = torch.randint(0, V, (1, 6)).expand(2, 6).clone()  # [2,6] aynı satırlar

    captures = {}
    def save(name):
        def hook(module, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(t, torch.Tensor):
                captures[name] = t.detach()
        return hook

    # Embedding
    h_emb = model.embedding.register_forward_hook(save("embedding"))
    h_pe = model.pos_encoding.register_forward_hook(save("after_pos"))
    handles = [h_emb, h_pe]
    for i, layer in enumerate(model.layers):
        handles.append(layer.register_forward_hook(save(f"layer_{i}")))
    h_onorm = model.output_norm.register_forward_hook(save("output_norm"))

    with torch.no_grad():
        _ = model(s)

    for h in handles:
        h.remove()
    h_onorm.remove()

    # Check equality at each stage
    stages = ["embedding", "after_pos", "layer_0", "layer_1", "output_norm"]
    for name in stages:
        if name not in captures:
            continue
        t = captures[name]
        if t.dim() < 2:
            continue
        r0, r1 = t[0], t[1]
        close = torch.allclose(r0, r1, atol=1e-5, rtol=1e-5)
        diff = (r0 - r1).abs().max().item()
        print(f"  {name:15s}: row0 vs row1 allclose={close}, max|diff|={diff:.2e}")
    # Final logits
    with torch.no_grad():
        logits, _ = model(s)
    close = torch.allclose(logits[0], logits[1], atol=1e-4, rtol=1e-4)
    print(f"  {'logits (final)':15s}: row0 vs row1 allclose={close}, max|diff|={(logits[0]-logits[1]).abs().max().item():.2e}")
    print()

def diagnose_causal_position1():
    """x_a vs x_b (aynı 0,1; farklı 2+) iken pozisyon 1 çıktısı nerede farklılaşıyor?"""
    print("=" * 60)
    print("TANI 2: CAUSAL POZİSYON 1 – Pozisyon 2+ değişince pos1 nerede etkileniyor?")
    print("=" * 60)
    torch.manual_seed(42)
    model = get_minimal_model()
    V = model.embedding.num_embeddings
    B, T = 2, 6
    x_a = torch.randint(0, V, (B, T))
    x_b = x_a.clone()
    x_b[:, 2:] = (x_a[:, 2:] + 1) % V

    captures_a = {}
    captures_b = {}
    def make_save(captures, name):
        def hook(module, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(t, torch.Tensor):
                captures[name] = t.detach()
        return hook

    stages_to_save = ["embedding", "after_pos", "layer_0", "layer_1", "output_norm"]
    hooks = []
    hooks.append(model.embedding.register_forward_hook(make_save(captures_a, "embedding")))
    hooks.append(model.pos_encoding.register_forward_hook(make_save(captures_a, "after_pos")))
    for i, layer in enumerate(model.layers):
        def _save(captures, idx):
            def hook(module, inp, out):
                t = out[0] if isinstance(out, (tuple, list)) else out
                if isinstance(t, torch.Tensor):
                    captures[f"layer_{idx}"] = t.detach()
            return hook
        hooks.append(model.layers[i].register_forward_hook(_save(captures_a, i)))
    hooks.append(model.output_norm.register_forward_hook(make_save(captures_a, "output_norm")))

    with torch.no_grad():
        _ = model(x_a)
    for h in hooks:
        h.remove()

    hooks = []
    hooks.append(model.embedding.register_forward_hook(make_save(captures_b, "embedding")))
    hooks.append(model.pos_encoding.register_forward_hook(make_save(captures_b, "after_pos")))
    for i in range(len(model.layers)):
        def _save(captures, idx):
            def hook(module, inp, out):
                t = out[0] if isinstance(out, (tuple, list)) else out
                if isinstance(t, torch.Tensor):
                    captures[f"layer_{idx}"] = t.detach()
            return hook
        hooks.append(model.layers[i].register_forward_hook(_save(captures_b, i)))
    hooks.append(model.output_norm.register_forward_hook(make_save(captures_b, "output_norm")))

    with torch.no_grad():
        _ = model(x_b)
    for h in hooks:
        h.remove()

    # Compare position 1 at each stage (for each batch index we compare a vs b)
    for name in stages_to_save:
        if name not in captures_a or name not in captures_b:
            continue
        ta, tb = captures_a[name], captures_b[name]
        if ta.dim() < 3:
            continue
        # [B, T, E] -> position 1: [B, E]
        pos1_a = ta[:, 1, :]   # (B, E)
        pos1_b = tb[:, 1, :]
        close = torch.allclose(pos1_a, pos1_b, atol=1e-5, rtol=1e-5)
        diff = (pos1_a - pos1_b).abs().max().item()
        print(f"  {name:15s}: pos1(x_a) vs pos1(x_b) allclose={close}, max|diff|={diff:.2e}")

    with torch.no_grad():
        logits_a, _ = model(x_a)
        logits_b, _ = model(x_b)
    pos1_la = logits_a[:, 1, :]
    pos1_lb = logits_b[:, 1, :]
    close = torch.allclose(pos1_la, pos1_lb, atol=1e-5, rtol=1e-5)
    print(f"  {'logits pos1':15s}: allclose={close}, max|diff|={(pos1_la - pos1_lb).abs().max().item():.2e}")
    print()

if __name__ == "__main__":
    diagnose_batch_independence()
    diagnose_causal_position1()
