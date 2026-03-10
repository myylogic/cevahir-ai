"""
E7 Checkpoint Generation Analysis
Investigate repetition and token 390 issues
"""

import sys
import torch
import os
sys.path.insert(0, r'C:\Users\Huawei\Desktop\cevahir_sinir_sistemi')

from model_management.model_manager import ModelManager
from tokenizer_management.core.tokenizer_core import TokenizerCore
from tokenizer_management.config import BPE_CONFIG

config = {
    "vocab_path": BPE_CONFIG["vocab_file"],
    "merges_path": BPE_CONFIG["merges_file"],
    "max_seq_length": 768,
    "vocab_size": 60000,
    "embed_dim": 256,
    "seq_proj_dim": 256,
    "num_heads": 4,
    "num_layers": 6,
    "dropout": 0.2,
    "device": "cpu",
    "use_rmsnorm": True,
    "use_swiglu": True,
    "use_kv_cache": True,
    "pe_mode": "rope",
    "tie_weights": True,
    "causal_mask": True,
    "pre_norm": True,
    "use_gradient_checkpointing": False,
    "model_module": "src.neural_network",
    "model_class": "CevahirNeuralNetwork",
}

print("=" * 80)
print("E7 GENERATION ANALYSIS")
print("=" * 80)

print("\n[1/4] Loading tokenizer...")
tokenizer = TokenizerCore(config)
vocab = tokenizer.get_vocab()

print("\n[2/4] Loading model...")
model_manager = ModelManager(config)
model_manager.initialize(build_optimizer=False, build_criterion=False, build_scheduler=False)

checkpoint_path = r"C:\Users\Huawei\Desktop\cevahir_sinir_sistemi\saved_models\cevahir_model.pth"
print(f"  Loading: {checkpoint_path}")

if not os.path.exists(checkpoint_path):
    print(f"  [ERROR] File not found!")
    sys.exit(1)

try:
    model_manager.load(checkpoint_path, weights_only=True)
    print(f"  [OK] Loaded")
except Exception as e:
    print(f"  [ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

model = model_manager.model
model.eval()
model = model.to(torch.device("cpu"))

print("\n[3/4] Testing generation...")

test_prompt = "Mutluluk nedir?"
_, token_ids = tokenizer.encode(test_prompt, mode="inference", include_whole_words=True)

print(f"Prompt: '{test_prompt}'")
print(f"Tokens: {token_ids}")

generated_ids = list(token_ids)
max_new = 20
temp = 0.8
top_k = 40
eos_id = 3
min_new = 5

print(f"\nParams: temp={temp}, top_k={top_k}, min={min_new}")
print("\nGeneration:")

with torch.no_grad():
    for step in range(max_new):
        inp = torch.tensor([generated_ids], dtype=torch.long)
        out = model(inp)
        if isinstance(out, tuple):
            out = out[0]
        
        logits = out[0, -1, :]
        logits = torch.clamp(logits, min=-50, max=50)
        
        if temp > 0:
            logits = logits / temp
        
        if top_k > 0:
            vals, inds = torch.topk(logits, min(top_k, logits.size(-1)))
            filt = torch.full_like(logits, float('-inf'))
            filt[inds] = vals
            logits = filt
        
        probs = torch.softmax(logits, dim=-1)
        pred = torch.multinomial(probs, 1).item()
        
        # Get top-5
        top5_p, top5_i = torch.topk(probs, 5)
        
        # Find token text
        txt = "?"
        for t, d in vocab.items():
            tid = d.get("id") if isinstance(d, dict) else d
            if tid == pred:
                txt = t
                break
        
        print(f"  Step {step+1}: {pred} = '{txt}' | Top prob: {top5_p[0].item():.3f}")
        
        if pred == eos_id and len(generated_ids) - len(token_ids) >= min_new:
            break
        elif pred == eos_id:
            continue
        
        generated_ids.append(pred)

resp_ids = generated_ids[len(token_ids):]
response = tokenizer.decode(resp_ids, method="bpe", remove_specials=True)

print(f"\n[4/4] Result:")
print(f"  Response: '{response}'")
print(f"  Tokens: {len(resp_ids)}")
print(f"  Token 390 count: {resp_ids.count(390)} ({resp_ids.count(390)/len(resp_ids)*100:.1f}%)")

# Check for repetitions
reps = 0
for i in range(len(resp_ids)-1):
    if resp_ids[i] == resp_ids[i+1]:
        reps += 1
print(f"  Consecutive repeats: {reps} ({reps/len(resp_ids)*100:.1f}%)")
