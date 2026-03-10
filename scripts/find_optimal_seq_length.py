"""
Find optimal sequence length for training data

Analyze actual content length (non-PAD tokens) to determine
the best max_seq_length that minimizes padding waste.
"""

import pickle
import numpy as np

cache_path = r"C:\Users\Huawei\Desktop\cevahir_sinir_sistemi\.cache\preprocessed_data\cached_data_900e5ad6662aaeab5f92acdad1ab1d3e_ba95397ce1cd1b55.pkl"

print("=" * 80)
print("OPTIMAL SEQUENCE LENGTH ANALYSIS")
print("=" * 80)

print(f"\nLoading cache...")
with open(cache_path, 'rb') as f:
    data = pickle.load(f)

print(f"Total examples: {len(data)}")

# Analyze content length (non-PAD tokens)
content_lengths = []

for example in data:
    if len(example) >= 2:
        tgt = example[1]
        # Count non-PAD tokens (token 0 is PAD)
        non_pad_count = sum(1 for t in tgt if t != 0)
        content_lengths.append(non_pad_count)

content_lengths = np.array(content_lengths)

print("\n" + "=" * 80)
print("CONTENT LENGTH STATISTICS")
print("=" * 80)

print(f"\nTotal sequences: {len(content_lengths)}")
print(f"Mean content length: {content_lengths.mean():.1f}")
print(f"Median content length: {np.median(content_lengths):.1f}")
print(f"Min content length: {content_lengths.min()}")
print(f"Max content length: {content_lengths.max()}")
print(f"Std dev: {content_lengths.std():.1f}")

# Percentiles
percentiles = [50, 75, 90, 95, 99]
print(f"\nPercentiles:")
for p in percentiles:
    val = np.percentile(content_lengths, p)
    print(f"  {p:2d}%: {val:.0f} tokens")

# Recommendation
print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

p95 = np.percentile(content_lengths, 95)
p99 = np.percentile(content_lengths, 99)

print(f"\nCurrent max_seq_length: 768")
print(f"95th percentile: {p95:.0f} tokens")
print(f"99th percentile: {p99:.0f} tokens")

# Calculate padding waste for different lengths
for test_len in [256, 384, 512, 768]:
    sequences_fit = (content_lengths <= test_len).sum()
    fit_pct = sequences_fit / len(content_lengths) * 100
    avg_padding = test_len - content_lengths[content_lengths <= test_len].mean()
    padding_pct = avg_padding / test_len * 100
    
    print(f"\nIf max_seq_length = {test_len}:")
    print(f"  Sequences that fit: {sequences_fit}/{len(content_lengths)} ({fit_pct:.1f}%)")
    print(f"  Average padding: {avg_padding:.0f} tokens ({padding_pct:.1f}%)")
    print(f"  Padding waste: {padding_pct:.1f}%")

# Optimal recommendation
optimal_length = int(np.percentile(content_lengths, 95))
# Round to nearest 128 for efficiency
optimal_length = ((optimal_length + 127) // 128) * 128

print(f"\n" + "=" * 80)
print("OPTIMAL RECOMMENDATION")
print("=" * 80)
print(f"\nRecommended max_seq_length: {optimal_length}")
print(f"  Covers 95% of sequences")
print(f"  Minimizes padding waste")
print(f"  Efficient for GPU (multiple of 128)")

# Impact analysis
current_pad_pct = 67.96
new_pad_pct = (optimal_length - content_lengths.mean()) / optimal_length * 100

print(f"\nImpact:")
print(f"  Current padding: 67.96%")
print(f"  New padding: ~{new_pad_pct:.1f}%")
print(f"  Reduction: {67.96 - new_pad_pct:.1f}%")
print(f"  Training speed: ~{768/optimal_length:.1f}x faster")
print(f"  Memory usage: ~{768/optimal_length:.1f}x less")
