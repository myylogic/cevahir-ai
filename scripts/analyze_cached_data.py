"""
Analyze cached training data to find:
1. Token 390 (SPACE) frequency in training data
2. Repetitive patterns in data
3. Common words that might cause repetition
"""

import pickle
import sys
from collections import Counter

cache_path = r"C:\Users\Huawei\Desktop\cevahir_sinir_sistemi\.cache\preprocessed_data\cached_data_8df4eb897cc9dad55060736aa5ae4cfd_f19e513a813b497d.pkl"

print("=" * 80)
print("CACHED DATA ANALYSIS")
print("=" * 80)

print(f"\nLoading: {cache_path}")

with open(cache_path, 'rb') as f:
    data = pickle.load(f)

print(f"Total examples: {len(data)}")
print(f"Sample format: {type(data[0]) if data else 'empty'}")

# Analyze first few examples
print("\n" + "=" * 80)
print("SAMPLE ANALYSIS (First 5 examples)")
print("=" * 80)

for i in range(min(5, len(data))):
    example = data[i]
    if len(example) == 3:
        inp, tgt, source_id = example
    elif len(example) == 2:
        inp, tgt = example
        source_id = None
    else:
        print(f"Example {i}: Unknown format")
        continue
    
    print(f"\nExample {i+1}:")
    print(f"  Input length: {len(inp)}")
    print(f"  Target length: {len(tgt)}")
    print(f"  Input (first 20): {inp[:20]}")
    print(f"  Target (first 20): {tgt[:20]}")
    
    # Count token 390
    count_390_inp = inp.count(390)
    count_390_tgt = tgt.count(390)
    print(f"  Token 390 in input: {count_390_inp}/{len(inp)} ({count_390_inp/len(inp)*100:.1f}%)")
    print(f"  Token 390 in target: {count_390_tgt}/{len(tgt)} ({count_390_tgt/len(tgt)*100:.1f}%)")

# Overall statistics
print("\n" + "=" * 80)
print("OVERALL STATISTICS")
print("=" * 80)

all_input_tokens = []
all_target_tokens = []

for example in data:
    if len(example) >= 2:
        inp, tgt = example[0], example[1]
        all_input_tokens.extend(inp)
        all_target_tokens.extend(tgt)

print(f"\nTotal input tokens: {len(all_input_tokens)}")
print(f"Total target tokens: {len(all_target_tokens)}")

# Token 390 frequency
count_390_total = all_target_tokens.count(390)
print(f"\nToken 390 (SPACE) frequency in targets:")
print(f"  Count: {count_390_total}")
print(f"  Percentage: {count_390_total/len(all_target_tokens)*100:.2f}%")

# Most common tokens
token_counts = Counter(all_target_tokens)
most_common = token_counts.most_common(20)

print(f"\nTop 20 most common tokens:")
for rank, (token_id, count) in enumerate(most_common, 1):
    pct = count / len(all_target_tokens) * 100
    print(f"  {rank:2d}. Token {token_id:5d}: {count:7d} times ({pct:5.2f}%)")

# Check if token 390 is in top 20
if 390 in [t for t, c in most_common]:
    rank_390 = [i for i, (t, c) in enumerate(most_common, 1) if t == 390][0]
    print(f"\n[INFO] Token 390 (SPACE) rank: {rank_390}/20")
else:
    print(f"\n[INFO] Token 390 (SPACE) not in top 20")

# Analysis
print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

space_freq = count_390_total / len(all_target_tokens)

if space_freq > 0.15:
    print(f"\n[CRITICAL] Token 390 (SPACE) frequency too high: {space_freq*100:.1f}%")
    print(f"  Expected: 5-10% for Turkish text")
    print(f"  Current: {space_freq*100:.1f}%")
    print(f"  ROOT CAUSE: Training data has too many spaces!")
    print(f"  Impact: Model learns to generate spaces frequently")
elif space_freq > 0.10:
    print(f"\n[WARNING] Token 390 (SPACE) frequency elevated: {space_freq*100:.1f}%")
    print(f"  Slightly high but acceptable")
else:
    print(f"\n[OK] Token 390 (SPACE) frequency normal: {space_freq*100:.1f}%")

# Check for repetitive patterns
print(f"\n[INFO] Checking for repetitive patterns in data...")
consecutive_repeats = 0
for i in range(len(all_target_tokens) - 1):
    if all_target_tokens[i] == all_target_tokens[i+1]:
        consecutive_repeats += 1

repeat_rate = consecutive_repeats / len(all_target_tokens)
print(f"  Consecutive repeats in training data: {consecutive_repeats} ({repeat_rate*100:.2f}%)")

if repeat_rate > 0.05:
    print(f"  [WARNING] Training data has repetitive patterns!")
    print(f"  Model is learning to repeat from data")
else:
    print(f"  [OK] Training data repetition rate normal")
