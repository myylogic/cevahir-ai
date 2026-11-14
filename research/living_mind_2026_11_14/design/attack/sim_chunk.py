#!/usr/bin/env python3
"""sim_chunk.py -- LEAD_DESIGN.md §3 "Chunking: an EXPECT(lag 1) chain u1->u2->...->um whose ties all have
e >= e_chunk and use >= m_chunk is compiled into one composed unit (a word from characters, ...)".

Stream: ~300k characters of real English (the licence texts shipped with the OS, read as plain data),
27 symbols (a-z and space).  Ties: first-order EXPECT(lag 1) between characters, learned with the design's rule.
Questions:
  (a) how many ties pass e >= e_chunk at all?  (b) how many CHAINS exist, and how many of them never occurred
  as a sequence (false chunks)?  (c) how many of the 100 most frequent words satisfy the chain criterion?
  (d) the fix: chunk only what was experienced -- extend the unit that just occurred by the symbol that just
      followed it, when that tie is strong RELATIVE TO THE BASE RATE; the chunk then owns its own ties.
"""
import glob, os, re, json, math, sys
from collections import Counter, defaultdict
import numpy as np

K = 1.0


def load_text():
    seen = set(); parts = []
    for f in sorted(glob.glob("/usr/share/common-licenses/*")):
        rp = os.path.realpath(f)
        if rp in seen or not os.path.isfile(rp):
            continue
        seen.add(rp)
        parts.append(open(rp, encoding="utf-8", errors="ignore").read())
    t = " ".join(parts).lower()
    t = re.sub(r"[^a-z]+", " ", t)
    return t


def learn_first_order(ids, n, rule):
    wp = np.zeros((n, n)); wn = np.zeros((n, n)); born = np.zeros((n, n), bool)
    for a, b in zip(ids[:-1], ids[1:]):
        row_p = wp[a]; row_n = wn[a]; bo = born[a]
        if rule == "design":
            e = (row_p + K / 2) / (row_p + row_n + K)
            inc = np.where(bo, e, 0.0)
        else:
            inc = bo.astype(float)
        inc[b] = 0.0
        row_n += inc
        if bo[b]:
            row_p[b] += 1.0
        else:
            bo[b] = True; row_p[b] = 1.0
    W = wp + wn
    e = (wp + K / 2) / (W + K)
    return wp, W, e


def chains(strong, m, simple=True):
    """all paths with m nodes in the directed graph `strong` (adjacency list); simple = no repeated node"""
    out = []
    def rec(path):
        if len(path) == m:
            out.append(tuple(path)); return
        for v in strong[path[-1]]:
            if simple and v in path:
                continue
            rec(path + [v])
    for u in range(len(strong)):
        rec([u])
        if len(out) > 2_000_000:
            break
    return out


def main():
    text = load_text(); N = len(text)
    alpha = sorted(set(text)); ix = {c: i for i, c in enumerate(alpha)}; n = len(alpha)
    ids = [ix[c] for c in text]
    words = Counter(w for w in text.split(" ") if len(w) >= 3)
    top = [w for w, _ in words.most_common(100)]
    print(f"stream: {N} characters, {n} symbols, {len(words)} distinct words (len>=3); top words: {top[:12]}")
    res = {"N": N}
    for rule in ("design", "plain"):
        wp, W, e = learn_first_order(ids, n, rule)
        P = wp / np.maximum(wp.sum(1, keepdims=True), 1)
        print(f"\n== first-order ties learned with the {'design rule (e-weighted negatives)' if rule=='design' else 'unweighted negative rule (f = how often it held)'} ==")
        for th in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            strong = [[v for v in range(n) if e[u, v] >= th and wp[u, v] >= 5] for u in range(n)]
            nt = sum(len(s) for s in strong)
            pairs = [alpha[u] + alpha[v] for u in range(n) for v in strong[u]]
            line = f" e_chunk={th}: {nt:3d} ties pass"
            tot_true = tot_false = 0; counts = []
            for m in range(3, 9):
                ch = chains(strong, m)
                occ = sum(1 for c in ch if "".join(alpha[i] for i in c) in text)
                counts.append((m, len(ch), len(ch) - occ)); tot_true += occ; tot_false += len(ch) - occ
            okw = [w for w in top if all(e[ix[a], ix[b]] >= th and wp[ix[a], ix[b]] >= 5 for a, b in zip(w[:-1], w[1:]))]
            line += f"; chains m=3..8 (simple paths): " + " ".join(f"{m}:{c}({f} never occurred)" for m, c, f in counts)
            line += f"; top-100 words that qualify: {len(okw)} {okw[:6]}"
            print(line)
            if th in (0.5,):
                print(f"      passing ties: {' '.join(repr(p) for p in pairs[:40])}")
            res[f"{rule}_{th}"] = dict(ties=nt, chains=counts, words=len(okw))
        if rule == "design":
            ex = [("t", "h"), ("h", "e"), ("e", " "), ("q", "u"), ("a", "n"), ("n", "d"), ("i", "n"), ("n", "g")]
            print("   examples  P(next|prev) -> design e : " +
                  ", ".join(f"{a}{b}: {P[ix[a], ix[b]]:.2f}->{e[ix[a], ix[b]]:.2f}" for a, b in ex))

    # ------------------------------------------------------------------ the fix: experienced, hierarchical chunking
    print("\n== fix: extend the unit that just occurred by the symbol that just followed, if the tie is strong relative to the")
    print("   base rate of that symbol (e - b >= delta) and was confirmed >= m_chunk times; the chunk owns its own ties ==")
    for delta, m_chunk in [(0.25, 10), (0.4, 10)]:
        unit_str = list(alpha)                 # unit id -> string
        ext = {}                               # (unit, atom) -> chunk unit
        n_u = defaultdict(float)               # exposure of a unit as "current unit"
        n_uv = defaultdict(float)              # (unit, atom) -> count
        base = np.zeros(n); cur = ids[0]; growth = []; false_chunks = 0
        for t in range(1, N):
            s = ids[t]; base[s] += 1
            n_u[cur] += 1; n_uv[(cur, s)] += 1
            nxt = ext.get((cur, s))
            if nxt is None:
                c_uv = n_uv[(cur, s)]; Wt = n_u[cur]
                b = base[s] / t
                e_uv = (c_uv + b * K) / (Wt + K)
                if c_uv >= m_chunk and e_uv - b >= delta:
                    unit_str.append(unit_str[cur] + alpha[s]); ext[(cur, s)] = len(unit_str) - 1
                cur = s
            else:
                cur = nxt
            if t in (10000, 30000, 100000, N - 1):
                growth.append((t, len(unit_str) - n))
        chunks = unit_str[n:]
        lens = Counter(len(c) for c in chunks)
        hit = [w for w in top if (w in chunks) or (w + " " in chunks) or (" " + w in chunks) or (" " + w + " " in chunks)]
        never = sum(1 for c in chunks if c not in text)
        print(f" delta={delta}, m_chunk={m_chunk}: chunks formed after (10k, 30k, 100k, all) characters: "
              f"{[g[1] for g in growth]}; never-occurred chunks: {never}; "
              f"top-100 words that became one unit: {len(hit)}; longest: {sorted(chunks, key=len)[-5:]}")
        res[f"fix_{delta}"] = dict(growth=growth, words=len(hit), never=never)
    json.dump(res, open(__file__.replace(".py", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
