# -*- coding: utf-8 -*-
"""Deterministik ön eğitim ve temel model kapıları (CEV-11 PROTOCOL §5, §6; V7).

Ön eğitim (§5)
--------------
* Her geçiş (pass) ``cev11.world.pass_sentences(world, pass_idx, rng)``'den gelir: P3'te tutulan hücreler dışında
  her (varlık, ilişki, hâl) hücresi bir kez; bağlam ve yazım her geçişte yeniden çekilir (üreteç görevidir).
* Düzyazı parçaları (``cev11.world.prose_chunks``, en çok 60 token) token payının %20'sini oluşturur: bir geçişte
  düzyazı tokenleri, cümle tokenlerinin ``share/(1-share)`` katına ulaşana dek, karıştırılmış parça sırasından
  (bitince yeniden karıştırılarak) alınır. "Token" = kayba giren (PAD olmayan hedef) token sayısı.
* Her dizi ``[BOS] metin [EOS]``; düzyazı parçasına EOS yalnız ``is_end`` ise eklenir. Sağdan PAD; kayıp PAD'i yok
  sayar.
* AdamW lr 2e-3, betas (0.9, 0.98), eps 1e-8, ağırlık azalması 0.01 — 1-B parametrelerde azalma yok. Gruplar AD
  ile seçilir (``named_parameters`` üzerinden; ``isinstance`` yok — çift-import tehlikesi). Ad kuralı ile boyut kuralı
  çelişirse hata verilir. Depo modelinde ``requires_grad=False`` olan parametreler (dikkat modülünün kullanılmayan
  ``attn.norm`` LayerNorm'u) eğitilmez ve meta'da "frozen" olarak listelenir. 100 ısınma adımı (doğrusal), sonra son
  adımda 0.1×'e inen kosinüs. Yığın 64 dizi, gradyan kırpma 1.0, 1 iş parçacığı. Model başlatma tohumu = ``seed``;
  veri sırası ``random.Random(seed + 1)``.
* cev11 kod özeti koşu başında alınır ve sonda yeniden denetlenir (``code_unchanged_during_run``).
* Toplam adım sayısı (kosinüs için) veri planının kuru bir ön geçişiyle kesin olarak hesaplanır; eğitimde aynı plan
  aynı RNG'den yeniden üretilir ve yığın sayıları birebir denetlenir.
* Yığınlama (``batching``): varsayılan ``"by_type"`` — geçişin cümleleri kendi içinde, düzyazı parçaları kendi içinde
  karıştırılıp 64'lük yığınlara bölünür, sonra yığın sırası karıştırılır. Gerekçe: ~%7'lik uzun düzyazı dizileri
  hemen her karışık yığını 62 tokene doldurur ve hesabın ~2/3'ü PAD'e gider (Phase-0 hız ölçümü dolgusuzdu).
  ``"mixed"`` tam karışık yığınlar verir. Bu seçim protokol metninde yoktur; yürütücünün onayına sunulmuştur.
* Çıktılar (``out_dir``): ``model.pt`` (+ ``model.tokenizer.json``), ``curve.jsonl`` (adım, kayıp, token, tok/s,
  geçen süre, lr, gradyan normu), ``exposure.json`` (maruz kalma defteri), ``meta.json`` (dünya özeti, tokenizer
  kimliği, cev11 kod sha256, geçiş, tohum, hiperparametreler, geçiş istatistikleri, ortam).

INTERFACES.md'ye eklemeler: ``pretrain``'e yalnız anahtar sözcüklü, varsayılanı protokol değeri olan parametreler
eklendi (heads, kv_heads, max_len, batch_size, lr, warmup, final_lr_frac, weight_decay, grad_clip, batching,
max_steps, log_every, threads) ve kardeş modüller için enjeksiyon noktaları (sentence_fn, chunk_fn, ledger;
varsayılan ``cev11.world``). ``evaluate_gates``'e anahtar sözcüklü classify, min_editable, batch, max_new eklendi.

Kapılar (§6) — ``evaluate_gates``
--------------------------------
G1 tüm dünya dizgeleri ve tutulmuş düzyazıda %100 gidiş-dönüş; ek olarak "boundary" ön koşulu: her dünya cümlesi
için ``encode(önek) + encode(" " + nesne + ".") == encode(cümle)`` (tam-üretim denetiminin eğitimdeki token
dizisiyle aynı şeyi sınaması için gerekli). G3 600 görülmüş biçimsel bağlamsız hücrede ≥ %95; G4 P3'te tutulan
hücrelerde ≥ %80; G5 600 gündelik hücrede ≥ %80 (IDENT / NONIDENT ayrı raporlanır); G6 600 önekli (P1/P2) biçimsel
hücrede ≥ %80; G7 ≥ 120 yoklama-havuzu varlığı A ilişkisinde 7 biçimsel kalıbın hepsinde ve 6 gündelik çekimli
kalıbın en az 3'ünde doğru (bağlamsız). Tümü ``decode.exact_check_batch`` ile, en çok 8 tokenlik açgözlü üretim
eşdeğeriyle (``max_new=8``) ölçülür.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import platform
import random
import re
import time
from fractions import Fraction
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from tokenizer_management.byte_bpe import BOS_ID, EOS_ID, PAD_ID

# --- PROTOCOL §5 sabitleri ---
LR = 2e-3
BETAS = (0.9, 0.98)
ADAM_EPS = 1e-8
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
FINAL_LR_FRAC = 0.1
BATCH_SIZE = 64
GRAD_CLIP = 1.0
PROSE_MAX_TOKENS = 60
BATCHING_MODES = ("by_type", "mixed")

# --- PROTOCOL §6 eşikleri ---
GATE_THRESHOLDS = {"G1": 1.0, "G3": 0.95, "G4": 0.80, "G5": 0.80, "G6": 0.80, "G7": 120}
GATE_MAX_NEW = 8

# Azalmasız (1-B) parametrelerin ad kuralı: normların ölçek/ağırlık/sapma terimleri ve sapmalar.
_NO_DECAY_NAME = re.compile(r"(^|\.)([a-z_]*norm[0-9]*)\.(weight|bias|scale)$|\.bias$")


# ---------------------------------------------------------------------------------------------------- iyileyici
def lr_at(step: int, total_steps: int, base_lr: float = LR, warmup: int = WARMUP_STEPS,
          final_frac: float = FINAL_LR_FRAC) -> float:
    """0 tabanlı ``step`` için öğrenme oranı: doğrusal ısınma, sonra son adımda ``final_frac``'a inen kosinüs."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    span = max(1, total_steps - 1 - warmup)
    progress = min(1.0, (step - warmup) / span)
    return base_lr * (final_frac + (1.0 - final_frac) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def param_groups(model, weight_decay: float = WEIGHT_DECAY) -> tuple[list[dict], list[str], list[str]]:
    """AdamW grupları; seçim ``named_parameters`` adlarıyla yapılır. Döner: (gruplar, azalmalı adlar, azalmasız adlar).

    Kural: ≥2-B → azalma; 1-B → azalma yok. Ad kuralı (``_NO_DECAY_NAME``) ile boyut kuralı her parametrede aynı
    sonucu vermelidir; vermezse (ör. beklenmeyen bir 1-B parametre) hata verilir — sessiz yanlış gruplama yok.
    """
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]   # bağlı ağırlık bir kez görünür
    decay, no_decay, conflicts = [], [], []
    for name, p in named:
        by_dim = p.ndim < 2
        by_name = bool(_NO_DECAY_NAME.search(name))
        if by_dim != by_name:
            conflicts.append(f"{name} (ndim={p.ndim})")
        (no_decay if by_dim else decay).append(name)
    if conflicts:
        raise ValueError("parameter grouping by name and by dimension disagree: " + ", ".join(conflicts))
    params = dict(named)
    if len({id(p) for p in params.values()}) != len(params):
        raise ValueError("a parameter tensor appears under two names")
    groups = [{"params": [params[n] for n in decay], "weight_decay": weight_decay},
              {"params": [params[n] for n in no_decay], "weight_decay": 0.0}]
    return groups, decay, no_decay


# ---------------------------------------------------------------------------------------------------- veri planı
class _ProseCycler:
    """Düzyazı parçalarını karıştırılmış sırayla verir; liste bitince aynı RNG ile yeniden karıştırır."""

    def __init__(self, seqs: list[tuple[int, ...]], rng: random.Random):
        self.seqs = seqs
        self.order = list(range(len(seqs)))
        rng.shuffle(self.order)
        self.cursor = 0
        self.wraps = 0

    def next(self, rng: random.Random) -> tuple[int, ...]:
        if self.cursor == len(self.order):
            self.order = list(range(len(self.seqs)))
            rng.shuffle(self.order)
            self.cursor = 0
            self.wraps += 1
        seq = self.seqs[self.order[self.cursor]]
        self.cursor += 1
        return seq


def _chunks(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _plan_pass(world, tok, pass_idx: int, rng: random.Random, sentence_fn: Callable, enc_cache: dict,
               prose: Optional[_ProseCycler], prose_ratio: float, batch_size: int, batching: str,
               max_len: int) -> tuple[list[list[tuple[tuple[int, ...], Optional[dict]]]], dict]:
    """Bir geçişin yığınları: [[(dizi, hücre metası | None), ...], ...] ve geçiş istatistikleri."""
    sents = sentence_fn(world, pass_idx, rng)
    if not sents:
        raise ValueError(f"pass {pass_idx}: the sentence source returned no sentences")
    held = world.p3_held
    s_items, s_tok = [], 0
    for text, meta in sents:
        # Koruma: P3'te tutulan bir hücre eğitime girerse G4 anlamsızlaşır — uzun koşudan önce dur.
        if meta and meta.get("rel") == "A" and (meta.get("name"), meta.get("case")) in held:
            raise ValueError(f"pass {pass_idx}: sentence source emitted a P3-held cell {meta!r}")
        ids = enc_cache.get(text)
        if ids is None:
            ids = (BOS_ID, *tok.encode(text), EOS_ID)
            if len(ids) - 1 > max_len:
                raise ValueError(f"training sentence needs {len(ids) - 1} input positions > max_len={max_len}")
            enc_cache[text] = ids
        s_items.append((ids, meta))
        s_tok += len(ids) - 1
    p_items, p_tok = [], 0
    wraps_before = prose.wraps if prose is not None else 0
    if prose is not None:
        budget = prose_ratio * s_tok
        while p_tok < budget:
            ids = prose.next(rng)
            p_items.append((ids, None))
            p_tok += len(ids) - 1
    if batching == "by_type":
        rng.shuffle(s_items)
        rng.shuffle(p_items)
        batches = _chunks(s_items, batch_size) + _chunks(p_items, batch_size)
        rng.shuffle(batches)
    elif batching == "mixed":
        items = s_items + p_items
        rng.shuffle(items)
        batches = _chunks(items, batch_size)
    else:
        raise ValueError(f"unknown batching mode {batching!r}")
    stats = {"pass": pass_idx, "n_sentences": len(s_items), "sentence_tokens": s_tok, "n_prose_chunks": len(p_items),
             "prose_tokens": p_tok, "prose_share": p_tok / (s_tok + p_tok), "n_batches": len(batches),
             "prose_reshuffles": (prose.wraps - wraps_before) if prose is not None else 0}
    return batches, stats


def _pad(batch: list[tuple[tuple[int, ...], Optional[dict]]]) -> torch.Tensor:
    width = max(len(ids) for ids, _ in batch)
    return torch.tensor([list(ids) + [PAD_ID] * (width - len(ids)) for ids, _ in batch], dtype=torch.long)


def _environment(threads: int) -> dict:
    import numpy
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": numpy.__version__,
            "torch_threads": threads, "torch_interop_threads": torch.get_num_interop_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "platform": platform.platform()}


# ---------------------------------------------------------------------------------------------------- ön eğitim
def pretrain(world, tok, prose_train: list[str], out_dir, passes: int, seed: int, log=print,
             layers: int = 6, d: int = 192, prose_share: float = 0.20, *,
             heads: int = 4, kv_heads: int = 4, max_len: int = 64, batch_size: int = BATCH_SIZE, lr: float = LR,
             warmup: int = WARMUP_STEPS, final_lr_frac: float = FINAL_LR_FRAC, weight_decay: float = WEIGHT_DECAY,
             grad_clip: float = GRAD_CLIP, batching: str = "by_type", max_steps: Optional[int] = None,
             log_every: int = 10, threads: int = 1, sentence_fn: Optional[Callable] = None,
             chunk_fn: Optional[Callable] = None, ledger=None) -> pathlib.Path:
    """PROTOCOL §5 ön eğitimi; checkpoint yolunu (``out_dir/model.pt``) döndürür.

    ``max_steps`` yalnız duman/V7 testleri içindir: iyileyici tam planın adım sayısıyla kurulur, eğitim erken kesilir
    ve ``meta["completed"]`` False olur.
    """
    from .model import build_model, code_sha256, model_sha256, save_checkpoint

    if passes < 1:
        raise ValueError("passes must be >= 1")
    if not 0.0 <= prose_share < 1.0:
        raise ValueError("prose_share must be in [0, 1)")
    if batching not in BATCHING_MODES:
        raise ValueError(f"batching must be one of {BATCHING_MODES}")
    if batch_size < 1 or log_every < 1 or threads < 1:
        raise ValueError("batch_size, log_every and threads must be positive")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive when given")
    if sentence_fn is None:
        from .world import pass_sentences as sentence_fn
    if ledger is None:
        from .world import ExposureLedger
        ledger = ExposureLedger()
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Kod özeti koşu BAŞINDA alınır (çalışan kod budur); sonda yeniden denetlenir.
    code_digest, code_files = code_sha256()
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        # --- düzyazı dizileri (BOS + parça [+ EOS yalnız paragraf sonunda]) ---
        chunk_seqs: list[tuple[int, ...]] = []
        if prose_share > 0.0:
            if chunk_fn is None:
                from .world import prose_chunks as chunk_fn
            chunks = chunk_fn(prose_train, tok, max_tokens=PROSE_MAX_TOKENS)
            if not chunks:
                raise ValueError("prose_share > 0 but there are no prose chunks")
            for text, is_end in chunks:
                body = tok.encode(text)
                if not body or len(body) > PROSE_MAX_TOKENS:
                    raise ValueError(f"prose chunk with {len(body)} tokens (allowed 1..{PROSE_MAX_TOKENS})")
                seq = (BOS_ID, *body, EOS_ID) if is_end else (BOS_ID, *body)
                if len(seq) - 1 > max_len:
                    raise ValueError(f"prose chunk needs {len(seq) - 1} input positions > max_len={max_len}")
                chunk_seqs.append(seq)
        prose_ratio = prose_share / (1.0 - prose_share)

        def fresh_state():
            rng = random.Random(seed + 1)
            return rng, (_ProseCycler(chunk_seqs, rng) if chunk_seqs else None)

        enc_cache: dict[str, tuple[int, ...]] = {}
        # --- kuru ön geçiş: kesin toplam adım sayısı ---
        rng, prose = fresh_state()
        plan_counts, plan_stats = [], []
        for p in range(passes):
            batches, stats = _plan_pass(world, tok, p, rng, sentence_fn, enc_cache, prose, prose_ratio,
                                        batch_size, batching, max_len)
            plan_counts.append(len(batches))
            plan_stats.append(stats)
        total_steps = sum(plan_counts)
        del batches

        # --- model ve iyileyici ---
        model, cfg = build_model(tok.vocab_size, layers=layers, d=d, heads=heads, kv_heads=kv_heads,
                                 max_len=max_len, seed=seed)
        model.train()
        groups, decay_names, no_decay_names = param_groups(model, weight_decay)
        opt = torch.optim.AdamW(groups, lr=lr, betas=BETAS, eps=ADAM_EPS, foreach=False)
        log(f"[pretrain] world={getattr(world, 'seed', '?')} seed={seed} passes={passes} steps={total_steps} "
            f"params={cfg['arch']['n_params']:,} batching={batching} prose_share={prose_share}")

        # --- eğitim: aynı plan aynı RNG'den yeniden üretilir ---
        rng, prose = fresh_state()
        step, tokens, slots = 0, 0, 0
        window: list[float] = []
        last_loss = float("nan")
        stopped = False
        t0 = time.perf_counter()
        with open(out_dir / "curve.jsonl", "w", encoding="utf-8") as curve:
            for p in range(passes):
                batches, stats = _plan_pass(world, tok, p, rng, sentence_fn, enc_cache, prose, prose_ratio,
                                            batch_size, batching, max_len)
                if len(batches) != plan_counts[p] or stats != plan_stats[p]:
                    raise RuntimeError(f"pass {p}: data plan differs from the dry run (non-deterministic source)")
                for batch in batches:
                    if max_steps is not None and step >= max_steps:
                        stopped = True
                        break
                    lr_now = lr_at(step, total_steps, lr, warmup, final_lr_frac)
                    for g in opt.param_groups:
                        g["lr"] = lr_now
                    x = _pad(batch)
                    inp, tgt = x[:, :-1], x[:, 1:]
                    opt.zero_grad(set_to_none=True)
                    logits = model(inp)[0]
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                                           ignore_index=PAD_ID)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite loss at step {step}")
                    loss.backward()
                    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
                    opt.step()
                    step += 1
                    tokens += int((tgt != PAD_ID).sum())
                    slots += tgt.numel()
                    for _, meta in batch:
                        if meta is not None:
                            ledger.add(meta)
                    last_loss = loss.item()
                    window.append(last_loss)
                    if step % log_every == 0 or step == total_steps or step == max_steps:
                        elapsed = time.perf_counter() - t0
                        curve.write(json.dumps({"step": step, "pass": p + 1, "loss": sum(window) / len(window),
                                                "loss_last": last_loss, "lr": lr_now, "grad_norm": float(gnorm),
                                                "tokens": tokens, "tok_s": tokens / max(elapsed, 1e-9),
                                                "elapsed": elapsed}) + "\n")
                        curve.flush()
                        window = []
                if stopped:
                    break
                elapsed = time.perf_counter() - t0
                log(f"[pretrain] pass {p + 1}/{passes} step {step}/{total_steps} loss {last_loss:.4f} "
                    f"tok {tokens:,} {tokens / max(elapsed, 1e-9):,.0f} tok/s {elapsed:.0f}s")
        elapsed = time.perf_counter() - t0

        code_end = code_sha256()[0]
        if code_end != code_digest:
            log(f"[pretrain] WARNING: cev11 code changed during the run ({code_digest[:12]} -> {code_end[:12]})")
        meta = {
            "study": "CEV-11 Paradigma", "kind": "pretrain",
            "world_seed": getattr(world, "seed", None), "world_hash": world.name_hash(),
            "tokenizer_identity": tok.identity(), "tokenizer_class": type(tok).__name__,
            "vocab_size": int(tok.vocab_size),
            "code_sha256": code_digest, "code_files": code_files,
            "code_sha256_at_end": code_end, "code_unchanged_during_run": code_end == code_digest,
            "passes": passes, "seed": seed, "init_seed": seed, "data_order_seed": seed + 1,
            "hyper": {"lr": lr, "betas": list(BETAS), "eps": ADAM_EPS, "weight_decay": weight_decay,
                      "warmup_steps": warmup, "final_lr_frac": final_lr_frac, "batch_size": batch_size,
                      "grad_clip": grad_clip, "prose_share": prose_share, "prose_max_tokens": PROSE_MAX_TOKENS,
                      "batching": batching, "layers": layers, "d": d, "heads": heads, "kv_heads": kv_heads,
                      "max_len": max_len, "threads": threads},
            "param_groups": {"decay": decay_names, "no_decay": no_decay_names,
                             "frozen": [n for n, p in model.named_parameters() if not p.requires_grad]},
            "total_steps": total_steps, "steps_done": step, "completed": step == total_steps,
            "tokens": tokens, "padding_efficiency": tokens / max(slots, 1),
            "final_loss": last_loss, "elapsed_s": elapsed, "train_tok_s": tokens / max(elapsed, 1e-9),
            "n_prose_chunks_available": len(chunk_seqs),
            "pass_stats": plan_stats,
            "n_params": cfg["arch"]["n_params"],
            "n_params_non_embedding": cfg["arch"]["n_params_non_embedding"],
            "model_sha256": model_sha256(model),
            "env": _environment(threads),
        }
        ckpt = out_dir / "model.pt"
        save_checkpoint(ckpt, model, cfg, tok, meta)
        (out_dir / "exposure.json").write_text(json.dumps(ledger.to_json(), ensure_ascii=False, sort_keys=True),
                                               encoding="utf-8")
        (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                                           encoding="utf-8")
        log(f"[pretrain] saved {ckpt} sha256={meta['model_sha256'][:16]} completed={meta['completed']}")
        return ckpt
    finally:
        torch.set_num_threads(prev_threads)


# ---------------------------------------------------------------------------------------------------- kapılar
def _sample(rng: random.Random, pool: list, n: int, gate: str) -> list:
    if len(pool) < n:
        raise ValueError(f"{gate}: need {n} cells but the world has only {len(pool)}")
    return rng.sample(pool, n)


def _meets(k: int, n: int, threshold: float) -> bool:
    """k/n ≥ eşik, kesirli aritmetikle (0.95 gibi eşiklerde kayan nokta sınır hatası olmasın)."""
    return n > 0 and Fraction(k, n) >= Fraction(str(threshold))


def _rate_entry(oks: list[bool], threshold: float) -> dict:
    k, n = sum(oks), len(oks)
    return {"value": k / n if n else None, "threshold": threshold, "pass": _meets(k, n, threshold),
            "n": n, "n_correct": k}


def _breakdown(keys: list, oks: list[bool]) -> dict:
    out: dict = {}
    for key, ok in zip(keys, oks):
        e = out.setdefault(key, [0, 0])
        e[0] += int(ok)
        e[1] += 1
    return {k: {"n": v[1], "n_correct": v[0], "value": v[0] / v[1]} for k, v in sorted(out.items())}


def evaluate_gates(model, tok, world, heldout_prose, n_cells: int = 600, seed: int = 0, *,
                   classify: Optional[Callable] = None, min_editable: int = GATE_THRESHOLDS["G7"],
                   batch: int = 64, max_new: int = GATE_MAX_NEW) -> dict:
    """PROTOCOL §6 kapıları (G1, G3–G7) ve "boundary" ön koşulu; değerler, eşikler, geçti/kaldı, düzenlenebilir liste.

    ``seed`` örnekleme tohumudur (protokol: yoklama örneklemesi w+3). ``classify(tok, name, case)`` varsayılan
    olarak ``cev11.tokenizers.classify_informal``'dır. Düzyazı metni hiçbir çıktıya yazılmaz (yalnız sayı/indeks).
    """
    from .decode import exact_check_batch
    from .morph_tr import CASES, INFLECTED
    from .world import CONTEXTS, frame_prefix

    if classify is None:
        from .tokenizers import classify_informal as classify
    heldout_prose = list(heldout_prose)
    if not heldout_prose:
        raise ValueError("G1 needs held-out prose")
    names = list(world.names)
    rels = sorted({r for (_, r) in world.facts})
    if "A" not in rels:
        raise ValueError("world has no relation A")
    missing = [(n, r) for n in names for r in rels if (n, r) not in world.facts]
    if missing:
        raise ValueError(f"world.facts is missing {len(missing)} (name, relation) pairs, e.g. {missing[:3]}")
    held = set(world.p3_held)
    if not held:
        raise ValueError("G4 needs P3-held cells")

    # --- G1: gidiş-dönüş + boundary ön koşulu ---
    n_world = bad_world = bad_boundary = 0
    boundary_examples: list[str] = []
    for name in names:
        for rel in rels:
            cont = " " + world.facts[(name, rel)] + "."
            cont_ids = tok.encode(cont)
            for case in CASES:
                for orth in (("formal",) if case == "nom" else ("formal", "informal")):
                    for ctx in CONTEXTS:
                        prefix = frame_prefix(name, rel, case, orth, ctx)
                        text = prefix + cont
                        ids = tok.encode(text)
                        n_world += 1
                        if tok.decode(ids) != text:
                            bad_world += 1
                        if tok.encode(prefix) + cont_ids != ids:
                            bad_boundary += 1
                            if len(boundary_examples) < 5:
                                boundary_examples.append(text)    # sentetik dünya dizgesi; düzyazı değil
    bad_held_idx = [i for i, t in enumerate(heldout_prose) if tok.decode(tok.encode(t)) != t]
    g1 = {"value": {"world": (n_world - bad_world) / n_world,
                    "heldout": (len(heldout_prose) - len(bad_held_idx)) / len(heldout_prose)},
          "threshold": GATE_THRESHOLDS["G1"], "pass": bad_world == 0 and not bad_held_idx,
          "n_world": n_world, "n_world_fail": bad_world, "n_heldout": len(heldout_prose),
          "n_heldout_fail": len(bad_held_idx), "heldout_fail_idx": bad_held_idx[:20]}
    boundary = {"value": (n_world - bad_boundary) / n_world, "threshold": 1.0, "pass": bad_boundary == 0,
                "n": n_world, "n_fail": bad_boundary, "examples": boundary_examples}

    # --- örneklem (sabit RNG sırası) ---
    rng = random.Random(seed)
    seen = [(n, r, c) for n in names for r in rels for c in CASES if not (r == "A" and (n, c) in held)]
    g3_cells = _sample(rng, seen, n_cells, "G3")
    g5_cells = _sample(rng, [x for x in seen if x[2] != "nom"], n_cells, "G5")
    g6_cells = _sample(rng, seen, n_cells, "G6")
    g6_ctx = [rng.choice(("P1", "P2")) for _ in g6_cells]
    g4_cells = sorted((n, "A", c) for (n, c) in held)
    probe = sorted(n for n in names if world.partition[n] == "probe")

    queries: list[tuple[str, str, str]] = []          # (önek, nesne, etiket)
    for n, r, c in g3_cells:
        queries.append((frame_prefix(n, r, c, "formal", "none"), world.facts[(n, r)], "G3"))
    for n, r, c in g4_cells:
        queries.append((frame_prefix(n, r, c, "formal", "none"), world.facts[(n, r)], "G4"))
    for n, r, c in g5_cells:
        queries.append((frame_prefix(n, r, c, "informal", "none"), world.facts[(n, r)], "G5"))
    for (n, r, c), ctx in zip(g6_cells, g6_ctx):
        queries.append((frame_prefix(n, r, c, "formal", ctx), world.facts[(n, r)], "G6"))
    for n in probe:
        obj = world.facts[(n, "A")]
        for c in CASES:
            queries.append((frame_prefix(n, "A", c, "formal", "none"), obj, "G7f"))
        for c in INFLECTED:
            queries.append((frame_prefix(n, "A", c, "informal", "none"), obj, "G7i"))
    too_long = sum(len(tok.encode(" " + o + ".")) > max_new for _, o, _ in queries)
    oks = exact_check_batch(model, tok, [q[0] for q in queries], [q[1] for q in queries], batch=batch,
                            max_new=max_new)
    by_tag: dict[str, list[bool]] = {}
    for (_, _, tag), ok in zip(queries, oks):
        by_tag.setdefault(tag, []).append(ok)

    g3 = _rate_entry(by_tag["G3"], GATE_THRESHOLDS["G3"])
    g3["by_case"] = _breakdown([c for _, _, c in g3_cells], by_tag["G3"])
    g3["by_rel"] = _breakdown([r for _, r, _ in g3_cells], by_tag["G3"])
    g4 = _rate_entry(by_tag["G4"], GATE_THRESHOLDS["G4"])
    g4["by_case"] = _breakdown([c for _, _, c in g4_cells], by_tag["G4"])

    classes = []
    for n, _, c in g5_cells:
        cls = classify(tok, n, c)
        if cls not in ("IDENT", "RESEG", "FUSED"):
            raise ValueError(f"classify returned {cls!r} for ({n!r}, {c!r})")
        classes.append(cls)
    g5 = _rate_entry(by_tag["G5"], GATE_THRESHOLDS["G5"])
    g5["by_class"] = _breakdown(classes, by_tag["G5"])
    nonident = [ok for cls, ok in zip(classes, by_tag["G5"]) if cls != "IDENT"]
    g5["by_class"]["NONIDENT"] = {"n": len(nonident), "n_correct": sum(nonident),
                                  "value": sum(nonident) / len(nonident) if nonident else None}
    g5["by_case"] = _breakdown([c for _, _, c in g5_cells], by_tag["G5"])
    g6 = _rate_entry(by_tag["G6"], GATE_THRESHOLDS["G6"])
    g6["by_ctx"] = _breakdown(g6_ctx, by_tag["G6"])
    g6["by_case"] = _breakdown([c for _, _, c in g6_cells], by_tag["G6"])

    nf, ni = len(CASES), len(INFLECTED)
    formal_ok, informal_ok = by_tag.get("G7f", []), by_tag.get("G7i", [])
    editable = [n for k, n in enumerate(probe)
                if all(formal_ok[k * nf:(k + 1) * nf]) and sum(informal_ok[k * ni:(k + 1) * ni]) >= 3]
    g7 = {"value": len(editable), "threshold": min_editable, "pass": len(editable) >= min_editable,
          "n_probe": len(probe), "n_formal_all7": sum(all(formal_ok[k * nf:(k + 1) * nf]) for k in range(len(probe)))}

    gates = {"G1": g1, "G3": g3, "G4": g4, "G5": g5, "G6": g6, "G7": g7}
    return {**gates, "boundary": boundary, "editable": editable,
            "all_pass": all(g["pass"] for g in gates.values()) and boundary["pass"],
            "settings": {"n_cells": n_cells, "seed": seed, "max_new": max_new, "batch": batch,
                         "too_long_targets": too_long, "threads": torch.get_num_threads(),
                         "n_queries": len(queries)}}


__all__ = ["LR", "BETAS", "WEIGHT_DECAY", "WARMUP_STEPS", "FINAL_LR_FRAC", "BATCH_SIZE", "GRAD_CLIP",
           "PROSE_MAX_TOKENS", "GATE_THRESHOLDS", "GATE_MAX_NEW", "lr_at", "param_groups", "pretrain",
           "evaluate_gates"]
