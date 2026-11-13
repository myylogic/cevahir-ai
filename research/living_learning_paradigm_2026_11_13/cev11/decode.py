# -*- coding: utf-8 -*-
"""Açgözlü (greedy) çözümleme ve toplu tam-üretim denetimi (CEV-11 PROTOCOL §6, §9; V6, V8).

Tanımlar (token düzeyinde):
* Gerçek açgözlü çözümleme: ``[BOS] + tok.encode(prefix)``'ten başlayıp her adımda ``argmax`` seçilir; en çok
  ``max_new`` token, ``stop_ids``'den biri üretilince durur (durdurucu token çıktıya dahildir).
* Tam üretim: açgözlü çıktının ilk ``len(hedef)`` tokeni ``tok.encode(" " + obj + ".")`` ile birebir aynıdır
  (nokta dahil). ``greedy_reproduces`` bunun gerçek çözümlemeli başvuru uygulamasıdır.
* Toplu denetim (``exact_check_batch``): önek + hedef üzerinde öğretmen zorlamalı TEK bir ileri geçiş; her hedef
  konumundaki ``argmax`` hedef tokenine eşitse True. Açgözlü çözümleme bu hedefi ancak ve ancak her adımda aynı
  argmax'ı seçerse üretir; ``max_new`` verilirse hedefi ``max_new``'dan uzun olanlar False sayılır (en çok
  ``max_new`` tokenlik gerçek çözümleme onları üretemez). Toplu satırlar sağdan PAD ile doldurulur; nedensel model
  için dizinin sonrasındaki dolgu önceki konumları (matematiksel olarak) etkilemez ve yalnız gerçek konumlar
  karşılaştırılır. Sonuç girdi listesinin deterministik bir işlevidir; yığın bileşimi logitleri kayan nokta düzeyinde
  (~1e-6) oynatabilir, bu yalnız tam eşitliğe çok yakın argmax'larda önem taşır — V8 bunu gerçek çözümlemeyle ölçer.

KV önbelleği: önbellekli çözümlemede önek tek seferde ``cache_position=arange(T)`` ile işlenir, sonra her yeni token
``cache_position=[T+adım]`` ile verilir (depodaki ``CevahirModelAPI._autoregressive_generate`` ile aynı yol).
Önbellek her çözümlemeden ÖNCE ve SONRA temizlenir; önbelleğin gerçekten dolduğu her adımda denetlenir (önbellek
kapalı bir modelde tek tokenlik ileri geçiş bağlamsız sessiz yanlış sonuç verirdi). Toplam konum sayısı modelin
``max_cache_len`` / ``pe_max_len`` sınırını aşarsa hata verilir (aşım, önbellekte kaydırmalı atmayı tetikler ve
önbellekli/önbelleksiz çıktılar ayrışırdı).

``torch.no_grad`` kullanılır, ``inference_mode`` değil: önbellek tamponları sonraki grad'lı kullanımlarda yerinde
sıfırlanabilmelidir.

INTERFACES.md'ye eklemeler (imzalar korunmuştur): ``exact_check_batch(..., max_new=None)`` isteğe bağlı parametresi;
yardımcılar ``exact_check_ids``, ``greedy_reproduces``, ``greedy_answer``.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

from tokenizer_management.byte_bpe import BOS_ID, EOS_ID, PAD_ID


def clear_cache(model) -> None:
    """``model.clear_kv_cache()`` varsa çağırır."""
    fn = getattr(model, "clear_kv_cache", None)
    if callable(fn):
        fn()


def _position_limit(model) -> Optional[int]:
    """Modelin desteklediği en çok konum sayısı (önbellek ve konum kodlaması sınırlarının küçüğü)."""
    cfg = getattr(model, "_ctor_cfg", None)
    if not cfg:
        return None
    limits = [int(cfg[k]) for k in ("max_cache_len", "pe_max_len") if cfg.get(k)]
    return min(limits) if limits else None


def _check_length(model, n_positions: int, what: str) -> None:
    limit = _position_limit(model)
    if limit is not None and n_positions > limit:
        raise ValueError(f"{what} needs {n_positions} positions but the model supports {limit}")


def _cache_lengths(model) -> list[int]:
    out = []
    for layer in getattr(model, "layers", ()):
        cache = getattr(getattr(layer, "attn", None), "kv_cache", None)
        out.append(-1 if cache is None else int(cache.seen_tokens))
    return out


def _assert_cache(model, expected: int) -> None:
    lengths = _cache_lengths(model)
    if not lengths or any(n != expected for n in lengths):
        raise RuntimeError(f"KV cache not populated as expected (want {expected} positions per layer, have "
                           f"{lengths}); the model may have use_kv_cache disabled - use use_cache=False")


@torch.no_grad()
def greedy_ids(model, ids: list[int], max_new: int = 8, use_cache: bool = True, stop_ids=(EOS_ID,)) -> list[int]:
    """Açgözlü çözümleme; yalnız YENİ tokenleri döndürür (durdurucu token üretildiyse sonda dahildir)."""
    ids = [int(i) for i in ids]
    if not ids:
        raise ValueError("ids must contain at least one token (normally BOS)")
    if max_new < 0:
        raise ValueError("max_new must be non-negative")
    if max_new == 0:
        return []
    _check_length(model, len(ids) + max_new - 1, "greedy decoding")
    stop = {int(s) for s in stop_ids}
    was_training = model.training
    model.eval()
    out: list[int] = []
    try:
        clear_cache(model)
        if use_cache:
            n = len(ids)
            logits = model(torch.tensor([ids], dtype=torch.long), use_cache=True,
                           cache_position=torch.arange(n, dtype=torch.long))[0]
            _assert_cache(model, n)
            for step in range(max_new):
                nxt = int(torch.argmax(logits[0, -1]))
                out.append(nxt)
                if nxt in stop or step == max_new - 1:
                    break
                logits = model(torch.tensor([[nxt]], dtype=torch.long), use_cache=True,
                               cache_position=torch.tensor([n + step], dtype=torch.long))[0]
                _assert_cache(model, n + step + 1)
        else:
            seq = list(ids)
            for _ in range(max_new):
                logits = model(torch.tensor([seq], dtype=torch.long))[0]
                nxt = int(torch.argmax(logits[0, -1]))
                out.append(nxt)
                seq.append(nxt)
                if nxt in stop:
                    break
    finally:
        clear_cache(model)
        model.train(was_training)
    return out


def greedy_text(model, tok, prefix: str, max_new: int = 8) -> str:
    """``[BOS] + encode(prefix)``'ten açgözlü devam metni (özel tokenler atlanır; yarım UTF-8 → U+FFFD)."""
    out = greedy_ids(model, [BOS_ID] + tok.encode(prefix), max_new=max_new)
    return tok.decode(out, errors="replace")


def greedy_answer(model, tok, prefix: str, max_new: int = 8) -> Optional[str]:
    """Modelin açgözlü nesne cevabı: devam metninde ilk '.'den önceki kısım (kırpılmış); '.' yoksa None."""
    text = greedy_text(model, tok, prefix, max_new=max_new)
    cut = text.find(".")
    return None if cut < 0 else text[:cut].strip()


def greedy_reproduces(model, tok, prefix: str, obj: str, max_new: int = 8, use_cache: bool = True) -> bool:
    """Gerçek açgözlü çözümlemeyle tam üretim (V8 başvurusu): ilk len(hedef) token == encode(" " + obj + ".")."""
    target = tok.encode(" " + obj + ".")
    if len(target) > max_new:
        return False
    out = greedy_ids(model, [BOS_ID] + tok.encode(prefix), max_new=len(target), use_cache=use_cache)
    return out == target


@torch.no_grad()
def exact_check_ids(model, prefix_ids: Sequence[Sequence[int]], target_ids: Sequence[Sequence[int]],
                    batch: int = 64, max_new: Optional[int] = None) -> list[bool]:
    """Token düzeyinde toplu tam-üretim denetimi (bkz. modül belgesi). Sıra korunur."""
    if len(prefix_ids) != len(target_ids):
        raise ValueError("prefix_ids and target_ids must have the same length")
    if batch <= 0:
        raise ValueError("batch must be positive")
    n = len(prefix_ids)
    prefixes = [[int(t) for t in p] for p in prefix_ids]
    targets = [[int(t) for t in t_] for t_ in target_ids]
    for i in range(n):
        if not prefixes[i] or not targets[i]:
            raise ValueError(f"empty prefix or target at index {i}")
        if PAD_ID in prefixes[i] or PAD_ID in targets[i]:
            raise ValueError(f"PAD token inside prefix/target at index {i}")
        _check_length(model, len(prefixes[i]) + len(targets[i]) - 1, f"exact check item {i}")
    results = [False] * n
    # Benzer uzunluklar aynı yığına (dolgu azalır); sıralama kararlıdır, sonuç girdinin sırasıyla döner.
    order = sorted(range(n), key=lambda i: len(prefixes[i]) + len(targets[i]))
    was_training = model.training
    model.eval()
    try:
        for start in range(0, n, batch):
            rows = order[start:start + batch]
            seqs = [prefixes[i] + targets[i][:-1] for i in rows]
            width = max(len(s) for s in seqs)
            x = torch.full((len(rows), width), PAD_ID, dtype=torch.long)
            for r, s in enumerate(seqs):
                x[r, :len(s)] = torch.tensor(s, dtype=torch.long)
            pred = model(x)[0].argmax(dim=-1)
            for r, i in enumerate(rows):
                p, t = len(prefixes[i]), targets[i]
                got = pred[r, p - 1:p - 1 + len(t)].tolist()
                results[i] = got == t and (max_new is None or len(t) <= max_new)
    finally:
        model.train(was_training)
    return results


def exact_check_batch(model, tok, prefixes: list[str], objects: list[str], batch: int = 64,
                      max_new: Optional[int] = None) -> list[bool]:
    """True iff ``[BOS] + encode(prefix)``'ten açgözlü çözümleme ``encode(" " + obj + ".")``'i birebir üretir."""
    if len(prefixes) != len(objects):
        raise ValueError("prefixes and objects must have the same length")
    prefix_ids = [[BOS_ID] + tok.encode(p) for p in prefixes]
    target_ids = [tok.encode(" " + o + ".") for o in objects]
    return exact_check_ids(model, prefix_ids, target_ids, batch=batch, max_new=max_new)


__all__ = ["greedy_ids", "greedy_text", "greedy_answer", "greedy_reproduces", "exact_check_ids",
           "exact_check_batch", "clear_cache"]
