# -*- coding: utf-8 -*-
"""CEV-11 core: model kurulumu/checkpoint, açgözlü çözümleme, toplu tam-üretim denetimi, ön eğitim, kapılar.

Hızlı (< 10 s), yalnız tmp_path; küçük model (D32/L2) ve sahte mini dünya kullanır. Kardeş modüllerin (world
üretimi, tokenizers) bitmiş olması gerekmez: geçiş cümleleri, düzyazı bölücü, defter ve sınıflayıcı yerel sahtedir;
yalnız sabit metin işlevleri (cev11.world.frame_prefix/sentence, cev11.morph_tr) kullanılır.

Kapsanan doğrulamalar: V6 (önbellekli = önbelleksiz açgözlü çıktı; fc2 kancasıyla da), V7 (8 eğitim adımı sabit iş
parçacığında bit düzeyinde yeniden üretilebilir), V8 mantığı (toplu denetim = gerçek açgözlü çözümleme), checkpoint
gidiş-dönüşü ve yapılandırma/tokenizer/ağırlık uyuşmazlığı korumaları.

Davranış testleri (bağımsız inceleme ekledi; kaydedilen değerlere değil etkilere bakar): kapıların doğru yazım /
bağlam / hücreleri sorduğu ve G7 eşikleri (exact_check yerine bir kâhin), düzyazı EOS kuralı ve token sayımı, veri
sırasının Random(seed + 1)'den geldiği, başlatma tohumu, lr çizelgesinin ve gradyan kırpmanın gerçekten uygulandığı,
bayat KV önbelleğinin açgözlü çözümlemeyi bozmadığı.
"""
import hashlib
import json
import pathlib
import random
import shutil
import sys

import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
STUDY = REPO / "research" / "living_learning_paradigm_2026_11_13"
for _p in (str(REPO), str(STUDY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cev11 import decode as cd  # noqa: E402
from cev11 import model as cm  # noqa: E402
from cev11 import train as ct  # noqa: E402
from cev11.morph_tr import CASES, INFLECTED  # noqa: E402
from cev11.world import CONTEXTS, frame_prefix, sentence  # noqa: E402
from tokenizer_management.byte_bpe import BOS_ID, EOS_ID, ByteLevelBPE  # noqa: E402

# ------------------------------------------------------------------------------------------- sahte mini dünya
NAMES = ("Bermut", "Dorhan", "Ezgür", "Kemsu", "Lirten", "Sılay")
OBJECTS = {"A": ("Akköy", "Belpınar", "Cömdere", "Durkent", "Erova"),
           "B": ("öğretmen", "doktor", "terzi", "aşçı", "pilot")}
PROSE = ["Kış gelince köyün yolları kapanır ve herkes evinde oturur.",
         "Bahar geldiğinde dereler coşar, tarlalar yeşerir ve çocuklar dışarıda oynar. Akşam olunca herkes "
         "sofraya toplanır, büyükler eski günleri anlatır ve küçükler sessizce dinler.",
         "Kitap okumak insanın ufkunu açar.",
         "Yaşlı adam her sabah çay içer ve gazetesini okur."]
HELDOUT = ["Yağmur dinince güneş bulutların arasından göründü.", "Pazar günü çarşı çok kalabalıktı."]


class FakeWorld:
    """World arayüzünün testte gereken kısmı (INTERFACES.md: seed, names, objects, facts, partition, p3_held)."""

    def __init__(self, seed=5):
        rnd = random.Random(seed)
        self.seed = seed
        self.names = tuple(sorted(NAMES))
        self.objects = OBJECTS
        self.facts = {(n, r): rnd.choice(OBJECTS[r]) for n in self.names for r in sorted(OBJECTS)}
        self.partition = {n: ("p3" if i == 0 else "c0" if i == 1 else "probe") for i, n in enumerate(self.names)}
        self.p3_held = frozenset({(self.names[0], "gen"), (self.names[0], "loc")})

    def name_hash(self):
        payload = [self.seed, list(self.names), sorted([list(k), v] for k, v in self.facts.items())]
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()


def fake_pass_sentences(world, pass_idx, rng):
    """Her hücre bir kez (P3'te tutulanlar hariç); bağlam .50/.25/.25, çekimli hâllerde gündelik yazım p=.25."""
    out = []
    for n in world.names:
        for r in sorted(world.objects):
            for c in CASES:
                if r == "A" and (n, c) in world.p3_held:
                    continue
                u = rng.random()
                ctx = "none" if u < 0.5 else ("P1" if u < 0.75 else "P2")
                orth = "informal" if c != "nom" and rng.random() < 0.25 else "formal"
                out.append((sentence(n, r, c, world.facts[(n, r)], orth, ctx),
                            {"name": n, "rel": r, "case": c, "orth": orth, "ctx": ctx, "pass": pass_idx}))
    return out


def fake_prose_chunks(texts, tok, max_tokens=60):
    out = []
    for t in texts:
        cur = []
        for w in t.split(" "):
            if cur and len(tok.encode(" ".join(cur + [w]))) > max_tokens:
                out.append((" ".join(cur), False))
                cur = []
            cur.append(w)
        out.append((" ".join(cur), True))
    return out


class FakeLedger:
    def __init__(self):
        self.counts = {}

    def add(self, meta):
        key = "|".join(str(meta[k]) for k in ("name", "rel", "case", "ctx", "orth"))
        self.counts[key] = self.counts.get(key, 0) + 1

    def to_json(self):
        return {"counts": self.counts}


def fake_classify(tok, name, case):
    """Kaba yerel sınıflayıcı (cev11.tokenizers.classify_informal yerine): kök tokenleri önekse IDENT."""
    from cev11.morph_tr import informal
    root, inf = tok.encode(name), tok.encode(informal(name, case))
    return "IDENT" if inf[:len(root)] == root else "FUSED"


def all_world_prefixes(world):
    """(önek, doğru nesne) — tüm hücreler × yazım × bağlam."""
    out = []
    for n in world.names:
        for r in sorted(world.objects):
            for c in CASES:
                for o in (("formal",) if c == "nom" else ("formal", "informal")):
                    for x in CONTEXTS:
                        out.append((frame_prefix(n, r, c, o, x), world.facts[(n, r)]))
    return out


TRAIN_KW = dict(layers=2, d=32, heads=4, kv_heads=4, max_len=64, batch_size=32, lr=2e-2, warmup=5,
                sentence_fn=fake_pass_sentences, chunk_fn=fake_prose_chunks, log=lambda s: None)


@pytest.fixture(scope="module")
def world():
    return FakeWorld()


@pytest.fixture(scope="module")
def tok(world):
    corpus = [s + o for s, o in ((p, " " + obj + ".") for p, obj in all_world_prefixes(world))] + PROSE + HELDOUT
    return ByteLevelBPE.train(corpus, vocab_size=400)


@pytest.fixture(scope="module")
def trained(world, tok, tmp_path_factory):
    """Küçük modeli sahte dünyada kısa süre eğitir (tüm modül tarafından paylaşılır)."""
    out = tmp_path_factory.mktemp("pretrain")
    ledger = FakeLedger()
    threads_before = torch.get_num_threads()
    ckpt = ct.pretrain(world, tok, PROSE, out, passes=16, seed=11, ledger=ledger, log_every=5, **TRAIN_KW)
    assert torch.get_num_threads() == threads_before          # pretrain thread ayarını geri yükler
    model, cfg, tok2, meta = cm.load_checkpoint(ckpt)
    return {"dir": out, "ckpt": ckpt, "model": model, "cfg": cfg, "tok": tok2, "meta": meta, "ledger": ledger}


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


# ------------------------------------------------------------------------------------------- iyileyici / gruplar
def test_lr_schedule_and_param_groups():
    total = 400
    lrs = [ct.lr_at(s, total) for s in range(total)]
    assert lrs[0] == pytest.approx(2e-3 / 100) and lrs[99] == pytest.approx(2e-3)
    assert all(a < b for a, b in zip(lrs[:99], lrs[1:100]))                 # doğrusal ısınma
    assert all(a >= b for a, b in zip(lrs[99:], lrs[100:]))                 # kosinüs azalması
    assert lrs[-1] == pytest.approx(0.1 * 2e-3)                             # son adım 0.1×

    model, _ = cm.build_model(300, layers=2, d=32, seed=0)
    groups, decay, no_decay = ct.param_groups(model)
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    assert frozen <= {f"layers.{i}.attn.norm.{w}" for i in range(2) for w in ("weight", "bias")}  # kullanılmayan
    assert set(decay) | set(no_decay) == set(params) and not set(decay) & set(no_decay)
    assert all(params[n].ndim >= 2 for n in decay) and all(params[n].ndim == 1 for n in no_decay)
    assert "embedding.embedding.weight" in decay and "output_layer.weight" not in params   # bağlı: tek kez
    assert {"output_norm.scale", "layers.0.norm1.scale"} <= set(no_decay)
    assert groups[0]["weight_decay"] == 0.01 and groups[1]["weight_decay"] == 0.0

    class Odd(torch.nn.Module):          # ad kuralına uymayan 1-B parametre → hata (sessiz yanlış gruplama yok)
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4, bias=False)
            self.gate = torch.nn.Parameter(torch.ones(4))
    with pytest.raises(ValueError, match="disagree"):
        ct.param_groups(Odd())


# ------------------------------------------------------------------------------------------- ön eğitim çıktıları
def test_pretrain_writes_checkpoint_curve_ledger_meta(trained, world, tok):
    out, meta = trained["dir"], trained["meta"]
    for name in ("model.pt", "model.tokenizer.json", "curve.jsonl", "exposure.json", "meta.json"):
        assert (out / name).is_file(), name
    assert meta == json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["world_hash"] == world.name_hash() and meta["tokenizer_identity"] == tok.identity()
    assert len(meta["code_sha256"]) == 64 and "model.py" in meta["code_files"]
    assert meta["passes"] == 16 and meta["seed"] == 11 and meta["data_order_seed"] == 12
    assert meta["completed"] and meta["steps_done"] == meta["total_steps"]
    assert meta["model_sha256"] == cm.model_sha256(trained["model"])
    curve = [json.loads(line) for line in (out / "curve.jsonl").read_text(encoding="utf-8").splitlines()]
    assert curve[-1]["step"] == meta["total_steps"]
    assert {"step", "loss", "tokens", "tok_s", "elapsed", "lr", "grad_norm"} <= set(curve[0])
    assert curve[-1]["loss"] < curve[0]["loss"]
    for st in meta["pass_stats"]:                         # düzyazı ≈ %20 token (bir parçalık aşım payı)
        assert 0.20 <= st["prose_share"] < 0.25
        assert st["n_sentences"] == len(world.names) * 2 * 7 - len(world.p3_held)
    exposures = sum(trained["ledger"].counts.values())
    assert exposures == sum(st["n_sentences"] for st in meta["pass_stats"])
    assert json.loads((out / "exposure.json").read_text(encoding="utf-8")) == trained["ledger"].to_json()
    for key in trained["ledger"].counts:                  # P3'te tutulan hücreler hiç eğitilmedi
        name, rel, case = key.split("|")[:3]
        assert not (rel == "A" and (name, case) in world.p3_held), key


def test_v7_eight_training_steps_bitwise_reproducible(world, tok, tmp_path):
    def run(sub, seed):
        ck = ct.pretrain(world, tok, PROSE, tmp_path / sub, passes=2, seed=seed, ledger=FakeLedger(),
                         max_steps=8, log_every=1, **{**TRAIN_KW, "batch_size": 16})
        meta = json.loads((tmp_path / sub / "meta.json").read_text(encoding="utf-8"))
        curve = [json.loads(x) for x in (tmp_path / sub / "curve.jsonl").read_text(encoding="utf-8").splitlines()]
        return ck, meta, curve

    _, m1, c1 = run("a", 3)
    _, m2, c2 = run("b", 3)
    _, m3, _ = run("c", 4)
    assert m1["steps_done"] == 8 and not m1["completed"] and m1["total_steps"] > 8
    assert len(c1) == 8 and c1[-1]["step"] == 8
    assert m1["model_sha256"] == m2["model_sha256"]
    assert [(r["loss"], r["grad_norm"], r["lr"], r["tokens"]) for r in c1] == \
           [(r["loss"], r["grad_norm"], r["lr"], r["tokens"]) for r in c2]
    assert m3["model_sha256"] != m1["model_sha256"]
    assert m1["env"]["torch_threads"] == 1
    # Çizelge gerçekten uygulanır: her adımın lr'si lr_at(0 tabanlı adım, toplam plan adımı) (ısınma 5 adım).
    assert [r["lr"] for r in c1] == [ct.lr_at(r["step"] - 1, m1["total_steps"], 2e-2, 5) for r in c1]
    assert c1[0]["lr"] < c1[4]["lr"] == 2e-2 > c1[7]["lr"]          # ısınma yükselir, kosinüs iner


def test_pretrain_applies_its_data_and_optimiser_contract(world, tok, tmp_path):
    """Kaydedilen değerler değil, etkiler: düzyazı EOS kuralı ve token sayımı, veri sırası tohumu (seed + 1),
    başlatma tohumu (seed), gradyan kırpmanın uygulanması."""
    def run(sub, seed=3, passes=1, **over):
        ledger = over.pop("ledger", None) or FakeLedger()
        ct.pretrain(world, tok, PROSE, tmp_path / sub, passes=passes, seed=seed, ledger=ledger,
                    **{**TRAIN_KW, **over})
        return json.loads((tmp_path / sub / "meta.json").read_text(encoding="utf-8"))

    # EOS yalnız is_end parçalarında; işlenen token = plandaki kayıp tokenleri (PAD sayılmaz).
    body_text = PROSE[0]
    n_body = len(tok.encode(body_text))
    for is_end in (False, True):
        meta = run(f"eos{int(is_end)}", chunk_fn=lambda texts, t, max_tokens=60, e=is_end: [(body_text, e)])
        st = meta["pass_stats"][0]
        assert st["n_prose_chunks"] > 0
        assert st["prose_tokens"] == st["n_prose_chunks"] * (n_body + int(is_end))
        assert meta["completed"] and meta["tokens"] == st["sentence_tokens"] + st["prose_tokens"]

    # Veri sırası: düzyazısız tek geçişte bağlam/yazım çekilişleri tam olarak Random(seed + 1)'den gelir.
    ledger = FakeLedger()
    meta = run("order", seed=7, prose_share=0.0, ledger=ledger)
    sents = fake_pass_sentences(world, 0, random.Random(7 + 1))
    expect = FakeLedger()
    for _, m in sents:
        expect.add(m)
    assert ledger.counts == expect.counts
    assert meta["pass_stats"][0]["sentence_tokens"] == sum(len(tok.encode(s)) + 1 for s, _ in sents) == meta["tokens"]
    # Sıra karıştırılır: ilk yığının hücreleri kanonik (ad × ilişki × hâl) sırada ardışık bir pencere değildir.
    first = FakeLedger()
    run("shuffle", seed=7, prose_share=0.0, ledger=first, max_steps=1, batch_size=16)
    canon = [(m["name"], m["rel"], m["case"]) for _, m in sents]
    idx = sorted(canon.index(tuple(k.split("|")[:3])) for k in first.counts)
    assert len(idx) > 1 and idx != list(range(idx[0], idx[0] + len(idx)))

    # Başlatma tohumu: lr = 0 ile bir adım ağırlıkları değiştirmez; özet build_model(seed) ile aynıdır.
    meta = run("lr0", seed=5, lr=0.0, max_steps=1)
    fresh, _ = cm.build_model(tok.vocab_size, layers=2, d=32, heads=4, kv_heads=4, max_len=64, seed=5)
    assert meta["model_sha256"] == cm.model_sha256(fresh)

    # Gradyan kırpma uygulanır: çok küçük ve çok büyük eşik farklı ağırlık verir (aynı veri, aynı başlangıç).
    lo = run("clip_lo", grad_clip=1e-6, max_steps=2)
    hi = run("clip_hi", grad_clip=1e6, max_steps=2)
    assert lo["model_sha256"] != hi["model_sha256"]


def test_pretrain_refuses_p3_held_cell_and_bad_settings(world, tok, tmp_path):
    def leaky(w, pass_idx, rng):                  # P3'te tutulan bir hücreyi sızdıran kaynak
        name, case = sorted(w.p3_held)[0]
        return fake_pass_sentences(w, pass_idx, rng) + [
            (sentence(name, "A", case, w.facts[(name, "A")]), {"name": name, "rel": "A", "case": case,
                                                               "orth": "formal", "ctx": "none"})]
    with pytest.raises(ValueError, match="P3-held"):
        ct.pretrain(world, tok, PROSE, tmp_path / "leak", passes=1, seed=0, ledger=FakeLedger(),
                    **{**TRAIN_KW, "sentence_fn": leaky})
    assert not (tmp_path / "leak" / "model.pt").exists()
    with pytest.raises(ValueError, match="no prose chunks"):
        ct.pretrain(world, tok, [], tmp_path / "noprose", passes=1, seed=0, ledger=FakeLedger(), **TRAIN_KW)
    with pytest.raises(ValueError, match="batching"):
        ct.pretrain(world, tok, PROSE, tmp_path / "b", passes=1, seed=0, batching="sorted", **TRAIN_KW)


# ------------------------------------------------------------------------------------------- çözümleme (V6)
def _random_prefixes(n, seed):
    words = sorted({w.strip(".,") for t in PROSE for w in t.split()} | set(NAMES) | {"memleketi", "mesleği"})
    rnd = random.Random(seed)
    return [" ".join(rnd.choice(words) for _ in range(rnd.randint(2, 7))) for _ in range(n)]


def test_v6_cached_equals_uncached_with_and_without_fc2_hook(trained, world):
    model, tok = trained["model"], trained["tok"]
    prefixes = [p for p, _ in all_world_prefixes(world)][::37] + _random_prefixes(8, seed=1)
    ids = [[BOS_ID] + tok.encode(p) for p in prefixes]

    def decode_all(use_cache):
        return [cd.greedy_ids(model, x, max_new=6, use_cache=use_cache) for x in ids]

    base_c, base_u = decode_all(True), decode_all(False)
    assert base_c == base_u
    fc2 = model.layers[0].ffn.fc2
    M = torch.randn(fc2.out_features, fc2.in_features, generator=torch.Generator().manual_seed(0)) * 0.5
    handle = fc2.register_forward_hook(lambda mod, inp, out: out + inp[0] @ M.T)
    try:
        hook_c, hook_u = decode_all(True), decode_all(False)
    finally:
        handle.remove()
    assert hook_c == hook_u                      # yazım (kanca) sonrası da önbellekli = önbelleksiz
    assert hook_c != base_c                      # kanca çıktıları gerçekten değiştiriyor
    assert decode_all(True) == base_c            # kanca sökülünce eski çıktılar
    assert cd.greedy_text(model, tok, prefixes[0], max_new=6) == tok.decode(base_c[0], errors="replace")


def test_greedy_guards(trained, world):
    model, tok = trained["model"], trained["tok"]
    prefix = frame_prefix(world.names[2], "A", "abl", "formal", "P1")
    ids = [BOS_ID] + tok.encode(prefix)
    assert cd.greedy_ids(model, ids, max_new=0) == []
    with pytest.raises(ValueError):
        cd.greedy_ids(model, [], max_new=3)
    with pytest.raises(ValueError, match="positions"):
        cd.greedy_ids(model, [BOS_ID] * 60, max_new=8)                 # 67 konum > max_cache_len 64
    model.train()
    out = cd.greedy_ids(model, ids, max_new=8)
    assert model.training                                             # kip geri yüklenir
    model.eval()
    if EOS_ID in out:
        assert out[-1] == EOS_ID and out.count(EOS_ID) == 1           # durdurucu dahil, sonra durur
    for obj in OBJECTS["A"]:                                          # tam üretim ⇒ cevap o nesnedir
        if cd.greedy_reproduces(model, tok, prefix, obj):
            assert cd.greedy_answer(model, tok, prefix) == obj
    # Önbelleği kapalı bir modelde önbellekli çözümleme sessizce yanlış olmaz, hata verir.
    other, _ = cm.build_model(tok.vocab_size, layers=2, d=32, seed=1)
    for layer in other.layers:
        layer.attn.use_kv_cache = False
    with pytest.raises(RuntimeError, match="KV cache"):
        cd.greedy_ids(other, ids, max_new=3, use_cache=True)
    assert cd.greedy_ids(other, ids, max_new=3, use_cache=False)


def test_greedy_ids_is_immune_to_a_stale_kv_cache(trained, world):
    """Başka bir çağrının bıraktığı (daha uzun) önbellek içeriği çözümlemeyi etkilemez; çözümleme sonrası boştur."""
    model, tok = trained["model"], trained["tok"]
    ids = [BOS_ID] + tok.encode(frame_prefix(world.names[3], "B", "dat"))
    clean = cd.greedy_ids(model, ids, max_new=6)
    stale = [BOS_ID] + tok.encode(PROSE[3])
    assert len(ids) + 6 < len(stale) <= 64
    with torch.no_grad():
        model.eval()
        model(torch.tensor([stale]), use_cache=True)
    assert model.layers[0].attn.kv_cache.seen_tokens == len(stale)
    assert cd.greedy_ids(model, ids, max_new=6) == clean
    assert cd.greedy_ids(model, ids, max_new=6, use_cache=False) == clean
    assert all(layer.attn.kv_cache.seen_tokens == 0 for layer in model.layers)


# ------------------------------------------------------------------------------------------- toplu denetim (V8)
def test_v8_exact_check_ids_matches_true_greedy_with_right_padding(trained):
    model, tok = trained["model"], trained["tok"]
    rnd = random.Random(7)
    prefixes, targets, expect = [], [], []
    for k in range(12):
        p = [BOS_ID] + [rnd.randrange(4, tok.vocab_size) for _ in range(rnd.randint(1, 20))]   # farklı uzunluklar
        g = cd.greedy_ids(model, p, max_new=6, stop_ids=())
        for j in (1, 3, 6):                                    # açgözlü devamın önekleri: True
            prefixes.append(p), targets.append(g[:j]), expect.append(True)
        bad = list(g[:4])
        bad[-1] = 4 + (bad[-1] - 3) % (tok.vocab_size - 4)     # son token değişik: False
        prefixes.append(p), targets.append(bad), expect.append(False)
    assert cd.exact_check_ids(model, prefixes, targets, batch=7) == expect
    assert cd.exact_check_ids(model, prefixes, targets, batch=64) == expect
    six = [i for i, (t, e) in enumerate(zip(targets, expect)) if e and len(t) == 6]
    p6, t6 = [prefixes[i] for i in six], [targets[i] for i in six]
    assert six and all(cd.exact_check_ids(model, p6, t6))
    assert not any(cd.exact_check_ids(model, p6, t6, max_new=5))    # 5 tokenlik çözümleme 6 tokeni üretemez


def test_v8_exact_check_batch_agrees_with_greedy(trained, world):
    model, tok = trained["model"], trained["tok"]
    pairs = all_world_prefixes(world)[::9]
    rnd = random.Random(3)
    prefixes, objects = [], []
    for p, obj in pairs:                                         # doğru ve yanlış nesneler
        prefixes += [p, p]
        objects += [obj, rnd.choice([o for o in OBJECTS["A"] + OBJECTS["B"] if o != obj])]
    for p in _random_prefixes(10, seed=2):                       # rastgele önekler: modelin kendi cevabı + rastgele
        ans = cd.greedy_answer(model, tok, p)
        if ans:
            prefixes.append(p), objects.append(ans)
        prefixes.append(p), objects.append(rnd.choice(OBJECTS["A"]))
    fast = cd.exact_check_batch(model, tok, prefixes, objects, batch=16, max_new=8)
    # Gerçek açgözlü çözümleme (önek başına bir kez, 8 token): token düzeyi ve metin düzeyi ölçütler.
    greedy = {p: cd.greedy_ids(model, [BOS_ID] + tok.encode(p), max_new=8) for p in dict.fromkeys(prefixes)}
    token_level = [greedy[p][:len(tok.encode(" " + o + "."))] == tok.encode(" " + o + ".")
                   for p, o in zip(prefixes, objects)]
    text_level = [tok.decode(greedy[p], errors="replace").startswith(" " + o + ".") for p, o in zip(prefixes, objects)]
    assert fast == token_level
    assert fast == text_level
    assert 0 < sum(fast) < len(fast)                             # hem doğru hem yanlış durumlar sınandı
    sub = list(range(0, len(prefixes), 5))                       # başvuru işlevinin kendisi (alt küme)
    assert [cd.greedy_reproduces(model, tok, prefixes[i], objects[i]) for i in sub] == [fast[i] for i in sub]


# ------------------------------------------------------------------------------------------- checkpoint
def test_checkpoint_round_trip(trained, tmp_path):
    model, cfg, tok, meta = trained["model"], trained["cfg"], trained["tok"], trained["meta"]
    fresh, fresh_cfg = cm.build_model(**{k: cfg["build"][k] for k in
                                         ("vocab_size", "layers", "d", "heads", "kv_heads", "max_len", "seed")})
    assert fresh_cfg == cfg                                       # cfg kurulum argümanlarından yeniden üretilir
    assert cfg["arch"]["layers"][0]["num_heads"] == 4 and cfg["arch"]["layers"][0]["ffn_dim"] == 256
    assert cfg["config"]["dropout"] == 0.0 and cfg["arch"]["tied_embeddings"]
    path = tmp_path / "again" / "model.pt"
    cm.save_checkpoint(path, model, cfg, tok, meta)
    m2, cfg2, tok2, meta2 = cm.load_checkpoint(path)
    assert cm.model_sha256(m2) == cm.model_sha256(model) == meta["model_sha256"]
    assert cfg2 == cfg and meta2 == meta and tok2.identity() == tok.identity() and not m2.training
    ids = [BOS_ID] + tok.encode("Dorhan'ın memleketi")
    assert cd.greedy_ids(m2, ids) == cd.greedy_ids(model, ids)

    class Named(torch.nn.Module):            # özet anahtar adlarını ve şekilleri de kapsar (yalnız baytları değil)
        def __init__(self, second, shape=(2, 3)):
            super().__init__()
            self.a = torch.nn.Parameter(torch.ones(2))
            setattr(self, second, torch.nn.Parameter(torch.zeros(shape)))
    assert cm.model_sha256(Named("b")) == cm.model_sha256(Named("b"))
    assert cm.model_sha256(Named("b")) != cm.model_sha256(Named("c"))
    assert cm.model_sha256(Named("b")) != cm.model_sha256(Named("b", shape=(3, 2)))


def _copy_ckpt(src: pathlib.Path, dst_dir: pathlib.Path) -> pathlib.Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst_dir / src.name)
    shutil.copy(cm.tokenizer_path(src), cm.tokenizer_path(dst_dir / src.name))
    return dst_dir / src.name


def test_checkpoint_config_mismatch_guard(trained, tmp_path):
    ck = torch.load(trained["ckpt"], map_location="cpu", weights_only=True)
    # Tehlike: 2 başlı bir model aynı şekilli ağırlıkları strict=True ile HATASIZ yükler ama başka işlev hesaplar.
    two_heads, _ = cm.build_model(ck["cfg"]["build"]["vocab_size"], layers=2, d=32, heads=2, kv_heads=2, seed=0)
    two_heads.load_state_dict(ck["state_dict"], strict=True)
    x = torch.tensor([[BOS_ID] + trained["tok"].encode("Dorhan'ın memleketi")])
    with torch.no_grad():
        assert not torch.allclose(two_heads.eval()(x)[0], trained["model"](x)[0])

    def tampered(sub, edit):
        c = torch.load(trained["ckpt"], map_location="cpu", weights_only=True)
        edit(c["cfg"])
        path = _copy_ckpt(trained["ckpt"], tmp_path / sub)
        torch.save(c, path)
        return path

    def heads_in_config_and_build(cfg):           # yeniden kurulum 2 baş verir; kayıtlı mimari 4 baş der
        cfg["config"]["num_heads"] = cfg["config"]["num_kv_heads"] = 2
        cfg["build"]["heads"] = cfg["build"]["kv_heads"] = 2

    def heads_in_config_only(cfg):                # yapılandırma ile kurulum argümanları çelişir
        cfg["config"]["num_heads"] = cfg["config"]["num_kv_heads"] = 2

    def soft_cap_only(cfg):                       # şekli değiştirmeyen başka bir işlev farkı
        cfg["config"]["logit_soft_cap"] = 10.0

    with pytest.raises(ValueError, match="differs from the saved one"):
        cm.load_checkpoint(tampered("a", heads_in_config_and_build))
    with pytest.raises(ValueError, match="requested architecture"):
        cm.load_checkpoint(tampered("b", heads_in_config_only))
    with pytest.raises(ValueError, match="differs from the saved one"):
        cm.load_checkpoint(tampered("c", soft_cap_only))


def test_checkpoint_tokenizer_digest_and_format_guards(trained, tmp_path, tok):
    # Başka bir tokenizer (aynı sözlük boyutu bile olsa) reddedilir.
    path = _copy_ckpt(trained["ckpt"], tmp_path / "tok")
    ByteLevelBPE.train(PROSE * 3, vocab_size=tok.vocab_size).save(cm.tokenizer_path(path))
    with pytest.raises(ValueError, match="tokenizer identity mismatch"):
        cm.load_checkpoint(path)
    path = _copy_ckpt(trained["ckpt"], tmp_path / "notok")
    cm.tokenizer_path(path).unlink()
    with pytest.raises(FileNotFoundError):
        cm.load_checkpoint(path)
    # Ağırlık değişip özet aynı kalırsa reddedilir.
    c = torch.load(trained["ckpt"], map_location="cpu", weights_only=True)
    c["state_dict"]["layers.0.ffn.fc2.weight"] = c["state_dict"]["layers.0.ffn.fc2.weight"] + 1e-3
    path = _copy_ckpt(trained["ckpt"], tmp_path / "digest")
    torch.save(c, path)
    with pytest.raises(ValueError, match="digest"):
        cm.load_checkpoint(path)
    # Eksik anahtar strict yüklemede reddedilir.
    c = torch.load(trained["ckpt"], map_location="cpu", weights_only=True)
    del c["state_dict"]["output_norm.scale"]
    path = _copy_ckpt(trained["ckpt"], tmp_path / "strict")
    torch.save(c, path)
    with pytest.raises(RuntimeError):
        cm.load_checkpoint(path)
    # Biçim dışı dosya ve modeli tanımlamayan cfg ile kayıt reddedilir.
    bad = tmp_path / "bad.pt"
    torch.save({"state_dict": c["state_dict"]}, bad)
    with pytest.raises(ValueError, match="not a cev11-checkpoint"):
        cm.load_checkpoint(bad)
    other_model, other_cfg = cm.build_model(tok.vocab_size, layers=1, d=32, seed=0)
    with pytest.raises(ValueError, match="does not describe"):
        cm.save_checkpoint(tmp_path / "x.pt", trained["model"], other_cfg, tok, {})


# ------------------------------------------------------------------------------------------- kapılar
def test_evaluate_gates_structure_and_consistency(trained, world):
    model, tok = trained["model"], trained["tok"]
    res = ct.evaluate_gates(model, tok, world, HELDOUT, n_cells=20, seed=3, classify=fake_classify, min_editable=1)
    for g in ("G1", "G3", "G4", "G5", "G6", "G7"):
        assert {"value", "threshold", "pass"} <= set(res[g]), g
    assert res["G1"]["pass"] and res["G1"]["n_world"] == len(all_world_prefixes(world))
    assert res["boundary"]["pass"]
    assert res["G3"]["threshold"] == 0.95 and res["G7"]["threshold"] == 1
    assert res["G3"]["n"] == res["G5"]["n"] == res["G6"]["n"] == 20
    held = sorted(world.p3_held)                                  # G4 = doğrudan tam denetimle aynı
    direct = cd.exact_check_batch(model, tok, [frame_prefix(n, "A", c) for n, c in held],
                                  [world.facts[(n, "A")] for n, _ in held], max_new=8)
    assert res["G4"]["n"] == len(held) and res["G4"]["n_correct"] == sum(direct)
    probe = [n for n in world.names if world.partition[n] == "probe"]
    assert set(res["editable"]) <= set(probe) and len(res["editable"]) == res["G7"]["value"]
    assert res["editable"] == sorted(res["editable"]) and res["G7"]["n_probe"] == len(probe)
    for n in res["editable"]:                                     # düzenlenebilir: 7 biçimsel kalıbın hepsi doğru
        assert all(cd.exact_check_batch(model, tok, [frame_prefix(n, "A", c) for c in CASES],
                                        [world.facts[(n, "A")]] * len(CASES)))
    cls = res["G5"]["by_class"]
    assert sum(v["n"] for k, v in cls.items() if k != "NONIDENT") == 20
    assert res["all_pass"] == (all(res[g]["pass"] for g in ("G1", "G3", "G4", "G5", "G6", "G7"))
                               and res["boundary"]["pass"])
    json.dumps(res)                                               # JSON'a yazılabilir
    with pytest.raises(ValueError, match="need 1000 cells"):
        ct.evaluate_gates(model, tok, world, HELDOUT, n_cells=1000, classify=fake_classify)


def _prefix_meta(world):
    """Önek dizgesi -> (ad, ilişki, hâl, yazım, bağlam). NOM'un gündelik dizgesi biçimselle aynıdır: 'formal'."""
    out = {}
    for n in world.names:
        for r in sorted(world.objects):
            for c in CASES:
                for o in ("formal", "informal"):
                    for x in CONTEXTS:
                        out.setdefault(frame_prefix(n, r, c, o, x), (n, r, c, "formal" if c == "nom" else o, x))
    return out


def test_gate_queries_follow_protocol_cells_orthography_and_context(world, tok, monkeypatch):
    """exact_check_batch yerine kural tabanlı bir kâhin: her kapının sorduğu yazım/bağlam/hücreler ve G7 eşikleri
    (7/7 biçimsel, ≥ 3/6 gündelik, A ilişkisi, bağlamsız) modelden ve örnekleme sırasından bağımsız sınanır."""
    meta_of = _prefix_meta(world)
    held = set(world.p3_held)
    probe = sorted(n for n in world.names if world.partition[n] == "probe")
    max_news = set()

    def run(rule):
        def oracle(model, tok_, prefixes, objects, batch=64, max_new=None):
            max_news.add(max_new)
            out = []
            for p, o in zip(prefixes, objects):
                n, r, c, orth, ctx = meta_of[p]
                assert o == world.facts[(n, r)]                   # kapılar her zaman doğru nesneyi sorar
                out.append(bool(rule(n, r, c, orth, ctx)))
            return out
        monkeypatch.setattr(cd, "exact_check_batch", oracle)
        return ct.evaluate_gates(None, tok, world, HELDOUT, n_cells=20, seed=3, classify=fake_classify,
                                 min_editable=1)

    def rates(res):
        return tuple(res[g]["n_correct"] / res[g]["n"] for g in ("G3", "G4", "G5", "G6"))

    res = run(lambda n, r, c, o, x: o == "formal" and x == "none")      # G3, G4: biçimsel bağlamsız
    assert rates(res) == (1.0, 1.0, 0.0, 0.0)
    assert res["editable"] == [] and res["G7"]["n_formal_all7"] == len(probe)
    res = run(lambda n, r, c, o, x: o == "informal" and x == "none")    # G5: gündelik bağlamsız, çekimli
    assert rates(res) == (0.0, 0.0, 1.0, 0.0)
    res = run(lambda n, r, c, o, x: o == "formal" and x != "none")      # G6: biçimsel, P1/P2
    assert rates(res) == (0.0, 0.0, 0.0, 1.0) and set(res["G6"]["by_ctx"]) <= {"P1", "P2"}
    res = run(lambda n, r, c, o, x: r == "A" and (n, c) in held)        # tutulan hücre yalnız G4'te
    assert rates(res) == (0.0, 1.0, 0.0, 0.0)
    res = run(lambda n, r, c, o, x: c == "nom")                         # G5'te NOM yok
    assert res["G5"]["n_correct"] == 0 and "nom" not in res["G5"]["by_case"]

    pattern = {probe[0]: (7, 3), probe[1]: (7, 2), probe[2]: (6, 6), probe[3]: (7, 6)}   # (biçimsel, gündelik)

    def g7_rule(n, r, c, o, x):
        if n not in pattern or r != "A" or x != "none":
            return False
        k_formal, k_informal = pattern[n]
        return (CASES.index(c) < k_formal) if o == "formal" else (INFLECTED.index(c) < k_informal)
    res = run(g7_rule)
    assert res["editable"] == sorted([probe[0], probe[3]]) and res["G7"]["n_formal_all7"] == 3
    assert max_news == {ct.GATE_MAX_NEW}
