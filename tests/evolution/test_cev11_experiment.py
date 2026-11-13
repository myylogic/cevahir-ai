"""CEV-11 edit engine plumbing on a tiny random model (fast; tmp_path only).

Checks that every arm runs end to end, produces a schema-valid record, and leaves the base model bit-identical.
No learning claim: the model is random and untrained.
"""
import json
import pathlib
import random
import sys

import pytest
import torch

STUDY = pathlib.Path(__file__).resolve().parents[2] / "research" / "living_learning_paradigm_2026_11_13"
sys.path[:0] = [str(STUDY)]

from cev11 import experiment as E  # noqa: E402
from cev11 import probes as P  # noqa: E402
from cev11.model import build_model, model_sha256  # noqa: E402
from cev11.world import make_world, sentence  # noqa: E402
from cev11.tokenizers import train_tokenizer  # noqa: E402
import analyze as A  # noqa: E402


@pytest.fixture(scope="module")
def tiny_base():
    torch.set_num_threads(1)
    world = make_world(11, n_names=120)
    rnd = random.Random(0)
    texts = [sentence(n, r, c, world.facts[(n, r)], o, x)
             for n in world.names[:60] for r in ("A", "B", "C") for c in ("nom", "gen", "abl")
             for o in ("formal", "informal") for x in ("none", "P1")]
    rnd.shuffle(texts)
    # every Turkish letter frequent enough to become a single token (strict offsets need whole characters)
    letters = " ".join(ch * 3 for ch in "çğıöşüÇĞİÖŞÜâîû") + " "
    tok = train_tokenizer(texts + ["Hava bugün çok güzeldi.", "Kapı yavaşça açıldı.", letters] * 20,
                          vocab_size=900, require_full_vocab=False)
    model, cfg = build_model(tok.vocab_size, layers=2, d=32, heads=4, kv_heads=4, max_len=64, seed=0)
    model.eval()
    base = E.Base(model, cfg, tok, {}, world, texts[:200], texts[200:220], pathlib.Path("."), model_sha256(model))
    return base


def test_all_arms_run_validate_and_restore(tiny_base):
    base = tiny_base
    w = base.world
    probe = sorted(n for n, p in w.partition.items() if p == "probe")
    items = P.edit_items(w, probe, seed=2, per_case=1)
    item = next(it for it in items if it["lived_case"] == "abl")
    answers = E.Answers(base)
    inst = E.Installer(base)
    prep = E.prepare_item(base, item, answers, seed=3)
    base.layer_cache["bpb_base"] = E.bits_per_byte(base, base.prose_held[:3])
    hp_key = {"step": 2.0, "cap_mult": 8, "max_steps": 5}
    for arm, layer, hp in [("BASE", None, {}), ("STD", 0, hp_key), ("ROOT", 0, hp_key), ("LAST", 0, hp_key),
                           ("PARA-ORTH", 0, hp_key), ("NORM", 0, hp_key),
                           ("FT-L", 0, {"lr": 1e-2, "eps": 0.02, "steps": 3}),
                           ("FT-ALL", None, {"lr": 1e-3, "steps": 3})]:
        rec = E.run_arm(base, inst, answers, prep, arm, layer, hp, world_seed=1000, run="pilot", config="t",
                        memo={}, with_tau=True, bpb_texts=base.prose_held[:3] if arm != "BASE" else None)
        A.validate_edit_record(rec)
        assert model_sha256(base.model) == base.base_sha, f"{arm} left the base model changed"
    # a base answer must never be computed while an edit is installed
    answers.locked = True
    with pytest.raises(RuntimeError):
        answers("Bu önek daha önce sorulmadı: memleketi")
    answers.locked = False


def test_root_key_is_identical_across_formal_frames(tiny_base):
    """V2 in practice: the ROOT-site key of a sentence-initial name does not depend on the case suffix."""
    from cev11.keysites import key_at, sites
    from cev11.world import frame_prefix
    base = tiny_base
    name = sorted(base.world.names)[3]
    keys = []
    for case in ("nom", "gen", "dat", "acc", "loc", "abl", "ins"):
        pre = frame_prefix(name, "A", case)
        keys.append(key_at(base.model, 0, base.tok, pre, sites(base.tok, pre, name)["ROOT"]))
    for k in keys[1:]:
        assert torch.equal(k, keys[0])
