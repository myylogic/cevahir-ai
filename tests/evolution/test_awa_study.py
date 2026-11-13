# -*- coding: utf-8 -*-
"""AWA-3 study pipeline (research/attention_order_law_2026_11_13): generator, arms, order law, drivers, statistics.

* VarDeref v2: paired strata, equal chance floors, frozen streams.
* Arms: asserted parameter counts, answer-row forwards against the full forward, analytic cost model.
* The order law with hand-set weights: lower-triangular operators resolve a chain iff it is dependency-sorted
  (exactly one of the (k - 1)! alias orders); the query-window walk resolves every order.
* Trainer: schedule, parameter groups, the recipe against an independent training loop (weights bit for bit),
  bitwise determinism, bitwise resumption, extension.
* Evaluation records and the preregistered statistics / decisions (exact tests, Hodges-Lehmann, T4).
* Phase-0 selection, extension ledger, freeze / manifest / rerun tooling, job files, command-line drivers.

CPU only and fast: models are built at full study size (551k parameters) but trained for a few steps on tiny
batches. Everything is written under tmp_path; nothing is written into the repository.
"""
from __future__ import annotations

import copy
import importlib.util
import itertools
import json
import logging
import math
import os
import pathlib
import random
import shutil
import sys

import pytest
import torch

sys.dont_write_bytecode = True
REPO = pathlib.Path(__file__).resolve().parents[2]
STUDY = REPO / "research" / "attention_order_law_2026_11_13"
for _p in (str(STUDY), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from awa3 import analysis, arms, evaluate, handset, jobs, layout, manifest, phases, stats, task, train  # noqa: E402


@pytest.fixture(autouse=True)
def _single_thread_quiet():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(previous)
    torch.set_num_threads(threads)


def tiny_recipe(steps=6, **kw):
    base = dict(lr=1e-3, steps=steps, schedule_steps=3000, batch=8, val_every=3, ckpt_every=3, val_items=8,
                log_every=1)
    base.update(kw)
    return train.Recipe(**base)


def test_study_scripts_do_not_shadow_modules_of_other_study_folders():
    """This file puts the study folder first on sys.path. A top-level module name shared with another study
    (CEV-11 has analyze.py and verify.py, which its tests import by bare name) would make that import depend on
    the collection order of the test files."""
    def top_level(folder):
        return {q.stem for q in folder.glob("*.py")} | {q.name for q in folder.iterdir()
                                                         if q.is_dir() and (q / "__init__.py").is_file()}

    mine = top_level(STUDY)
    assert mine == {"awa3", "make_jobs", "run_analyze", "run_eval", "run_extend", "run_phase0_select", "run_train",
                    "run_verify"}
    for other in sorted(q for q in (REPO / "research").iterdir() if q.is_dir() and q != STUDY):
        assert not (mine & top_level(other)), (other.name, sorted(mine & top_level(other)))
    assert str(STUDY) in sys.path
    for name in ("analyze", "verify"):
        spec = importlib.util.find_spec(name)
        assert spec is None or STUDY not in pathlib.Path(spec.origin).resolve().parents, (name, spec.origin)


# ============================================================================== task: VarDeref v2
def test_vocabulary_and_length_are_fixed():
    assert task.VOCAB == 72 and task.SEQ_LEN == 67 and task.M_FACTS == 16
    assert (task.EQ, task.SEMI, task.QM, task.BOS, task.EOS, task.PAD) == (64, 65, 66, 67, 68, 69)
    assert task.STRATA == ("sorted", "oneswap", "anti", "shuffled")
    assert [task.eval_size(k) for k in task.TEST_DEPTHS] == [500, 500, 500, 500, 500, 1000, 1000, 500, 500]


def test_strata_invariants():
    rng = random.Random("unit|strata")
    for _ in range(300):
        k = rng.randint(1, 12)
        p = task.base_program(rng, k)
        lhs = [f[2] for f in p["facts"]]
        assert len(lhs) == len(set(lhs)) == task.M_FACTS, "unique LHS"
        assert task.interpret(p) == (p["answer"], k), "acyclic, exact depth, answer = interpreter"
        assert len(set(p["lits"])) == len(p["lits"]), "distinct literal per chain"
        qa = task.query_aliases(p)
        base = task.stratum_order(p, "shuffled")
        L = p["n_lit"]
        assert all(p["facts"][fi][3] >= task.N_VAR for fi in base[:L]), "literal block first"
        assert all(p["facts"][fi][3] < task.N_VAR for fi in base[L:]), "alias block after it"
        for s in task.STRATA:
            od = task.stratum_order(p, s)
            assert sorted(od) == list(range(task.M_FACTS))
            assert od[:L] == base[:L], "literal block identical in every stratum"
            assert all(od[q] == base[q] for q in range(task.M_FACTS) if base[q] not in qa), "distractors fixed"
            toks = task.tokens(p, od)
            assert len(toks) == task.SEQ_LEN and toks[0] == task.BOS and toks[-2:] == [task.QM, p["qvar"]]
            inv = task.out_of_order_adjacent(p, od)
            assert task.followable_lower_triangular(p, od) == (inv == 0)
            if s == "sorted":
                assert inv == 0
            if k >= 3 and s == "anti":
                assert inv == k - 2
            if k >= 3 and s == "oneswap":
                assert inv == 1
                pos = {fi: q for q, fi in enumerate(od)}
                ranks = sorted(range(k - 1), key=lambda m: -pos[qa[m]])
                assert sum(r != m for r, m in zip(ranks, range(k - 1))) == 2, "one adjacent transposition"
            if k <= 2:
                assert od == base, "the strata coincide at k <= 2"
            spans = task.chain_spans(p, od)
            assert len(spans) == k and toks[spans[0].start] == p["qvar"] and toks[spans[-1].start + 2] == p["answer"]
            types = task.position_types(p, od)
            assert len(types) == task.SEQ_LEN and types.count("semi") == 16
            assert types.count("rhs_lit") == L and types.count("rhs_var") == 16 - L
    with pytest.raises(ValueError):
        task.stratum_order(p, "reverse")
    with pytest.raises(ValueError):
        task.base_program(rng, 17)


@pytest.mark.parametrize("k", [3, 6, 8])
def test_chance_floors_are_identical_in_all_four_strata(k):
    progs = task.eval_set(k)[:300]
    rows = {s: task.heuristic_floors(progs, s) for s in task.STRATA}
    for s in task.STRATA:
        for heuristic in ("uniform", "last", "first"):          # guesses that ignore the chain
            assert rows[s][heuristic] == rows["shuffled"][heuristic]
    assert rows["sorted"]["lowertri_followable"] == 1.0
    assert rows["oneswap"]["lowertri_followable"] == rows["anti"]["lowertri_followable"] == 0.0


def test_frozen_streams_floors_and_no_overlap_with_training():
    # digests recorded with CPython 3.13; a change means the frozen items changed
    assert task.set_sha256(task.item_set("test", 6, 50)) == \
        "c1e1eb71f8c54bb0e2f4c67776fc3c7cb97d6e481323b418637095dfe035edf2"
    assert task.set_sha256(task.item_set("val", 3, 20)) == \
        "c6db146216b9e219f9d586f7a6c280327d3979cc31c203884b516d684f7b20dd"
    assert task.item_set("test", 6, 5) == task.eval_set(6)[:5]
    assert task.item_set("val", 6, 5) != task.item_set("test", 6, 5)
    assert len(task.val_set(3)) == 600 and len(task.val_set(6)) == 300
    assert round(task.floor(6), 3) == 0.161 and round(task.floor(8), 3) == 0.193      # PROTOCOL section 5
    assert round(task.followable_fraction(3), 3) == 0.546 and task.followable_fraction(8) == 0.0
    hashes = task.frozen_set_hashes()
    assert len(hashes) == 18 and "test|k=6|n=1000" in hashes and "val|k=3|n=600" in hashes
    assert task.train_overlap(0, 3000) == 0
    x1, y1 = task.make_batch(random.Random("train|7"), 5)
    x2, y2 = task.make_batch(random.Random("train|7"), 5)
    assert torch.equal(x1, x2) and torch.equal(y1, y2) and x1.shape == (5, 67)
    assert ((y1 >= task.N_VAR) & (y1 < task.N_VAR + task.N_LIT)).all()


# ===================================================================================== the order law
def test_order_law_with_hand_set_weights():
    table = handset.order_law_table(depths=range(1, 9), trials=40, K=8)
    for (k, stratum), row in table.items():
        assert row["AWA"] == 1.0, (k, stratum, row)                       # any order, any depth <= K + 1
        for name in ("AWA-own", "ChaCAL"):                                 # lower-triangular operators
            assert row[name] == row["followable"], (k, stratum, row)       # succeed iff dependency-sorted
            if stratum == "sorted":
                assert row[name] == 1.0
            if stratum in ("oneswap", "anti") and k >= 3:
                assert row[name] == 0.0
    assert 0.25 < table[(3, "shuffled")]["ChaCAL"] < 0.75
    assert table[(8, "shuffled")]["ChaCAL"] == 0.0


@pytest.mark.parametrize("k", [3, 4, 5])
def test_order_law_exactly_one_of_the_factorial_orders_is_followable(k):
    """Under uniform shuffling of the k - 1 aliases the success probability is exactly 1 / (k - 1)!."""
    for index in range(3):
        prog = task.item_set("unit-factorial", k, 3)[index]
        base = task.stratum_order(prog, "shuffled")
        aliases = task.query_aliases(prog)
        slots = sorted(base.index(fi) for fi in aliases)
        solved = {name: 0 for name in handset.OPERATORS}
        n_orders = 0
        for perm in itertools.permutations(aliases):
            order = list(base)
            for slot, fi in zip(slots, perm):
                order[slot] = fi
            n_orders += 1
            result = handset.resolve(prog, order, K=8)
            assert result["AWA-own"] == result["ChaCAL"] == task.followable_lower_triangular(prog, order)
            for name, ok in result.items():
                solved[name] += ok
        assert n_orders == math.factorial(k - 1)
        assert solved["AWA"] == n_orders
        assert solved["AWA-own"] == solved["ChaCAL"] == 1
        assert solved["ChaCAL"] / n_orders == handset.sorted_probability(k)


def test_shuffled_training_stream_is_dependency_sorted_with_probability_one_over_factorial():
    rng = random.Random("unit|sorted-share")
    for k in (3, 4, 5):
        n = 4000
        share = sum(task.followable_lower_triangular(p, p["order"])
                    for p in (task.base_program(rng, k) for _ in range(n))) / n
        expected = handset.sorted_probability(k)
        assert abs(share - expected) < 4 * math.sqrt(expected * (1 - expected) / n), (k, share, expected)
    assert handset.sorted_probability(1) == handset.sorted_probability(2) == 1.0


def test_walk_budget_limits_the_resolvable_depth():
    prog = task.item_set("unit-budget", 6, 1)[0]
    order = task.stratum_order(prog, "anti")
    assert handset.resolve(prog, order, K=5)["AWA"] is True        # depth 6 needs K >= 5 hops after hop 1
    assert handset.resolve(prog, order, K=3)["AWA"] is False


# ============================================================================================== arms
@pytest.fixture(scope="module")
def built():
    torch.set_num_threads(1)
    logging.disable(logging.CRITICAL)
    try:
        return {arm: arms.build(arm, 0) for arm in arms.ARMS}
    finally:
        logging.disable(logging.NOTSET)


def test_arm_parameter_counts_and_architecture(built):
    assert {arm: spec["params"] for arm, spec in arms.ARMS.items()} == {
        "AWA-2L": 551_308, "AWA-own-2L": 551_308, "ChaCAL-2L": 551_300, "LOOPED-2L": 551_296,
        "ARLOOP-2L": 551_296, "STD-2L-pm": 551_296}
    for arm, (model, config) in built.items():
        assert arms.count_parameters(model) == arms.ARMS[arm]["params"]
        arms.check_architecture(model, arm)
        operator = arms.ARMS[arm]["operator"]
        assert config.get("attention_operator", "standard") == operator
        assert [layer.attn.attention_operator for layer in model.layers] == ["standard", operator]
    spread = [spec["params"] for spec in arms.ARMS.values()]
    assert max(spread) - min(spread) == 12, "the arms are parameter-matched to within 12 parameters"
    assert sum(spec["seeds"] for spec in arms.ARMS.values()) == 26
    with pytest.raises(ValueError, match="does not match the protocol"):
        arms.check_architecture(built["STD-2L-pm"][0], "AWA-2L")
    with pytest.raises(ValueError, match="unknown arm"):
        arms.build("STD-4L", 0)


def test_arms_share_initial_weights_for_a_seed_and_differ_across_seeds(built):
    state = {arm: model.state_dict() for arm, (model, _) in built.items()}
    for arm in ("ChaCAL-2L", "LOOPED-2L", "ARLOOP-2L"):                  # same shapes as STD-2L-pm
        assert all(torch.equal(v, state[arm][k]) for k, v in state["STD-2L-pm"].items())
    assert all(torch.equal(v, state["AWA-own-2L"][k]) for k, v in state["AWA-2L"].items())
    # the walk arms have another ffn_dim: only weights drawn before the first FFN coincide with the other group
    for key in ("embedding.embedding.weight", "layers.0.attn.query_proj.weight", "layers.0.attn.value_proj.weight"):
        assert torch.equal(state["AWA-2L"][key], state["STD-2L-pm"][key])
    assert not torch.equal(state["AWA-2L"]["layers.1.attn.query_proj.weight"],
                           state["STD-2L-pm"]["layers.1.attn.query_proj.weight"])
    other, _ = arms.build("STD-2L-pm", 1)
    assert not torch.equal(other.state_dict()["embedding.embedding.weight"], state["STD-2L-pm"]["embedding.embedding.weight"])
    torch.manual_seed(123)
    before = torch.rand(1)
    torch.manual_seed(123)
    arms.build("STD-2L-pm", 5)
    assert torch.equal(torch.rand(1), before), "building does not consume the caller's random stream"


def test_answer_logits_equal_the_full_forward_and_the_loop_reductions(built):
    x, _ = task.make_batch(random.Random("unit|forward"), 4)
    rows = torch.tensor([task.ANSWER_POS])
    with torch.no_grad():
        std_model = built["STD-2L-pm"][0].eval()
        std = arms.answer_logits(std_model, "STD-2L-pm", x)
        assert float((std - std_model(x)[0][:, -1]).abs().max()) < 2e-6
        for arm in ("AWA-2L", "AWA-own-2L", "ChaCAL-2L"):
            model = built[arm][0].eval()
            budget = 4 if arms.ARMS[arm]["forward"] == "walk" else None
            assert float((arms.answer_logits(model, arm, x, budget) - model(x)[0][:, -1]).abs().max()) < 2e-6
        awa = built["AWA-2L"][0]
        axis = arms.answer_logits(awa, "AWA-2L", x, 6, return_all=True)
        assert len(axis) == 7
        assert float((axis[3] - arms.answer_logits(awa, "AWA-2L", x, 3)).abs().max()) < 1e-6
        # LOOPED and ARLOOP share STD-2L-pm's weights at seed 0: T = 1 is the standard model
        looped, arloop = built["LOOPED-2L"][0].eval(), built["ARLOOP-2L"][0].eval()
        assert float((arms.answer_logits(looped, "LOOPED-2L", x, 1) - std).abs().max()) < 1e-6
        assert float((arms.answer_logits(arloop, "ARLOOP-2L", x, 1) - std).abs().max()) < 1e-6
        all_t = arms.answer_logits(looped, "LOOPED-2L", x, 3, return_all=True)
        assert len(all_t) == 3
        for T in (1, 2, 3):
            assert float((all_t[T - 1] - arms.answer_logits(looped, "LOOPED-2L", x, T)).abs().max()) < 1e-5
        h = looped.layers[0](arms._embed(looped, x), causal_mask=True)[0]
        for _ in range(3):
            h = looped.layers[1](h, causal_mask=True)[0]
        assert float((looped._output_head(h[:, -1]) - all_t[2]).abs().max()) < 1e-5
        ar = arms.answer_logits(arloop, "ARLOOP-2L", x, 2, return_all=True)
        h = arloop.layers[0](arms._embed(arloop, x), causal_mask=True)[0]
        y = arms.layer_forward_rows(arloop.layers[1], h, rows)
        h2 = h.clone()
        h2[:, -1] = y[:, 0]                                        # only the answer row changes
        manual = arloop._output_head(arms.layer_forward_rows(arloop.layers[1], h2, rows))[:, 0]
        assert float((ar[1] - manual).abs().max()) < 1e-6
        assert float((ar[1] - arms.answer_logits(arloop, "ARLOOP-2L", x, 2)).abs().max()) < 1e-6
    with pytest.raises(ValueError, match="needs a budget"):
        arms.answer_logits(awa, "AWA-2L", x)
    with pytest.raises(ValueError, match="no budget axis"):
        arms.answer_logits(std_model, "STD-2L-pm", x, return_all=True)
    with pytest.raises(ValueError, match="T >= 1"):
        arms.answer_logits(looped, "LOOPED-2L", x, 0)


def test_analytic_cost_model():
    n, D, H, V = 67, 128, 4, 72
    std = arms.macs_per_answer("STD-2L-pm")
    full = n * (4 * D * D + 2 * n * D + 3 * D * 534)
    row = 2 * D * D + 2 * n * D + 3 * D * 534
    assert std == full + 2 * n * D * D + row + V * D
    assert arms.macs_per_answer("LOOPED-2L", 1) == arms.macs_per_answer("ARLOOP-2L", 1) == std
    assert arms.macs_per_answer("LOOPED-2L", 12) - std == 11 * full
    assert arms.macs_per_answer("ARLOOP-2L", 12) - std == 11 * (2 * D * D + row)
    hop = arms.macs_per_answer("AWA-2L", 13) - arms.macs_per_answer("AWA-2L", 12)
    assert hop == H * n * n
    assert arms.macs_per_answer("AWA-2L", 12) == arms.macs_per_answer("AWA-own-2L", 12)
    # a walk at K = 12 costs about a tenth of twelve looped blocks and about as much as ARLOOP at T = 12
    assert arms.macs_per_answer("LOOPED-2L", 12) > 9 * arms.macs_per_answer("AWA-2L", 12)
    assert 0.8 < arms.macs_per_answer("AWA-2L", 12) / arms.macs_per_answer("ARLOOP-2L", 12) < 1.25
    assert arms.macs_per_answer("ChaCAL-2L") > std
    with pytest.raises(ValueError):
        arms.macs_per_answer("ARLOOP-2L", 0)


# =========================================================================================== trainer
def test_learning_rate_schedule_and_parameter_groups(built):
    recipe = train.Recipe(lr=1e-3)
    assert recipe.steps == recipe.schedule_steps == 3000 and recipe.batch == 64 and recipe.warmup == 100
    assert tuple(recipe.betas) == (0.9, 0.98) and recipe.eps == 1e-8 and recipe.weight_decay == 0.01
    assert recipe.clip == 1.0 and tuple(recipe.depths) == (1, 2, 3) and tuple(recipe.budgets) == (3, 4, 5)
    assert train.lr_at(1, recipe) == pytest.approx(1e-5) and train.lr_at(100, recipe) == pytest.approx(1e-3)
    assert train.lr_at(1550, recipe) == pytest.approx(1e-3 * (0.1 + 0.9 * 0.5))
    assert train.lr_at(3000, recipe) == pytest.approx(1e-4)
    assert train.lr_at(4500, recipe) == pytest.approx(1e-4), "extension steps stay at the floor"
    pilot = phases.recipe_for("0", 2e-3)
    assert pilot.steps == 800 and pilot.schedule_steps == 3000
    assert train.lr_at(800, pilot) == train.lr_at(800, train.Recipe(lr=2e-3)), "a pilot is the start of a main run"
    assert phases.recipe_for("1", 1e-3) == train.Recipe(lr=1e-3)
    for phase in ("0", "1"):
        with pytest.raises(ValueError, match="no overrides"):
            phases.recipe_for(phase, 1e-3, steps=10)
    model = built["AWA-2L"][0]
    decay, no_decay = train.param_groups(model, 0.01)
    assert decay["weight_decay"] == 0.01 and no_decay["weight_decay"] == 0.0
    names = {id(p): n for n, p in model.named_parameters()}
    undecayed = {names[id(p)] for p in no_decay["params"]}
    assert {"layers.1.attn.walk_rho", "layers.1.attn.walk_tau", "layers.1.attn.walk_gate.bias"} <= undecayed
    assert all(p.ndim >= 2 for p in decay["params"]) and all(p.ndim < 2 for p in no_decay["params"])
    assert "layers.1.attn.walk_pointer.weight" in {names[id(p)] for p in decay["params"]}
    total = sum(p.numel() for g in (decay, no_decay) for p in g["params"])
    assert total == arms.count_parameters(model), "every parameter once (tied weights are not duplicated)"


def reference_training(arm, seed, steps, *, lr, batch, warmup, horizon, floor, betas, eps, weight_decay, clip,
                       vector_decay=0.0):
    """PROTOCOL section 7 written out a second time, without awa3/train.py: AdamW with the given betas and eps;
    weight decay on matrices only (``vector_decay`` is 0 in the protocol); linear warm-up, cosine to ``floor`` x lr
    at step ``horizon``, constant after; gradient clipping; data from ``train|seed``, budgets from
    ``budget|seed`` (drawn in every arm). Returns (weight digest, gradient norms before clipping, learning rates)."""
    model, _ = arms.build(arm, seed)
    model.train()
    matrices = [q for q in model.parameters() if q.ndim >= 2]
    vectors = [q for q in model.parameters() if q.ndim < 2]
    optim = torch.optim.AdamW([{"params": matrices, "weight_decay": weight_decay},
                               {"params": vectors, "weight_decay": vector_decay}], lr=lr, betas=betas, eps=eps)
    data, budgets = random.Random(f"train|{seed}"), random.Random(f"budget|{seed}")
    has_budget = arms.ARMS[arm]["forward"] in ("walk", "looped", "arloop")
    norms, rates = [], []
    for step in range(1, steps + 1):
        x, y = task.make_batch(data, batch, (1, 2, 3))
        budget = budgets.choice((3, 4, 5))
        loss = torch.nn.functional.cross_entropy(arms.answer_logits(model, arm, x, budget if has_budget else None), y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip)))
        if step <= warmup:
            rate = lr * step / warmup
        else:
            progress = min(1.0, (step - warmup) / (horizon - warmup))
            rate = lr * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress)))
        for group in optim.param_groups:
            group["lr"] = rate
        optim.step()
        rates.append(rate)
    return train.model_sha256(model), norms, rates


def test_trainer_applies_the_protocol_recipe_bit_for_bit(tmp_path):
    """The trained weights equal those of an independent loop that implements PROTOCOL section 7: the schedule
    reaches the optimiser, gradients are clipped, betas and eps are the recipe's, vectors are not decayed."""
    # (1) the frozen numbers of the protocol; only the step count, batch and validation size are small
    recipe = train.Recipe(lr=1e-3, steps=5, batch=8, val_every=5, ckpt_every=5, val_items=8, log_every=1)
    meta = train.train(tmp_path / "protocol", "AWA-2L", 2, recipe)
    digest, norms, rates = reference_training("AWA-2L", 2, 5, lr=1e-3, batch=8, warmup=100, horizon=3000, floor=0.1,
                                              betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01, clip=1.0)
    assert meta["model_sha256"] == digest
    curve = [r for r in train.read_jsonl(tmp_path / "protocol" / "curve.jsonl") if r["type"] == "train"]
    assert [r["grad_norm"] for r in curve] == norms and [r["lr"] for r in curve] == rates
    assert rates == pytest.approx([1e-5, 2e-5, 3e-5, 4e-5, 5e-5])
    assert min(norms) > 1.0, "the protocol's clipping threshold is active in these steps"
    # (2) every recipe field reaches the optimiser: other values, warm-up, cosine and floor all inside 7 steps
    custom = dict(lr=2e-3, warmup=2, betas=(0.8, 0.95), eps=1e-5, weight_decay=0.2, clip=0.05)
    for arm in ("AWA-2L", "ChaCAL-2L"):
        recipe = train.Recipe(steps=7, schedule_steps=5, lr_floor=0.25, batch=8, val_every=7, ckpt_every=7,
                              val_items=8, log_every=1, **custom)
        meta = train.train(tmp_path / f"custom_{arm}", arm, 3, recipe)
        digest, norms, rates = reference_training(arm, 3, 7, batch=8, horizon=5, floor=0.25, **custom)
        assert meta["model_sha256"] == digest, arm
        assert min(norms) > custom["clip"], "every step of this run is clipped"
        assert rates[:2] == pytest.approx([1e-3, 2e-3]) and rates[4:] == pytest.approx([5e-4] * 3)
        assert 5e-4 < rates[3] < rates[2] < 2e-3
        # the reference itself is sensitive to each ingredient (otherwise the equality above would prove nothing)
        for change in (dict(clip=1e9), dict(betas=(0.9, 0.999)), dict(eps=1e-8), dict(weight_decay=0.0),
                       dict(vector_decay=0.2), dict(warmup=1)):
            assert reference_training(arm, 3, 7, batch=8, horizon=5, floor=0.25, **{**custom, **change})[0] != digest, \
                (arm, change)


def test_training_is_bitwise_deterministic_and_resumable(tmp_path):
    recipe = tiny_recipe(steps=6)
    full = train.train(tmp_path / "full", "AWA-2L", 3, recipe)
    again = train.train(tmp_path / "again", "AWA-2L", 3, recipe)
    assert full["finished"] and full["steps"] == 6 and full["n_params"] == 551_308
    assert full["model_sha256"] == again["model_sha256"]
    curve = train.read_jsonl(tmp_path / "full" / "curve.jsonl")
    assert train.strip_nondeterministic(curve) == train.strip_nondeterministic(
        train.read_jsonl(tmp_path / "again" / "curve.jsonl"))
    assert [r["step"] for r in curve if r["type"] == "train"] == [1, 2, 3, 4, 5, 6]
    assert [r["step"] for r in curve if r["type"] == "val"] == [0, 3, 6]
    assert all(r["budget"] in (3, 4, 5) for r in curve if r["type"] == "train")
    assert curve[1]["lr"] == pytest.approx(1e-5) and len(curve[0]["lambda"]) == 4
    # killed after 4 steps (last checkpoint at step 3, one more record already written), then resumed
    part = train.train(tmp_path / "resumed", "AWA-2L", 3, recipe, stop_after=3)
    assert part == {"finished": False, "step": 3}
    train._append_jsonl(tmp_path / "resumed" / "curve.jsonl", {"type": "train", "step": 4, "loss": 0.0})
    resumed = train.train(tmp_path / "resumed", "AWA-2L", 3, recipe)
    assert resumed["model_sha256"] == full["model_sha256"]
    assert train.strip_nondeterministic(train.read_jsonl(tmp_path / "resumed" / "curve.jsonl")) == \
        train.strip_nondeterministic(curve)
    assert train.strip_nondeterministic(resumed) == train.strip_nondeterministic(full)
    # a finished run started again does nothing; another seed or recipe is refused
    assert train.train(tmp_path / "full", "AWA-2L", 3, recipe)["model_sha256"] == full["model_sha256"]
    with pytest.raises(ValueError, match="belongs to another run"):
        train.train(tmp_path / "full", "AWA-2L", 4, recipe)
    with pytest.raises(ValueError, match="belongs to another run"):
        train.train(tmp_path / "full", "AWA-2L", 3, tiny_recipe(steps=6, lr=2e-3))
    other_seed = train.train(tmp_path / "seed4", "AWA-2L", 4, recipe)
    assert other_seed["model_sha256"] != full["model_sha256"]
    model, meta = train.load_run(tmp_path / "full")
    assert train.model_sha256(model) == meta["model_sha256"] == full["model_sha256"] and not model.training


def test_extension_continues_the_same_run_and_common_random_numbers(tmp_path):
    recipe = tiny_recipe(steps=4, val_every=2, ckpt_every=2)
    base = train.train(tmp_path / "run", "ARLOOP-2L", 5, recipe)
    assert not base["extended"] and base["competence"]["extendable"] and not base["competence"]["competent"]
    extended = train.train(tmp_path / "run", "ARLOOP-2L", 5, recipe, total_steps=6)
    assert extended["extended"] and extended["steps"] == 6 and extended["planned_steps"] == 4
    straight = train.train(tmp_path / "straight", "ARLOOP-2L", 5, recipe, total_steps=6)
    assert straight["model_sha256"] == extended["model_sha256"]
    with pytest.raises(ValueError, match="cannot be smaller"):
        train.train(tmp_path / "run", "ARLOOP-2L", 5, recipe, total_steps=3)
    # the budget stream is drawn in every arm, so arms with the same seed see the same data and budgets
    std = train.train(tmp_path / "std", "STD-2L-pm", 5, recipe)
    looped = train.train(tmp_path / "looped", "LOOPED-2L", 5, recipe)
    budgets = [r["budget"] for r in train.read_jsonl(tmp_path / "run" / "curve.jsonl") if r["type"] == "train"]
    assert budgets[:4] == [r["budget"] for r in train.read_jsonl(tmp_path / "looped" / "curve.jsonl")
                           if r["type"] == "train"]
    rng = random.Random("budget|5")
    assert budgets == [rng.choice((3, 4, 5)) for _ in range(6)]
    assert all(r["budget"] is None for r in train.read_jsonl(tmp_path / "std" / "curve.jsonl") if r["type"] == "train")
    assert std["competence"] == {"metric": "k1_acc", "value": std["final_val"]["k1_acc"], "threshold": 0.90,
                                 "competent": False, "extendable": False}
    assert looped["final_val"]["competence_stratum"] == "shuffled"
    # the cost of an extension can be read from the run's own records (used by the extension ledger)
    assert phases.extension_cost(tmp_path / "run", extended) == \
        extended["timing"]["cpu_train_s"] - base["timing"]["cpu_train_s"]
    # a logging cadence that does not divide the planned steps: the last planned step is still recorded, and an
    # extended run has the same curve as one that ran straight through
    sparse = tiny_recipe(steps=4, val_every=2, ckpt_every=2, log_every=3)
    first = train.train(tmp_path / "sparse", "STD-2L-pm", 6, sparse)
    more = train.train(tmp_path / "sparse", "STD-2L-pm", 6, sparse, total_steps=7)
    through = train.train(tmp_path / "sparse_through", "STD-2L-pm", 6, sparse, total_steps=7)
    curves = [train.read_jsonl(tmp_path / name / "curve.jsonl") for name in ("sparse", "sparse_through")]
    assert [r["step"] for r in curves[0] if r["type"] == "train"] == [3, 4, 6, 7]
    assert train.strip_nondeterministic(curves[0]) == train.strip_nondeterministic(curves[1])
    assert more["model_sha256"] == through["model_sha256"] != first["model_sha256"]
    assert phases.extension_cost(tmp_path / "sparse", more) == \
        more["timing"]["cpu_train_s"] - first["timing"]["cpu_train_s"]


def test_competence_rules():
    val = {"competence": 0.93, "competence_stratum": "sorted", "k1_acc": 0.5}
    assert train.competence_verdict("ChaCAL-2L", val)["competent"] is True
    assert train.competence_verdict("ChaCAL-2L", {**val, "competence": 0.89})["competent"] is False
    assert train.competence_verdict("STD-2L-pm", val)["competent"] is False
    assert train.competence_verdict("STD-2L-pm", {**val, "k1_acc": 0.9})["competent"] is True
    assert [arms.competence_stratum(a) for a in arms.ARM_ORDER] == ["shuffled", "sorted", "sorted", "shuffled",
                                                                    "shuffled", "shuffled"]


# ================================================================================= evaluation records
@pytest.fixture(scope="module")
def smoke_runs(tmp_path_factory):
    """One briefly trained run per arm with a truncated evaluation (the structure of a real result set)."""
    torch.set_num_threads(1)
    logging.disable(logging.CRITICAL)
    root = tmp_path_factory.mktemp("awa3_smoke")
    runs = {}
    try:
        for arm in arms.ARM_ORDER:
            run_dir = layout.run_dir(root, "smoke", arm, 0, 1e-3)
            meta = train.train(run_dir, arm, 0, tiny_recipe(steps=2, val_every=2, ckpt_every=2))
            model, _ = train.load_run(run_dir)
            record = evaluate.evaluate_run(model, arm, limit=6)
            record.update(seed=0, steps=meta["steps"], model_sha256=meta["model_sha256"])
            train.write_json(run_dir / "eval.json", record)
            runs[run_dir.name] = {"meta": meta, "eval": record}
    finally:
        logging.disable(logging.NOTSET)
    return root, runs


def test_evaluation_record_structure(smoke_runs):
    _, runs = smoke_runs
    by_arm = {r["eval"]["arm"]: r["eval"] for r in runs.values()}
    assert by_arm["AWA-2L"]["budgets"] == list(range(25)) and by_arm["AWA-own-2L"]["budgets"] == list(range(25))
    assert by_arm["LOOPED-2L"]["budgets"] == list(range(1, 17)) and by_arm["ARLOOP-2L"]["budgets"] == list(range(1, 25))
    assert by_arm["ChaCAL-2L"]["budgets"] == by_arm["STD-2L-pm"]["budgets"] == [0]
    for arm, ev in by_arm.items():
        assert ev["truncated_to"] == 6 and set(ev["test"]) == {str(k) for k in task.TEST_DEPTHS}
        assert set(ev["val"]) == {str(k) for k in task.VAL_ORACLE_DEPTHS}
        for k, cell in ev["test"].items():
            assert set(cell) == set(task.STRATA)
            for stratum in task.STRATA:
                assert cell[stratum]["n"] == 6 and len(cell[stratum]["correct"]) == len(ev["budgets"])
                assert all(0 <= c <= 6 for c in cell[stratum]["correct"])
            if int(k) <= 2:
                assert all(cell[s] == cell["shuffled"] for s in task.STRATA)
        assert set(ev["macs"]) == {str(b) for b in ev["budgets"]}
        assert ev["by_disorder"]["3"] and sum(g["n"] for g in ev["by_disorder"]["6"].values()) == 6
        json.dumps(ev, allow_nan=False)
    awa = by_arm["AWA-2L"]
    for k in awa["test"]:                                        # walks: the ablation lambda := 0 is budget K = 0
        for s in task.STRATA:
            assert awa["zero_operator"][k][s]["correct"] == awa["test"][k][s]["correct"][0]
    assert by_arm["ChaCAL-2L"]["zero_operator"] is not None and by_arm["STD-2L-pm"]["zero_operator"] is None
    assert len(awa["gate"]["lambda"]) == 4 and set(awa["gate"]["mean_c"]) == set(task.POSITION_TYPES)
    assert len(by_arm["ChaCAL-2L"]["gate"]["gamma"]) == 4 and by_arm["LOOPED-2L"]["gate"] == {}
    traces = awa["traces"]
    assert traces["deltas"] == [-1, 0, 1] and by_arm["STD-2L-pm"]["traces"] is None
    for split in ("val", "test"):
        assert set(traces[split]) == {"3", "4", "6"}
        for k, cell in traces[split].items():
            assert set(cell) == set(task.STRATA) | {"pooled"} and cell["pooled"]["n"] == 24
            table = cell["pooled"]["mean"]
            assert len(table) == 4 and len(table[0]) == 3 and len(table[0][0]) == int(k)
            assert table[0][0][int(k) - 1] is None and table[0][2][0] is None      # element index outside the chain
            assert all(0.0 <= v <= 1.0 + 1e-9 for head in table for row in head for v in row if v is not None)


def test_analysis_runs_on_smoke_records_and_refuses_them_by_default(smoke_runs, tmp_path):
    root, runs = smoke_runs
    with pytest.raises(ValueError, match="truncated evaluation records"):
        analysis.compute_decisions(runs)
    decisions = analysis.compute_decisions(runs, allow_truncated=True)
    assert decisions["P-M"]["decision"] == "INCOMPLETE" and decisions["P-T"]["decision"] == "INCOMPLETE"
    assert len(decisions["runs"]["missing"]) == 26 and len(decisions["per_run"]) == 6
    assert decisions == analysis.compute_decisions(copy.deepcopy(runs), allow_truncated=True), "pure function"
    assert analysis.load_runs(layout.phase_dir(root, "smoke")).keys() == runs.keys()
    # the command-line tools on a copy of the result set: analysis, manifest, check, recomputation
    results = tmp_path / "results"
    shutil.copytree(root, results)
    assert analysis.main(["--results", str(results), "--phase", "smoke"]) == 1
    assert analysis.main(["--results", str(results), "--phase", "smoke", "--allow-truncated"]) == 0
    phase_dir = layout.phase_dir(results, "smoke")
    stored = json.loads((phase_dir / "decisions.json").read_text(encoding="utf-8"))
    assert stored["analysis_code_sha256"].keys() == {"awa3/analysis.py", "awa3/stats.py"}
    assert "| P-M | INCOMPLETE |" in (phase_dir / "tables" / "summary.md").read_text(encoding="utf-8")
    assert manifest.main(["--manifest", "--results", str(results)]) == 0
    assert manifest.main(["--check", "--results", str(results)]) == 0
    report = manifest.rerun(results, "smoke")
    assert report["ok"] and report["decision_differences"] == [] and report["analysis_code_matches_freeze"] is None
    # a changed record or a changed decision is detected
    stored["per_run"]["AWA-2L_lr0.001_s0"]["Y"] += 0.25
    (phase_dir / "decisions.json").write_text(json.dumps(stored), encoding="utf-8")
    tampered = manifest.rerun(results, "smoke")
    assert not tampered["ok"] and tampered["manifest"]["mismatched"] == ["results/phasesmoke/decisions.json"]
    assert any("AWA-2L_lr0.001_s0.Y" in d for d in tampered["decision_differences"])
    assert manifest.main(["--check", "--results", str(results)]) == 1


# ======================================================================================== statistics
def test_exact_t4_test():
    awa, comp = [0.95, 0.93, 0.97, 0.91, 0.99], [0.20, 0.18, 0.22, 0.19, 0.21]
    assert stats.exact_t4(awa, comp, 0.15) == {"T4": 5, "p": 1 / 252, "n_splits": 252}
    one_failed = [0.95, 0.93, 0.97, 0.91, 0.10]
    result = stats.exact_t4(one_failed, comp, 0.15)
    assert result["T4"] == 4 and result["p"] == pytest.approx(6 / 252)             # tolerates one failed seed
    assert stats.exact_t4([0.95, 0.93, 0.97, 0.15, 0.10], comp, 0.15)["T4"] == 3
    assert stats.exact_t4([0.95, 0.93, 0.97, 0.15, 0.10], comp, 0.15)["p"] > 0.05
    assert stats.exact_t4([0.40, 0.41, 0.42, 0.43, 0.44], comp, 0.15)["T4"] == 5    # shift 0.15 < the gap
    assert stats.exact_t4([0.40, 0.41, 0.42, 0.43, 0.44], comp, 0.25)["T4"] == 0
    assert stats.exact_t4([0.40, 0.41, 0.42, 0.43, 0.44], comp, 0.25)["p"] == 1.0
    assert stats.exact_t4([0.36, 0.38, 0.37, 0.39, 0.40], comp, 0.15)["T4"] == 3    # 0.37 - 0.15 == max(comp): strict
    assert stats.exact_t4([1.0], [0.0])["p"] == 0.5
    with pytest.raises(ValueError):
        stats.exact_t4([], comp)


def test_exact_mann_whitney_and_hodges_lehmann():
    a, b = [0.95, 0.93, 0.97, 0.91, 0.99], [0.20, 0.18, 0.22, 0.19, 0.21]
    assert stats.mann_whitney_exact(a, b, 0.0, "greater") == {"U": 25.0, "p": 1 / 252, "n_splits": 252,
                                                               "alternative": "greater"}
    assert stats.mann_whitney_exact(a, b, 0.0, "two-sided")["p"] == pytest.approx(2 / 252)
    assert stats.mann_whitney_exact(a, b[:4], 0.0, "two-sided")["p"] == pytest.approx(2 / 126)    # 5 vs 4
    assert stats.mann_whitney_exact(a, b, 0.75, "greater")["p"] > 0.2                             # shifted null
    assert stats.mann_whitney_exact([1, 2], [1, 2], 0.0, "two-sided")["p"] == 1.0
    with pytest.raises(ValueError):
        stats.mann_whitney_exact(a, b, 0.0, "less")
    cdf = stats.u_null_cdf(5, 5)
    assert len(cdf) == 26 and cdf[0] == pytest.approx(1 / 252) and cdf[4] == pytest.approx(12 / 252)
    assert cdf[-1] == pytest.approx(1.0)
    hl = stats.hodges_lehmann(a, b)
    diffs = sorted(x - y for x in a for y in b)
    assert hl["estimate"] == pytest.approx(diffs[12]) and hl["order_statistic"] == 5
    assert hl["lower"] == pytest.approx(diffs[4]) and hl["confidence"] == pytest.approx(1 - 12 / 252)   # 95.2%
    two = stats.hodges_lehmann(a, b, two_sided=True)
    assert two["order_statistic"] == 3 and two["lower"] == pytest.approx(diffs[2])
    assert two["upper"] == pytest.approx(diffs[22]) and two["confidence"] == pytest.approx(1 - 2 * 4 / 252)
    tiny = stats.hodges_lehmann([1.0], [0.0])
    assert tiny["estimate"] == 1.0 and tiny["lower"] == -math.inf and tiny["confidence"] is None
    assert stats.median([3, 1, 2]) == 2 and stats.median([4, 1, 2, 3]) == 2.5


def test_holm_student_t_and_bounds():
    assert stats.holm({"a": 0.01, "b": 0.04}) == {"a": 0.02, "b": 0.04}
    assert stats.holm({"a": 0.03, "b": 0.02, "c": 0.5}) == pytest.approx({"b": 0.06, "a": 0.06, "c": 0.5})
    with pytest.raises(ValueError):
        stats.holm({"a": 1.5})
    for df, q in ((1, 6.3138), (2, 2.9200), (4, 2.1318), (9, 1.8331)):
        assert stats.student_t_ppf(0.95, df) == pytest.approx(q, abs=2e-4)
    assert stats.student_t_sf(2.1318, 4) == pytest.approx(0.05, abs=1e-4)
    assert stats.student_t_sf(0.0, 3) == pytest.approx(0.5) and stats.student_t_sf(-2.0, 3) > 0.9
    assert stats.betainc(2.0, 3.0, 0.4) == pytest.approx(0.5248, abs=1e-4)
    values = [0.02, 0.05, -0.01, 0.03, 0.01]
    m = sum(values) / 5
    se = math.sqrt(sum((v - m) ** 2 for v in values) / 4 / 5)
    assert stats.t_upper(values) == pytest.approx(m + 2.1318 * se, abs=1e-5)
    assert stats.t_lower(values) == pytest.approx(m - 2.1318 * se, abs=1e-5)
    assert stats.t_test_greater([0.6, 0.62, 0.58, 0.61, 0.59], 0.15) < 1e-5
    assert stats.t_test_greater(values, 0.15) > 0.99
    assert stats.t_test_greater([0.5, 0.5], 0.15) == 0.0 and stats.t_test_greater([0.1, 0.1], 0.15) == 1.0
    with pytest.raises(ValueError):
        stats.t_upper([0.1])


def test_per_hop_error_law_and_classifier():
    floors = {k: task.floor(k) for k in (2, 3, 6, 8)}
    acc = {k: stats.predict_accuracy(k, floors[k], 0.04) for k in (2, 3)}
    fit = stats.fit_epsilon(acc, floors)
    assert fit["eps"] == pytest.approx(0.04, abs=1e-4) and fit["sse"] < 1e-9
    assert stats.predict_accuracy(1, 0.1, 0.5) == 1.0
    assert stats.fit_epsilon({2: 1.0, 3: 1.0}, floors)["eps"] == 0.0
    assert stats.classify_per_hop(0.85, 0.85, 0.5) == "precise"
    assert stats.classify_per_hop(0.60, 0.62, 0.66) == "compounding"
    assert stats.classify_per_hop(0.30, 0.32, 0.80) == "depth cliff"
    assert stats.classify_per_hop(0.60, 0.62, 0.52) == "unclassified"
    assert stats.majority(["precise", "precise", "depth cliff"]) == "precise"
    assert stats.majority(["precise", "depth cliff"]) is None
    with pytest.raises(ValueError):
        stats.fit_epsilon({1: 1.0}, floors)


# ============================================================================ decisions (synthetic)
FLOORS = {k: task.floor(k) for k in task.TEST_DEPTHS}
FOLLOW = {k: task.followable_fraction(k) for k in task.TEST_DEPTHS}
AXIS = {"walk": list(range(25)), "looped": list(range(1, 17)), "arloop": list(range(1, 25)), "exact": [0], "std": [0]}


def fake_run(arm, seed, accuracy, *, competent=True, n=1000, val_accuracy=None):
    """A synthetic result: ``accuracy(k, stratum, budget)`` gives every test cell, ``val_accuracy`` every
    validation cell (default: the same function)."""
    forward = arms.ARMS[arm]["forward"]
    axis = AXIS[forward]

    def grid(depths, cell=accuracy):
        return {str(k): {s: {"n": n, "correct": [round(n * cell(k, s, b)) for b in axis]} for s in task.STRATA}
                for k in depths}

    ev = {"format": "awa3-eval", "arm": arm, "forward": forward, "budgets": axis, "truncated_to": None,
          "floors": {str(k): FLOORS[k] for k in task.TEST_DEPTHS}, "followable": {str(k): FOLLOW[k] for k in task.TEST_DEPTHS},
          "macs": {str(b): arms.macs_per_answer(arm, b) for b in axis},
          "test": grid(task.TEST_DEPTHS), "val": grid(task.VAL_ORACLE_DEPTHS, val_accuracy or accuracy),
          "traces": None}
    if forward == "walk":
        ev["zero_operator"] = {str(k): {s: {"n": n, "correct": round(n * accuracy(k, s, 0))} for s in task.STRATA}
                               for k in task.TEST_DEPTHS}
    meta = {"arm": arm, "seed": seed, "steps": 3000, "extended": False,
            "competence": {"competent": competent, "value": 0.99 if competent else 0.4}}
    return {"meta": meta, "eval": ev}


def walk_acc(top):
    """An ideal walk: needs K >= k - 1 hops, then ``top`` in every stratum; chance below."""
    return lambda k, s, b: (1.0 if k == 1 else top) if b >= k - 1 else FLOORS[k]


def lower_triangular_acc(sorted_acc=0.97, leak=0.0):
    def accuracy(k, s, b):
        if k <= 2 or s == "sorted":
            return sorted_acc
        if s == "shuffled":
            return FOLLOW[k] * sorted_acc + (1 - FOLLOW[k]) * (FLOORS[k] + leak)
        return FLOORS[k] + leak
    return accuracy


def loop_acc(top, need=1):
    return lambda k, s, b: (1.0 if k == 1 else top) if b >= need * (k - 1) else FLOORS[k]


def std_acc(k, s, b):
    return 0.99 if k == 1 else FLOORS[k] + 0.02


def result_set(awa_tops=(0.95, 0.96, 0.94, 0.97, 0.93), **kw):
    runs = {}
    for seed, top in enumerate(awa_tops):
        runs[f"AWA-2L_s{seed}"] = fake_run("AWA-2L", seed, walk_acc(top), competent=kw.get("awa_competent", [True] * 5)[seed])
    for seed in range(5):
        jitter = 0.002 * seed
        runs[f"ChaCAL-2L_s{seed}"] = fake_run("ChaCAL-2L", seed, lower_triangular_acc(0.97 - jitter, kw.get("chacal_leak", 0.0) + jitter),
                                              competent=kw.get("chacal_competent", True))
        runs[f"AWA-own-2L_s{seed}"] = fake_run("AWA-own-2L", seed, lower_triangular_acc(0.96 - jitter, jitter))
    for seed in range(4):
        runs[f"LOOPED-2L_s{seed}"] = fake_run("LOOPED-2L", seed, loop_acc(0.90 + 0.01 * seed))
        runs[f"ARLOOP-2L_s{seed}"] = fake_run("ARLOOP-2L", seed, loop_acc(0.50 + 0.01 * seed))
    for seed in range(3):
        runs[f"STD-2L-pm_s{seed}"] = fake_run("STD-2L-pm", seed, std_acc)
    return runs


def test_decisions_on_the_predicted_outcome():
    d = analysis.compute_decisions(result_set())
    assert d["runs"]["missing"] == [] and len(d["per_run"]) == 26
    assert d["G0"]["passed"] and d["G0"]["control"]["status"] == "B holds"
    y = [d["per_run"][f"AWA-2L_s{s}"]["Y"] for s in range(5)]
    assert y == pytest.approx([0.95, 0.96, 0.94, 0.97, 0.93])
    assert d["per_run"]["ChaCAL-2L_s0"]["Y"] == pytest.approx((FLOORS[6] + FLOORS[8]) / 2, abs=1e-3)
    assert d["per_run"]["AWA-2L_s0"]["Y_chance_corrected"] == pytest.approx(
        ((0.95 - FLOORS[6]) / (1 - FLOORS[6]) + (0.95 - FLOORS[8]) / (1 - FLOORS[8])) / 2, abs=1e-3)
    pm = d["P-M"]
    assert pm["decision"] == "CONFIRMED"
    for key, comparator in (("a", "ChaCAL-2L"), ("b", "AWA-own-2L")):
        c = pm["contrasts"][key]
        assert c["comparator"] == comparator and c["passes"] and c["awa_successes"] == 5
        assert c["T4"] == {"T4": 5, "p": pytest.approx(1 / 252), "n_splits": 252}
        assert c["hodges_lehmann"]["estimate"] > 0.7 and c["hodges_lehmann"]["confidence"] == pytest.approx(0.952, abs=1e-3)
    pt = d["P-T"]
    assert pt["decision"] == "SUPPORTED" and pt["assumption_B"] == "B holds"
    for arm in ("ChaCAL-2L", "AWA-own-2L"):
        row = pt["arms"][arm]
        assert row["informative_depths"] == [3, 4, 6, 8] and row["supported"] and not row["refuted"]
        assert max(row["upper_bounds"].values()) <= 0.10 and row["mixture_upper_bound"] <= 0.05
    assert d["H3b"]["decision"] == "SUPPORTED" and d["H3b"]["mean_gap"] == pytest.approx(0.0, abs=1e-9)
    assert d["H4"]["a"]["decision"] == "SUPPORTED"
    assert d["H4"]["a"]["per_seed"]["0"]["offsets"] == {str(k): -1 for k in (2, 3, 4, 5, 6, 8)}
    assert d["H4"]["b"]["decision"] == "SUPPORTED" and d["H4"]["c"]["decision"] == "INCOMPLETE"
    assert d["H5"]["decision"] == "SUPPORTED" and d["H5"]["worst_drop"] == pytest.approx(0.0, abs=1e-9)
    assert d["per_hop_classifier"]["study_label"] == "precise"
    h6 = d["H6"]["baselines"]
    assert h6["ARLOOP-2L"]["capability"] == "AWA beats the baseline" and h6["ARLOOP-2L"]["p_holm"] <= 0.05
    assert h6["LOOPED-2L"]["capability"] == "no decided capability difference"
    assert h6["LOOPED-2L"]["mann_whitney"]["n_splits"] == 126
    cost = h6["LOOPED-2L"]["cost_at_matched_accuracy"]
    assert cost["reading"] == "AWA cheaper" and cost["log_cost_ratio"]["estimate"] > math.log(2.0)
    assert cost["awa"][0]["budget"] == 7 and cost["baseline"][0]["budget"] == 7      # depth 8 needs 7 hops / loops
    # the reading of H6 (PROTOCOL 3.8): (i) decides when it is significant, otherwise (ii)
    assert h6["ARLOOP-2L"]["reading"] == "AWA beats ARLOOP-2L"
    assert h6["LOOPED-2L"]["reading"] == "AWA matches LOOPED-2L at >= 2x lower cost"
    assert d["H6"]["decision"] == ("LOOPED-2L: AWA matches LOOPED-2L at >= 2x lower cost; "
                                   "ARLOOP-2L: AWA beats ARLOOP-2L")
    assert pm["refuted_by"] == [] and pm["inconclusive_contrasts"] == []
    summary = analysis.render_summary(d, result_set())
    assert "| H6 vs LOOPED-2L | AWA matches LOOPED-2L at >= 2x lower cost |" in summary
    assert "| H6 vs ARLOOP-2L | AWA beats ARLOOP-2L |" in summary and "| P-M | CONFIRMED |" in summary


def test_primary_decision_rules():
    # one failed AWA seed is tolerated: T4 = 4, exact p = 6/252
    d = analysis.compute_decisions(result_set(awa_tops=(0.95, 0.96, 0.94, 0.97, 0.20)))
    assert d["P-M"]["decision"] == "CONFIRMED"
    assert d["P-M"]["contrasts"]["a"]["T4"]["T4"] == 4 and d["P-M"]["contrasts"]["a"]["T4"]["p"] == pytest.approx(6 / 252)
    # two failed seeds: neither confirmed nor refuted
    d = analysis.compute_decisions(result_set(awa_tops=(0.95, 0.96, 0.94, 0.30, 0.20)))
    assert d["P-M"]["decision"] == "INCONCLUSIVE" and d["P-M"]["contrasts"]["a"]["criteria"] == {
        "seed_count": False, "T4": False, "HL": True}
    assert d["per_hop_classifier"]["awa_labels"].count("precise") == 3
    # no seed generalises: refuted
    d = analysis.compute_decisions(result_set(awa_tops=(0.30, 0.25, 0.20, 0.22, 0.28)))
    assert d["P-M"]["decision"] == "REFUTED"
    # in-distribution depths solved, deeper chains at the floor: refuted, labelled "depth cliff"
    runs = result_set()
    for seed in range(5):
        runs[f"AWA-2L_s{seed}"] = fake_run("AWA-2L", seed, lambda k, s, b: 0.96 if k <= 3 and b >= k - 1 else FLOORS[k])
    d = analysis.compute_decisions(runs)
    assert d["P-M"]["decision"] == "REFUTED" and d["per_hop_classifier"]["study_label"] == "depth cliff"
    assert d["H6"]["baselines"]["LOOPED-2L"]["cost_at_matched_accuracy"]["reading"].startswith("not testable")
    # high accuracy but below the success threshold, large effect: inconclusive, not confirmed
    d = analysis.compute_decisions(result_set(awa_tops=(0.75, 0.76, 0.74, 0.77, 0.73)))
    assert d["P-M"]["decision"] == "INCONCLUSIVE"
    # the learnability gate fails: P-M is not tested, P-T still is
    d = analysis.compute_decisions(result_set(awa_competent=[True, True, True, False, False]))
    assert not d["G0"]["passed"] and d["P-M"]["decision"] == "NOT TESTED" and d["P-T"]["decision"] == "SUPPORTED"
    # a comparator that is not ID-competent makes the contrast inconclusive
    d = analysis.compute_decisions(result_set(chacal_competent=False))
    assert d["P-M"]["decision"] == "INCONCLUSIVE"
    assert d["P-M"]["contrasts"]["a"]["comparator_not_id_competent"] == [0, 1, 2, 3, 4]
    assert d["P-M"]["contrasts"]["a"]["passes"] and d["P-M"]["contrasts"]["a"]["inconclusive"]
    assert d["P-M"]["inconclusive_contrasts"] == ["a"] and d["P-M"]["refuted_by"] == []
    # missing runs: no decision
    runs = result_set()
    del runs["AWA-own-2L_s4"]
    d = analysis.compute_decisions(runs)
    assert d["P-M"]["decision"] == "INCOMPLETE" and d["P-T"]["decision"] == "INCOMPLETE"
    assert d["runs"]["missing"] == ["AWA-own-2L_s4"]


def test_a_comparator_that_is_not_id_competent_can_neither_confirm_nor_refute():
    """PROTOCOL 12.3: such a contrast is INCONCLUSIVE. Refutation then needs the AWA-only clause (no comparator
    in it) or HL < 0.15 in the other contrast, if that comparator is ID-competent."""
    def with_own(competent, tops=(0.95, 0.96, 0.94, 0.97, 0.93), **kw):
        runs = result_set(awa_tops=tops, **kw)
        for seed in range(5):                                       # an "own-window" arm that does as well as AWA
            runs[f"AWA-own-2L_s{seed}"] = fake_run("AWA-own-2L", seed, walk_acc(0.90 + 0.002 * seed),
                                                   competent=competent)
        return analysis.compute_decisions(runs)["P-M"]

    # HL(AWA - AWA-own) is about 0.05 < 0.15. With competent comparator runs that refutes H2 ...
    pm = with_own(True)
    assert pm["contrasts"]["b"]["hodges_lehmann"]["estimate"] == pytest.approx(0.046, abs=0.01)
    assert pm["decision"] == "REFUTED" and pm["refuted_by"] == ["contrast (b): Hodges-Lehmann estimate < 0.15"]
    # ... with comparator runs that are not ID-competent it does not: the contrast is inconclusive
    pm = with_own(False)
    assert pm["contrasts"]["b"]["inconclusive"] and not pm["contrasts"]["a"]["inconclusive"]
    assert pm["contrasts"]["a"]["passes"] and pm["contrasts"]["b"]["hodges_lehmann"]["estimate"] < 0.15
    assert pm["decision"] == "INCONCLUSIVE" and pm["refuted_by"] == [] and pm["inconclusive_contrasts"] == ["b"]
    # one non-competent run is enough
    runs = result_set()
    for seed in range(5):
        runs[f"AWA-own-2L_s{seed}"] = fake_run("AWA-own-2L", seed, walk_acc(0.90 + 0.002 * seed), competent=seed != 3)
    pm = analysis.compute_decisions(runs)["P-M"]
    assert pm["decision"] == "INCONCLUSIVE" and pm["contrasts"]["b"]["comparator_not_id_competent"] == [3]
    # the other contrast keeps its force: ChaCAL not competent, AWA-own competent and as good as AWA
    pm = with_own(True, chacal_competent=False)
    assert pm["inconclusive_contrasts"] == ["a"]
    assert pm["decision"] == "REFUTED" and pm["refuted_by"] == ["contrast (b): Hodges-Lehmann estimate < 0.15"]
    # both comparators not competent: nothing can be said from the contrasts
    pm = with_own(False, chacal_competent=False)
    assert pm["inconclusive_contrasts"] == ["a", "b"] and pm["decision"] == "INCONCLUSIVE"
    # the AWA-only clause involves no comparator: AWA at the floor is refuted whatever the comparators did
    floor_tops = (0.30, 0.25, 0.20, 0.22, 0.28)
    pm = with_own(False, tops=floor_tops, chacal_competent=False)
    assert pm["decision"] == "REFUTED"
    assert pm["refuted_by"] == ["AWA endpoint: at most 1 seed with Y >= 0.80 and median Y < 0.50"]
    pm = analysis.compute_decisions(result_set(awa_tops=floor_tops, chacal_competent=False))["P-M"]
    assert pm["decision"] == "REFUTED" and pm["inconclusive_contrasts"] == ["a"]
    assert pm["refuted_by"] == ["AWA endpoint: at most 1 seed with Y >= 0.80 and median Y < 0.50",
                                "contrast (b): Hodges-Lehmann estimate < 0.15"]
    # two failed AWA seeds with a high median do not trigger the AWA-only clause
    pm = with_own(False, tops=(0.95, 0.96, 0.94, 0.30, 0.20), chacal_competent=False)
    assert pm["decision"] == "INCONCLUSIVE" and pm["refuted_by"] == []


def test_order_law_decision_rules():
    # a lower-triangular arm clearly above the floor on one-swap / anti refutes the law
    d = analysis.compute_decisions(result_set(chacal_leak=0.40))
    assert d["P-T"]["decision"] == "REFUTED" and d["P-T"]["arms"]["ChaCAL-2L"]["refuted"]
    assert not d["P-T"]["arms"]["AWA-own-2L"]["refuted"]
    assert min(d["P-T"]["arms"]["ChaCAL-2L"]["lower_bounds"].values()) > 0.15
    # ... but not before every seed is in: with 2 of 5 ChaCAL runs the same excess gives no decision
    runs = result_set(chacal_leak=0.40)
    for seed in (2, 3, 4):
        del runs[f"ChaCAL-2L_s{seed}"]
    d = analysis.compute_decisions(runs)
    assert d["P-T"]["arms"]["ChaCAL-2L"]["status"] == "tested"
    assert min(d["P-T"]["arms"]["ChaCAL-2L"]["excess_over_floor"]["anti"]) > 0.35
    assert d["P-T"]["decision"] == "INCOMPLETE" and d["P-M"]["decision"] == "INCOMPLETE"
    # a small leak: neither supported (upper bound above 0.10) nor refuted
    d = analysis.compute_decisions(result_set(chacal_leak=0.12))
    assert d["P-T"]["decision"] == "INCONCLUSIVE" and not d["P-T"]["arms"]["ChaCAL-2L"]["supported"]
    # an arm that cannot do sorted chains at k = 3 is not testable
    runs = result_set()
    for seed in range(5):
        runs[f"ChaCAL-2L_s{seed}"] = fake_run("ChaCAL-2L", seed, lambda k, s, b: 0.99 if k <= 2 else FLOORS[k] + 0.05)
    d = analysis.compute_decisions(runs)
    assert d["P-T"]["arms"]["ChaCAL-2L"]["status"] == "not testable" and d["P-T"]["decision"] == "NOT TESTABLE"
    # assumption B: the standard model composing two facts is flagged
    runs = result_set()
    for seed in range(3):
        runs[f"STD-2L-pm_s{seed}"] = fake_run("STD-2L-pm", seed, lambda k, s, b: 0.99 if k == 1 else FLOORS[k] + 0.30)
    assert analysis.compute_decisions(runs)["G0"]["control"]["status"] == "B false"
    runs = result_set()
    for seed in range(3):
        runs[f"STD-2L-pm_s{seed}"] = fake_run("STD-2L-pm", seed, std_acc, competent=seed == 0)
    assert analysis.compute_decisions(runs)["G0"]["control"]["status"] == "uninformative"
    # order invariance of AWA: a 20-point gap between strata refutes H3b
    runs = result_set()
    for seed in range(5):
        runs[f"AWA-2L_s{seed}"] = fake_run("AWA-2L", seed, lambda k, s, b: 0.95 - (0.2 if s == "anti" and k >= 3 else 0.0)
                                           if b >= k - 1 else FLOORS[k])
    d = analysis.compute_decisions(runs)
    assert d["H3b"]["decision"] == "REFUTED" and d["H3b"]["mean_gap"] == pytest.approx(0.2, abs=1e-6)
    # overthinking: accuracy that falls at large budgets refutes H5
    runs = result_set()
    for seed in range(5):
        runs[f"AWA-2L_s{seed}"] = fake_run("AWA-2L", seed, lambda k, s, b: FLOORS[k] if b < k - 1 else
                                           0.95 if b < 2 * k else 0.85)
    assert analysis.compute_decisions(runs)["H5"]["decision"] == "REFUTED"


def test_h3a_refutation_p_value_doubles_the_smaller_one_sided_p_and_is_holm_adjusted():
    """PROTOCOL 12.4: per arm, p = 2 x min(one-sided p on one-swap, on anti), capped at 1; Holm over the two
    arms. The excess here is significant one-sided (p about 0.015) but not after the two corrections."""
    runs = result_set()
    for seed, leak in enumerate((0.15, 0.17, 0.19, 0.17, 0.18)):
        runs[f"ChaCAL-2L_s{seed}"] = fake_run("ChaCAL-2L", seed, lower_triangular_acc(0.97, leak))
    d = analysis.compute_decisions(runs)
    row, other = d["P-T"]["arms"]["ChaCAL-2L"], d["P-T"]["arms"]["AWA-own-2L"]
    one_sided = {s: stats.t_test_greater(row["excess_over_floor"][s], 0.15) for s in ("oneswap", "anti")}
    smaller = min(one_sided.values())
    assert 0.0125 < smaller < 0.025, one_sided
    assert row["p_refute"] == pytest.approx(2.0 * smaller)
    assert other["p_refute"] == 1.0, "the cap at 1"
    assert row["p_refute_holm"] == pytest.approx(4.0 * smaller) and other["p_refute_holm"] == 1.0
    assert row["p_refute_holm"] > 0.05 and not row["refuted"]
    assert d["P-T"]["decision"] == "INCONCLUSIVE"
    # a clear excess survives both corrections
    runs = result_set()
    for seed, leak in enumerate((0.25, 0.27, 0.29, 0.27, 0.28)):
        runs[f"ChaCAL-2L_s{seed}"] = fake_run("ChaCAL-2L", seed, lower_triangular_acc(0.97, leak))
    d = analysis.compute_decisions(runs)
    assert d["P-T"]["arms"]["ChaCAL-2L"]["p_refute_holm"] < 0.01 and d["P-T"]["decision"] == "REFUTED"


def test_oracle_budget_of_the_loops_is_chosen_on_validation():
    """PROTOCOL 12.6: per depth, the T that maximises the worst-stratum VALIDATION accuracy (ties: smallest T).
    Here the test split would choose other budgets."""
    def test_cells(k, s, b):
        return 0.90 if b == 5 else 0.50                              # the test split peaks at T = 5

    def val_cells(k, s, b):
        if k == 6:                                                   # T = 9 and T = 10 tie at 0.80: the smaller
            return 0.80 if b in (9, 10) else 0.30
        if b == 12:                                                  # best mean, but one stratum fails
            return 0.20 if s == "anti" else 1.00
        return 0.70 if b == 11 else 0.30

    run = fake_run("LOOPED-2L", 0, test_cells, val_accuracy=val_cells)
    ev = run["eval"]
    assert analysis.oracle_budget(ev, 6) == 9 and analysis.oracle_budget(ev, 8) == 11
    assert analysis.plateau_budgets(ev) == {6: 9, 8: 11}
    summary = analysis.per_run_summary(ev, run["meta"])
    assert summary["plateau_budgets"] == {"6": 9, "8": 11}
    assert summary["Y_plateau"] == pytest.approx(0.50), "test accuracy at the validation-chosen budgets"
    assert summary["Y"] == pytest.approx(0.50), "T = 12"
    # with the same cells on both splits the choice is the test peak, as in every other synthetic run here
    assert analysis.plateau_budgets(fake_run("LOOPED-2L", 0, test_cells)["eval"]) == {6: 5, 8: 5}
    # H6 compares AWA with the baseline at that budget
    runs = result_set()
    for seed in range(4):
        runs[f"LOOPED-2L_s{seed}"] = fake_run("LOOPED-2L", seed, test_cells, val_accuracy=val_cells)
    h6 = analysis.compute_decisions(runs)["H6"]["baselines"]["LOOPED-2L"]
    assert h6["Y_baseline_oracle_T"] == pytest.approx([0.50] * 4) and h6["reading"] == "AWA beats LOOPED-2L"


def test_h6_readings():
    """PROTOCOL 3.8 "Reading" and the choices of PROTOCOL 12.9."""
    def entry(capability="no decided capability difference", cost="comparable", descriptive=False):
        return {"capability": capability, "descriptive_only": descriptive,
                "cost_at_matched_accuracy": {"reading": cost}}

    arm = "LOOPED-2L"
    assert analysis.h6_reading(arm, entry("AWA beats the baseline", "LOOPED-2L cheaper")) == "AWA beats LOOPED-2L"
    assert analysis.h6_reading(arm, entry("the baseline beats AWA", "AWA cheaper")) == "LOOPED-2L beats AWA"
    assert analysis.h6_reading(arm, entry(cost="AWA cheaper")) == "AWA matches LOOPED-2L at >= 2x lower cost"
    assert analysis.h6_reading(arm, entry(cost="LOOPED-2L cheaper")) == "LOOPED-2L matches AWA at >= 2x lower cost"
    assert analysis.h6_reading(arm, entry()) == "equivalent"
    assert analysis.h6_reading(arm, entry(cost="AWA does not reach matched accuracy")) == \
        "not decided (AWA does not reach matched accuracy)"
    assert analysis.h6_reading(arm, entry("AWA beats the baseline", descriptive=True)).startswith("descriptive only")
    assert analysis.h6_reading(arm, {"missing": True}) == "INCOMPLETE"
    # through the analysis: a baseline that is as good and about as expensive as AWA is "equivalent"
    runs = result_set()
    for seed in range(4):
        runs[f"ARLOOP-2L_s{seed}"] = fake_run("ARLOOP-2L", seed, loop_acc(0.94 + 0.005 * seed))
    d = analysis.compute_decisions(runs)
    assert d["H6"]["baselines"]["ARLOOP-2L"]["capability"] == "no decided capability difference"
    assert d["H6"]["baselines"]["ARLOOP-2L"]["cost_at_matched_accuracy"]["reading"] == "comparable"
    assert d["H6"]["baselines"]["ARLOOP-2L"]["reading"] == "equivalent"
    # a baseline that beats AWA
    d = analysis.compute_decisions(result_set(awa_tops=(0.55, 0.56, 0.54, 0.57, 0.53)))
    assert d["H6"]["baselines"]["LOOPED-2L"]["reading"] == "LOOPED-2L beats AWA"
    # the matched accuracy is not above the chance floor: (ii) cannot decide
    runs = result_set()
    for arm_name, n_seeds in (("AWA-2L", 5), ("LOOPED-2L", 4)):
        for seed in range(n_seeds):
            runs[f"{arm_name}_s{seed}"] = fake_run(arm_name, seed, lambda k, s, b: 0.99 if k <= 3 else FLOORS[k])
    d = analysis.compute_decisions(runs)
    assert d["H6"]["baselines"]["LOOPED-2L"]["reading"] == \
        "not decided (not testable: the matched accuracy is not above the chance floor)"
    # a baseline with fewer seeds than planned is descriptive only; one that did not run at all leaves H6 open
    runs = result_set()
    del runs["LOOPED-2L_s3"]
    d = analysis.compute_decisions(runs)
    assert d["H6"]["baselines"]["LOOPED-2L"]["reading"].startswith("descriptive only")
    assert d["H6"]["decision"].startswith("LOOPED-2L: descriptive only") and "ARLOOP-2L: AWA beats" in d["H6"]["decision"]
    for seed in range(3):
        del runs[f"LOOPED-2L_s{seed}"]
    d = analysis.compute_decisions(runs)
    assert d["H6"]["decision"] == "INCOMPLETE" and d["H6"]["baselines"]["LOOPED-2L"]["reading"] == "INCOMPLETE"
    assert "| H6 | INCOMPLETE |" in analysis.render_summary(d, runs)
    runs = result_set()
    del runs["AWA-2L_s4"]
    h6 = analysis.compute_decisions(runs)["H6"]
    assert h6["decision"] == "INCOMPLETE" and {e["reading"] for e in h6["baselines"].values()} == {"INCOMPLETE"}


# ================================================================================ phase 0 and phase 1
def write_pilot(results, arm, lr, *, loss, k2_acc=0.1, k2_loss=3.0, k2_loss_200=3.2, steps=800, extension=None):
    run_dir = layout.run_dir(results, "0", arm, layout.PILOT_SEED, lr)
    run_dir.mkdir(parents=True)

    def val(step, loss_, acc, kloss):
        cells = {str(k): {s: {"loss": kloss, "acc": acc} for s in ("shuffled", "sorted")} for k in (1, 2, 3)}
        return {"type": "val", "step": step, "cells": cells, "competence": acc, "competence_loss": loss_}

    records = [val(200, loss + 1, 0.0, k2_loss_200), val(steps, loss, k2_acc, k2_loss)]
    final_steps = steps
    if extension:
        records.append(val(1600, loss, extension["k2_acc"], extension["k2_loss"]))
        final_steps = 1600
    (run_dir / "curve.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    (run_dir / "meta.json").write_text(json.dumps({"finished": True, "steps": final_steps,
                                                    "timing": {"s_per_step": 0.2}}), encoding="utf-8")
    return run_dir


def test_phase0_selection_rule_and_go_decision(tmp_path):
    assert phases.select_lr({"a": {"lr": 5e-4, "loss": 1.00}, "b": {"lr": 1e-3, "loss": 0.99}, "c": {"lr": 2e-3, "loss": 1.5}}) == 5e-4
    assert phases.select_lr({"a": {"lr": 5e-4, "loss": 1.03}, "b": {"lr": 1e-3, "loss": 0.99}}) == 1e-3
    assert phases.go_rule({"k2_acc": 0.30, "k2_loss": 9.0, "k2_loss_200": 1.0})
    assert phases.go_rule({"k2_acc": 0.0, "k2_loss": 0.8, "k2_loss_200": 1.0})
    assert not phases.go_rule({"k2_acc": 0.29, "k2_loss": 0.81, "k2_loss_200": 1.0})
    results = tmp_path / "go"
    for arm in arms.PILOT_ARMS:
        for lr, loss in zip(layout.PILOT_LRS, (1.2, 1.0, 1.01)):
            write_pilot(results, arm, lr, loss=loss, k2_acc=0.35 if (arm, lr) == ("AWA-2L", 2e-3) else 0.1)
    with pytest.raises(FileNotFoundError):
        phases.selected_lr(results, "AWA-2L")
    selection = phases.select_phase0(results)
    assert selection["go"] and not selection["awa_extended"] and selection["phase1_arms"] == list(arms.ARM_ORDER)
    assert selection["lrs"] == {**{arm: 1e-3 for arm in arms.PILOT_ARMS}, "STD-2L-pm": 1e-3}
    train.write_json(layout.selection_path(results), selection)
    assert phases.selected_lr(results, "LOOPED-2L") == 1e-3
    # NO-GO path: nothing passes at step 800, the best-lr AWA pilot is extended once, the rule is re-applied
    results = tmp_path / "nogo"
    for arm in arms.PILOT_ARMS:
        for lr, loss in zip(layout.PILOT_LRS, (1.2, 1.0, 1.1)):
            write_pilot(results, arm, lr, loss=loss)
    pending = phases.select_phase0(results)
    assert not pending["go"] and pending["extension_pending"] and not pending["awa_extended"]
    calls = []

    def extend(lr):
        calls.append(lr)
        shutil.rmtree(layout.run_dir(results, "0", "AWA-2L", layout.PILOT_SEED, lr))
        write_pilot(results, "AWA-2L", lr, loss=1.0, extension={"k2_acc": 0.05, "k2_loss": 3.1})

    selection = phases.select_phase0(results, extend=extend)
    assert calls == [1e-3] and selection["awa_extended"] and not selection["go"]
    assert selection["phase1_arms"] == ["ChaCAL-2L", "AWA-own-2L", "STD-2L-pm"]
    assert selection["pilot"]["AWA-2L"]["0.001"]["extension"]["steps"] == 1600
    assert phases.select_phase0(results, extend=extend) == selection and calls == [1e-3], "extended only once"
    train.write_json(layout.selection_path(results), selection)
    with pytest.raises(ValueError, match="not part of Phase 1"):
        phases.selected_lr(results, "AWA-2L")
    assert phases.selected_lr(results, "STD-2L-pm") == 1e-3
    shutil.rmtree(layout.run_dir(results, "0", "ARLOOP-2L", layout.PILOT_SEED, 2e-3))
    with pytest.raises(ValueError, match="unfinished pilot runs"):
        phases.select_phase0(results)


def test_extension_job_follows_priority_and_the_cpu_reserve(tmp_path, monkeypatch):
    results = tmp_path

    def write_run(arm, seed, competent, s_per_step=1.0):
        run_dir = layout.phase_dir(results, "1") / "runs" / f"{arm}_s{seed}"
        run_dir.mkdir(parents=True)
        (run_dir / "meta.json").write_text(json.dumps({
            "finished": True, "extended": False, "competence": {"competent": competent},
            "recipe": {"lr": 1e-3}, "timing": {"s_per_step": s_per_step, "cpu_train_s": 3000 * s_per_step}}),
            encoding="utf-8")

    write_run("ARLOOP-2L", 1, False)
    write_run("AWA-2L", 2, False)
    write_run("AWA-2L", 0, True)
    write_run("ChaCAL-2L", 3, False)
    write_run("LOOPED-2L", 0, False, s_per_step=0.9)
    write_run("STD-2L-pm", 0, False)                                # exempt
    queue = phases.extension_queue(results)
    assert [(q["arm"], q["seed"]) for q in queue] == [("AWA-2L", 2), ("ChaCAL-2L", 3), ("LOOPED-2L", 0), ("ARLOOP-2L", 1)]
    called = []

    def fake_train(run_dir, arm, seed, recipe, total_steps=None, progress=None):
        called.append((arm, seed, total_steps, recipe.lr, recipe.steps))
        return {"timing": {"cpu_train_s": 4500.0}, "competence": {"competent": True}}

    monkeypatch.setattr(train, "train", fake_train)
    ledger = phases.run_extensions(results)                         # reserve: 3,600 CPU-s; each extension 1,500
    assert called == [("AWA-2L", 2, 4500, 1e-3, 3000), ("ChaCAL-2L", 3, 4500, 1e-3, 3000)]
    assert ledger["spent_cpu_s"] == 3000.0 and ledger["reserve_cpu_s"] == 3600.0
    assert ledger["runs"]["AWA-2L_s2"] == {"funded": True, "cpu_s": 1500.0, "competent_after": True}
    assert ledger["runs"]["LOOPED-2L_s0"] == {"funded": False, "projected_cpu_s": 1350.0}
    assert ledger["runs"]["ARLOOP-2L_s1"]["funded"] is False and "STD-2L-pm_s0" not in ledger["runs"]
    assert phases.run_extensions(results) == ledger and len(called) == 2, "restarting the job repeats nothing"


def test_extension_ledger_books_an_extension_that_finished_just_before_a_kill(tmp_path, monkeypatch):
    """The job died after an extended run's meta.json was rewritten and before the ledger was: the run is no
    longer in the queue. A restart must still book its cost, or the reserve funds more than it holds."""
    results = tmp_path

    def write_run(arm, seed, *, extended, s_per_step=1.0, curve=True):
        run_dir = layout.phase_dir(results, "1") / "runs" / f"{arm}_s{seed}"
        run_dir.mkdir(parents=True)
        steps = 4500 if extended else 3000
        (run_dir / "meta.json").write_text(json.dumps({
            "finished": True, "extended": extended, "steps": steps, "planned_steps": 3000,
            "competence": {"competent": extended}, "recipe": {"lr": 1e-3},
            "timing": {"s_per_step": s_per_step, "cpu_train_s": steps * s_per_step}}), encoding="utf-8")
        if curve:
            records = [{"type": "train", "step": step, "cpu_s": step * s_per_step} for step in (2990, 3000, 4500)[:2 + extended]]
            (run_dir / "curve.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        return run_dir

    write_run("AWA-2L", 2, extended=True)                           # booked before the kill
    write_run("ChaCAL-2L", 3, extended=True)                        # finished, never booked
    write_run("LOOPED-2L", 0, extended=False, s_per_step=0.9)       # still failing: 1,350 CPU-s projected
    ledger_path = layout.phase_dir(results, "1") / "extensions.json"
    train.write_json(ledger_path, {"format": "awa3-extensions", "reserve_cpu_s": 3600.0, "spent_cpu_s": 1500.0,
                                   "runs": {"AWA-2L_s2": {"funded": True, "cpu_s": 1500.0, "competent_after": True}}})
    assert [(q["arm"], q["seed"]) for q in phases.extension_queue(results)] == [("LOOPED-2L", 0)]
    assert [(q["arm"], q["seed"]) for q in phases.extended_runs(results)] == [("AWA-2L", 2), ("ChaCAL-2L", 3)]
    called = []
    monkeypatch.setattr(train, "train", lambda *a, **k: called.append(a) or {})
    ledger = phases.run_extensions(results)
    assert ledger["runs"]["ChaCAL-2L_s3"] == {"funded": True, "cpu_s": 1500.0, "recovered": True,
                                              "competent_after": True}
    assert ledger["runs"]["AWA-2L_s2"] == {"funded": True, "cpu_s": 1500.0, "competent_after": True}
    assert ledger["spent_cpu_s"] == 3000.0, "both extensions are booked"
    assert ledger["runs"]["LOOPED-2L_s0"] == {"funded": False, "projected_cpu_s": 1350.0}, \
        "3,000 + 1,350 exceeds the reserve of 3,600 (1,500 + 1,350 would not have)"
    assert called == [] and json.loads(ledger_path.read_text(encoding="utf-8")) == ledger
    assert phases.run_extensions(results) == ledger, "booked once"
    # an extended run whose curve cannot show the cost is an error, not a silent zero
    write_run("AWA-own-2L", 1, extended=True, curve=False)
    with pytest.raises(RuntimeError, match="cost of the extension cannot be booked"):
        phases.run_extensions(results)


# ============================================================================ freeze, manifest, jobs
def copy_study(tmp_path):
    study = tmp_path / "research" / "attention_order_law_2026_11_13"
    shutil.copytree(STUDY, study, ignore=shutil.ignore_patterns("results", "__pycache__", "freeze_manifest.sha256",
                                                                "frozen_sets.json"))
    return study, tmp_path / "results"


def test_freeze_requires_a_frozen_protocol_and_pins_code_sets_and_selection(tmp_path):
    study, results = copy_study(tmp_path)
    selection = results / "phase0" / "selection.json"
    selection.parent.mkdir(parents=True)
    selection.write_text(json.dumps({"format": phases.SELECTION_FORMAT, "go": True, "extension_pending": False}))
    protocol = study / "PROTOCOL.md"
    text = protocol.read_text(encoding="utf-8")
    with pytest.raises(RuntimeError, match="NOT FROZEN"):
        manifest.write_freeze(results, study=study, repo=REPO)
    assert not manifest.check_freeze(results, study=study, repo=REPO)["ok"], "no freeze, no Phase 1"
    protocol.write_text(text.replace("NOT FROZEN", "FROZEN (unit test)"), encoding="utf-8")
    report = manifest.write_freeze(results, study=study, repo=REPO)
    listed = {label for _, label in manifest.read_manifest(study / "freeze_manifest.sha256")}
    assert report["n_files"] == len(listed)

    def outside(name):                                           # label of a repository file, relative to the study
        return pathlib.Path(os.path.relpath(REPO / name, study)).as_posix()

    assert {"PROTOCOL.md", "frozen_sets.json", "awa3/analysis.py", "awa3/task.py", "jobs_phase1.json",
            "results/phase0/selection.json", outside("tests/evolution/test_awa_study.py"),
            outside("tests/evolution/test_awa_attention.py"), outside("src/neural_network.py"),
            outside("src/neural_network_module/ortak_katman_module/attention_manager_module/absorbing_walk.py"),
            outside("model_management/config_schema.py")} <= listed
    assert len([label for label in listed if label.startswith("..")]) == len(manifest.TEST_FILES + manifest.ENGINE_FILES)
    assert "AMENDMENTS.md" not in listed and "freeze_manifest.sha256" not in listed
    frozen = json.loads((study / "frozen_sets.json").read_text(encoding="utf-8"))
    assert len(frozen["sets"]) == 18 and frozen["floors"]["6"] == task.floor(6)
    assert manifest.check_freeze(results, study=study, repo=REPO)["ok"]
    with pytest.raises(RuntimeError, match="frozen"):
        manifest.write_freeze(results, study=study, repo=REPO)
    (study / "AMENDMENTS.md").write_text("post-freeze note\n", encoding="utf-8")       # the log stays writable
    assert manifest.check_freeze(results, study=study, repo=REPO)["ok"]
    # any change of code, protocol, selection or evaluation sets breaks the check
    for target, label in ((study / "awa3" / "stats.py", "awa3/stats.py"), (protocol, "PROTOCOL.md"),
                          (selection, "results/phase0/selection.json")):
        original = target.read_bytes()
        target.write_bytes(original + b"\n# changed\n")
        bad = manifest.check_freeze(results, study=study, repo=REPO)
        assert not bad["ok"] and bad["mismatched"] == [label]
        target.write_bytes(original)
    sets_path = study / "frozen_sets.json"
    frozen["sets"]["test|k=6|n=1000"] = "0" * 64
    original = sets_path.read_bytes()
    sets_path.write_text(json.dumps(frozen, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    bad = manifest.check_freeze(results, study=study, repo=REPO)
    assert not bad["ok"] and not bad["sets_ok"]
    sets_path.write_bytes(original)
    (study / "awa3" / "extra.py").write_text("# not in the freeze\n", encoding="utf-8")
    bad = manifest.check_freeze(results, study=study, repo=REPO)
    assert not bad["ok"] and bad["unlisted"] == ["awa3/extra.py"]


def test_the_committed_study_is_not_frozen_and_phase1_refuses_to_start(tmp_path, capsys):
    protocol = (STUDY / "PROTOCOL.md").read_text(encoding="utf-8")
    assert ("NOT FROZEN" in protocol) != (STUDY / "freeze_manifest.sha256").exists()
    run_train = _load_script("run_train")
    code = run_train.main(["--phase", "1", "--arm", "AWA-2L", "--seed", "0", "--results", str(tmp_path)])
    if "NOT FROZEN" in protocol:
        assert code == 2 and "refusing to start a Phase-1 run" in capsys.readouterr().err
    assert not (tmp_path / "phase1").exists()
    for bad in (["--phase", "1", "--arm", "AWA-2L", "--seed", "7"], ["--phase", "1", "--arm", "AWA-2L", "--seed", "0", "--lr", "1e-3"],
                ["--phase", "0", "--arm", "STD-2L-pm", "--lr", "1e-3"], ["--phase", "0", "--arm", "AWA-2L", "--lr", "3e-3"],
                ["--phase", "smoke", "--arm", "AWA-2L"]):
        with pytest.raises(SystemExit):
            run_train.main(bad + ["--results", str(tmp_path)])


def _load_script(name):
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"awa3_script_{name}", STUDY / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_command_line_drivers_end_to_end(tmp_path, capsys):
    results = tmp_path / "results"
    run_train, run_eval = _load_script("run_train"), _load_script("run_eval")
    common = ["--phase", "smoke", "--arm", "ChaCAL-2L", "--lr", "1e-3", "--results", str(results)]
    assert run_train.main(common + ["--steps", "2", "--val-every", "2", "--ckpt-every", "2", "--val-items", "8"]) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    run_dir = pathlib.Path(out["run_dir"])
    assert run_dir == results / "phasesmoke" / "runs" / "ChaCAL-2L_lr0.001_s900" and out["steps"] == 2
    assert {p.name for p in run_dir.iterdir()} == {"curve.jsonl", "ckpt.pt", "meta.json"}
    assert run_eval.main(common + ["--limit", "4"]) == 0
    record = json.loads((run_dir / "eval.json").read_text(encoding="utf-8"))
    assert record["arm"] == "ChaCAL-2L" and record["truncated_to"] == 4 and record["seed"] == 900
    assert record["model_sha256"] == out["model_sha256"]
    assert _load_script("run_analyze").main(["--results", str(results), "--phase", "smoke", "--allow-truncated"]) == 0
    assert _load_script("run_verify").main(["--manifest", "--results", str(results)]) == 0
    assert _load_script("run_verify").main(["--rerun", "--phase", "smoke", "--results", str(results)]) == 0
    summary = (results / "phasesmoke" / "tables" / "summary.md").read_text(encoding="utf-8")
    assert "| H6 | INCOMPLETE |" in summary and "generated by run_analyze.py" in summary
    with pytest.raises(SystemExit):
        run_eval.main(["--phase", "1", "--arm", "AWA-2L", "--seed", "0", "--limit", "4", "--results", str(results)])


def test_job_files_match_the_generator_and_form_a_valid_plan():
    built_jobs = jobs.build()
    assert set(built_jobs) == {"jobs_phase0.json", "jobs_phase1.json", "jobs_phase1_nogo.json"}
    assert jobs.main(["--check"]) == 0, "run make_jobs.py --write after changing the plan"
    assert {name: len(v) for name, v in built_jobs.items()} == {
        "jobs_phase0.json": 16, "jobs_phase1.json": 57, "jobs_phase1_nogo.json": 31}
    for name, plan in built_jobs.items():
        assert json.loads((STUDY / name).read_text(encoding="utf-8")) == plan
        seen = set()
        for job in plan:
            assert set(job) == {"id", "cmd", "deps", "log"}
            assert job["id"] not in seen and all(dep in seen for dep in job["deps"]), "listed in a runnable order"
            seen.add(job["id"])
            words = job["cmd"].split()
            assert words[0] == "python" and words[1].startswith("research/attention_order_law_2026_11_13/")
            assert (REPO / words[1]).is_file()
            assert job["log"] == f"research/attention_order_law_2026_11_13/results/job_logs/{job['id']}.log"
    assert (STUDY / "results" / "job_logs" / ".gitkeep").is_file(), "the log folder is part of the checkout"
    p0 = built_jobs["jobs_phase0.json"]
    assert sum("run_train.py --phase 0" in j["cmd"] for j in p0) == 15 and p0[-1]["id"] == "p0_select"
    assert len(p0[-1]["deps"]) == 15
    p1 = {j["id"]: j for j in built_jobs["jobs_phase1.json"]}
    trains = [j for j in built_jobs["jobs_phase1.json"] if j["id"].startswith("p1_train_")]
    assert len(trains) == 26 and all(j["deps"] == ["p1_freeze_check"] for j in trains)
    assert [j["id"] for j in trains[:6]] == [f"p1_train_{arm}_s0" for arm in arms.ARM_ORDER], "seeds round-robin"
    assert trains[-1]["id"] == "p1_train_AWA-own-2L_s4"
    assert sorted(p1["p1_extend"]["deps"]) == sorted(j["id"] for j in trains)
    evals = [j for j in p1.values() if j["id"].startswith("p1_eval_")]
    assert len(evals) == 26 and all(j["deps"] == ["p1_extend"] for j in evals), "no evaluation before the extensions"
    assert sorted(p1["p1_analyze"]["deps"]) == sorted(j["id"] for j in evals)
    assert p1["p1_manifest"]["deps"] == ["p1_analyze"] and p1["p1_verify"]["deps"] == ["p1_manifest"]
    nogo = [j["id"] for j in built_jobs["jobs_phase1_nogo.json"] if j["id"].startswith("p1_train_")]
    assert len(nogo) == 13 and not any("AWA-2L" in j or "LOOPED" in j or "ARLOOP" in j for j in nogo)
