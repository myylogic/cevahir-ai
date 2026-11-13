# -*- coding: utf-8 -*-
"""Absorbing-Walk Attention (AWA) in the engine: optional attention operator of CevahirNeuralNetwork.

What is pinned here
-------------------
* Operator OFF (the default): the model is bit-identical to the engine before the operator existed
  (state_dict, logits, KV-cache decode, gradients and the position of the random stream). Three checks:
  a differential run against the engine exported from git at 0f9be09 (any machine that has the commit),
  digests recorded on ``main`` at 0f9be09 (the recording environment only: bit patterns depend on the
  machine), and recorded float64 summaries (any build of the same torch release). No new config keys,
  no new parameters.
* Mathematics: the K-step walk against an independent per-query reference and against the closed form
  a_i (I - C P(i))^-1 S(i); mass conservation; causality; the own-window ablation; the ChaCAL-style
  resolvent and the reduction own-window walk -> resolvent; lambda = 0 and K = 0 give standard attention.
* Engine paths: rows-only last layer == full forward, KV-cache decode == full forward (awa), a clear
  error for the operators without a cache path, GQA / QK-norm / sliding window / padding masks,
  gradients (gradcheck, live pointer at small lambda, gradient checkpointing), half-precision autocast
  (the operator stays in float32), the walk budget K in saved checkpoints.

CPU only, tiny models; the only files written are under pytest's tmp directories.
"""
from __future__ import annotations

import functools
import hashlib
import inspect
import io
import json
import logging
import math
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tarfile

import pytest
import torch

from model_management.checkpoint_contract import model_config, validate_model_identity
from model_management.config_schema import normalize_model_config, tiny_model_config
from model_management.model_manager import ModelManager
from src.neural_network import CevahirNeuralNetwork
from src.neural_network_module.ortak_katman_module.attention_manager_module import absorbing_walk as aw
from src.neural_network_module.ortak_katman_module.attention_manager_module.multi_head_attention import (
    MultiHeadAttention,
)

OPERATORS = ("awa", "awa_own", "chacal")
REPO = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _single_thread_quiet():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(previous)
    torch.set_num_threads(threads)


def build(seed: int = 0, **overrides):
    """A model exactly as the studies build it: tiny_model_config + ModelManager.initialize."""
    torch.manual_seed(seed)
    mm = ModelManager(dict(tiny_model_config(**overrides)))
    mm.initialize(build_optimizer=False, build_criterion=False, build_scheduler=False)
    return mm.model, mm.effective_config


def tokens(vocab: int = 128, batch: int = 3, length: int = 20, seed: int = 1) -> torch.Tensor:
    return torch.randint(0, vocab, (batch, length), generator=torch.Generator().manual_seed(seed))


def randomize_operator(model, seed: int = 7) -> None:
    """Non-trivial gate, null and gamma values, so that every term of the operators is exercised."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for layer in model.layers:
            attn = layer.attn
            if attn.attention_operator in aw.WALK_OPERATORS:
                attn.walk_gate.weight.copy_(0.4 * torch.randn(attn.walk_gate.weight.shape, generator=g))
                attn.walk_gate.bias.copy_(torch.randn(attn.walk_gate.bias.shape, generator=g))
                attn.walk_tau.copy_(torch.randn(attn.walk_tau.shape, generator=g))
                attn.walk_rho.copy_(torch.randn(attn.walk_rho.shape, generator=g) + 1.0)
            elif attn.attention_operator == "chacal":
                attn.chacal_rho.copy_(torch.randn(attn.chacal_rho.shape, generator=g))


# ------------------------------------------------------------------ operator OFF: bit-identical
# One claim, three checks, from the strongest to the most portable:
#   1. Differential (any environment that has the commit): the engine at BASE_COMMIT (main before the
#      operator existed) is exported from git and measured in a fresh process, next to the working tree in
#      another fresh process; every digest must be equal. CI fetches the full history for this.
#   2. Recorded digests: sha256 of state, logits, cached decode and gradients, recorded on main at 0f9be09
#      BEFORE the operator code existed (torch 2.15.0+cu132, x86_64, one thread). The bit patterns of a
#      forward pass depend on the BLAS kernels of the machine, so they are compared only there.
#   3. Recorded values (any build tag of torch 2.15.0 on x86_64, e.g. the +cpu wheels of CI): the position
#      of the global random stream right after the build, exactly (the build must consume exactly the
#      random numbers it consumed before), and float64 summaries of state, logits, cached decode and
#      gradients to a relative 1e-4. This check cannot see last-bit changes; checks 1 and 2 can.
BASE_COMMIT = "0f9be0957bb12d8ef1fbfc22a9c95dd94f175d49"
ENGINE_PATHS = ("src", "model_management")              # everything a ModelManager build imports
OPERATOR_FILE = "src/neural_network_module/ortak_katman_module/attention_manager_module/absorbing_walk.py"
GOLDEN_ENV = ("2.15.0+cu132", "x86_64")
GOLDEN_RELEASE = ("2.15.0", "x86_64")
GOLDEN_CONFIG_KEYS = ("4c61cd878fd723afdc404a399bede83d130403991c252b4c755ccb666d2007da", 47)
SUMMARY_TOLERANCE = 1e-4
GOLDEN_CASES = {
    "tiny_default": dict(),
    "cev11_like": dict(vocab_size=4096, embed_dim=192, seq_proj_dim=192, num_heads=4, num_kv_heads=4,
                       num_layers=6, ffn_dim=None, max_seq_length=64, max_cache_len=64, dropout=0.0,
                       use_gradient_checkpointing=False),
    "study_std": dict(vocab_size=72, embed_dim=128, seq_proj_dim=128, num_heads=4, num_kv_heads=4, num_layers=2,
                      ffn_dim=534, max_seq_length=128, max_cache_len=128, dropout=0.0,
                      use_gradient_checkpointing=False),
    "gqa_qknorm_window_manual": dict(vocab_size=96, embed_dim=64, seq_proj_dim=64, num_heads=4, num_kv_heads=2,
                                     num_layers=3, use_qk_norm=True, sliding_window=8, use_pytorch_sdpa=False,
                                     attn_logit_cap=5.0, max_seq_length=64, max_cache_len=64),
}
GOLDEN = {
    "tiny_default": {
        "rng_after_build": "c244bcd48bd0e4314480b427fff0c3283518f42e84d9ad81301beb30fe935921",
        "state": "82716ea53c6fe4d495322f1837b67fcb8a4891e5a42a1343a351d4bee80ddb39", "state_keys": 25,
        "logits": "3e00ab3194c9c08280bd95f064c907c07d6c97bbf34d21f2dcdbd5c1ae9bae3d",
        "cached_logits": "2a2346f5dc4b7b66c14f08dade41970f59b15589f331112f7fc23d80e79d26b1",
        "grads": "2c1f419530e795cffc52acd403671848177a6a2fdb6541a8eb1704c41ee26700"},
    "cev11_like": {
        "rng_after_build": "47c19dcc6758766decbc7437dd61d319955427fc74d26c966f2a3eea187ba68b",
        "state": "23e39afbd76b41108202804f806c2e25719017778c201db530ef5306d15da333", "state_keys": 65,
        "logits": "fd53848795643fa057d1e793542b2363e3051e913efa6770131a77a7755ebc5b",
        "cached_logits": "5306fd1ef49aa519e5913de44109a34000dc5b787d82909117743eec5a1230ea",
        "grads": "2b7788de39de887cd8a696935a938863b064697d0845f6313de4107aacf75637"},
    "study_std": {
        "rng_after_build": "b2a9677b4414d1c91edd16cad8f75c1951194f205f3d24f0e25eecbda686d0f3",
        "state": "8b5a52609f25b0bb96404457da497e1fccbf51e1a608d60e9ac27faf2849cd2c", "state_keys": 25,
        "logits": "0e765c4946c0291121a964e69f1946817a9622a301d5818c76097160b43337cd",
        "cached_logits": "22e1e8a39d0659b1533527a2214cdda96c7906b17e79c7fe687f3ce48924b8f7",
        "grads": "02a5e647c3a4882cc7ec5ec2476cdcc8dcb85ce6bcb26be857f0c375266a5574"},
    "gqa_qknorm_window_manual": {
        "rng_after_build": "2bf05471d0d05d224ad0e45fa2042aef7b87b1463d1ad00ac789998082968b45",
        "state": "22298695a459d2ce1d6ac2e608be14369414dc0dc448931f755bd9af6451a4d3", "state_keys": 41,
        "logits": "deb3ceec2b72131efbc83589bf080555c49fe5fe1a0b42b1d69530427ba518f1",
        "cached_logits": "390d9aead3c34de896c732bdc9e58d3999d321cbc6021a13cc37e004fb1be59b",
        "grads": "9e582b9e7a4bb8baa013a0550a847d3a7e801282f20de9caf4815352912ba702"},
}


# [sum |t|, sum t, sum t_i cos(i)] in float64, measured on the engine exported at 0f9be09, in a run whose
# digests equalled GOLDEN above
GOLDEN_SUMMARY = {
    "tiny_default": {
        "state": [1550.64997989317, 246.47913354636745, 8.544933302458418],
        "logits": [1172.4734084158117, 92.11786766445584, 18.705201810947898],
        "cached_logits": [1172.4734056959132, 92.11786794539512, 18.705201684833813],
        "grads": [37.38084783244423, -0.015605915202430154, 0.26130777384382486]},
    "cev11_like": {
        "state": [81407.41960343638, 3887.589967648195, -53.79577643783795],
        "logits": [35518.358581342494, 173.33222225824875, -62.69436830937582],
        "cached_logits": [35518.35863443975, 173.3322189567491, -62.69436666347866],
        "grads": [420.1086316536854, -0.04771753187655425, -1.3748343969682222]},
    "study_std": {
        "state": [14468.106778064695, 1080.4904807421735, -19.554326587199192],
        "logits": [745.9776328394453, 112.83077161626147, -17.280701849796223],
        "cached_logits": [745.9776271767696, 112.83077001605852, -17.280699585870764],
        "grads": [149.9808877125669, 0.26197888048366724, 1.4653884140731066]},
    "gqa_qknorm_window_manual": {
        "state": [4388.442063790761, 833.0995854827011, 6.905254067553748],
        "logits": [927.6255711385311, 120.40672538836952, -23.488424254763643],
        "cached_logits": [927.6255686884979, 120.40672369867389, -23.48842412589504],
        "grads": [76.11601076229047, 0.0943269277462278, 0.34800451351202677]},
}


def _tensor_digest(t: torch.Tensor) -> str:
    t = t.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return hashlib.sha256(repr(tuple(t.shape)).encode() + b"\0" + t.numpy().tobytes()).hexdigest()


def _state_digest(model) -> str:
    h = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state):
        t = state[key].detach().to(device="cpu", dtype=torch.float32).contiguous()
        h.update(key.encode("utf-8") + b"\0" + repr(tuple(t.shape)).encode("ascii") + b"\0")
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def _summary(tensors) -> list:
    """[sum |t|, sum t, sum t_i cos(i)] of the concatenated tensors in float64: moves with any real change."""
    v = torch.cat([t.detach().to(torch.float64).reshape(-1) for t in tensors])
    w = torch.cos(torch.arange(v.numel(), dtype=torch.float64))
    return [float(v.abs().sum()), float(v.sum()), float((v * w).sum())]


def _golden_measure(model, vocab: int) -> dict:
    """{"digests": ..., "summary": ...} of state, logits, cached decode and gradients (fixed input)."""
    state = model.state_dict()
    digests = {"state": _state_digest(model), "state_keys": len(state)}
    summary = {"state": _summary([state[k] for k in sorted(state)])}
    x = torch.randint(0, vocab, (3, 24), generator=torch.Generator().manual_seed(1234))
    model.eval()
    with torch.no_grad():
        logits = model(x)[0]
        model.clear_kv_cache()
        steps = [model(x[:, :8], use_cache=True)[0]]
        steps += [model(x[:, i:i + 1], use_cache=True)[0] for i in range(8, 24)]
        cached = torch.cat(steps, dim=1)
        model.clear_kv_cache()
    digests["logits"], digests["cached_logits"] = _tensor_digest(logits), _tensor_digest(cached)
    summary["logits"], summary["cached_logits"] = _summary([logits]), _summary([cached])
    model.train()
    logits, _ = model(x)
    torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1)).backward()
    h = hashlib.sha256()
    grads = []
    for name, p in sorted(model.named_parameters()):
        if p.grad is not None:
            h.update(name.encode() + b"\0" + p.grad.detach().contiguous().numpy().tobytes())
            grads.append(p.grad)
    digests["grads"] = h.hexdigest()
    summary["grads"] = _summary(grads)
    return {"digests": digests, "summary": summary}


def _measure_built(model, cfg) -> dict:
    """Everything that is compared with the engine before AWA. Call it right after build(): it first reads
    the position of the global random stream."""
    rng_after_build = hashlib.sha256(torch.rand(8).numpy().tobytes()).hexdigest()
    keys = sorted(cfg)
    measured = _golden_measure(model, cfg["vocab_size"])
    measured["digests"] = {"rng_after_build": rng_after_build, **measured["digests"]}
    measured["config_keys"] = [hashlib.sha256("\n".join(keys).encode()).hexdigest(), len(keys)]
    return measured


@functools.lru_cache(maxsize=None)
def _working_tree(case: str) -> dict:
    """The golden case measured in this process (once), with the facts that need the live model."""
    model, cfg = build(0, **GOLDEN_CASES[case])
    measured = _measure_built(model, cfg)
    measured["operator_state_keys"] = [k for k in model.state_dict() if "walk_" in k or "chacal_" in k]
    measured["operator"] = (model.attention_operator, list(model.attention_operator_layers),
                            [layer.attn.attention_operator for layer in model.layers])
    return measured


@pytest.mark.parametrize("case", sorted(GOLDEN_CASES))
def test_operator_off_is_bit_identical_to_the_engine_before_awa(case):
    measured = _working_tree(case)
    # environment-independent part: no new config keys, no operator parameters, operator reported off
    assert tuple(measured["config_keys"]) == GOLDEN_CONFIG_KEYS
    assert not measured["operator_state_keys"]
    assert measured["digests"]["state_keys"] == GOLDEN[case]["state_keys"]
    operator, layers, per_layer = measured["operator"]
    assert operator == "standard" and layers == [] and set(per_layer) == {"standard"}
    if (torch.__version__, platform.machine()) != GOLDEN_ENV:
        pytest.skip(f"digests were recorded with torch {GOLDEN_ENV[0]} on {GOLDEN_ENV[1]}; "
                    f"this is torch {torch.__version__} on {platform.machine()} (the differential test and "
                    f"the recorded-values test cover other environments)")
    assert measured["digests"] == GOLDEN[case]


@pytest.mark.parametrize("case", sorted(GOLDEN_CASES))
def test_operator_off_reproduces_the_values_recorded_before_awa_on_any_build_of_the_same_torch(case):
    """The +cpu wheels of CI are not the recording build: bit patterns may differ, the numbers may not."""
    if (torch.__version__.split("+")[0], platform.machine()) != GOLDEN_RELEASE:
        pytest.skip(f"values were recorded with torch {GOLDEN_RELEASE[0]} on {GOLDEN_RELEASE[1]}; "
                    f"this is torch {torch.__version__} on {platform.machine()}")
    measured = _working_tree(case)
    assert measured["digests"]["rng_after_build"] == GOLDEN[case]["rng_after_build"], \
        "the build consumed other random numbers than before"
    assert measured["summary"].keys() == GOLDEN_SUMMARY[case].keys()
    for name, recorded in GOLDEN_SUMMARY[case].items():
        scale = SUMMARY_TOLERANCE * recorded[0]
        assert recorded[0] > 0.0
        assert all(abs(a - b) <= scale for a, b in zip(measured["summary"][name], recorded)), \
            (case, name, measured["summary"][name], recorded)


def _measure_script() -> str:
    """A stand-alone script with the measuring code of this file (the same source, so the two cannot drift)."""
    functions = (build, _tensor_digest, _state_digest, _summary, _golden_measure, _measure_built)
    return "\n".join([
        "from __future__ import annotations",
        "import hashlib, json, logging, pathlib, sys",
        "import torch",
        "from model_management.config_schema import tiny_model_config",
        "from model_management.model_manager import ModelManager",
        "import model_management, src",
        "", "",
        "\n\n".join(inspect.getsource(f) for f in functions),
        "",
        "if __name__ == '__main__':",
        "    torch.set_num_threads(1)",
        "    logging.disable(logging.CRITICAL)",
        "    tree = pathlib.Path(sys.argv[1]).resolve()",
        "    for module in (model_management, src):",
        "        origin = pathlib.Path(list(module.__path__)[0]).resolve()",
        "        assert origin.parent == tree, f'{module.__name__} was imported from {origin}, not from {tree}'",
        "    out = {name: _measure_built(*build(0, **overrides)) for name, overrides in json.loads(sys.argv[2]).items()}",
        "    print('\\nAWA-MEASURE ' + json.dumps(out, sort_keys=True))",
        "",
    ])


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True)


@pytest.fixture(scope="module")
def engine_before_and_after(tmp_path_factory):
    """Golden-case measurements of the engine at BASE_COMMIT and of the working tree, each in a fresh process."""
    if shutil.which("git") is None or _git("cat-file", "-e", BASE_COMMIT + "^{commit}").returncode != 0:
        pytest.skip(f"commit {BASE_COMMIT[:7]} (the engine before AWA) is not in the local git history "
                    f"(shallow clone or source archive): the differential check cannot run here")
    root = tmp_path_factory.mktemp("awa_engine_before")
    archive = _git("archive", "--format=tar", BASE_COMMIT, *ENGINE_PATHS)
    assert archive.returncode == 0, archive.stderr.decode(errors="replace")
    base = root / "base"
    base.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        tar.extractall(base, filter="data")
    assert not (base / OPERATOR_FILE).exists(), "the base commit must not contain the operator"
    assert (REPO / OPERATOR_FILE).is_file()
    script = root / "script" / "measure.py"
    script.parent.mkdir()
    script.write_text(_measure_script(), encoding="utf-8")
    procs = {}
    for name, tree in (("before", base), ("after", REPO)):
        env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1",
               "PYTHONPATH": os.pathsep.join(filter(None, [str(tree), os.environ.get("PYTHONPATH")]))}
        procs[name] = subprocess.Popen([sys.executable, str(script), str(tree), json.dumps(GOLDEN_CASES)],
                                       cwd=str(script.parent), env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True)
    results = {}
    for name, proc in procs.items():
        stdout, stderr = proc.communicate(timeout=840)
        assert proc.returncode == 0, f"measuring the engine {name} AWA failed:\n{stderr[-4000:]}"
        line = [ln for ln in stdout.splitlines() if ln.startswith("AWA-MEASURE ")][-1]
        results[name] = json.loads(line[len("AWA-MEASURE "):])
    return results


@pytest.mark.parametrize("case", sorted(GOLDEN_CASES))
def test_operator_off_is_bit_identical_to_the_engine_at_the_base_commit_in_this_environment(
        case, engine_before_and_after):
    """Same seed, same process setup, engine before and after: identical state_dict, logits, cached decode,
    gradients, config keys and position of the random stream. No recorded value is involved."""
    before, after = (engine_before_and_after[side][case] for side in ("before", "after"))
    assert before["digests"].keys() == {"rng_after_build", "state", "state_keys", "logits", "cached_logits", "grads"}
    assert after["digests"] == before["digests"]
    assert after["config_keys"] == before["config_keys"] == list(GOLDEN_CONFIG_KEYS)
    assert after["summary"] == before["summary"]


def test_operator_off_leaves_configs_untouched_and_explicit_standard_is_the_same_model():
    cfg = tiny_model_config()
    assert "attention_operator" not in cfg and "attention_operator_options" not in cfg
    assert normalize_model_config(cfg) == cfg                       # saved effective configs rebuild unchanged
    base, base_cfg = build(3)
    assert "attention_operator" not in base_cfg
    assert "attention_operator" not in model_config(base) and "attention_operator" not in base._ctor_cfg
    explicit, explicit_cfg = build(3, attention_operator="standard")
    assert explicit_cfg["attention_operator"] == "standard" and "attention_operator_options" not in explicit_cfg
    assert "attention_operator" not in explicit._ctor_cfg
    assert _state_digest(base) == _state_digest(explicit)
    x = tokens()
    base.eval(), explicit.eval()
    with torch.no_grad():
        assert torch.equal(base(x)[0], explicit(x)[0])
    validate_model_identity(explicit, model_config(base))            # a default checkpoint loads into either
    validate_model_identity(base, {**model_config(base), "attention_operator": "standard"})


# ------------------------------------------------------------------ construction and config path
@pytest.mark.parametrize("operator", OPERATORS)
def test_operator_models_share_every_standard_weight_and_add_the_stated_parameters(operator):
    std, _ = build(5)
    model, cfg = build(5, attention_operator=operator)
    D, H = 32, 4
    extra = {k: v for k, v in model.state_dict().items() if k not in std.state_dict()}
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in std.state_dict().items())
    assert all(k.startswith("layers.1.attn.") for k in extra), "default: the last layer only"
    n_extra = sum(v.numel() for v in extra.values())
    assert n_extra == (H if operator == "chacal" else D * D + H * (D + 1) + 2 * H)
    assert cfg["attention_operator"] == operator
    assert cfg["attention_operator_options"] == {**aw.DEFAULT_OPERATOR_OPTIONS, "layers": [1]}
    assert model.attention_operator_layers == [1]
    assert model.layers[0].attn.attention_operator == "standard"
    assert model.layers[1].attn.attention_operator == operator
    if operator != "chacal":
        lam = aw.walk_lambda(model.layers[1].attn).detach()
        assert torch.allclose(lam, torch.full((H,), 0.5), atol=1e-6) and float(lam.min()) > 0.0
        assert float(model.layers[1].attn.walk_gate.weight.detach().abs().max()) == 0.0
    else:
        assert torch.allclose(aw.chacal_gamma(model.layers[1].attn).detach(), torch.full((H,), 0.5), atol=1e-6)
    # the stored constructor config rebuilds the same architecture and identifies the operator
    rebuilt = CevahirNeuralNetwork(**model_config(model))
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    validate_model_identity(rebuilt, model_config(model))
    other = "awa_own" if operator != "awa_own" else "awa"
    wrong, _ = build(5, attention_operator=other)
    with pytest.raises(ValueError, match="attention_operator"):
        validate_model_identity(wrong, model_config(model))
    with pytest.raises(ValueError, match="attention_operator"):
        validate_model_identity(std, model_config(model))


@pytest.mark.parametrize("operator", OPERATORS)
def test_checkpoint_round_trip_through_the_manager_and_the_loader(operator, tmp_path):
    """Save with ModelManager, restore without any external config: same operator, same function."""
    from model_management.model_loader import ModelLoader

    torch.manual_seed(21)
    source = ModelManager(dict(tiny_model_config(attention_operator=operator,
                                                 attention_operator_options={"num_hops": 3})))
    source.initialize(build_optimizer=False, build_criterion=False, build_scheduler=False)
    randomize_operator(source.model)
    source.model.eval()
    path = source.save(str(tmp_path / "model.pth"))
    x = tokens()
    with torch.no_grad():
        expected = source.model(x)[0]
        restored = ModelLoader.load_model(type(source.model), path, device="cpu").eval()
        assert restored.attention_operator == operator and restored.attention_operator_layers == [1]
        if operator != "chacal":
            assert restored.layers[1].attn.walk_num_hops == 3
        assert torch.equal(restored(x)[0], expected)
    # a model of the same shapes but another operator refuses the weights before touching them
    other = "awa_own" if operator == "awa" else "awa" if operator == "awa_own" else None
    if other is not None:
        target = ModelManager(dict(tiny_model_config(attention_operator=other,
                                                     attention_operator_options={"num_hops": 3})))
        target.initialize(build_optimizer=False, build_criterion=False, build_scheduler=False)
        before = {k: v.clone() for k, v in target.model.state_dict().items()}
        with pytest.raises(RuntimeError) as error:
            target.load(path, weights_only=True)
        assert "attention_operator" in str(error.value.__cause__)
        assert all(torch.equal(v, target.model.state_dict()[k]) for k, v in before.items())


def test_operator_layers_options_and_validation():
    model, cfg = build(0, num_layers=3, attention_operator="awa",
                       attention_operator_options={"layers": [0, -1], "num_hops": 2, "lambda_init": 0.25,
                                                   "tau_init": 1.5, "score_cap": 10.0})
    assert model.attention_operator_layers == [0, 2] == cfg["attention_operator_options"]["layers"]
    assert [layer.attn.attention_operator for layer in model.layers] == ["awa", "standard", "awa"]
    attn = model.layers[0].attn
    assert attn.walk_num_hops == 2 and attn.walk_score_cap == 10.0 and attn.walk_exclude_self is True
    assert torch.allclose(aw.walk_lambda(attn), torch.full((4,), 0.25), atol=1e-6)
    assert torch.allclose(attn.walk_tau, torch.full((4,), 1.5))
    assert normalize_model_config(cfg, strict=False) == cfg
    assert normalize_model_config({"vocab_size": 16, "attention_operator": "chacal"}, strict=True)[
        "attention_operator_options"]["layers"] == [7]
    bad = [
        dict(attention_operator="walk"),
        dict(attention_operator="awa", attention_operator_options={"hops": 3}),
        dict(attention_operator="awa", attention_operator_options={"num_hops": -1}),
        dict(attention_operator="awa", attention_operator_options={"num_hops": True}),
        dict(attention_operator="awa", attention_operator_options={"lambda_init": 0.99}),
        dict(attention_operator="awa", attention_operator_options={"lambda_init": 0.0}),
        dict(attention_operator="awa", attention_operator_options={"tau_init": float("nan")}),
        dict(attention_operator="awa", attention_operator_options={"score_cap": -1.0}),
        dict(attention_operator="awa", attention_operator_options={"exclude_self": 1}),
        dict(attention_operator="awa", attention_operator_options={"layers": [2]}),
        dict(attention_operator="awa", attention_operator_options={"layers": [1, -1]}),
        dict(attention_operator="awa", attention_operator_options={"layers": []}),
        dict(attention_operator="awa", attention_operator_options={"layers": "1"}),
        dict(attention_operator="chacal", attention_operator_options={"gamma_init": 1.0}),
        dict(attention_operator="awa", attention_operator_options=[1]),
        dict(attention_operator="standard", attention_operator_options={"num_hops": 3}),
        dict(attention_operator_options={"num_hops": 3}),
    ]
    for overrides in bad:
        with pytest.raises(ValueError, match="attention_operator"):
            tiny_model_config(**overrides)
        with pytest.raises(ValueError, match="attention_operator"):
            CevahirNeuralNetwork(learning_rate=1e-3, dropout=0.0, vocab_size=16, embed_dim=16, seq_proj_dim=16,
                                 num_heads=2, num_layers=2, log_level=50, **overrides)
    module = MultiHeadAttention(embed_dim=16, num_heads=2, dropout=0.0)
    module.set_attention_operator("awa")
    with pytest.raises(ValueError, match="already carries"):
        module.set_attention_operator("chacal")


def test_set_attention_hops_changes_the_budget_at_inference():
    model, _ = build(2, attention_operator="awa")
    randomize_operator(model)
    model.eval()
    x = tokens()
    with torch.no_grad():
        k4 = model(x)[0]
        model.set_attention_hops(9)
        assert model.layers[1].attn.walk_num_hops == 9
        k9 = model(x)[0]
        model.set_attention_hops(0)
        k0 = model(x, return_attention_weights=True)[0]
    assert float((k4 - k9).abs().max()) > 1e-6
    std, _ = build(2)
    std.eval()
    with torch.no_grad():
        assert torch.equal(k0, std(x, return_attention_weights=True)[0]), "K = 0 is the standard manual path"
    with pytest.raises(ValueError, match="no walk-operator layer"):
        std.set_attention_hops(3)
    with pytest.raises(ValueError, match="non-negative integer"):
        model.set_attention_hops(-1)


@pytest.mark.parametrize("operator", aw.WALK_OPERATORS)
def test_the_walk_budget_set_at_inference_is_saved_and_is_not_part_of_the_checkpoint_identity(operator, tmp_path):
    """set_attention_hops(K), save, load: the restored model walks K steps. Weights saved at one K load into a
    model configured with another K (the budget is weight-tied), which keeps its own K."""
    from model_management.model_loader import ModelLoader

    def manager(**options):
        torch.manual_seed(22)
        mm = ModelManager(dict(tiny_model_config(attention_operator=operator, attention_operator_options=options)))
        mm.initialize(build_optimizer=False, build_criterion=False, build_scheduler=False)
        return mm

    source = manager()
    randomize_operator(source.model)
    source.model.eval()
    x = tokens()
    with torch.no_grad():
        at_default = source.model(x)[0]                              # K = 4, the configured budget
        source.model.set_attention_hops(9)
        at_nine = source.model(x)[0]
    assert float((at_default - at_nine).abs().max()) > 1e-6
    assert model_config(source.model)["attention_operator_options"]["num_hops"] == 9
    path = source.save(str(tmp_path / "model.pth"))
    with torch.no_grad():
        restored = ModelLoader.load_model(type(source.model), path, device="cpu").eval()
        assert restored.layers[1].attn.walk_num_hops == 9
        assert torch.equal(restored(x)[0], at_nine), "the saved budget comes back with the weights"
        # a model built with the original configuration (K = 4) accepts the weights and keeps its own budget
        target = manager()
        target.load(path, weights_only=True)
        target.model.eval()
        assert target.model.layers[1].attn.walk_num_hops == 4
        assert torch.equal(target.model(x)[0], at_default)
        target.model.set_attention_hops(9)
        assert torch.equal(target.model(x)[0], at_nine)
    validate_model_identity(target.model, model_config(source.model))
    validate_model_identity(manager(num_hops=2).model, model_config(source.model))
    # every other option stays part of the identity: it changes the function without changing a shape
    for options in ({"score_cap": 5.0}, {"exclude_self": False}, {"num_hops": 9, "tau_init": 1.0}):
        other = manager(**options)
        before = {k: v.clone() for k, v in other.model.state_dict().items()}
        with pytest.raises(RuntimeError) as error:
            other.load(path, weights_only=True)
        assert "attention_operator_options" in str(error.value.__cause__)
        assert all(torch.equal(v, other.model.state_dict()[k]) for k, v in before.items())


# ------------------------------------------------------------------ reductions to standard attention
def test_lambda_zero_equals_standard_attention():
    std, _ = build(4)
    model, _ = build(4, attention_operator="awa")
    randomize_operator(model)
    attn = model.layers[1].attn
    x = tokens()
    std.eval(), model.eval()
    with torch.no_grad():
        fused = std(x)[0]                                           # F.scaled_dot_product_attention path
        manual, manual_w = std(x, return_attention_weights=True)    # explicit softmax path
        attn.walk_rho.fill_(float("-inf"))                          # lambda = 0 exactly
        assert float(aw.walk_lambda(attn).abs().max()) == 0.0
        zero, zero_w = model(x, return_attention_weights=True)
        assert torch.equal(zero, manual) and torch.equal(zero_w, manual_w), "bit-exact on the manual path"
        assert float((zero - fused).abs().max()) <= 1e-6
        attn.walk_rho.fill_(aw._logit(1e-3 / aw.LAMBDA_MAX))        # a small gate moves the output a little
        small = model(x)[0]
        delta = float((small - fused).abs().max())
        assert 1e-8 < delta < 1e-2
        attn.walk_rho.fill_(0.0)
        attn.walk_tau.fill_(60.0)                                    # tau -> +inf: every hop takes the null option
        assert float((model(x)[0] - fused).abs().max()) <= 1e-6


def test_chacal_gamma_zero_and_own_window_lambda_zero_equal_standard_attention():
    std, _ = build(4)
    x = tokens()
    std.eval()
    with torch.no_grad():
        fused = std(x)[0]
    for operator, name in (("chacal", "chacal_rho"), ("awa_own", "walk_rho")):
        model, _ = build(4, attention_operator=operator)
        randomize_operator(model)
        model.eval()
        with torch.no_grad():
            getattr(model.layers[1].attn, name).fill_(float("-inf"))
            assert float((model(x)[0] - fused).abs().max()) <= 1e-6, operator


# ------------------------------------------------------------------ the operators (float64 references)
def _random_problem(n=11, B=2, H=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    scores = torch.randn(B, H, n, n, generator=g, dtype=torch.float64) * 2
    causal = torch.ones(n, n, dtype=torch.bool).tril()
    A = torch.softmax(scores.masked_fill(~causal, float("-inf")), -1)
    E = torch.exp(torch.randn(B, H, n, n, generator=g, dtype=torch.float64) * 2)
    E = E.masked_fill(torch.eye(n, dtype=torch.bool), 0.0)
    c = 0.9 * torch.rand(B, H, n, generator=g, dtype=torch.float64)
    tau = torch.randn(H, generator=g, dtype=torch.float64)
    return scores, A, E, c, tau, causal


def _reference_walk(a, E, c, null, K, window):
    """Independent per-query implementation with the explicit matrix P(i) (one query row, one head)."""
    n = E.shape[0]
    w = window.to(E.dtype)
    Z = null + (E * w.view(1, n)).sum(-1)                       # Z_j = e^tau + sum_{l in W_i} E_jl
    P = E * w.view(1, n) / Z.view(n, 1)
    p_null = null / Z
    pi, out = a.clone(), torch.zeros_like(a)
    for _ in range(K):
        out = out + pi * (1 - c + c * p_null)
        pi = (pi * c) @ P
    return out + pi


def test_query_window_walk_matches_the_per_query_reference_and_the_closed_form():
    _, A, E, c, tau, causal = _random_problem()
    B, H, n, _ = A.shape
    null = torch.exp(tau)
    allowed = causal.view(1, 1, n, n)
    walk = aw.absorbing_walk(A, E, c, null, 5, allowed=allowed)
    worst = 0.0
    for b in range(B):
        for h in range(H):
            for i in range(n):
                ref = _reference_walk(A[b, h, i], E[b, h], c[b, h], null[h], 5, causal[i])
                worst = max(worst, float((ref - walk[b, h, i]).abs().max()))
    assert worst < 1e-12, worst
    assert float((walk.sum(-1) - 1).abs().max()) < 1e-12, "rows stay probability vectors"
    assert float(walk.masked_select(~causal.expand_as(walk)).abs().max()) == 0.0, "nothing outside the window"
    # one pass gives the whole K axis; budget 0 is hop 1 itself
    axis = aw.absorbing_walk(A, E, c, null, 5, allowed=allowed, return_all=True)
    assert len(axis) == 6 and axis[0] is A and torch.equal(axis[5], walk)
    assert torch.equal(axis[3], aw.absorbing_walk(A, E, c, null, 3, allowed=allowed))
    # K -> infinity is a_i (I - C P(i))^-1 S(i); the K-step error is bounded by 2 * max(c)^K
    closed = aw.absorbing_walk_resolvent(A, E, c, null, allowed=allowed)
    assert float((aw.absorbing_walk(A, E, c, null, 400, allowed=allowed) - closed).abs().max()) < 1e-12
    for K in (1, 3, 8):
        err = float((axis[K] if K <= 5 else aw.absorbing_walk(A, E, c, null, K, allowed=allowed)).sub(closed)
                    .abs().sum(-1).max())
        assert err <= 2 * 0.9 ** K + 1e-12, (K, err)
    assert float((closed.sum(-1) - 1).abs().max()) < 1e-12
    # the hop trace is the moving mass pi(m)
    trace = []
    aw.absorbing_walk(A, E, c, null, 3, allowed=allowed, trace=trace)
    assert len(trace) == 4 and trace[0] is A
    assert float((trace[1] - torch.matmul(A * c.unsqueeze(-2) / (
        torch.matmul(allowed.double(), E.transpose(-1, -2)) + null.view(1, H, 1, 1)), E) * allowed).abs().max()) < 1e-12
    with pytest.raises(ValueError, match="non-negative"):
        aw.absorbing_walk(A, E, c, null, -1)


def test_walk_respects_sliding_windows_and_padding():
    _, _, E, c, tau, causal = _random_problem(n=10, seed=3)
    B, H, n, _ = E.shape
    pos = torch.arange(n)
    window = causal & ((pos.view(n, 1) - pos.view(1, n)) < 4)       # sliding window of 4
    pad = torch.zeros(B, n, dtype=torch.bool)
    pad[1, 2] = pad[1, 5] = True                                     # batch 1: keys 2 and 5 are padding
    allowed = window.view(1, 1, n, n) & ~pad.view(B, 1, 1, n)
    scores = torch.randn(B, H, n, n, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    A = torch.softmax(scores.masked_fill(~allowed, float("-inf")), -1)
    null = torch.exp(tau)
    walk = aw.absorbing_walk(A, E, c, null, 6, allowed=allowed)
    assert float(walk.masked_select(~allowed.expand_as(walk)).abs().max()) == 0.0
    assert float((walk.sum(-1) - 1).abs().max()) < 1e-12
    worst = 0.0
    for b in range(B):
        for h in range(H):
            for i in range(n):
                ref = _reference_walk(A[b, h, i], E[b, h], c[b, h], null[h], 6, allowed[b, 0, i])
                worst = max(worst, float((ref - walk[b, h, i]).abs().max()))
    assert worst < 1e-12, worst
    assert float((aw.absorbing_walk(A, E, c, null, 500, allowed=allowed)
                  - aw.absorbing_walk_resolvent(A, E, c, null, allowed=allowed)).abs().max()) < 1e-12


def test_own_window_walk_is_lower_triangular_and_differs_from_the_query_window_walk():
    _, A, E, c, tau, causal = _random_problem(seed=4)
    B, H, n, _ = A.shape
    null = torch.exp(tau)
    own = aw.absorbing_walk(A, E, c, null, 5, own_window=causal.view(1, 1, n, n))
    # reference: one fixed lower-triangular transition matrix for all queries
    Et = E * causal
    P = Et / (null.view(1, H, 1, 1) + Et.sum(-1, keepdim=True))
    p_null = null.view(1, H, 1) / (null.view(1, H, 1) + Et.sum(-1))
    pi, ref = A, torch.zeros_like(A)
    for _ in range(5):
        ref = ref + pi * (1 - c + c * p_null).unsqueeze(-2)
        pi = (pi * c.unsqueeze(-2)) @ P
    ref = ref + pi
    assert float((own - ref).abs().max()) < 1e-12
    assert float(own.masked_select(~causal.expand_as(own)).abs().max()) == 0.0
    assert float((own.sum(-1) - 1).abs().max()) < 1e-12
    assert float(P.masked_select(~causal.expand_as(P)).abs().max()) == 0.0, "every factor is lower-triangular"
    query = aw.absorbing_walk(A, E, c, null, 5, allowed=causal.view(1, 1, n, n))
    assert float((own - query).abs().max()) > 1e-3, "the normalisation window is the single difference"
    # the last query's window is everything, but its hops still differ (own windows are prefixes)
    assert float((own[:, :, 0] - query[:, :, 0]).abs().max()) < 1e-12, "position 0 has nowhere to hop"


def test_chacal_resolvent_brute_force_and_reduction_from_the_own_window_walk():
    scores, A, _, _, _, causal = _random_problem(n=9, seed=1)
    B, H, n, _ = A.shape
    gamma = torch.tensor([0.3, 0.6, 0.9], dtype=torch.float64)
    rows = torch.tensor([[n - 1, 3], [5, n - 1]])
    eye = torch.eye(n, dtype=torch.float64)
    brute = (1 - gamma.view(1, H, 1, 1)) * A @ torch.linalg.inv(eye - gamma.view(1, H, 1, 1) * A)
    full = aw.chacal_resolvent(A, gamma)
    fast = aw.chacal_resolvent(A, gamma, rows)
    assert float((full - brute).abs().max()) < 1e-12
    assert float((aw._gather_rows(brute, rows) - fast).abs().max()) < 1e-12
    assert float((aw.chacal_resolvent(A, gamma, lower_triangular=False) - brute).abs().max()) < 1e-12
    assert float((full.sum(-1) - 1).abs().max()) < 1e-12, "mass conservation"
    assert float(full.masked_select(~causal.expand_as(full)).abs().max()) == 0.0
    assert float((aw.chacal_resolvent(A, torch.zeros(H, dtype=torch.float64)) - A).abs().max()) < 1e-15
    # resolvent = own-window walk with pointer logits = attention logits, self-hops, no cap, no null,
    # c = gamma and K -> infinity
    E = torch.exp(scores)
    walk = aw.absorbing_walk(aw._gather_rows(A, rows), E, gamma.view(1, H, 1).expand(B, H, n),
                             torch.exp(torch.full((H,), -1e4, dtype=torch.float64)) + 1e-300, 600,
                             own_window=causal.view(1, 1, n, n))
    assert float((walk - fast).abs().max()) < 1e-10
    with pytest.raises(ValueError, match="square"):
        aw.chacal_resolvent(A[:, :, :4], gamma)


def test_transition_weights_cap_and_self_exclusion():
    g = torch.Generator().manual_seed(2)
    t = torch.randn(1, 2, 6, 8, generator=g, dtype=torch.float64) * 30
    k = torch.randn(1, 2, 6, 8, generator=g, dtype=torch.float64) * 30
    raw = (t @ k.transpose(-1, -2)) / math.sqrt(8)
    E = aw.transition_weights(t, k, 20.0, True)
    off = ~torch.eye(6, dtype=torch.bool)
    assert float(E.diagonal(dim1=-2, dim2=-1).abs().max()) == 0.0
    assert torch.allclose(E[..., off], torch.exp(20.0 * torch.tanh(raw / 20.0))[..., off], rtol=1e-12)
    assert float(E.max()) <= math.exp(20.0) and float(E[..., off].min()) >= math.exp(-20.0)
    assert float(raw.abs().max()) > 100.0, "the cap is exercised"
    uncapped = aw.transition_weights(t / 30, k / 30, 0.0, False)
    assert torch.allclose(uncapped, torch.exp(raw / 900), rtol=1e-12)


# ------------------------------------------------------------------ module against the reference
VARIANTS = {
    "default": dict(),
    "gqa_qknorm_window_manual": dict(num_kv_heads=2, use_qk_norm=True, sliding_window=6, use_pytorch_sdpa=False,
                                     attn_logit_cap=5.0),
    "mqa_three_layers": dict(num_kv_heads=1, num_layers=3),
}


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_module_computes_the_reference_walk_with_gqa_qknorm_window(variant):
    """The effective weights the engine returns equal the float64 reference built from its own parts."""
    model, _ = build(6, attention_operator="awa", attention_operator_options={"num_hops": 5},
                     **VARIANTS[variant])
    randomize_operator(model)
    model = model.double().eval()
    layer, attn = model.layers[-1], model.layers[-1].attn
    x = tokens(length=14)
    seen = {}
    hook = layer.register_forward_pre_hook(lambda mod, args: seen.setdefault("h", args[0]))
    with torch.no_grad():
        _, eff = model(x, return_attention_weights=True)
        hook.remove()
        h = layer.norm1(seen["h"])
        B, n, _ = h.shape
        H, Hk, d = attn.num_heads, attn.num_kv_heads, attn.head_dim
        q = attn.query_proj(h).view(B, n, H, d).transpose(1, 2)
        k = attn.key_proj(h).view(B, n, Hk, d).transpose(1, 2)
        if attn.use_qk_norm:
            q, k = attn.q_norm(q), attn.k_norm(k)
        k = k.repeat_interleave(attn.num_kv_groups, dim=1)
        kbar = k
        pos = torch.arange(n)
        q = attn.positional_encoding.apply_rotary_pos_emb(q, pos)
        k = attn.positional_encoding.apply_rotary_pos_emb(k, pos)
        s = q @ k.transpose(-1, -2) / math.sqrt(d)
        if attn.attn_logit_cap > 0:
            s = torch.tanh(s / attn.attn_logit_cap) * attn.attn_logit_cap
        window = pos.view(1, n) <= pos.view(n, 1)
        if attn.sliding_window is not None:
            window = window & ((pos.view(n, 1) - pos.view(1, n)) < attn.sliding_window)
        A = torch.softmax(s.masked_fill(~window, float("-inf")), -1)
        t = attn.walk_pointer(h).view(B, n, H, d).transpose(1, 2)
        E = torch.exp(20.0 * torch.tanh(t @ kbar.transpose(-1, -2) / (20.0 * math.sqrt(d))))
        E = E.masked_fill(torch.eye(n, dtype=torch.bool), 0.0)
        c = aw.LAMBDA_MAX * torch.sigmoid(attn.walk_rho).view(1, H, 1) * torch.sigmoid(attn.walk_gate(h)).transpose(1, 2)
        assert torch.allclose(c, aw.continue_probability(attn, h), atol=1e-12)
        null = torch.exp(attn.walk_tau)
        worst = 0.0
        for b in range(B):
            for head in range(H):
                for i in range(n):
                    ref = _reference_walk(A[b, head, i], E[b, head], c[b, head], null[head], 5, window[i])
                    worst = max(worst, float((ref - eff[b, head, i]).abs().max()))
    assert worst < 1e-10, worst
    assert float((eff.sum(-1) - 1).abs().max()) < 1e-10
    assert float(eff.masked_select(~window.expand_as(eff)).abs().max()) == 0.0


@pytest.mark.parametrize("operator", OPERATORS)
def test_padding_masks_keep_mass_inside_the_window(operator):
    model, _ = build(8, attention_operator=operator)
    randomize_operator(model)
    model.eval()
    x = tokens(batch=3, length=12)
    blocked = torch.zeros(3, 12, dtype=torch.bool)
    blocked[0, 9:] = True
    blocked[2, 5:] = True                                    # (B, S) key padding, True = blocked
    with torch.no_grad():
        _, eff = model(x, mask=blocked, return_attention_weights=True)
        _, plain = model(x, return_attention_weights=True)
    pos = torch.arange(12)
    window = (pos.view(1, 12) <= pos.view(12, 1)).view(1, 1, 12, 12) & ~blocked.view(3, 1, 1, 12)
    assert float(eff.masked_select(~window.expand_as(eff)).abs().max()) == 0.0
    assert float((eff.sum(-1) - 1).abs().max()) < 1e-5
    assert float((eff - plain)[1].abs().max()) < 1e-6, "an unpadded batch element is unaffected"
    assert float((eff - plain)[0].abs().max()) > 1e-4


def test_cross_attention_awa_walks_over_the_keys_and_square_operators_refuse():
    torch.manual_seed(0)
    module = MultiHeadAttention(embed_dim=16, num_heads=2, dropout=0.0).eval()
    module.set_attention_operator("awa", num_hops=3)
    q, src = torch.randn(2, 5, 16), torch.randn(2, 9, 16)
    with torch.no_grad():
        out, eff = module(q, src, src, return_attention_weights=True)
    assert out.shape == (2, 5, 16) and eff.shape == (2, 2, 5, 9)
    assert float((eff.sum(-1) - 1).abs().max()) < 1e-5
    for operator in ("awa_own", "chacal"):
        other = MultiHeadAttention(embed_dim=16, num_heads=2, dropout=0.0).eval()
        other.set_attention_operator(operator)
        with pytest.raises(ValueError, match="self-attention only"):
            other(q, src, src)
        assert other(q).shape == (2, 5, 16)


# ------------------------------------------------------------------ rows-only last layer
ROWS_VARIANTS = {
    "default": dict(),
    "gqa_qknorm_window_cap": dict(num_kv_heads=2, use_qk_norm=True, sliding_window=6, attn_logit_cap=5.0),
    "parallel_residual": dict(parallel_residual=True),
    "post_norm_layernorm_gelu": dict(pre_norm=False, use_rmsnorm=False, use_swiglu=False),
    "three_layers": dict(num_layers=3),
}


@pytest.mark.parametrize("operator", ("standard",) + OPERATORS)
@pytest.mark.parametrize("variant", sorted(ROWS_VARIANTS))
def test_rows_only_last_layer_equals_the_full_forward(operator, variant):
    model, _ = build(9, attention_operator=operator, **ROWS_VARIANTS[variant])
    randomize_operator(model)
    model.eval()
    x = tokens(length=17)
    rows = torch.tensor([[16, 3], [7, 7], [12, 0]])
    blocked = torch.zeros(3, 17, dtype=torch.bool)
    blocked[1, 2] = True
    with torch.no_grad():
        for mask in (None, blocked):
            full = model(x, mask=mask)[0]
            part = model.forward_rows(x, rows, mask=mask)
            want = full.gather(1, rows.unsqueeze(-1).expand(3, 2, full.shape[-1]))
            assert part.shape == want.shape
            assert float((part - want).abs().max()) < 2e-6, (operator, variant, mask is not None)
        shared = model.forward_rows(x, torch.tensor([16, 5]))        # one row list for the whole batch
        assert float((shared - model(x)[0][:, [16, 5]]).abs().max()) < 2e-6
    with pytest.raises(ValueError, match="rows"):
        model.forward_rows(x, torch.tensor([17]))
    with pytest.raises(ValueError, match="rows"):
        model.forward_rows(x, torch.tensor([0.5]))


def test_rows_only_gives_the_whole_hop_axis_from_one_pass():
    model, _ = build(10, attention_operator="awa")
    randomize_operator(model)
    std, _ = build(10)
    model.eval(), std.eval()
    x = tokens(length=15)
    rows = torch.tensor([14])
    with torch.no_grad():
        axis = model.forward_rows(x, rows, num_hops=6, return_all_hops=True)
        assert len(axis) == 7
        for K in (0, 3, 6):
            assert float((axis[K] - model.forward_rows(x, rows, num_hops=K)).abs().max()) < 1e-6
        assert torch.equal(axis[0], std.forward_rows(x, rows)), "budget 0 is plain attention"
        assert float((axis[4] - model.forward_rows(x, rows)).abs().max()) < 1e-6, "default budget = walk_num_hops"
        with pytest.raises(ValueError, match="walk operator"):
            std.forward_rows(x, rows, return_all_hops=True)
    moe, _ = build(0, use_moe=True, num_experts=2, moe_top_k=1)
    with pytest.raises(NotImplementedError, match="MoE"):
        moe.forward_rows(x, rows)


# ------------------------------------------------------------------ KV cache
CACHE_VARIANTS = {
    "default": dict(),
    "sliding_window": dict(sliding_window=7),
    "gqa_qknorm": dict(num_kv_heads=2, use_qk_norm=True),
    "two_awa_layers": dict(num_layers=3, attention_operator_options={"layers": [0, 2]}),
}


@pytest.mark.parametrize("variant", sorted(CACHE_VARIANTS))
def test_awa_kv_cache_decode_equals_the_full_forward(variant):
    model, _ = build(11, attention_operator="awa", **CACHE_VARIANTS[variant])
    randomize_operator(model)
    model.eval()
    x = tokens(batch=2, length=36)
    with torch.no_grad():
        full = model(x)[0]
        for _ in range(2):                                   # second round: clear_kv_cache() leaves no stale state
            model.clear_kv_cache()
            steps = [model(x[:, :4], use_cache=True)[0]]
            steps += [model(x[:, i:i + 1], use_cache=True)[0] for i in range(4, 36)]     # 32 decode steps
            assert float((torch.cat(steps, dim=1) - full).abs().max()) < 1e-5
        for layer in model.layers:
            cache = layer.attn.kv_cache
            if layer.attn.attention_operator == "awa":
                assert torch.equal(cache._operator_state["positions"], cache.positions)
            else:
                assert not hasattr(cache, "_operator_state")
        # another sequence with another batch size right after: no leakage from the old cache
        y = tokens(batch=3, length=9, seed=5)
        model.clear_kv_cache()
        steps = [model(y[:, i:i + 1], use_cache=True)[0] for i in range(9)]
        assert float((torch.cat(steps, dim=1) - model(y)[0]).abs().max()) < 1e-5


def test_awa_kv_cache_survives_eviction_and_detects_missing_walk_state():
    model, _ = build(12, attention_operator="awa", max_cache_len=8, max_seq_length=64, kv_num_sink_tokens=2)
    randomize_operator(model)
    model.eval()
    x = tokens(batch=1, length=20)
    attn = model.layers[1].attn
    with torch.no_grad():
        model.clear_kv_cache()
        for i in range(20):
            logits = model(x[:, i:i + 1], use_cache=True)[0]
            assert torch.isfinite(logits).all()
            state = attn.kv_cache._operator_state
            assert torch.equal(state["positions"], attn.kv_cache.positions)
            assert state["pointer"].shape[2] == attn.kv_cache.cache_len <= 8
        assert attn.kv_cache.positions.tolist() == [0, 1, 14, 15, 16, 17, 18, 19], "sinks + sliding tail"
        # a cache whose walk state was lost must not be decoded with silently wrong transitions
        del attn.kv_cache._operator_state
        with pytest.raises(RuntimeError, match="out of sync with the KV cache"):
            model(x[:, :1], use_cache=True, cache_position=torch.tensor([20]))


@pytest.mark.parametrize("operator", ("awa_own", "chacal"))
def test_operators_without_a_cache_path_raise_a_clear_error(operator):
    model, _ = build(13, attention_operator=operator)
    x = tokens(batch=1, length=6)
    model.eval()
    with torch.no_grad():
        with pytest.raises(NotImplementedError, match="no KV-cache path"):
            model(x, use_cache=True)
        assert model(x)[0].shape == (1, 6, 128)
    model.train()                                            # the cache is inference-only: training ignores it
    assert model(x, use_cache=True)[0].shape == (1, 6, 128)


# ------------------------------------------------------------------ half-precision autocast
def _active_autocast(dtype) -> bool:
    """Whether CPU autocast really runs matmuls in ``dtype`` in this torch build."""
    with torch.autocast("cpu", dtype=dtype):
        return torch.mm(torch.ones(2, 2), torch.ones(2, 2)).dtype == dtype


HALF_DTYPES = pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))


@HALF_DTYPES
def test_walk_primitives_compute_in_float32_inside_half_precision_autocast(dtype):
    """E reaches e^20 = 4.9e8, far above the fp16 range (65504): inside autocast the primitives must give
    exactly what they give outside it, in float32."""
    if not _active_autocast(dtype):
        pytest.skip(f"CPU autocast with {dtype} is not available in torch {torch.__version__}")
    g = torch.Generator().manual_seed(5)
    pointer, address = torch.randn(2, 3, 9, 8, generator=g) * 30, torch.randn(2, 3, 9, 8, generator=g) * 30
    _, A, _, c, tau, causal = _random_problem(n=9, B=2, H=3, seed=6)
    A, c, null, allowed = A.float(), c.float(), torch.exp(tau.float()), causal.view(1, 1, 9, 9)
    gamma = torch.tensor([0.3, 0.6, 0.9])
    E = aw.transition_weights(pointer, address, 20.0, True)
    assert E.dtype == torch.float32 and 1e8 < float(E.max()) <= math.exp(20.0) * (1 + 1e-6)
    outside = {
        "E": E,
        "awa": aw.absorbing_walk(A, E, c, null, 5, allowed=allowed),
        "awa_own": aw.absorbing_walk(A, E, c, null, 5, own_window=allowed),
        "closed": aw.absorbing_walk_resolvent(A, E, c, null, allowed=allowed),
        "chacal": aw.chacal_resolvent(A, gamma),
    }
    with torch.autocast("cpu", dtype=dtype):
        assert torch.matmul(A, E).dtype == dtype, "autocast is active around the primitives"
        inside = {
            "E": aw.transition_weights(pointer, address, 20.0, True),
            "awa": aw.absorbing_walk(A, E, c, null, 5, allowed=allowed),
            "awa_own": aw.absorbing_walk(A, E, c, null, 5, own_window=allowed),
            "closed": aw.absorbing_walk_resolvent(A, E, c, null, allowed=allowed),
            "chacal": aw.chacal_resolvent(A, gamma),
        }
        # half-precision arguments are widened before anything is multiplied
        from_half = aw.absorbing_walk(A.to(dtype), E, c.to(dtype), null.to(dtype), 5, allowed=allowed)
        E_from_half = aw.transition_weights((pointer / 30).to(dtype), (address / 30).to(dtype), 20.0, True)
        assert torch.matmul(A, E).dtype == dtype, "the caller's autocast state is restored"
    for name, value in inside.items():
        assert value.dtype == torch.float32 and torch.isfinite(value).all(), name
        assert torch.equal(value, outside[name]), name
    assert from_half.dtype == torch.float32 and torch.isfinite(from_half).all()
    assert float((from_half.sum(-1) - A.to(dtype).float().sum(-1)).abs().max()) < 1e-5, "mass of the fp16 rows is kept"
    assert E_from_half.dtype == torch.float32 and torch.isfinite(E_from_half).all()


@HALF_DTYPES
@pytest.mark.parametrize("operator", OPERATORS)
def test_operators_stay_finite_and_accurate_under_half_precision_autocast(operator, dtype):
    """The engine's GPU training loops run under autocast(float16). The operator must not turn that into NaN:
    forward, rows-only forward, cached decode (awa) and backward stay finite and close to float32."""
    if not _active_autocast(dtype):
        pytest.skip(f"CPU autocast with {dtype} is not available in torch {torch.__version__}")
    model, _ = build(17, attention_operator=operator)
    randomize_operator(model)
    attn = model.layers[1].attn
    x = tokens()
    tolerance = 0.02 if dtype == torch.float16 else 0.06            # measured: 5e-4 (fp16), 3e-3 (bf16)
    model.eval()
    with torch.no_grad():
        if operator != "chacal":
            attn.walk_pointer.weight.mul_(5.0)                       # pointer scores reach the cap
            seen = {}
            hook = model.layers[1].register_forward_pre_hook(lambda mod, args: seen.setdefault("h", args[0]))
            model(x)
            hook.remove()
            h = model.layers[1].norm1(seen["h"])
            pointer = attn.walk_pointer(h).view(3, 20, 4, 8).transpose(1, 2)
            address = attn.key_proj(h).view(3, 20, 4, 8).transpose(1, 2)
            assert float(aw.transition_weights(pointer, address, 20.0, True).max()) > 1e6, \
                "the transition weights of this model do not fit fp16 (max 65504)"
        reference, reference_weights = model(x, return_attention_weights=True)
        rows = torch.tensor([19, 4])
        with torch.autocast("cpu", dtype=dtype):
            out, weights = model(x, return_attention_weights=True)
            part = model.forward_rows(x, rows)
            if operator == "awa":
                model.clear_kv_cache()
                steps = [model(x[:, :5], use_cache=True)[0]]
                steps += [model(x[:, i:i + 1], use_cache=True)[0] for i in range(5, 20)]
                model.clear_kv_cache()
                cached = torch.cat(steps, dim=1)
                assert torch.isfinite(cached).all()
                assert float((cached.float() - reference).abs().max()) < tolerance
    assert out.dtype == dtype, "the rest of the model does run in half precision"
    assert weights.dtype == torch.float32, "the operator's effective weights are float32"
    assert torch.isfinite(out).all() and torch.isfinite(weights).all() and torch.isfinite(part).all()
    assert float((weights.sum(-1) - 1).abs().max()) < 1e-3, "rows stay probability vectors"
    assert float((weights - reference_weights).abs().max()) < tolerance
    assert float((out.float() - reference).abs().max()) < tolerance
    assert float((part.float() - reference[:, [19, 4]]).abs().max()) < tolerance
    # training step under autocast: finite loss, finite gradients, live operator parameters
    model.train()
    with torch.autocast("cpu", dtype=dtype):
        logits = model(x)[0]
    loss = torch.nn.functional.cross_entropy(logits.float()[:, :-1].reshape(-1, 128), x[:, 1:].reshape(-1))
    loss.backward()
    assert math.isfinite(float(loss.detach()))
    names = ("chacal_rho",) if operator == "chacal" else ("walk_pointer.weight", "walk_gate.weight", "walk_rho",
                                                            "walk_tau")
    params = dict(attn.named_parameters())
    for name in names:
        assert params[name].grad is not None and float(params[name].grad.abs().sum()) > 0.0, name
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


# ------------------------------------------------------------------ gradients
def test_gradcheck_of_the_walk_and_the_resolvent():
    g = torch.Generator().manual_seed(0)
    n, H = 5, 2
    causal = torch.ones(n, n, dtype=torch.bool).tril().view(1, 1, n, n)
    scores = torch.randn(1, H, n, n, generator=g, dtype=torch.float64, requires_grad=True)
    pointer = torch.randn(1, H, n, 4, generator=g, dtype=torch.float64, requires_grad=True)
    address = torch.randn(1, H, n, 4, generator=g, dtype=torch.float64, requires_grad=True)
    gate = torch.randn(1, H, n, generator=g, dtype=torch.float64, requires_grad=True)
    rho = torch.randn(H, generator=g, dtype=torch.float64, requires_grad=True)
    tau = torch.randn(H, generator=g, dtype=torch.float64, requires_grad=True)

    def run(scores, pointer, address, gate, rho, tau, own=False):
        A = torch.softmax(scores.masked_fill(~causal, float("-inf")), -1)
        E = aw.transition_weights(pointer, address, 20.0, True)
        c = aw.LAMBDA_MAX * torch.sigmoid(rho).view(1, H, 1) * torch.sigmoid(gate)
        window = dict(own_window=causal) if own else dict(allowed=causal)
        return aw.absorbing_walk(A, E, c, torch.exp(tau), 3, **window)

    inputs = (scores, pointer, address, gate, rho, tau)
    assert torch.autograd.gradcheck(run, inputs, eps=1e-6, atol=1e-5)
    assert torch.autograd.gradcheck(lambda *a: run(*a, own=True), inputs, eps=1e-6, atol=1e-5)
    assert torch.autograd.gradcheck(
        lambda s, r: aw.chacal_resolvent(torch.softmax(s.masked_fill(~causal, float("-inf")), -1),
                                         aw.GAMMA_MAX * torch.sigmoid(r)), (scores, rho), eps=1e-6, atol=1e-5)

    def closed(scores, pointer, address, gate, rho, tau):
        A = torch.softmax(scores.masked_fill(~causal, float("-inf")), -1)
        E = aw.transition_weights(pointer, address, 20.0, True)
        c = aw.LAMBDA_MAX * torch.sigmoid(rho).view(1, H, 1) * torch.sigmoid(gate)
        return aw.absorbing_walk_resolvent(A, E, c, torch.exp(tau), allowed=causal)

    assert torch.autograd.gradcheck(closed, inputs, eps=1e-6, atol=1e-5)


@pytest.mark.parametrize("operator", OPERATORS)
def test_every_operator_parameter_receives_gradient_through_the_model(operator):
    model, _ = build(14, attention_operator=operator)
    model.train()
    x = tokens()
    for rows_only in (False, True):
        model.zero_grad(set_to_none=True)
        if rows_only:
            logits = model.forward_rows(x, torch.tensor([19]))[:, 0]
        else:
            logits = model(x)[0][:, -1]
        torch.nn.functional.cross_entropy(logits, x[:, 0]).backward()
        attn = model.layers[1].attn
        names = ("chacal_rho",) if operator == "chacal" else ("walk_pointer.weight", "walk_gate.weight",
                                                                "walk_gate.bias", "walk_rho", "walk_tau")
        params = dict(attn.named_parameters())
        for name in names:
            grad = params[name].grad
            assert grad is not None and torch.isfinite(grad).all() and float(grad.abs().sum()) > 0.0, (name, rows_only)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_gate_parametrisation_keeps_the_pointer_alive_at_small_lambda():
    model, _ = build(15, attention_operator="awa", attention_operator_options={"lambda_init": 1e-3, "num_hops": 3})
    attn = model.layers[1].attn
    model.train()
    x = tokens()
    model.forward_rows(x, torch.tensor([19])).logsumexp(-1).sum().backward()
    assert float(attn.walk_pointer.weight.grad.abs().sum()) > 0.0
    assert float(aw.walk_lambda(attn).detach().min()) > 0.0


def test_gradient_checkpointing_gives_the_same_gradients_with_an_awa_layer():
    grads = {}
    x = tokens()
    for flag in (False, True):
        model, _ = build(16, attention_operator="awa", use_gradient_checkpointing=flag)
        randomize_operator(model)
        model.train()
        assert all(layer.use_gradient_checkpointing is flag for layer in model.layers)
        logits = model(x)[0]
        torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 128), x[:, 1:].reshape(-1)).backward()
        grads[flag] = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    assert grads[False].keys() == grads[True].keys() and "layers.1.attn.walk_pointer.weight" in grads[True]
    assert max(float((grads[False][n] - grads[True][n]).abs().max()) for n in grads[True]) <= 1e-6


def test_attention_dropout_is_applied_after_the_walk():
    torch.manual_seed(0)
    module = MultiHeadAttention(embed_dim=16, num_heads=2, dropout=0.5)
    module.set_attention_operator("awa", num_hops=3)
    x = torch.randn(2, 7, 16)
    module.train()
    _, dropped = module(x, causal_mask=True, return_attention_weights=True)
    module.eval()
    with torch.no_grad():
        _, clean = module(x, causal_mask=True, return_attention_weights=True)
    kept = dropped != 0
    assert kept.any() and (~kept).any()
    assert torch.allclose(dropped[kept], 2.0 * clean[kept], atol=1e-6), "surviving entries are scaled walk weights"
