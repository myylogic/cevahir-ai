# -*- coding: utf-8 -*-
"""Model kurulumu, kanonik ağırlık özeti ve denetimli checkpoint (CEV-11 PROTOCOL §2).

Model yalnız ``model_management.config_schema.tiny_model_config`` + ``ModelManager.initialize`` yoluyla kurulur
(bugünkü pilotun yolu); hiçbir depo dosyası değiştirilmez. Birincil yapılandırma: vocab 4096, D = seq_proj = 192,
4 baş, 4 KV başı, 6 katman, ``ffn_dim=None`` (SwiGLU → 512), RMSNorm ön-norm, RoPE, bağlı gömmeler, dropout 0.0,
gradient checkpointing kapalı, ``max_seq_length = max_cache_len = 64``, fp32.

Checkpoint güvenliği (geçmiş bir hata): farklı baş sayısıyla yeniden kurulan bir model, aynı şekilli ağırlıkları
``load_state_dict(strict=True)`` ile hatasız yükler ama BAŞKA bir işlev hesaplar (Q/K/V/O matrisleri baş sayısından
bağımsız olarak [D, D] şeklindedir). Bu yüzden:

* ``build_model`` canlı modülleri okuyarak mimariyi (``architecture``) çıkarır ve istenen değerlerle karşılaştırır;
* ``save_checkpoint`` kaydedilen ``cfg``'nin canlı modeli tanımladığını doğrular;
* ``load_checkpoint`` modeli VARSAYILANLARDAN DEĞİL kaydedilen tam yapılandırmadan yeniden kurar, yeniden kurulan
  ``cfg``'nin (tam etkin yapılandırma + canlı mimari + kurulum argümanları) kaydedilenle birebir aynı olmasını şart
  koşar, ağırlık özetini (``model_sha256``) ve tokenizer kimliğini doğrular; herhangi bir uyuşmazlıkta hata verir.

INTERFACES.md'ye göre eklemeler (imzalar korunmuştur):
* ``build_model`` arayüzdeki ezmelere ek olarak ``use_gradient_checkpointing=False`` geçirir (PROTOCOL §2'de kapalı;
  tiny ön ayarında zaten kapalıdır — açık yazmak ön ayar değişirse sessiz sapmayı önler).
* ``build_model`` küresel torch RNG'yi ``torch.random.fork_rng`` içinde tohumlar; çağıranın RNG durumu değişmez.
* ``load_checkpoint`` modeli ``eval()`` kipinde döndürür.
* Yardımcılar: ``architecture``, ``tokenizer_path``, ``code_sha256``, ``CKPT_FORMAT``.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import pathlib
from typing import Any

import torch

CKPT_FORMAT = "cev11-checkpoint"
CKPT_VERSION = 1
CFG_FORMAT = "cev11-model-cfg-v1"

# Kurulum sırasında depo modüllerinin INFO günlüklerini bastırmak için (yalnız geçici; eski düzey geri yüklenir).
_NOISY_LOGGERS = ("ModelManager", "model_initializer")

# Tokenizer sınıfları: kayıtlı sınıf adına göre açık eşleme (bilinmeyen sınıf → hata, sessiz yedek yok).
_TOKENIZER_CLASSES = ("ByteLevelBPE", "FusedFreqBPE")


# ---------------------------------------------------------------------------------------------------- yardımcılar
def _json_canonical(obj: Any) -> Any:
    """JSON'a çevrilemeyen bir değer varsa hata verir; demetler listeye döner (kayıt/yükleme simetrisi için)."""
    return json.loads(json.dumps(obj, sort_keys=True, allow_nan=False))


@contextlib.contextmanager
def _quiet_repo_loggers(level: int = logging.WARNING):
    saved = {}
    for name in _NOISY_LOGGERS:
        lg = logging.getLogger(name)
        saved[name] = lg.level
        lg.setLevel(max(level, lg.level))
    try:
        yield
    finally:
        for name, lvl in saved.items():
            logging.getLogger(name).setLevel(lvl)


def _diff(a: Any, b: Any, path: str = "") -> list[str]:
    """İki iç içe yapı arasındaki farkların okunur listesi (hata iletileri için)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out: list[str] = []
        for k in sorted(set(a) | set(b), key=str):
            p = f"{path}.{k}" if path else str(k)
            if k not in a:
                out.append(f"{p}: missing in saved, rebuilt={b[k]!r}")
            elif k not in b:
                out.append(f"{p}: saved={a[k]!r}, missing in rebuilt")
            else:
                out.extend(_diff(a[k], b[k], p))
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(_diff(x, y, f"{path}[{i}]"))
        return out
    return [] if a == b else [f"{path}: saved={a!r}, rebuilt={b!r}"]


def tokenizer_path(ckpt_path) -> pathlib.Path:
    """Checkpoint'in yanındaki tokenizer dosyası: ``model.pt`` → ``model.tokenizer.json``."""
    p = pathlib.Path(ckpt_path)
    return p.with_name(p.stem + ".tokenizer.json")


def code_sha256(pkg_dir=None) -> tuple[str, dict[str, str]]:
    """cev11 paketinin kod özeti: ``__pycache__`` dışındaki .py/.tsv/.json/.txt dosyaları, göreli yol sırasıyla.

    Döner: (toplam sha256, {göreli yol: dosya sha256}).
    """
    root = pathlib.Path(pkg_dir) if pkg_dir is not None else pathlib.Path(__file__).resolve().parent
    files = sorted(p for p in root.rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".tsv", ".json", ".txt"))
    total = hashlib.sha256()
    per_file: dict[str, str] = {}
    for p in files:
        rel = p.relative_to(root).as_posix()
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        per_file[rel] = digest
        total.update(rel.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\n")
    return total.hexdigest(), per_file


# ---------------------------------------------------------------------------------------------------- mimari
def architecture(model) -> dict:
    """İşlevi belirleyen mimariyi CANLI modüllerden okur (yapılandırmadan değil).

    Baş sayısı, KV başı, baş boyutu, FFN genişliği, katman sayısı, norm türü, bağlı gömme, dropout'lar vb.
    Aynı şekilli ağırlıklara rağmen farklı işlev hesaplayan bir yeniden kurulumu yakalamak için kullanılır.
    """
    layers = []
    for layer in model.layers:
        a, f = layer.attn, layer.ffn
        layers.append({
            "num_heads": int(a.num_heads), "num_kv_heads": int(a.num_kv_heads), "head_dim": int(a.head_dim),
            "embed_dim": int(a.embed_dim), "use_rope": bool(a.use_rope), "use_qk_norm": bool(a.use_qk_norm),
            "sliding_window": a.sliding_window, "attn_logit_cap": float(a.attn_logit_cap),
            "use_kv_cache": bool(a.use_kv_cache), "max_cache_len": int(a.max_cache_len),
            "kv_eviction_strategy": str(a.kv_eviction_strategy),
            "ffn_dim": int(f.ffn_dim), "ffn_activation": str(f.activation_name), "ffn_bias": bool(f.use_bias),
            "fc2_shape": [int(s) for s in f.fc2.weight.shape],
            "norm": type(layer.norm1).__name__, "pre_norm": bool(layer.pre_norm),
            "parallel_residual": bool(layer.parallel_residual), "use_moe": bool(layer.use_moe),
            "gradient_checkpointing": bool(layer.use_gradient_checkpointing),
            "drop_path_rate": float(layer.drop_path_rate),
        })
    emb = model.embedding.embedding.weight
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "num_layers": len(layers),
        "vocab_size": int(emb.shape[0]),
        "embed_dim": int(emb.shape[1]),
        "tied_embeddings": model.output_layer.weight is emb,
        "output_norm": type(model.output_norm).__name__,
        "logit_soft_cap": float(model.logit_soft_cap),
        "pe_mode": str(model.pos_encoding.mode),
        "causal": bool(model.causal_mask),
        "dropouts": sorted({float(m.p) for m in model.modules() if isinstance(m, torch.nn.Dropout)}),
        "n_params": int(n_params),
        "n_params_non_embedding": int(n_params - emb.numel()),
        "layers": layers,
    }


def _check_architecture(arch: dict, build: dict) -> None:
    """Canlı mimari, kurulum argümanlarının istediğiyle aynı değilse hata verir."""
    from src.neural_network_module.architecture_contracts import resolve_ffn_dim

    d, heads, kv = build["d"], build["heads"], build["kv_heads"]
    want = {"num_heads": heads, "num_kv_heads": kv, "head_dim": d // heads, "embed_dim": d,
            "ffn_dim": resolve_ffn_dim(d, None, gated=True), "ffn_activation": "swiglu", "norm": "RMSNorm",
            "pre_norm": True, "use_rope": True, "use_kv_cache": True, "max_cache_len": build["max_len"],
            "parallel_residual": False, "use_moe": False, "gradient_checkpointing": False, "drop_path_rate": 0.0,
            "sliding_window": None}
    problems = []
    if arch["num_layers"] != build["layers"]:
        problems.append(f"num_layers={arch['num_layers']} (wanted {build['layers']})")
    if arch["vocab_size"] != build["vocab_size"] or arch["embed_dim"] != d:
        problems.append(f"embedding {arch['vocab_size']}x{arch['embed_dim']} (wanted {build['vocab_size']}x{d})")
    if not arch["tied_embeddings"]:
        problems.append("embeddings are not tied")
    if arch["dropouts"] not in ([], [0.0]):
        problems.append(f"dropout probabilities {arch['dropouts']} (wanted 0.0)")
    if arch["pe_mode"] != "rope" or arch["output_norm"] != "RMSNorm" or not arch["causal"]:
        problems.append(f"pe_mode={arch['pe_mode']} output_norm={arch['output_norm']} causal={arch['causal']}")
    for i, layer in enumerate(arch["layers"]):
        for k, v in want.items():
            if layer[k] != v:
                problems.append(f"layers[{i}].{k}={layer[k]!r} (wanted {v!r})")
    if problems:
        raise ValueError("built model does not match the requested architecture: " + "; ".join(problems))


def _build_from_config(config: dict, build: dict):
    """Tam bir yapılandırma sözlüğünden ModelManager ile kurar; (model, cfg) döner."""
    from model_management.model_manager import ModelManager

    with _quiet_repo_loggers(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(build["seed"]))
        mm = ModelManager(dict(config))
        mm.initialize(build_optimizer=False, build_criterion=False, build_scheduler=False)
    model = mm.model
    if mm.effective_config is None:
        raise RuntimeError("ModelManager did not record an effective configuration")
    arch = architecture(model)
    _check_architecture(arch, build)
    cfg = _json_canonical({"format": CFG_FORMAT, "build": build, "config": mm.effective_config, "arch": arch})
    return model, cfg


# ---------------------------------------------------------------------------------------------------- arayüz
def build_model(vocab_size: int, layers: int = 6, d: int = 192, heads: int = 4, kv_heads: int = 4,
                max_len: int = 64, seed: int = 0) -> tuple[torch.nn.Module, dict]:
    """PROTOCOL §2 modelini kurar. Döner: (model, cfg).

    ``cfg`` = {"format", "build": kurulum argümanları, "config": ModelManager'ın tam etkin yapılandırması,
    "arch": canlı modüllerden okunan mimari}. Model eğitim (train) kipindedir; ağırlıklar ``seed`` ile belirlenir.
    """
    from model_management.config_schema import tiny_model_config

    build = {"vocab_size": int(vocab_size), "layers": int(layers), "d": int(d), "heads": int(heads),
             "kv_heads": int(kv_heads), "max_len": int(max_len), "seed": int(seed)}
    config = tiny_model_config(vocab_size=build["vocab_size"], embed_dim=build["d"], seq_proj_dim=build["d"],
                               num_heads=build["heads"], num_kv_heads=build["kv_heads"], num_layers=build["layers"],
                               ffn_dim=None, max_seq_length=build["max_len"], max_cache_len=build["max_len"],
                               dropout=0.0, use_gradient_checkpointing=False)
    return _build_from_config(config, build)


def model_sha256(model) -> str:
    """Kanonik ağırlık özeti: sıralı state_dict anahtarları; her tensör için ad, şekil ve fp32 bitişik baytlar."""
    h = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state):
        t = state[key]
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"state_dict entry {key!r} is not a tensor ({type(t).__name__})")
        t = t.detach().to(device="cpu", dtype=torch.float32).contiguous()
        h.update(key.encode("utf-8") + b"\0" + repr(tuple(t.shape)).encode("ascii") + b"\0")
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def save_checkpoint(path, model, cfg: dict, tok, meta: dict) -> None:
    """Ağırlıklar + cfg + ağırlık özeti + tokenizer kimliği + meta; tokenizer yanına (``*.tokenizer.json``) yazılır.

    ``cfg`` canlı modeli tanımlamıyorsa (mimari okuması farklıysa) kaydetmez, hata verir. Yazım atomiktir.
    """
    path = pathlib.Path(path)
    if not isinstance(cfg, dict) or cfg.get("format") != CFG_FORMAT:
        raise ValueError("cfg must come from build_model/load_checkpoint")
    live = _json_canonical(architecture(model))
    if live != cfg["arch"]:
        raise ValueError("cfg does not describe this model: " + "; ".join(_diff(cfg["arch"], live)))
    if type(tok).__name__ not in _TOKENIZER_CLASSES:
        raise TypeError(f"unsupported tokenizer class {type(tok).__name__}")
    if int(tok.vocab_size) != int(cfg["build"]["vocab_size"]):
        raise ValueError(f"tokenizer vocab {tok.vocab_size} != model vocab {cfg['build']['vocab_size']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tpath = tokenizer_path(path)
    tok.save(tpath)
    payload = {
        "format": CKPT_FORMAT, "version": CKPT_VERSION,
        "cfg": _json_canonical(cfg),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_sha256": model_sha256(model),
        "tokenizer": {"file": tpath.name, "identity": tok.identity(), "class": type(tok).__name__},
        "meta": _json_canonical(meta),
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _load_tokenizer(tpath: pathlib.Path, info: dict):
    cls_name = info.get("class")
    if cls_name == "ByteLevelBPE":
        from tokenizer_management.byte_bpe import ByteLevelBPE as cls
    elif cls_name == "FusedFreqBPE":
        from .tokenizers import FusedFreqBPE as cls
    else:
        raise ValueError(f"unknown tokenizer class in checkpoint: {cls_name!r}")
    if not tpath.is_file():
        raise FileNotFoundError(f"tokenizer file missing next to checkpoint: {tpath}")
    tok = cls.load(tpath)
    if type(tok).__name__ != cls_name:
        raise ValueError(f"tokenizer class {type(tok).__name__} != saved {cls_name}")
    if tok.identity() != info.get("identity"):
        raise ValueError("tokenizer identity mismatch: the tokenizer file next to the checkpoint is not the one "
                         "the model was trained with")
    return tok


def load_checkpoint(path) -> tuple[torch.nn.Module, dict, Any, dict]:
    """Checkpoint'i yükler: (model [eval kipinde], cfg, tokenizer, meta).

    Model kaydedilen TAM yapılandırmadan yeniden kurulur (varsayılanlardan değil); yeniden kurulan cfg kaydedilenle
    birebir aynı değilse, ağırlık özeti ya da tokenizer kimliği tutmazsa hata verir.
    """
    path = pathlib.Path(path)
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ck, dict) or ck.get("format") != CKPT_FORMAT or ck.get("version") != CKPT_VERSION:
        raise ValueError(f"not a {CKPT_FORMAT} v{CKPT_VERSION} file: {path}")
    cfg = ck["cfg"]
    if not isinstance(cfg, dict) or cfg.get("format") != CFG_FORMAT:
        raise ValueError("checkpoint cfg has an unknown format")
    model, rebuilt = _build_from_config(cfg["config"], cfg["build"])
    if rebuilt != cfg:
        raise ValueError("rebuilt model config differs from the saved one (refusing to load weights into a "
                         "different function): " + "; ".join(_diff(cfg, rebuilt)))
    model.load_state_dict(ck["state_dict"], strict=True)
    digest = model_sha256(model)
    if digest != ck["model_sha256"]:
        raise ValueError(f"weight digest mismatch after loading: {digest} != saved {ck['model_sha256']}")
    tok = _load_tokenizer(path.parent / ck["tokenizer"]["file"], ck["tokenizer"])
    if int(tok.vocab_size) != int(cfg["build"]["vocab_size"]):
        raise ValueError("tokenizer vocab size does not match the model")
    model.eval()
    return model, cfg, tok, ck["meta"]


__all__ = ["CKPT_FORMAT", "CKPT_VERSION", "CFG_FORMAT", "build_model", "model_sha256", "save_checkpoint",
           "load_checkpoint", "architecture", "tokenizer_path", "code_sha256"]
