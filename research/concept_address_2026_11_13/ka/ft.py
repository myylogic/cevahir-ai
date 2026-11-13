# -*- coding: utf-8 -*-
"""Fine-tuning comparators: single edits and sequential streams (PROTOCOL §3.1 reference arms, §5 FT streams;
KA-H1 reference, KA-H2).

The optimisers are CEV-11's frozen ``cev11.baselines.ft_layer`` (FT-L: Adam on one fc2 weight with an l-inf clamp)
and ``ft_all`` (FT-ALL: Adam on every parameter), with the configurations CEV-11 selected on its discovery worlds
(``results/selection.json`` of the frozen study; never re-tuned here).

A stream applies the same optimiser fact after fact to the CURRENT weights. The KL anchor of each write is the
model as it is before that fact (``ft_layer`` / ``ft_all`` take the base distribution from the weights they start
from), on the next relation of the same entity in the lived frame.

Right after each write the stream checks, with CEV-11's exact check, whether the fact just written is produced on
its lived prompt. The share over the facts written so far is the record's ``newest_efficacy``: the lived-form
efficacy on the newest fact that gate G-KA1c asks of the FT comparators (PROTOCOL §8).
"""
from __future__ import annotations

import json
import pathlib
from typing import Callable, Dict, List, Optional, Sequence

import torch

from . import CEV11_DIR
from .streams import Fact, kl_relation
from cev11.world import frame_prefix
from tokenizer_management.byte_bpe import BOS_ID

FT_ARMS = ("FT-L", "FT-ALL")


def hp_from_config(cfg: str) -> dict:
    hp = {}
    for part in cfg.split(","):
        k, _, v = part.partition("=")
        if k:
            hp[k] = int(v) if v.isdigit() else float(v)
    return hp


def cev11_ft_configs(path: Optional[pathlib.Path] = None) -> Dict[str, dict]:
    """FT-L and FT-ALL as selected by CEV-11's frozen discovery selection: ``{arm: {"layer", "hp", "config"}}``."""
    path = pathlib.Path(path) if path is not None else CEV11_DIR / "results" / "selection.json"
    sel = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for arm in FT_ARMS:
        a = sel["arms"][arm]
        if a.get("status") != "selected":
            raise ValueError(f"CEV-11 did not select a configuration for {arm}")
        out[arm] = {"layer": a["layer"], "hp": hp_from_config(a["config"]), "config": a["config"]}
    return out


def fact_prompt(tok, fact: Fact):
    """(prompts, target ids, KL prompt ids) of one stream fact: the lived sentence, PRED site."""
    ids = [BOS_ID] + list(tok.encode(fact.prefix))
    target = list(tok.encode(" " + fact.new + "."))
    kl = [BOS_ID] + list(tok.encode(frame_prefix(fact.name, kl_relation(fact.rel), fact.lived, "formal", "none")))
    return [(ids, len(ids) - 1)], target, kl


def ft_step(model, tok, fact: Fact, arm: str, cfg: dict) -> dict:
    """Apply one FT write to the current weights (in place). Returns the optimiser's info dict."""
    from cev11.baselines import ft_all, ft_layer
    prompts, target, kl = fact_prompt(tok, fact)
    hp = cfg["hp"]
    info: dict = {}
    if arm == "FT-L":
        layer = int(cfg["layer"])
        dW = ft_layer(model, layer, prompts, target, hp["lr"], int(hp.get("steps", 60)), hp["eps"],
                      kl_prompt_ids=kl, info=info)
        with torch.no_grad():
            model.layers[layer].ffn.fc2.weight.add_(dW)
    elif arm == "FT-ALL":
        state = ft_all(model, prompts, target, hp["lr"], int(hp.get("steps", 40)), kl_prompt_ids=kl, info=info)
        model.load_state_dict(state, strict=True)
    else:
        raise ValueError(f"unknown FT arm {arm!r}")
    clear = getattr(model, "clear_kv_cache", None)
    if callable(clear):
        clear()
    return info


def newest_produced(model, tok, fact: Fact) -> bool:
    """True iff the model as it is now produces the fact's new object on its lived prompt (exact check)."""
    from cev11.decode import exact_check_ids
    prompts, target, _ = fact_prompt(tok, fact)
    return bool(exact_check_ids(model, [prompts[0][0]], [target])[0])


def ft_stream(model, tok, facts: Sequence[Fact], arm: str, cfg: dict, checkpoints: Sequence[int],
              at_checkpoint: Callable[[int], dict]) -> List[dict]:
    """Sequential fine-tuning over ``facts``; ``at_checkpoint(N)`` is called with the model holding N facts.

    The model is restored bitwise at the end (also on failure). Returns one record per checkpoint; each carries
    ``newest_efficacy``, the share of the facts written so far that were produced right after their own write.
    """
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    cps = sorted(set(int(c) for c in checkpoints if c <= len(facts)))
    out: List[dict] = []
    steps, stopped, produced = [], 0, 0
    try:
        for i, f in enumerate(facts[:max(cps) if cps else 0], start=1):
            info = ft_step(model, tok, f, arm, cfg)
            steps.append(int(info.get("steps", 0)))
            stopped += int(bool(info.get("stopped")))
            produced += int(newest_produced(model, tok, f))
            if i in cps:
                rec = at_checkpoint(i)
                rec.update({"arm": arm, "config": cfg["config"], "layer": cfg["layer"], "N": i,
                            "mean_steps": sum(steps) / len(steps), "stopped_share": stopped / i,
                            "newest_efficacy": produced / i})
                out.append(rec)
    finally:
        model.load_state_dict(original, strict=True)
        clear = getattr(model, "clear_kv_cache", None)
        if callable(clear):
            clear()
    return out


__all__ = ["FT_ARMS", "hp_from_config", "cev11_ft_configs", "fact_prompt", "ft_step", "newest_produced", "ft_stream"]
