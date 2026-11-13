"""Run and evaluation keys.

A run specification is content: the effective training configuration with every path replaced
by the key of what the path points to. A run is an occurrence: two runs of one specification have
one ``runspec`` key and two ``run`` ids. An evaluation is a recipe: model, evaluation set, metric
generation, decoding settings and seed; its results are attributes, not key inputs.
"""
import os
import time

from cevahir_ai.keys.canonical import as_int, structured_digest
from cevahir_ai.keys.grammar import Key

EVAL_METRICS = "ce-loss/1+gen-probe/1"
RUN_INPUT_ROLES = ("tok", "cache", "corpus", "parent", "arch", "code", "env", "split")


def runspec_key(config, inputs):
    """``config``: plain training settings (no paths); ``inputs``: {role: key} for tokenizer, cache, parent ...

    Output directories, device names and log settings do not belong in ``config``: they identify
    nothing. Returns ``(key, payload)``.
    """
    unknown = set(inputs) - set(RUN_INPUT_ROLES)
    if unknown:
        raise ValueError(f"unknown run input roles: {sorted(unknown)}")
    payload = {"config": dict(config),
               "inputs": {role: str(Key.parse(str(key))) for role, key in inputs.items() if key is not None}}
    return Key("runspec", "cfg-v1", structured_digest("runspec", "cfg-v1", payload)), payload


def uuid7_hex(now_ms=None, random_bytes=None):
    """A time-ordered random id (UUID version 7 layout) as 32 hex digits."""
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    rand = bytearray(os.urandom(10) if random_bytes is None else random_bytes)
    value = bytearray(now_ms.to_bytes(6, "big") + rand)
    value[6] = (value[6] & 0x0F) | 0x70
    value[8] = (value[8] & 0x3F) | 0x80
    return bytes(value).hex()


def new_run_key():
    return Key("run", "id-v1", uuid7_hex())


def eval_recipe_key(model, evalset, *, decoding, seed, metrics=EVAL_METRICS):
    model_key, evalset_key = Key.parse(str(model)), Key.parse(str(evalset))
    if model_key.kind != "model" or evalset_key.kind != "evalset":
        raise ValueError("an evaluation needs a model key and an evaluation-set key")
    payload = {"model": str(model_key), "evalset": str(evalset_key), "metrics": str(metrics),
               "decoding": dict(decoding), "seed": as_int(seed, "seed")}
    return Key("eval", "recipe-v1", structured_digest("eval", "recipe-v1", payload)), payload
