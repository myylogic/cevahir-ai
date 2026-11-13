"""The compatibility gate: one pure function behind every governed load.

``check_compatibility(facts, purpose, gate)`` combines facts into a verdict. It does no I/O and
needs no torch: providers (``cevahir_ai.adapters.inspection``) read files and tensors and hand
over ``CheckResult`` facts, so the gate's behaviour can be pinned by tables.

Result statuses:
  pass         the fact holds
  fail         the fact is established and does not hold
  unknown      the fact cannot be established (no declared architecture, no sidecar ...)
  unsupported  the artefact is outside every scheme (quantized tensors, pickled modules)
  skipped      not applicable to this artefact or target
A required check with no fact at all (its provider did not run) makes the verdict ``unverifiable``.

Built-in policy. A missing architecture denies every purpose that loads weights into a model:
a bare state dict of a 4-head model has the shapes of an 8-head model, so shapes prove nothing.
The dtype check (C9) is required by every such purpose as well: ``load_state_dict`` casts an fp16
file into an fp32 model without a word, so a dtype difference loads only when the caller allows
the up-cast. For C7, ``unknown`` means the file carries no tokenizer identity at all (a legacy
file, a warning for working purposes); an identity that nothing verified is a ``fail``.
"""
from cevahir_ai.domain.verdicts import CHECKS, PURPOSES, CheckResult, GatePolicy, Verdict

_WEIGHTS = ("C12", "C1", "C3", "C4", "C5", "C7", "C9")

DEFAULT_GATE_POLICY = GatePolicy(
    required={
        "inspect": ("C12", "C1"),
        "export": ("C12", "C1"),
        "evaluate": _WEIGHTS,
        "finetune": _WEIGHTS,
        "resume": _WEIGHTS + ("C10",),
        "serve": _WEIGHTS + ("C11",),
        "publish": _WEIGHTS + ("C11",),
    },
    on_unknown={
        ("inspect", "C1"): "warn",
        # Legacy checkpoints carry no file digest, no weight digest and often no tokenizer identity.
        # Refusing them would invalidate every existing checkpoint, so for working purposes the
        # absence is a warning, not a denial.
        ("evaluate", "C1"): "warn", ("evaluate", "C3"): "warn", ("evaluate", "C7"): "warn",
        ("finetune", "C1"): "warn", ("finetune", "C3"): "warn", ("finetune", "C7"): "warn",
        ("resume", "C1"): "warn", ("resume", "C3"): "warn", ("resume", "C7"): "warn",
        # serve, publish and export: every unknown denies (the default).
    },
)

_TR = {"compatible": "uyumlu", "compatible-with-warnings": "uyarılarla uyumlu",
       "incompatible": "uyumsuz", "unverifiable": "doğrulanamıyor"}


def check_compatibility(facts, purpose, gate=None):
    """``facts``: mapping check id -> CheckResult (or an iterable of CheckResult)."""
    if purpose not in PURPOSES:
        raise ValueError(f"unknown purpose: {purpose!r}; expected one of {PURPOSES}")
    gate = DEFAULT_GATE_POLICY if gate is None else gate
    if not isinstance(facts, dict):
        facts = {fact.id: fact for fact in facts}
    for check_id, fact in facts.items():
        if check_id not in CHECKS or not isinstance(fact, CheckResult) or fact.id != check_id:
            raise ValueError(f"malformed fact for {check_id!r}")
    required = tuple(gate.required[purpose])
    failing, warnings, missing = [], [], []
    for check_id in required:
        fact = facts.get(check_id)
        if fact is None:
            missing.append(check_id)
        elif fact.status in ("fail", "unsupported"):
            failing.append(check_id)
        elif fact.status == "unknown":
            action = gate.unknown_action(purpose, check_id)
            if action == "fail":
                failing.append(check_id)
            elif action == "warn":
                warnings.append(check_id)
            elif action != "pass":
                raise ValueError(f"unknown on_unknown action: {action!r}")
    if failing:
        verdict = "incompatible"
    elif missing:
        verdict = "unverifiable"
    elif warnings:
        verdict = "compatible-with-warnings"
    else:
        verdict = "compatible"
    ordered = [facts[check_id] for check_id in required if check_id in facts]
    ordered += [fact for check_id, fact in sorted(facts.items(), key=lambda item: int(item[0][1:]))
                if check_id not in required]
    reasons = [f"{check_id} {CHECKS[check_id]}: {facts[check_id].detail or facts[check_id].status}"
               for check_id in failing]
    reasons += [f"{check_id} {CHECKS[check_id]}: not checked" for check_id in missing]
    notes = [f"{check_id} {CHECKS[check_id]}: {facts[check_id].detail or 'unknown'}" for check_id in warnings]
    summary_en = f"{verdict} for '{purpose}'"
    summary_tr = f"'{purpose}' amacı için {_TR[verdict]}"
    if reasons:
        summary_en += ": " + "; ".join(reasons)
        summary_tr += ": " + "; ".join(reasons)
    elif notes:
        summary_en += " (" + "; ".join(notes) + ")"
        summary_tr += " (" + "; ".join(notes) + ")"
    return Verdict(verdict=verdict, purpose=purpose, checks=tuple(ordered), required=required,
                   failing=tuple(failing + missing), warnings=tuple(warnings),
                   summary_en=summary_en, summary_tr=summary_tr)
