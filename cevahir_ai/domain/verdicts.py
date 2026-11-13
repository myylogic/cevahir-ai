"""Verdict types of the compatibility gate."""
from dataclasses import dataclass, field

PURPOSES = ("inspect", "evaluate", "serve", "resume", "finetune", "publish", "export")
STATUSES = ("pass", "fail", "unknown", "unsupported", "skipped")
VERDICTS = ("compatible", "compatible-with-warnings", "incompatible", "unverifiable")

# Check ids and names. C2, C6, C8, C13, C14 and C15 need the registry, signatures or policies of
# later phases; they are named here so that their ids never change meaning.
CHECKS = {
    "C1": "file.integrity", "C2": "manifest.signature", "C3": "weights.digest", "C4": "wschema.match",
    "C5": "arch.declared", "C6": "arch.observed", "C7": "tok.identity", "C8": "tok.behavior",
    "C9": "dtype", "C10": "resume.signatures", "C11": "context.window", "C12": "envelope.safety",
    "C13": "provenance.present", "C14": "eval.passed", "C15": "path.protected",
}
IMPLEMENTED_CHECKS = ("C1", "C3", "C4", "C5", "C7", "C9", "C10", "C11", "C12")


@dataclass(frozen=True)
class CheckResult:
    id: str
    status: str
    detail: str = ""
    evidence: tuple = ()

    def __post_init__(self):
        if self.id not in CHECKS:
            raise ValueError(f"unknown check id: {self.id}")
        if self.status not in STATUSES:
            raise ValueError(f"unknown check status: {self.status}")

    @property
    def name(self):
        return CHECKS[self.id]

    def to_dict(self):
        return {"id": self.id, "name": self.name, "status": self.status, "detail": self.detail,
                "evidence": [str(item) for item in self.evidence]}


@dataclass(frozen=True)
class GatePolicy:
    """Which checks a purpose requires, and what an ``unknown`` result means for it."""
    required: dict
    on_unknown: dict = field(default_factory=dict)       # (purpose, check id) -> "pass" | "warn" | "fail"
    default_on_unknown: str = "fail"

    def unknown_action(self, purpose, check_id):
        return self.on_unknown.get((purpose, check_id), self.default_on_unknown)


@dataclass(frozen=True)
class Verdict:
    verdict: str
    purpose: str
    checks: tuple          # every CheckResult considered, required first
    required: tuple        # ids required for this purpose
    failing: tuple         # ids that deny
    warnings: tuple        # ids that only warn
    summary_en: str
    summary_tr: str

    @property
    def ok(self):
        return self.verdict in ("compatible", "compatible-with-warnings")

    def to_dict(self):
        return {"verdict": self.verdict, "purpose": self.purpose, "required": list(self.required),
                "failing": list(self.failing), "warnings": list(self.warnings),
                "checks": [check.to_dict() for check in self.checks],
                "summary_en": self.summary_en, "summary_tr": self.summary_tr}
