"""Opt-in research controls; identities must name the actual loaded artifacts."""
from dataclasses import dataclass, field
import math

from .runtime import BudgetLimits


@dataclass
class ResearchConfig:
    mode: str = "off"  # off | budget | shadow | adaptive
    model_revision: str = ""
    tokenizer_revision: str = ""
    limits: BudgetLimits = field(default_factory=BudgetLimits)
    max_records_per_scope: int = 256
    min_support: int = 3
    cost_weight: float = 0.1
    decay_half_life_seconds: float = 604800.0
    exploration_rate: float = 0.0
    enable_neural_routing: bool = False
    routing_bias_by_domain: dict = field(default_factory=dict)

    def validate(self):
        if self.mode not in {"off", "budget", "shadow", "adaptive"}:
            raise ValueError("research.mode must be off, budget, shadow or adaptive")
        if self.mode != "off" and not all(
            isinstance(x, str) and x.strip()
            for x in (self.model_revision, self.tokenizer_revision)
        ):
            raise ValueError("Research requires explicit model and tokenizer revisions")
        if not isinstance(self.limits, BudgetLimits):
            raise ValueError("research.limits must be BudgetLimits")
        self.limits.__post_init__()
        for name in ("max_records_per_scope", "min_support"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"research.{name} must be a positive integer")
        for name in ("cost_weight", "decay_half_life_seconds", "exploration_rate"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"research.{name} must be finite")
        if not 0 <= self.cost_weight <= 1 or self.decay_half_life_seconds <= 0 or not 0 <= self.exploration_rate <= 1:
            raise ValueError("Invalid research cost, decay or exploration setting")
        if self.min_support > self.max_records_per_scope:
            raise ValueError("research.min_support exceeds retained experience capacity")
        if not isinstance(self.enable_neural_routing, bool) or not isinstance(self.routing_bias_by_domain, dict):
            raise ValueError("Invalid research routing configuration")
        for domain, vector in self.routing_bias_by_domain.items():
            if not isinstance(domain, str) or not domain or not isinstance(vector, (list, tuple)) or not vector:
                raise ValueError("Routing profiles must map domain names to nonempty expert vectors")
            if any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(x) or abs(x) > 1 for x in vector):
                raise ValueError("Routing prior entries must be finite and in [-1, 1]")
        if self.enable_neural_routing and self.mode == "off":
            raise ValueError("Neural routing requires an enabled research mode")
