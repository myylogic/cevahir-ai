"""Common sync/async stage scopes and explicit degradation metadata."""
from contextlib import contextmanager
from .runtime import BudgetExhausted, current_runtime, runtime_stage

STAGES = {"FeatureExtraction": "features", "PolicyRouting": "policy",
          "Deliberation": "deliberation", "ContextBuilding": "context",
          "Generation": "answer", "SelfConsistency": "alternatives",
          "Critic": "refinement", "MemoryUpdate": "memory"}


@contextmanager
def processing_stage(context, handler_name):
    name = handler_name.removeprefix("Async")
    runtime = current_runtime()
    with runtime_stage(STAGES.get(name, name)):
        try:
            yield
        except BudgetExhausted as exc:
            context.errors.append(f"budget_limited:{name}:{exc.reason}")
            if runtime is not None:
                runtime.plan.setdefault("fallbacks", []).append({"stage": name, "reason": exc.reason})
        finally:
            if runtime is not None and name == "Deliberation" and context.selected_thought is None:
                if context.policy_output and context.policy_output.mode != "direct":
                    runtime.plan["executed_strategy"] = "direct"
                    runtime.plan.setdefault("fallbacks", []).append({"stage": name, "reason": "no_selected_thought"})
                    context.policy_output.mode = "direct"
