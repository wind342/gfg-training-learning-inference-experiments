"""Sealed optimizer-hook intervention for the capability mechanism.

The intervention delays formation by setting only the current optimizer-group
learning rates to zero for a finite number of optimizer applications, then
restoring the rates observed at the hook.  Gradients and Adam moments are
deliberately left evolving and are recorded in the audit state.
"""


HOLD_OPTIMIZER_STEPS = 1200
SHIFT_LOW = 600
SHIFT_HIGH = 1800


def _optimizer_groups(context):
    if isinstance(context, dict):
        optimizer = context.get("optimizer")
        if optimizer is not None and hasattr(optimizer, "param_groups"):
            return optimizer.param_groups
        for key in ("optimizer_param_groups", "param_groups",
                    "optimizer_group_hyperparameters"):
            groups = context.get(key)
            if groups is not None:
                return groups
    optimizer = getattr(context, "optimizer", None)
    if optimizer is not None and hasattr(optimizer, "param_groups"):
        return optimizer.param_groups
    groups = getattr(context, "optimizer_param_groups", None)
    if groups is not None:
        return groups
    groups = getattr(context, "param_groups", None)
    return [] if groups is None else groups


def _group_lr(group):
    if isinstance(group, dict):
        return float(group.get("lr", 0.0))
    return float(getattr(group, "lr"))


def _set_group_lr(group, value):
    if isinstance(group, dict):
        group["lr"] = float(value)
    else:
        setattr(group, "lr", float(value))


class TrainingIntervention:
    @staticmethod
    def initialize(mechanism_state, forecast):
        if isinstance(mechanism_state, dict):
            source_state = mechanism_state.get("initial", mechanism_state)
            cut_step = int(source_state.get("cut_step", 0))
            rule_logit = float(source_state.get("rule_logit", 0.0))
            decay_progress = float(source_state.get("decay_progress", 0.0))
            baseline_lr = float(source_state.get("learning_rate", 0.003))
        else:
            cut_step = 0
            rule_logit = 0.0
            decay_progress = 0.0
            baseline_lr = 0.003
        low = forecast.get("transition_step_low_500") if isinstance(forecast, dict) else None
        high = forecast.get("transition_step_high_500") if isinstance(forecast, dict) else None
        baseline_center = (int((low + high) // 2)
                           if isinstance(low, int) and isinstance(high, int)
                           else None)
        return {
            "schema": "gfg-training-intervention-state-v1",
            "direction": "DELAY",
            "declared_hook": "before_optimizer_step",
            "remaining_hold_steps": HOLD_OPTIMIZER_STEPS,
            "hold_steps_total": HOLD_OPTIMIZER_STEPS,
            "hook_calls": 0,
            "baseline_learning_rates": [],
            "fallback_learning_rate": baseline_lr,
            "rates_captured": False,
            "rates_restored": False,
            "mutation_applied": False,
            "missing_context_count": 0,
            "cut_step": cut_step,
            "baseline_transition_center": baseline_center,
            "predicted_shift_low": SHIFT_LOW,
            "predicted_shift_high": SHIFT_HIGH,
            "predicted_transition_step_low": (None if low is None else int(low) + SHIFT_LOW),
            "predicted_transition_step_high": (None if high is None else int(high) + SHIFT_HIGH),
            "rule_logit_at_intervention": rule_logit,
            "decay_progress_at_intervention": decay_progress,
            "rule_logit_change_while_held": 0.0,
            "decay_progress_change_while_held": 0.0,
            "gradients_continue_evolving": True,
            "adam_exp_avg_continues_evolving": True,
            "adam_exp_avg_sq_continues_evolving": True,
            "adam_step_continues_evolving": True,
            "predicted_stability_effect": "NO_CHANGE",
            "stability_effect_rationale": (
                "The pause holds parameter formation and decay progress but does not "
                "freeze gradients or Adam moments. The evidence supports delayed "
                "formation, but not a directional change in the long-run transient-"
                "degradation/recovery class after learning-rate restoration."
            ),
        }

    @staticmethod
    def apply(stage, context, state):
        if stage != "before_optimizer_step":
            return state
        groups = list(_optimizer_groups(context))
        state["hook_calls"] = int(state.get("hook_calls", 0)) + 1
        if not groups:
            state["missing_context_count"] = int(state.get("missing_context_count", 0)) + 1
            return state
        if not state.get("rates_captured"):
            state["baseline_learning_rates"] = [_group_lr(group) for group in groups]
            state["rates_captured"] = True
        remaining = int(state.get("remaining_hold_steps", 0))
        if remaining > 0:
            for group in groups:
                _set_group_lr(group, 0.0)
            state["remaining_hold_steps"] = remaining - 1
            state["mutation_applied"] = True
            state["parameter_updates_held"] = int(state.get("parameter_updates_held", 0)) + 1
            return state
        if not state.get("rates_restored"):
            baseline = list(state.get("baseline_learning_rates") or [])
            fallback = float(state.get("fallback_learning_rate", 0.003))
            for index, group in enumerate(groups):
                _set_group_lr(group, baseline[index] if index < len(baseline) else fallback)
            state["rates_restored"] = True
        return state
