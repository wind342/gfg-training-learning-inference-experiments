"""A sealed gradient/optimizer-state hold implemented only at allowed hooks."""


def _optimizer(context):
    if isinstance(context, dict):
        return context.get("optimizer")
    return getattr(context, "optimizer", None)


def _as_values(value):
    if value is None:
        return []
    if isinstance(value, dict):
        return list(value.values())
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _parameters(context):
    candidates = []
    if isinstance(context, dict):
        candidates.extend(_as_values(context.get("parameters")))
        model = context.get("model") or context.get("module")
    else:
        candidates.extend(_as_values(getattr(context, "parameters", None)))
        model = getattr(context, "model", None) or getattr(context, "module", None)
    if model is not None and callable(getattr(model, "parameters", None)):
        candidates.extend(list(model.parameters()))
    optimizer = _optimizer(context)
    if optimizer is not None:
        for group in getattr(optimizer, "param_groups", []):
            candidates.extend(list(group.get("params", [])))
    unique = []
    seen = set()
    for parameter in candidates:
        marker = id(parameter)
        if marker not in seen and hasattr(parameter, "grad"):
            unique.append(parameter)
            seen.add(marker)
    return unique


def _suppress_current_gradients(context):
    parameters = _parameters(context)
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad = None
    if parameters:
        return len(parameters)
    gradients = context.get("gradients") if isinstance(context, dict) else getattr(
        context, "gradients", None)
    zeroed = 0
    for gradient in _as_values(gradients):
        if gradient is not None and callable(getattr(gradient, "zero_", None)):
            gradient.zero_()
            zeroed += 1
    # Returning zero deliberately requests the learning-rate fallback: zero-valued
    # gradients alone would still advance Adam state and decoupled weight decay.
    return 0


def _set_zero_lr(context, state):
    optimizer = _optimizer(context)
    if optimizer is None:
        return False
    groups = getattr(optimizer, "param_groups", [])
    state["fallback_original_lrs"] = [float(group.get("lr", 0.0)) for group in groups]
    for group in groups:
        group["lr"] = 0.0
    state["fallback_lr_active"] = True
    return True


def _restore_lr(context, state):
    if not state.get("fallback_lr_active"):
        return
    optimizer = _optimizer(context)
    if optimizer is None:
        return
    groups = getattr(optimizer, "param_groups", [])
    for group, learning_rate in zip(groups, state.get("fallback_original_lrs", [])):
        group["lr"] = float(learning_rate)
    state["fallback_lr_active"] = False


class TrainingIntervention:
    """Hold parameters and per-parameter AdamW state for 900 updates.

    Setting every current parameter gradient to None makes AdamW skip the
    parameter, including its step, exp_avg, exp_avg_sq, decoupled weight decay,
    and parameter write.  A zero-learning-rate fallback is used only if the
    native hook does not expose parameters.
    """

    @staticmethod
    def initialize(mechanism_state, forecast):
        cut = mechanism_state.get("cut_step")
        if cut is None:
            cut = forecast.get("mechanism_state", {}).get("initial", {}).get(
                "optimizer_step", 0)
        return {
            "schema": "gradient-and-adam-state-hold-v1",
            "initialized_at_optimizer_step": int(cut),
            "hold_updates": 900,
            "held_updates": 0,
            "active": True,
            "suppressed_gradient_count": 0,
            "fallback_lr_active": False,
            "fallback_original_lrs": [],
            "direction": "DELAY",
            "predicted_direction": "DELAY",
            "predicted_shift_interval": {"step_low": 800, "step_high": 1000},
            "predicted_rule_state_change": "rule margins and renewal phase remain frozen for 900 optimizer occurrences",
            "predicted_stability_effect": "NO_CHANGE",
            "stability_effect_rationale": "all operative parameter and per-parameter Adam states resume from the same state after a finite hold",
        }

    @staticmethod
    def apply(stage, context, state):
        if not state.get("active", False):
            return state
        if stage == "after_backward":
            state["suppressed_gradient_count"] = _suppress_current_gradients(context)
        elif stage == "before_optimizer_step":
            count = _suppress_current_gradients(context)
            state["suppressed_gradient_count"] = max(
                int(state.get("suppressed_gradient_count", 0)), count)
            if count == 0:
                _set_zero_lr(context, state)
        elif stage == "after_optimizer_step":
            _restore_lr(context, state)
            state["held_updates"] = int(state["held_updates"]) + 1
            if state["held_updates"] >= int(state["hold_updates"]):
                state["active"] = False
        return state
