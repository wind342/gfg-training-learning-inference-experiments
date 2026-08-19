"""Sealed hook-only intervention for a reversible 800-update state pause."""


def _get(container, name, default=None):
    if isinstance(container, dict):
        return container.get(name, default)
    return getattr(container, name, default)


def _optimizer(context):
    if _get(context, "param_groups") is not None:
        return context
    candidate = _get(context, "optimizer")
    if candidate is not None:
        return candidate
    training_state = _get(context, "training_state")
    if training_state is not None:
        return _get(training_state, "optimizer")
    return None


def _groups(optimizer):
    if optimizer is None:
        return []
    groups = _get(optimizer, "param_groups", [])
    return list(groups) if groups is not None else []


def _group_get(group, name, default=None):
    if isinstance(group, dict):
        return group.get(name, default)
    return getattr(group, name, default)


def _group_set(group, name, value):
    if isinstance(group, dict):
        group[name] = value
    else:
        setattr(group, name, value)


def _freeze_gradients(groups):
    changed = 0
    for group in groups:
        parameters = _group_get(group, "params", [])
        for parameter in list(parameters):
            if hasattr(parameter, "grad"):
                parameter.grad = None
                changed += 1
    return changed


class TrainingIntervention:
    """Pause parameter and Adam state evolution, then restore native rates."""

    @staticmethod
    def initialize(mechanism_state, forecast):
        if isinstance(mechanism_state, dict) and "initial_state" in mechanism_state:
            initial = mechanism_state["initial_state"]
        else:
            initial = mechanism_state if isinstance(mechanism_state, dict) else {}
        return {
            "schema": "training-intervention-state-v1",
            "declared_hook": "before_optimizer_step",
            "freeze_updates_total": 800,
            "freeze_updates_remaining": 800,
            "applied_updates": 0,
            "saved_group_lrs": [],
            "rates_captured": False,
            "rates_restored": False,
            "last_changed_parameter_count": 0,
            "baseline_cut_step": int(initial.get("cut_step", 0)),
            "baseline_gain_l2": float(initial.get("gain_l2", 0.0)),
            "effective_update_deficit": 0,
            "direction": "DELAY",
            "transition_step_shift_low": 700,
            "transition_step_shift_high": 900,
            "predicted_stability_effect": "NO_CHANGE",
        }

    @staticmethod
    def apply(stage, context, state):
        if stage != "before_optimizer_step":
            return state
        result = dict(state)
        optimizer = _optimizer(context)
        groups = _groups(optimizer)

        if not result.get("rates_captured", False):
            result["saved_group_lrs"] = [
                float(_group_get(group, "lr", 0.0)) for group in groups
            ]
            result["rates_captured"] = bool(groups)

        remaining = int(result.get("freeze_updates_remaining", 0))
        if remaining > 0:
            for group in groups:
                _group_set(group, "lr", 0.0)
            changed = _freeze_gradients(groups)
            # A None gradient makes AdamW skip the parameter entirely, so its
            # parameter, exp_avg, exp_avg_sq and step state all remain fixed.
            result["last_changed_parameter_count"] = int(changed)
            if changed > 0:
                result["freeze_updates_remaining"] = remaining - 1
                result["applied_updates"] = int(result.get("applied_updates", 0)) + 1
                result["effective_update_deficit"] = int(
                    result["applied_updates"])
            return result

        if result.get("rates_captured", False) and not result.get("rates_restored", False):
            saved = list(result.get("saved_group_lrs", []))
            for index, group in enumerate(groups):
                if index < len(saved):
                    _group_set(group, "lr", float(saved[index]))
            result["rates_restored"] = True
        return result
