"""Sealed training-hook intervention: an audited finite optimizer pause."""


def _get(context, name, default=None):
    if isinstance(context, dict):
        return context.get(name, default)
    return getattr(context, name, default)


def _zero_tensor(value):
    if value is None:
        return False
    if isinstance(value, dict):
        changed = False
        for item in value.values():
            changed = _zero_tensor(item) or changed
        return changed
    if isinstance(value, (list, tuple)):
        changed = False
        for item in value:
            changed = _zero_tensor(item) or changed
        return changed
    if hasattr(value, "zero_"):
        value.zero_()
        return True
    if hasattr(value, "mul_"):
        value.mul_(0.0)
        return True
    return False


def _zero_current_gradients(context):
    changed = False
    for name in ("gradients", "current_gradients", "parameter_gradients"):
        changed = _zero_tensor(_get(context, name)) or changed
    model = _get(context, "model")
    if model is not None and hasattr(model, "parameters"):
        for parameter in model.parameters():
            gradient = getattr(parameter, "grad", None)
            changed = _zero_tensor(gradient) or changed
    for group in _groups(context):
        parameters = group.get("params", []) if isinstance(group, dict) else []
        for parameter in parameters:
            changed = _zero_tensor(getattr(parameter, "grad", None)) or changed
    return changed


def _groups(context):
    optimizer = _get(context, "optimizer")
    if optimizer is not None:
        groups = getattr(optimizer, "param_groups", None)
        if groups is not None:
            return groups
    groups = _get(context, "optimizer_param_groups")
    if groups is None:
        groups = _get(context, "param_groups")
    return groups if groups is not None else []


class TrainingIntervention:
    """Pause parameter motion for 800 updates, then restore native learning rates."""

    @staticmethod
    def initialize(mechanism_state, forecast):
        initial = mechanism_state.get("initial_state", mechanism_state)
        return {
            "schema": "gfg-optimizer-pause-intervention-state-v1",
            "pause_updates": 800,
            "completed_pause_updates": 0,
            "original_learning_rates": None,
            "gradient_zero_mutations": 0,
            "learning_rate_mutations": 0,
            "baseline_cycles_remaining": int(initial.get("cycles_remaining", 1)),
            "formation_progress_frozen_updates": 0,
            "first_moment_retention": 1.0,
            "second_moment_retention": 1.0,
            "phase": "PAUSE"
        }

    @staticmethod
    def apply(stage, context, state):
        result = dict(state)
        pause = int(result["pause_updates"])
        completed = int(result["completed_pause_updates"])

        if stage == "after_backward" and completed < pause:
            if _zero_current_gradients(context):
                result["gradient_zero_mutations"] = int(
                    result["gradient_zero_mutations"]) + 1

        if stage == "before_optimizer_step":
            groups = _groups(context)
            if result["original_learning_rates"] is None and groups:
                result["original_learning_rates"] = [
                    float(group.get("lr", 0.0)) for group in groups]
            original = result["original_learning_rates"]
            if completed < pause:
                for group in groups:
                    group["lr"] = 0.0
                result["learning_rate_mutations"] = int(
                    result["learning_rate_mutations"]) + len(groups)
                completed += 1
                result["completed_pause_updates"] = completed
                result["formation_progress_frozen_updates"] = completed
                # With zero gradients AdamW moments continue to decay while
                # parameters and weight-decay motion are held by lr=0.
                result["first_moment_retention"] = float(0.9 ** completed)
                result["second_moment_retention"] = float(0.98 ** completed)
                result["phase"] = "PAUSE" if completed < pause else "RESTART_PENDING"
            elif original is not None:
                for index, group in enumerate(groups):
                    if index < len(original):
                        group["lr"] = float(original[index])
                result["phase"] = "RESUMED"
        return result
