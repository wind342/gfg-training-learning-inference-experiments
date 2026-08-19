"""A sealed optimizer-clock hold intervention.

For a fixed number of optimizer updates, every optimizer parameter group has
learning rate zero.  This mutates only the frozen API's permitted optimizer
group hyperparameters.  No validation object is inspected or injected.
"""


def _get(container, key, default=None):
    if isinstance(container, dict):
        return container.get(key, default)
    return getattr(container, key, default)


def _groups(context):
    for key in ("optimizer", "current_optimizer"):
        optimizer = _get(context, key)
        if optimizer is not None:
            groups = _get(optimizer, "param_groups")
            if groups is not None:
                return list(groups)
    for key in ("param_groups", "optimizer_param_groups",
                "optimizer_groups", "optimizer_group_hyperparameters"):
        groups = _get(context, key)
        if groups is not None:
            return list(groups)
    direct = _get(context, "param_groups")
    return list(direct) if direct is not None else []


def _group_lr(group):
    value = _get(group, "lr")
    return None if value is None else float(value)


def _set_group_lr(group, value):
    if isinstance(group, dict):
        group["lr"] = float(value)
    else:
        setattr(group, "lr", float(value))


class TrainingIntervention:
    PAUSE_OPTIMIZER_STEPS = 1300

    @staticmethod
    def initialize(mechanism_state, forecast):
        return {
            "schema": "optimizer-clock-hold-state-v1",
            "direction": "DELAY",
            "predicted_direction": "DELAY",
            "predicted_shift_low": 600,
            "predicted_shift_high": 1800,
            "transition_step_shift_low": 600,
            "transition_step_shift_high": 1800,
            "predicted_mechanism_state_change": (
                "hold SYMMETRY_NUCLEATION and postpone the next two "
                "gradient-burst rewrites"),
            "target_hook": "before_optimizer_step",
            "pause_optimizer_steps": TrainingIntervention.PAUSE_OPTIMIZER_STEPS,
            "paused_steps": 0,
            "saved_learning_rates": None,
            "active": True,
            "restored": False,
        }

    @staticmethod
    def apply(stage, context, state):
        if state is None:
            return None
        result = dict(state)
        if stage != "before_optimizer_step":
            return result

        groups = _groups(context)
        paused = int(result.get("paused_steps", 0))
        limit = int(result.get(
            "pause_optimizer_steps", TrainingIntervention.PAUSE_OPTIMIZER_STEPS))
        if paused < limit:
            if result.get("saved_learning_rates") is None:
                saved = [_group_lr(group) for group in groups]
                if not groups or any(value is None for value in saved):
                    raise ValueError("optimizer parameter-group learning rates unavailable")
                result["saved_learning_rates"] = saved
            for group in groups:
                _set_group_lr(group, 0.0)
            result["paused_steps"] = paused + 1
            result["active"] = True
            return result

        if not result.get("restored", False):
            saved = list(result.get("saved_learning_rates") or [])
            if len(saved) != len(groups):
                raise ValueError("optimizer parameter-group identity changed")
            for group, learning_rate in zip(groups, saved):
                _set_group_lr(group, learning_rate)
            result["restored"] = True
        result["active"] = False
        return result
