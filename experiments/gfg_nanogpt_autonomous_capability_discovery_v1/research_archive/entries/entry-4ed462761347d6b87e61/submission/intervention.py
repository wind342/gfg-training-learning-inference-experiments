"""Sealed pre-transition optimizer hold intervention."""


def _optimizer_from_context(context):
    if isinstance(context, dict):
        for key in ("optimizer", "training_optimizer", "current_optimizer"):
            if key in context:
                return context[key]
    for name in ("optimizer", "training_optimizer", "current_optimizer"):
        if hasattr(context, name):
            return getattr(context, name)
    return None


def _parameter_groups(optimizer):
    if optimizer is None:
        return []
    if isinstance(optimizer, dict):
        return optimizer.get("param_groups", [])
    return getattr(optimizer, "param_groups", [])


def _group_learning_rate(group):
    if isinstance(group, dict):
        return float(group.get("lr", 0.0))
    return float(getattr(group, "lr"))


def _set_group_learning_rate(group, value):
    if isinstance(group, dict):
        group["lr"] = float(value)
    else:
        setattr(group, "lr", float(value))


class TrainingIntervention:
    """Delay formation by holding parameter updates for 1600 steps.

    The hook changes only optimizer-group learning rates.  Gradients and Adam
    moments continue to be formed, so the intervention state explicitly tracks
    that the optimizer reservoir is adapting while rule-bearing parameters are
    held.
    """

    @staticmethod
    def initialize(mechanism_state, forecast):
        return {
            "schema": "training-intervention-state-v1",
            "direction": "DELAY",
            "hold_steps_total": 1600,
            "hold_steps_remaining": 1600,
            "held_steps": 0,
            "original_learning_rates": None,
            "learning_rates_restored": False,
            "rule_probability_at_start": float(mechanism_state["rule_probability"]),
            "rule_probability_progress_during_hold": 0.0,
            "optimizer_reservoir_continues_to_update": True,
            "predicted_transition_shift_low": 1300,
            "predicted_transition_shift_high": 1900,
            "transition_step_shift_low": 1300,
            "transition_step_shift_high": 1900,
            "predicted_stability_effect": "NO_CHANGE",
        }

    @staticmethod
    def apply(stage, context, state):
        if stage != "before_optimizer_step":
            return state
        optimizer = _optimizer_from_context(context)
        groups = _parameter_groups(optimizer)
        if not groups:
            raise ValueError("INTERVENTION_CONTEXT_HAS_NO_OPTIMIZER_GROUPS")
        if state["original_learning_rates"] is None:
            state["original_learning_rates"] = [
                _group_learning_rate(group) for group in groups
            ]

        if int(state["hold_steps_remaining"]) > 0:
            for group in groups:
                _set_group_learning_rate(group, 0.0)
            state["hold_steps_remaining"] = int(state["hold_steps_remaining"]) - 1
            state["held_steps"] = int(state["held_steps"]) + 1
            # Parameters do not move at lr=0, so rule progress is held.  Adam
            # first/second moments still read the current gradients and adapt.
            state["rule_probability_progress_during_hold"] = 0.0
            return state

        if not state["learning_rates_restored"]:
            original = state["original_learning_rates"]
            if len(original) != len(groups):
                raise ValueError("OPTIMIZER_GROUP_COUNT_CHANGED")
            for group, learning_rate in zip(groups, original):
                _set_group_learning_rate(group, learning_rate)
            state["learning_rates_restored"] = True
        return state
