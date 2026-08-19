"""Sealed training-hook intervention for the rule-oscillator mechanism."""


def _group_container(context):
    if context is None:
        return []
    optimizer = None
    if isinstance(context, dict):
        optimizer = context.get("optimizer")
        for key in ("optimizer_param_groups", "optimizer_groups",
                    "optimizer_group_hyperparameters", "param_groups"):
            if key in context:
                value = context[key]
                if isinstance(value, dict) and "lr" in value:
                    return [value]
                if isinstance(value, dict):
                    return list(value.values())
                return list(value)
    else:
        optimizer = getattr(context, "optimizer", None)
        for key in ("optimizer_param_groups", "optimizer_groups",
                    "optimizer_group_hyperparameters", "param_groups"):
            value = getattr(context, key, None)
            if value is not None:
                return list(value)
    if optimizer is not None:
        if isinstance(optimizer, dict):
            value = optimizer.get("param_groups", [])
        else:
            value = getattr(optimizer, "param_groups", [])
        return list(value)
    return []


def _get_lr(group):
    if isinstance(group, dict):
        return float(group.get("lr", 0.0))
    return float(getattr(group, "lr"))


def _set_lr(group, value):
    if isinstance(group, dict):
        group["lr"] = float(value)
    else:
        setattr(group, "lr", float(value))


class TrainingIntervention:
    """Pause parameter motion for 1600 updates, then restore native learning rates."""

    @staticmethod
    def initialize(mechanism_state, forecast):
        return {
            "schema": "gfg-lr-pause-intervention-state-v1",
            "direction": "DELAY",
            "predicted_shift_step_low": 1400,
            "predicted_shift_step_high": 2000,
            "predicted_stability_effect": "NO_CHANGE",
            "pause_updates_remaining": 1600,
            "paused_updates": 0,
            "original_learning_rates": [],
            "rates_captured": False,
            "rates_restored": False,
            "mutation_applied": False,
            "mechanism_variable": "rule_log_odds",
            "initial_rule_log_odds": float(
                mechanism_state.get("rule_log_odds", 0.0)),
            "expected_rule_log_odds_change_during_pause": 0.0,
            "baseline_transition_estimate": int(
                mechanism_state.get("transition_estimate", 10000)),
        }

    @staticmethod
    def apply(stage, context, state):
        if stage != "before_optimizer_step":
            return state
        groups = _group_container(context)
        if not groups:
            state["mutation_applied"] = False
            return state
        if not state["rates_captured"]:
            state["original_learning_rates"] = [_get_lr(group)
                                                 for group in groups]
            state["rates_captured"] = True
        if state["pause_updates_remaining"] > 0:
            for group in groups:
                _set_lr(group, 0.0)
            state["pause_updates_remaining"] -= 1
            state["paused_updates"] += 1
            state["mutation_applied"] = True
            return state
        if not state["rates_restored"]:
            saved = state["original_learning_rates"]
            for index, group in enumerate(groups):
                value = saved[index] if index < len(saved) else saved[-1]
                _set_lr(group, value)
            state["rates_restored"] = True
        state["mutation_applied"] = False
        return state
