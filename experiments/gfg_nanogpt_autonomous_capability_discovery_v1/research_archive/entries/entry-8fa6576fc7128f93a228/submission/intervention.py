"""Sealed optimizer-skip intervention using only the frozen hook API."""


def _context_value(context, name):
    if isinstance(context, dict):
        return context.get(name)
    return getattr(context, name, None)


def _candidate_parameters(context):
    optimizer = _context_value(context, "optimizer")
    if optimizer is not None and hasattr(optimizer, "param_groups"):
        for group in optimizer.param_groups:
            for parameter in group.get("params", []):
                yield parameter
        return
    model = _context_value(context, "model")
    if model is not None and hasattr(model, "parameters"):
        for parameter in model.parameters():
            yield parameter
        return
    parameters = _context_value(context, "parameters")
    if parameters is not None:
        values = parameters.values() if isinstance(parameters, dict) else parameters
        for parameter in values:
            yield parameter


class TrainingIntervention:
    """Delay formation by skipping 800 parameter and Adam-state updates."""

    HOLD_OPTIMIZER_STEPS = 800

    @staticmethod
    def initialize(mechanism_state, forecast):
        initial = mechanism_state
        if isinstance(mechanism_state, dict) and "initial" in mechanism_state:
            initial = mechanism_state["initial"]
        return {
            "schema": "training-intervention-state-v1",
            "direction": "DELAY",
            "hold_optimizer_steps": TrainingIntervention.HOLD_OPTIMIZER_STEPS,
            "held_optimizer_steps": 0,
            "active": True,
            "affected_parameter_count": 0,
            "formation_progress_at_start": float(
                initial.get("final_layer_norm_gain", 0.0)
            ) if isinstance(initial, dict) else 0.0,
            "circuit_norm_at_start": float(
                initial.get("circuit_norm", 0.0)
            ) if isinstance(initial, dict) else 0.0,
            "cycle_period_at_start": float(
                initial.get("cycle_period", 0.0)
            ) if isinstance(initial, dict) else 0.0,
            "held_mechanism_fields": [
                "rule_margins",
                "final_layer_norm_gain",
                "circuit_norm",
                "adam_exp_avg",
                "adam_exp_avg_sq",
                "adam_parameter_step"
            ],
            "continuing_external_fields": [
                "rng_state",
                "data_order",
                "gradient_relaxation_phase"
            ],
            "predicted_shift_step_low": 700,
            "predicted_shift_step_high": 1100,
            "predicted_stability_effect": "NO_CHANGE",
            "stability_effect_rationale": (
                "Setting current gradients to None makes Adam skip both parameter "
                "and moment/variance/step updates during the hold, so formation "
                "progress and circuit fragility are held together. RNG, data order, "
                "and measured gradient phase continue to evolve, which widens the "
                "shift and phase uncertainty; the expected post-formation stability "
                "direction remains unchanged because reserve and Adam state are held."
            ),
        }

    @staticmethod
    def apply(stage, context, state):
        if stage != "before_optimizer_step":
            return state
        held = int(state.get("held_optimizer_steps", 0))
        limit = int(state.get("hold_optimizer_steps", 0))
        if held >= limit:
            state["active"] = False
            return state
        affected = 0
        for parameter in _candidate_parameters(context):
            if hasattr(parameter, "grad"):
                parameter.grad = None
                affected += 1
        if affected == 0:
            raise ValueError("INTERVENTION_CONTEXT_EXPOSED_NO_CURRENT_GRADIENTS")
        state["held_optimizer_steps"] = held + 1
        state["affected_parameter_count"] = affected
        state["active"] = state["held_optimizer_steps"] < limit
        return state
