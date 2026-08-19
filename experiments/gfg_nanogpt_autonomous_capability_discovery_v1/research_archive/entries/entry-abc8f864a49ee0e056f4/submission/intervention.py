"""A hook-only delay intervention that pauses parameter learning."""


def _get(container, name, default=None):
    if isinstance(container, dict):
        return container.get(name, default)
    return getattr(container, name, default)


def _optimizer(context):
    optimizer = _get(context, "optimizer")
    if optimizer is not None:
        return optimizer
    training_state = _get(context, "training_state")
    if training_state is not None:
        return _get(training_state, "optimizer")
    return None


def _groups(optimizer):
    groups = _get(optimizer, "param_groups", [])
    return list(groups) if groups is not None else []


def _set_group_value(group, key, value):
    if isinstance(group, dict):
        group[key] = value
    else:
        setattr(group, key, value)


def _group_value(group, key, default=None):
    if isinstance(group, dict):
        return group.get(key, default)
    return getattr(group, key, default)


def _zero_tensor(value):
    if value is None:
        return
    zero = getattr(value, "zero_", None)
    if callable(zero):
        zero()
        return
    if isinstance(value, dict):
        for key in list(value):
            item = value[key]
            if isinstance(item, (int, float)):
                value[key] = 0.0
            else:
                _zero_tensor(item)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            if isinstance(item, (int, float)):
                value[index] = 0.0
            else:
                _zero_tensor(item)


def _zero_current_gradients(context, optimizer):
    for key in ("gradients", "current_gradients", "parameter_gradients"):
        gradients = _get(context, key)
        if gradients is not None:
            _zero_tensor(gradients)
    model = _get(context, "model")
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        for parameter in parameters():
            _zero_tensor(getattr(parameter, "grad", None))
    if optimizer is not None:
        for group in _groups(optimizer):
            for parameter in _group_value(group, "params", []):
                _zero_tensor(getattr(parameter, "grad", None))


class TrainingIntervention:
    PAUSE_OPTIMIZER_STEPS = 1200

    @staticmethod
    def initialize(mechanism_state, forecast):
        cut_step = None
        if isinstance(mechanism_state, dict):
            cut_step = mechanism_state.get("cut_step")
        if cut_step is None and isinstance(forecast, dict):
            cut_step = forecast.get("forecast_cut_step")
        return {
            "schema": "zero-lr-delay-intervention-state-v1",
            "direction": "DELAY",
            "transition_step_shift_low": 1000,
            "transition_step_shift_high": 1800,
            "predicted_mechanism_change": (
                "RULE_ASSEMBLY formation_progress is held constant during "
                "the pause, then resumes from the same parameter result."),
            "initialized_at_optimizer_step": cut_step,
            "pause_optimizer_steps": TrainingIntervention.PAUSE_OPTIMIZER_STEPS,
            "paused_steps": 0,
            "original_learning_rates": [],
            "optimizer_observed": False,
            "phase": "ARMED",
        }

    @staticmethod
    def apply(stage, context, state):
        if state is None:
            return None
        paused = int(state.get("paused_steps", 0))
        duration = int(state.get(
            "pause_optimizer_steps",
            TrainingIntervention.PAUSE_OPTIMIZER_STEPS))
        active = paused < duration
        optimizer = _optimizer(context)
        if active and stage == "after_backward":
            _zero_current_gradients(context, optimizer)
        if stage == "before_optimizer_step":
            if optimizer is not None:
                groups = _groups(optimizer)
                if not state.get("original_learning_rates"):
                    state["original_learning_rates"] = [
                        float(_group_value(group, "lr", 0.0))
                        for group in groups]
                state["optimizer_observed"] = True
                if active:
                    _zero_current_gradients(context, optimizer)
                    for group in groups:
                        _set_group_value(group, "lr", 0.0)
                    state["phase"] = "PAUSED"
                else:
                    original = state.get("original_learning_rates", [])
                    for index, group in enumerate(groups):
                        if index < len(original):
                            _set_group_value(group, "lr", float(original[index]))
                    state["phase"] = "RELEASED"
        if stage == "after_optimizer_step" and active:
            state["paused_steps"] = paused + 1
            if state["paused_steps"] >= duration:
                state["phase"] = "READY_TO_RELEASE"
        return state
