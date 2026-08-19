"""Sealed hook-only intervention: pause 800 optimizer updates.

At ``before_optimizer_step`` the current gradients are detached and the
current optimizer learning rates are set to zero.  With gradients absent,
AdamW does not advance per-parameter moments, step counters, parameters, or
decoupled weight decay.  The original rates are restored on the next update
after the finite pause.
"""


def _field(value, name, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _optimizer(context):
    candidate = _field(context, "optimizer", None)
    if candidate is not None:
        return candidate
    if _field(context, "param_groups", None) is not None:
        return context
    return None


def _groups(context):
    optimizer = _optimizer(context)
    if optimizer is not None:
        groups = _field(optimizer, "param_groups", None)
        if groups is not None:
            return list(groups)
    groups = _field(context, "optimizer_group_hyperparameters", None)
    return list(groups) if groups is not None else []


def _parameters(context, groups):
    found = []
    seen = set()
    for group in groups:
        params = _field(group, "params", [])
        for parameter in params:
            marker = id(parameter)
            if marker not in seen:
                seen.add(marker)
                found.append(parameter)
    model = _field(context, "model", None)
    if model is not None and hasattr(model, "parameters"):
        for parameter in model.parameters():
            marker = id(parameter)
            if marker not in seen:
                seen.add(marker)
                found.append(parameter)
    params = _field(context, "parameters", None)
    if params is not None:
        values = params.values() if isinstance(params, dict) else params
        for parameter in values:
            marker = id(parameter)
            if marker not in seen:
                seen.add(marker)
                found.append(parameter)
    return found


def _get_lr(group):
    if isinstance(group, dict):
        return float(group.get("lr", 0.0))
    return float(getattr(group, "lr", 0.0))


def _set_lr(group, value):
    if isinstance(group, dict):
        group["lr"] = float(value)
    else:
        setattr(group, "lr", float(value))


def _remove_gradient(parameter):
    if isinstance(parameter, dict):
        if "grad" in parameter:
            parameter["grad"] = None
    elif hasattr(parameter, "grad"):
        parameter.grad = None


def _zero_exposed_gradients(context):
    gradients = _field(context, "gradients", None)
    if gradients is None:
        return
    values = gradients.values() if isinstance(gradients, dict) else gradients
    for gradient in values:
        if gradient is not None and hasattr(gradient, "zero_"):
            gradient.zero_()


class TrainingIntervention:
    @staticmethod
    def initialize(mechanism_state, forecast):
        return {
            "schema": "gfg-training-intervention-state-v1",
            "pause_optimizer_updates": 800,
            "optimizer_updates_paused": 0,
            "active": True,
            "restore_pending": False,
            "original_learning_rates": None,
            "formation_progress_updates_held": 0,
            "oscillator_phase_updates_held": 0,
        }

    @staticmethod
    def apply(stage, context, state):
        if state is None:
            return None
        next_state = dict(state)
        if stage != "before_optimizer_step":
            return next_state

        groups = _groups(context)
        if next_state.get("restore_pending"):
            rates = next_state.get("original_learning_rates") or []
            for index, group in enumerate(groups):
                if index < len(rates):
                    _set_lr(group, rates[index])
            next_state["restore_pending"] = False
            next_state["active"] = False
            return next_state

        paused = int(next_state.get("optimizer_updates_paused", 0))
        limit = int(next_state.get("pause_optimizer_updates", 800))
        if not next_state.get("active", True) or paused >= limit:
            return next_state

        if next_state.get("original_learning_rates") is None:
            next_state["original_learning_rates"] = [
                _get_lr(group) for group in groups
            ]
        for group in groups:
            _set_lr(group, 0.0)
        for parameter in _parameters(context, groups):
            _remove_gradient(parameter)
        _zero_exposed_gradients(context)

        paused += 1
        next_state["optimizer_updates_paused"] = paused
        next_state["formation_progress_updates_held"] = paused
        next_state["oscillator_phase_updates_held"] = paused
        if paused >= limit:
            next_state["restore_pending"] = True
        return next_state
