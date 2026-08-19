"""A finite, hook-only intervention that delays effective optimizer time."""


class TrainingIntervention:
    """Skip 800 parameter updates without advancing per-parameter Adam state."""

    @staticmethod
    def initialize(mechanism_state, forecast):
        return {
            "schema": "gradient-pause-intervention-state-v1",
            "direction": "DELAY",
            "pause_updates": 800,
            "skipped_updates": 0,
            "active": True,
            "completed": False,
        }

    @staticmethod
    def _optimizer(context):
        if isinstance(context, dict):
            return context.get("optimizer")
        if hasattr(context, "param_groups"):
            return context
        return getattr(context, "optimizer", None)

    @staticmethod
    def _model(context):
        if isinstance(context, dict):
            return context.get("model")
        if hasattr(context, "parameters") and not hasattr(
                context, "param_groups"):
            return context
        return getattr(context, "model", None)

    @staticmethod
    def _parameters(context):
        model = TrainingIntervention._model(context)
        if model is not None and hasattr(model, "parameters"):
            return list(model.parameters())
        optimizer = TrainingIntervention._optimizer(context)
        if optimizer is not None and hasattr(optimizer, "param_groups"):
            parameters = []
            for group in optimizer.param_groups:
                parameters.extend(list(group.get("params", [])))
            return parameters
        if isinstance(context, dict):
            parameters = context.get("parameters", [])
        else:
            parameters = getattr(context, "parameters", [])
        if callable(parameters):
            parameters = parameters()
        return list(parameters)

    @staticmethod
    def apply(stage, context, state):
        next_state = dict(state)
        if stage != "before_optimizer_step":
            return next_state
        if int(next_state["skipped_updates"]) >= int(
                next_state["pause_updates"]):
            next_state["active"] = False
            next_state["completed"] = True
            return next_state

        # AdamW skips a parameter, including its moments and decoupled weight
        # decay, when grad is None.  This pauses effective optimizer time and
        # does not retain a native object in serialized intervention state.
        optimizer = TrainingIntervention._optimizer(context)
        if optimizer is not None and hasattr(optimizer, "zero_grad"):
            optimizer.zero_grad(set_to_none=True)
        for parameter in TrainingIntervention._parameters(context):
            if hasattr(parameter, "grad"):
                parameter.grad = None

        if isinstance(context, dict) and "gradients" in context:
            gradients = context["gradients"]
            if isinstance(gradients, dict):
                for key in list(gradients):
                    gradients[key] = None
            elif isinstance(gradients, list):
                for index in range(len(gradients)):
                    gradients[index] = None

        next_state["skipped_updates"] = int(
            next_state["skipped_updates"]) + 1
        if next_state["skipped_updates"] >= next_state["pause_updates"]:
            next_state["active"] = False
            next_state["completed"] = True
        return next_state
