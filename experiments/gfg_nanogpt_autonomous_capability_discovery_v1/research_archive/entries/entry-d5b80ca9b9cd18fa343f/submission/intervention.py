"""One frozen-hook intervention: an 800-step zero-LR parameter pause."""


class TrainingIntervention:
    SCHEMA = "gfg-zero-lr-pause-intervention-v1"
    PAUSE_STEPS = 800

    @staticmethod
    def initialize(mechanism_state, forecast):
        return {
            "schema": TrainingIntervention.SCHEMA,
            "hook": "before_optimizer_step",
            "remaining_pause_steps": TrainingIntervention.PAUSE_STEPS,
            "pause_steps_applied": 0,
            "base_learning_rates": None,
            "restored": False,
            "mutation_succeeded": False,
            "parameter_progress_steps_suppressed": 0,
            "optimizer_age_steps_advanced": 0,
            "direction": "DELAY",
            "predicted_direction": "DELAY",
            "shift_step_low": 600,
            "shift_step_high": 1100,
            "transition_step_shift_low": 600,
            "transition_step_shift_high": 1100,
            "transition_step_shift_interval": {
                "step_low": 600,
                "step_high": 1100,
            },
            "predicted_mechanism_state_change": {
                "parameter_progress_steps_suppressed": 800,
                "optimizer_age_steps_advanced": 800,
                "rule_margin": "held_near_intervention_start",
                "optimizer_stress": "rephased",
            },
            "predicted_shift_step_low": 600,
            "predicted_shift_step_high": 1100,
            "predicted_stability_effect": "NO_CHANGE",
            "stability_effect_rationale": (
                "The pause rephases Adam moments but does not remove the "
                "long-run moment/gradient stress mechanism."),
        }

    @staticmethod
    def _groups(context):
        candidates = []
        if isinstance(context, dict):
            for key in ("optimizer", "native_optimizer"):
                if key in context:
                    candidates.append(context[key])
            for key in ("optimizer_param_groups",
                        "optimizer_group_hyperparameters", "param_groups"):
                value = context.get(key)
                if isinstance(value, list):
                    return value
        else:
            candidates.append(context)
            try:
                candidates.append(context.optimizer)
            except Exception:
                pass
        for candidate in candidates:
            if isinstance(candidate, dict):
                groups = candidate.get("param_groups")
            else:
                groups = getattr(candidate, "param_groups", None)
            if isinstance(groups, list):
                return groups
        return None

    @staticmethod
    def apply(stage, context, state):
        if state is None or state.get("schema") != TrainingIntervention.SCHEMA:
            raise ValueError("UNRECOGNIZED_INTERVENTION_STATE")
        if stage != "before_optimizer_step":
            return state
        groups = TrainingIntervention._groups(context)
        if not groups:
            return state
        if state["base_learning_rates"] is None:
            state["base_learning_rates"] = [
                float(group.get("lr", 0.0)) for group in groups]
        if state["remaining_pause_steps"] > 0:
            for group in groups:
                group["lr"] = 0.0
            state["remaining_pause_steps"] -= 1
            state["pause_steps_applied"] += 1
            state["parameter_progress_steps_suppressed"] += 1
            # Adam first/second moments and step counters are intentionally
            # left native and therefore continue to evolve during the pause.
            state["optimizer_age_steps_advanced"] += 1
            state["mutation_succeeded"] = True
            return state
        if not state["restored"]:
            for index, group in enumerate(groups):
                if index < len(state["base_learning_rates"]):
                    group["lr"] = state["base_learning_rates"][index]
            state["restored"] = True
        return state
