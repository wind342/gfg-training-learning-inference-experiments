"""One sealed intervention through the frozen before-gradient-clip hook."""


class TrainingIntervention:
    target_max_norm = 0.25

    def initialize(self, mechanism_state, forecast):
        return {
            "schema": "gfg-clipped-impulse-intervention-state-v1",
            "hook": "before_gradient_clip",
            "target_max_norm": self.target_max_norm,
            "applications": 0,
            "mutation_observed": False,
            "predicted_direction": "DELAY",
            "predicted_transition_step_shift_interval": {
                "step_low": 600,
                "step_high": 4000,
            },
            "predicted_stability_effect": "IMPROVE",
            "formation_impulse_scale": 0.25,
            "stability_impulse_scale": 0.25,
            "mechanism_state_change": {
                "formation_state": {"impulse_scale": 0.25},
                "stability_state": {"impulse_scale": 0.25},
            },
        }

    def _set_max_norm(self, value):
        changed = False
        if isinstance(value, dict):
            for key in ("max_norm", "gradient_clip_max_norm", "clip_max_norm"):
                if key in value:
                    value[key] = self.target_max_norm
                    changed = True
            for key in ("gradient_clip_control", "gradient_clipping", "clip_control"):
                nested = value.get(key)
                if nested is not None:
                    changed = self._set_max_norm(nested) or changed
            if not changed:
                value["max_norm"] = self.target_max_norm
                changed = True
        else:
            for name in ("max_norm", "gradient_clip_max_norm", "clip_max_norm"):
                if hasattr(value, name):
                    try:
                        setattr(value, name, self.target_max_norm)
                        changed = True
                    except (AttributeError, TypeError, ValueError):
                        pass
            for name in ("gradient_clip_control", "gradient_clipping", "clip_control"):
                if hasattr(value, name):
                    changed = self._set_max_norm(getattr(value, name)) or changed
        return changed

    def apply(self, stage, context, state):
        if stage != "before_gradient_clip":
            return state
        next_state = dict(state)
        changed = self._set_max_norm(context)
        next_state["applications"] = int(state.get("applications", 0)) + 1
        next_state["mutation_observed"] = bool(
            state.get("mutation_observed", False) or changed
        )
        # These are the exact corresponding fields used by mechanism.py when
        # the counterfactual state is replayed.
        next_state["formation_impulse_scale"] = 0.25
        next_state["stability_impulse_scale"] = 0.25
        next_state["mechanism_state_change"] = {
            "formation_state": {"impulse_scale": 0.25},
            "stability_state": {"impulse_scale": 0.25},
        }
        return next_state
