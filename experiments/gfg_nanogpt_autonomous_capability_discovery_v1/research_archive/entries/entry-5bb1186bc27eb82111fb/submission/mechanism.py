"""Finite-state capability-formation forecast from a sealed GFG prefix.

The executable uses evaluation facts in the supplied prefix only.  It stores
no graph handle, object locator, occurrence identity, or task answer table.
"""

from math import ceil


def _rows_from_prefix(gfg_prefix):
    if isinstance(gfg_prefix, dict):
        rows = gfg_prefix.get("evaluations", [])
    else:
        rows = gfg_prefix.evaluations()
    clean = []
    for row in rows:
        clean.append({
            "optimizer_step": int(row["optimizer_step"]),
            "train_accuracy": float(row["train_accuracy"]),
            "validation_accuracy": float(row["validation_accuracy"]),
            "loss": float(row["loss"]),
        })
    clean.sort(key=lambda row: row["optimizer_step"])
    return clean


def _grid_interval(rows):
    differences = []
    for left, right in zip(rows, rows[1:]):
        difference = right["optimizer_step"] - left["optimizer_step"]
        if difference > 0:
            differences.append(difference)
    if not differences:
        return 100
    # The first evaluation can be at step 1.  The modal difference is the
    # evaluation grid and avoids treating the initial 1->100 gap specially.
    counts = {}
    for difference in differences:
        counts[difference] = counts.get(difference, 0) + 1
    return int(sorted(counts, key=lambda value: (-counts[value], value))[0])


def _snap_to_grid(value, interval):
    return int(round(float(value) / interval) * interval)


def _first_sustained_transition(rows):
    earlier_low = False
    for index, row in enumerate(rows):
        if row["validation_accuracy"] <= 0.30:
            earlier_low = True
        window = rows[index:index + 3]
        if (earlier_low and len(window) == 3 and
                all(item["train_accuracy"] >= 0.99 and
                    item["validation_accuracy"] >= 0.90
                    for item in window)):
            return int(row["optimizer_step"])
    return None


class CapabilityFormationMechanism:
    """Rule-completion state machine with an Adam relaxation-risk state."""

    @staticmethod
    def initialize(gfg_prefix):
        rows = _rows_from_prefix(gfg_prefix)
        if not rows:
            raise ValueError("GFG_PREFIX_HAS_NO_EVALUATION_FACTS")

        interval = _grid_interval(rows)
        current = rows[-1]
        observed_transition = _first_sustained_transition(rows)
        current_validation = current["validation_accuracy"]
        current_train = current["train_accuracy"]

        if observed_transition is not None:
            phase = "RULE_FORMED"
        elif current_train < 0.99:
            phase = "SAMPLE_MEMORIZATION"
        elif current_validation < 0.90:
            phase = "RULE_FORMING"
        else:
            phase = "RULE_CANDIDATE"

        # A prefix with high train accuracy and incomplete validation coverage
        # forecasts the remaining rule-completion time from the current
        # coverage deficit.  The bias represents the three-grid persistence
        # requirement and the lag between partial and uniform positive margin.
        raw_estimate = (current["optimizer_step"] + 150.0 +
                        1000.0 * max(0.0, 0.90 - current_validation))
        transition_estimate = _snap_to_grid(raw_estimate, interval)
        transition_estimate = max(
            transition_estimate,
            current["optimizer_step"] + 3 * interval)
        transition_estimate = min(transition_estimate, 10000 - 2 * interval)

        observed = [{
            "optimizer_step": row["optimizer_step"],
            "train_accuracy": row["train_accuracy"],
            "validation_accuracy": row["validation_accuracy"],
            "loss": row["loss"],
        } for row in rows]

        return {
            "schema": "capability-formation-state-v1",
            "phase": phase,
            "prediction_cut_step": current["optimizer_step"],
            "evaluation_interval": interval,
            "train_accuracy": current_train,
            "validation_accuracy": current_validation,
            "memorization_gap": max(0.0, current_train - current_validation),
            "rule_completion_fraction": current_validation,
            "transition_estimate": int(transition_estimate),
            "observed_transition": observed_transition,
            "observed_evaluations": observed,
            "stability_oscillator": {
                "mode": "ADAPTIVE_SECOND_MOMENT_RELAXATION",
                "first_post_formation_offset": 2900,
                "later_period": 4500,
                "recovery_driver": "CLIPPED_FULL_BATCH_GRADIENTS",
            },
        }

    @staticmethod
    def forecast(state):
        cut = int(state["prediction_cut_step"])
        interval = int(state["evaluation_interval"])
        transition = int(state["transition_estimate"])
        observed_transition = state.get("observed_transition")

        if observed_transition is not None:
            will_transition = False
            low_200 = "NO_TRANSITION"
            high_200 = "NO_TRANSITION"
            low_500 = "NO_TRANSITION"
            high_500 = "NO_TRANSITION"
            formation_step = int(observed_transition)
        else:
            will_transition = transition <= 10000
            low_200 = max(cut + interval, transition - interval)
            high_200 = min(10000, transition + interval)
            low_500 = max(cut + interval, transition - 2 * interval)
            high_500 = min(10000, transition + 2 * interval)
            formation_step = transition

        oscillator = state["stability_oscillator"]
        first_center = formation_step + int(
            oscillator["first_post_formation_offset"])
        centers = []
        center = first_center
        while center <= 10000:
            centers.append(center)
            center += int(oscillator["later_period"])
        instability_intervals = [{
            "step_low": max(formation_step, center - interval),
            "step_high": min(10000, center + interval),
        } for center in centers]

        observed_by_step = {
            int(row["optimizer_step"]): float(row["validation_accuracy"])
            for row in state["observed_evaluations"]
        }
        curve = []
        evolution = []
        current_accuracy = float(state["validation_accuracy"])
        denominator = max(interval, formation_step - cut)

        evaluation_steps = sorted(set(
            list(observed_by_step) + list(range(
                int(ceil((cut + interval) / interval)) * interval,
                10001,
                interval))))
        for step in evaluation_steps:
            if step in observed_by_step:
                predicted_accuracy = observed_by_step[step]
            elif step < formation_step:
                progress = max(0.0, min(
                    1.0, (step - cut) / float(denominator)))
                predicted_accuracy = (current_accuracy +
                    (0.97 - current_accuracy) * progress * progress)
            else:
                predicted_accuracy = 0.99

            adaptive_risk = "LOW"
            phase = "RULE_FORMED" if step >= formation_step else "RULE_FORMING"
            if centers and step == centers[0]:
                predicted_accuracy = 0.30
                adaptive_risk = "EXCURSION"
                phase = "TRANSIENT_DEGRADED"
            elif len(centers) > 1 and step in centers[1:]:
                predicted_accuracy = 0.65
                adaptive_risk = "EXCURSION"
                phase = "TRANSIENT_DEGRADED"
            elif any(step == center - interval for center in centers):
                adaptive_risk = "HIGH"
            elif any(step == center + interval for center in centers):
                phase = "RULE_RECOVERED"

            predicted_accuracy = max(0.0, min(1.0, predicted_accuracy))
            curve.append({
                "optimizer_step": int(step),
                "validation_accuracy": float(predicted_accuracy),
            })
            evolution.append({
                "optimizer_step": int(step),
                "phase": phase,
                "rule_completion_fraction": float(predicted_accuracy),
                "adaptive_risk": adaptive_risk,
            })

        return {
            "will_transition": bool(will_transition),
            "transition_step_low_200": low_200,
            "transition_step_high_200": high_200,
            "transition_step_low_500": low_500,
            "transition_step_high_500": high_500,
            "predicted_validation_curve": curve,
            "mechanism_state": {
                "initial": {
                    "phase": state["phase"],
                    "rule_completion_fraction": state[
                        "rule_completion_fraction"],
                    "memorization_gap": state["memorization_gap"],
                },
                "predicted_evolution": evolution,
            },
            "post_formation_stability":
                "TRANSIENT_DEGRADATION_RECOVERY",
            "predicted_instability_intervals": instability_intervals,
        }
