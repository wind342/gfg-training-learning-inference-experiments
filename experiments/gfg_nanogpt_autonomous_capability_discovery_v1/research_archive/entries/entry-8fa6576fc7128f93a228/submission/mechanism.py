"""Sealed prefix-only executable mechanism for capability formation."""

import math


def _as_list(value):
    data = value.tolist() if hasattr(value, "tolist") else value
    return data


def _flatten(value):
    value = _as_list(value)
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_flatten(item))
        return result
    return [float(value)]


def _l2(value):
    return math.sqrt(sum(x * x for x in _flatten(value)))


def _clamp(value, low, high):
    return max(low, min(high, value))


def _median(values):
    ordered = sorted(float(x) for x in values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _latest(rows, step):
    eligible = [row for row in rows if int(row.get("optimizer_step", -1)) <= step]
    if not eligible:
        raise ValueError("PREFIX_MISSING_REQUIRED_MATERIALIZED_OBJECT")
    return max(eligible, key=lambda row: int(row.get("optimizer_step", -1)))


def _burst_onsets(occurrences):
    """Debounce gradient-norm excursions into relaxation-cycle onsets."""
    ordered = sorted(occurrences, key=lambda row: int(row["optimizer_step"]))
    onsets = []
    last = -10**9
    for row in ordered:
        norm = float(row.get("payload", {}).get("total_norm", 0.0))
        step = int(row["optimizer_step"])
        if norm >= 6.0 and step - last >= 60:
            onsets.append(step)
            last = step
    return onsets


def _future_cycles(cut_step, onsets, period_override=None):
    if period_override is not None:
        period = float(period_override)
        center = float(onsets[-1]) if onsets else float(cut_step)
    elif len(onsets) >= 2:
        period = float(onsets[-1] - onsets[-2])
        center = float(onsets[-1])
    elif onsets:
        period = 300.0
        center = float(onsets[-1])
    else:
        period = 320.0
        center = float(cut_step)
    period = _clamp(period, 250.0, 380.0)
    initial_period = period
    cycles = []
    while center <= 10000.0:
        period = min(380.0, period * 1.15)
        center += period
        if center - 100.0 > 10000.0:
            break
        if center > cut_step:
            cycles.append({"center": center, "period": period})
    return cycles, initial_period


def _phase_load(step, cycles):
    best = 0.0
    for cycle in cycles:
        distance = float(step) - float(cycle["center"])
        if -100.0 <= distance <= 150.0:
            width = 100.0 if distance < 0.0 else 150.0
            best = max(best, 1.0 - abs(distance) / width)
    return best


class CapabilityFormationMechanism:
    """Prefix-conditioned margin/gain formation and stability state law."""

    MARGIN_GAIN_PER_FINAL_NORM = 2.0
    CIRCUIT_DECAY_PER_STEP = 0.00026
    BASELINE_ACCURACY = 0.995

    @staticmethod
    def initialize(gfg_prefix):
        evaluations = gfg_prefix.evaluations()
        if not evaluations:
            raise ValueError("PREFIX_HAS_NO_EVALUATION")
        evaluations = sorted(evaluations, key=lambda row: int(row["optimizer_step"]))
        cut_step = int(evaluations[-1]["optimizer_step"])

        parameter_objects = gfg_prefix.objects(
            role="parameter_version", materialized=True
        )
        final_norm_objects = [
            row for row in parameter_objects
            if row.get("name") == "transformer.ln_f.weight"
            and int(row.get("optimizer_step", -1)) > 0
            and int(row.get("optimizer_step", -1)) <= cut_step
        ]
        final_norm_objects.sort(key=lambda row: int(row["optimizer_step"]))
        if not final_norm_objects:
            raise ValueError("PREFIX_MISSING_FINAL_LAYER_NORM_STATE")
        final_history = []
        for row in final_norm_objects[-3:]:
            final_history.append({
                "optimizer_step": int(row["optimizer_step"]),
                "norm": _l2(gfg_prefix.load_tensor(row)),
            })
        current_final_norm = final_history[-1]["norm"]
        if len(final_history) >= 2:
            elapsed = final_history[-1]["optimizer_step"] - final_history[0]["optimizer_step"]
            final_norm_rate = (
                (final_history[-1]["norm"] - final_history[0]["norm"]) / elapsed
                if elapsed > 0 else 0.007
            )
        else:
            final_norm_rate = 0.007
        final_norm_rate = _clamp(final_norm_rate, 0.003, 0.015)

        circuit_objects = [
            row for row in parameter_objects
            if row.get("name") == "transformer.wte.weight"
            and int(row.get("optimizer_step", -1)) <= cut_step
        ]
        current_circuit_norm = _l2(
            gfg_prefix.load_tensor(_latest(circuit_objects, cut_step))
        )

        optimizer_objects = gfg_prefix.objects(
            role="optimizer_state", min_step=cut_step,
            max_step=cut_step, materialized=True
        )
        optimizer_by_name = {row.get("name"): row for row in optimizer_objects}
        pressure_sum = 0.0
        pressure_count = 0
        for name, mean_row in optimizer_by_name.items():
            if not name or not name.endswith(".exp_avg"):
                continue
            square_row = optimizer_by_name.get(name + "_sq")
            if square_row is None:
                continue
            mean_values = _flatten(gfg_prefix.load_tensor(mean_row))
            square_values = _flatten(gfg_prefix.load_tensor(square_row))
            for mean_value, square_value in zip(mean_values, square_values):
                normalized = mean_value / (math.sqrt(max(0.0, square_value)) + 1e-8)
                pressure_sum += normalized * normalized
                pressure_count += 1
        adam_pressure = math.sqrt(pressure_sum / pressure_count) if pressure_count else 0.0

        logits_rows = gfg_prefix.objects(
            role="validation_logits", min_step=cut_step,
            max_step=cut_step, materialized=True
        )
        logits_row = _latest(logits_rows, cut_step)
        logits = _as_list(gfg_prefix.load_tensor(logits_row))

        target_rows = gfg_prefix.objects(
            name_contains="validation_targets", materialized=True
        )
        target_row = _latest(target_rows, cut_step)
        targets = _as_list(gfg_prefix.load_tensor(target_row))
        if not logits or len(logits) != len(targets):
            raise ValueError("PREFIX_VALIDATION_BINDING_SHAPE_MISMATCH")
        rule_margins = []
        for logit_row, target_row_values in zip(logits, targets):
            correct = int(target_row_values[-1])
            correct_logit = float(logit_row[correct])
            other = max(
                float(value) for index, value in enumerate(logit_row)
                if index != correct
            )
            rule_margins.append(correct_logit - other)
        rule_margins.sort()

        gradient_occurrences = gfg_prefix.occurrences(
            occurrence_type="gradient_clip_application"
        )
        gradient_occurrences = [
            row for row in gradient_occurrences
            if int(row["optimizer_step"]) <= cut_step
        ]
        onsets = _burst_onsets(gradient_occurrences)
        _, cycle_period = _future_cycles(cut_step, onsets)

        observed_curve = [
            {
                "optimizer_step": int(row["optimizer_step"]),
                "validation_accuracy": float(row["validation_accuracy"]),
            }
            for row in evaluations
        ]
        observed_train_accuracy = [
            float(row["train_accuracy"]) for row in evaluations[-3:]
        ]
        return {
            "schema": "capability-formation-state-v1",
            "cut_step": cut_step,
            "evaluation_interval": 100,
            "observed_curve": observed_curve,
            "observed_train_accuracy_last_three": observed_train_accuracy,
            "rule_margins": [float(value) for value in rule_margins],
            "final_layer_norm_gain": float(current_final_norm),
            "final_layer_norm_rate": float(final_norm_rate),
            "circuit_norm": float(current_circuit_norm),
            "adam_pressure": float(adam_pressure),
            "margin_gain_per_final_norm": CapabilityFormationMechanism.MARGIN_GAIN_PER_FINAL_NORM,
            "circuit_decay_per_step": CapabilityFormationMechanism.CIRCUIT_DECAY_PER_STEP,
            "burst_onsets": [int(step) for step in onsets[-4:]],
            "cycle_period": float(cycle_period),
        }

    @staticmethod
    def forecast(state):
        cut = int(state["cut_step"])
        interval = int(state["evaluation_interval"])
        current_gain = float(state["final_layer_norm_gain"])
        gain_rate = float(state["final_layer_norm_rate"])
        current_circuit = float(state["circuit_norm"])
        current_adam_pressure = float(state["adam_pressure"])
        margin_gain = float(state["margin_gain_per_final_norm"])
        circuit_decay = float(state["circuit_decay_per_step"])
        margins = [float(value) for value in state["rule_margins"]]
        cycles, _ = _future_cycles(
            cut, state["burst_onsets"], state["cycle_period"]
        )

        curve = [dict(row) for row in state["observed_curve"]]
        evolution = []
        instability_intervals = []
        for cycle in cycles:
            center = float(cycle["center"])
            delta = max(0.0, center - cut)
            gain = current_gain + gain_rate * delta
            circuit = current_circuit * math.exp(-circuit_decay * delta)
            fragility = _clamp((gain / max(circuit, 1e-9) - 6.0) / 6.0, 0.0, 1.0)
            if fragility >= 0.20:
                instability_intervals.append({
                    "step_low": max(cut + 1, int(round(center - 100.0))),
                    "step_high": min(10000, int(round(center + 150.0))),
                })

        first_future = ((cut // interval) + 1) * interval
        for step in range(first_future, 10001, interval):
            delta = float(step - cut)
            gain = current_gain + gain_rate * delta
            circuit = current_circuit * math.exp(-circuit_decay * delta)
            margin_shift = margin_gain * (gain - current_gain)
            base_accuracy = sum(
                1 for value in margins if value + margin_shift > 0.0
            ) / float(len(margins))
            base_accuracy = min(
                CapabilityFormationMechanism.BASELINE_ACCURACY,
                base_accuracy,
            )
            fragility = _clamp(
                (gain / max(circuit, 1e-9) - 6.0) / 6.0, 0.0, 1.0
            )
            phase = _phase_load(step, cycles)
            predicted_peak_pressure = 0.10 + 0.30 * fragility
            optimizer_pressure = (
                current_adam_pressure * math.exp(-delta / 20.0)
                + phase * predicted_peak_pressure
            )
            pressure_fraction = _clamp(
                optimizer_pressure / max(0.10, predicted_peak_pressure),
                0.0,
                1.0,
            )
            shock_load = 0.08 * fragility * pressure_fraction
            accuracy = _clamp(base_accuracy - shock_load, 0.0, 1.0)
            curve.append({
                "optimizer_step": int(step),
                "validation_accuracy": float(accuracy),
            })
            evolution.append({
                "optimizer_step": int(step),
                "final_layer_norm_gain": float(gain),
                "rule_margin_shift": float(margin_shift),
                "rule_accuracy_base": float(base_accuracy),
                "circuit_norm": float(circuit),
                "circuit_fragility": float(fragility),
                "oscillator_phase_load": float(phase),
                "optimizer_pressure": float(optimizer_pressure),
                "shock_load": float(shock_load),
                "predicted_validation_accuracy": float(accuracy),
            })

        curve.sort(key=lambda row: int(row["optimizer_step"]))
        transition = None
        earlier_low = False
        last_three_train_good = all(
            value >= 0.99 for value in state["observed_train_accuracy_last_three"]
        )
        for index, row in enumerate(curve):
            if float(row["validation_accuracy"]) <= 0.30:
                earlier_low = True
            if index + 2 < len(curve) and earlier_low and last_three_train_good:
                window = curve[index:index + 3]
                if all(float(item["validation_accuracy"]) >= 0.90 for item in window):
                    transition = int(row["optimizer_step"])
                    break

        will_transition = transition is not None and transition <= 10000
        if will_transition:
            low_200 = max(1, transition - 100)
            high_200 = min(10000, transition + 100)
            low_500 = max(1, transition - 200)
            high_500 = min(10000, transition + 200)
            stability = (
                "TRANSIENT_DEGRADATION_RECOVERY"
                if instability_intervals else "STABLE"
            )
        else:
            low_200 = high_200 = "NO_TRANSITION"
            low_500 = high_500 = "NO_TRANSITION"
            stability = "UNDETERMINED"

        return {
            "will_transition": bool(will_transition),
            "transition_step_low_200": low_200,
            "transition_step_high_200": high_200,
            "transition_step_low_500": low_500,
            "transition_step_high_500": high_500,
            "predicted_validation_curve": curve,
            "mechanism_state": {
                "initial": state,
                "evolution": evolution,
            },
            "post_formation_stability": stability,
            "predicted_instability_intervals": instability_intervals,
        }
