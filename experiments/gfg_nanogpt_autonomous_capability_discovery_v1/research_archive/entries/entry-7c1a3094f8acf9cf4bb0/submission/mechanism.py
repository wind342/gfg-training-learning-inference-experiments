"""Sealed prefix-only capability-formation mechanism.

The implementation deliberately uses only the supplied prefix.  It models the
weight-decay/Adam relaxation oscillator seen in the primitive gradient and
parameter-version facts, and a rule-circuit progress variable observed through
evaluation outcomes and validation margins.
"""

import math


def _number(value):
    """Return a JSON-safe scalar from a Python or numerical scalar."""
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _flat(value):
    if hasattr(value, "reshape"):
        value = value.reshape(-1)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return [_number(value)]
    result = []
    stack = list(reversed(value))
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            stack.extend(reversed(item))
        else:
            result.append(_number(item))
    return result


def _quantile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(float(x) for x in values)
    position = (len(ordered) - 1) * fraction
    low = int(math.floor(position))
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _shape(value):
    shape = getattr(value, "shape", None)
    if shape is None:
        return ()
    return tuple(int(x) for x in shape)


def _transition_in_prefix(evaluations):
    for index in range(len(evaluations) - 2):
        window = evaluations[index:index + 3]
        if all(float(x["train_accuracy"]) >= 0.99 and
               float(x["validation_accuracy"]) >= 0.90 for x in window):
            earlier = evaluations[:index]
            if any(float(x["validation_accuracy"]) <= 0.30 for x in earlier):
                return int(window[0]["optimizer_step"])
    return None


def _burst_runs(samples):
    """Compress primitive global-gradient records into significant bursts."""
    runs = []
    current = []
    for step, value in sorted(samples):
        if value > 1.0:
            if current and step != current[-1][0] + 1:
                runs.append(current)
                current = []
            current.append((step, value))
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    result = []
    for run in runs:
        peak_step, peak_value = max(run, key=lambda item: item[1])
        if len(run) >= 5 and peak_value >= 2.5:
            result.append({
                "center_step": int(peak_step),
                "duration": int(len(run)),
                "peak_norm": float(peak_value),
            })
    return result


def _margin_state(prefix, cut_step):
    """Read the latest exact validation-logit formation and target identities."""
    try:
        logits_rows = prefix.objects(
            role="validation_logits", max_step=cut_step, materialized=True)
        target_rows = prefix.objects(
            role="evaluation_validation_targets", max_step=cut_step,
            materialized=True)
        if not logits_rows or not target_rows:
            return None
        latest_logits = max(logits_rows, key=lambda row: int(row["optimizer_step"]))
        same_step = int(latest_logits["optimizer_step"])
        compatible = [row for row in target_rows
                      if int(row["optimizer_step"]) == same_step]
        latest_targets = compatible[-1] if compatible else target_rows[-1]
        logits_tensor = prefix.load_tensor(latest_logits)
        targets_tensor = prefix.load_tensor(latest_targets)
        logits = _flat(logits_tensor)
        targets = _flat(targets_tensor)
        logits_shape = _shape(logits_tensor)
        targets_shape = _shape(targets_tensor)
        if len(logits_shape) != 2 or not targets_shape:
            return None
        rows, classes = logits_shape
        target_width = targets_shape[-1]
        margins = []
        for index in range(rows):
            label = int(targets[index * target_width + target_width - 1])
            if label < 0 or label >= classes:
                continue
            start = index * classes
            row = logits[start:start + classes]
            correct = row[label]
            strongest_other = max(row[:label] + row[label + 1:])
            margins.append(correct - strongest_other)
        if not margins:
            return None
        return {
            "median": float(_quantile(margins, 0.50)),
            "q10": float(_quantile(margins, 0.10)),
            "count": int(len(margins)),
        }
    except Exception:
        # Evaluations remain a complete, contract-defined fallback state.
        return None


def _optimizer_configuration(prefix, cut_step):
    learning_rate = 0.003
    weight_decay = 1.0
    try:
        rows = prefix.objects(role="optimizer_configuration", max_step=cut_step)
        if rows:
            payload = rows[0].get("literal_payload", {})
            groups = payload.get("param_groups", [])
            decayed = [group for group in groups
                       if float(group.get("weight_decay", 0.0)) > 0.0]
            if decayed:
                learning_rate = float(decayed[0].get("lr", learning_rate))
                weight_decay = float(decayed[0].get(
                    "weight_decay", weight_decay))
    except Exception:
        pass
    return learning_rate, weight_decay


def _weight_reserve(prefix, cut_step):
    """Normalized scale of the tied token/output weight formation state."""
    try:
        rows = prefix.objects(
            role="parameter_version", name_contains="wte.weight",
            max_step=cut_step, materialized=True)
        if not rows:
            return None
        norms = []
        for row in rows:
            tensor = prefix.load_tensor(row)
            values = _flat(tensor)
            norm = math.sqrt(sum(value * value for value in values))
            norms.append((int(row["optimizer_step"]), norm))
        norms.sort()
        maximum = max(value for _, value in norms)
        if maximum <= 0.0:
            return None
        return float(norms[-1][1] / maximum)
    except Exception:
        return None


def _predict_burst(state):
    centers = state["burst_centers_tail"]
    base_period = state["asymptotic_burst_period"]
    cut_step = state["cut_step"]
    if len(centers) >= 3:
        previous_period = centers[-2] - centers[-3]
        period = centers[-1] - centers[-2]
        acceleration = period - previous_period
        if period < 0.85 * base_period and acceleration > 0:
            next_period = period + 0.84 * acceleration
        else:
            next_period = period + 0.25 * (base_period - period)
        next_period = max(0.75 * period,
                          min(1.35 * base_period, next_period))
        return int(round(centers[-1] + next_period)), float(next_period)
    if len(centers) == 2:
        period = centers[-1] - centers[-2]
        next_period = min(1.35 * base_period,
                          period + max(80.0, 0.40 * period))
        return int(round(centers[-1] + next_period)), float(next_period)
    if centers:
        return int(round(centers[-1] + base_period)), float(base_period)
    return int(round(cut_step + base_period)), float(base_period)


def _grid_up(step, interval):
    return int(math.ceil(float(step) / interval) * interval)


def _future_pulses(state, formation_burst, formation_period):
    """Execute the optimizer/decay oscillator update law to step 10000."""
    base_period = state["asymptotic_burst_period"]
    reserve = state["weight_scale_reserve"]
    cycle_decay = state["reserve_multiplier_per_cycle"]
    center = float(formation_burst)
    period = float(formation_period)
    pulses = []
    cycle = 0
    while center <= 10000 + base_period:
        if cycle > 0:
            reserve *= cycle_decay
            pulses.append({
                "cycle": int(cycle),
                "center_step": int(round(center)),
                "reserve": float(max(0.0, reserve)),
            })
        period += 0.25 * (base_period - period)
        center += period
        cycle += 1
    return pulses


class CapabilityFormationMechanism:
    """Finite state law initialized only from a complete GFG prefix."""

    @staticmethod
    def initialize(gfg_prefix):
        evaluations = [dict(row) for row in gfg_prefix.evaluations()]
        if not evaluations:
            raise ValueError("PREFIX_HAS_NO_CAPABILITY_EVALUATION")
        evaluations.sort(key=lambda row: int(row["optimizer_step"]))
        cut_step = int(evaluations[-1]["optimizer_step"])

        differences = [
            int(evaluations[index]["optimizer_step"]) -
            int(evaluations[index - 1]["optimizer_step"])
            for index in range(2, len(evaluations))
        ]
        interval = int(round(_quantile(differences, 0.50))) if differences else 100
        if interval <= 0:
            interval = 100

        gradient_samples = []
        try:
            rows = gfg_prefix.objects(
                role="gradient_total_norm", max_step=cut_step,
                materialized=True)
            for row in rows:
                tensor = gfg_prefix.load_tensor(row)
                values = _flat(tensor)
                if values:
                    gradient_samples.append((
                        int(row["optimizer_step"]), float(values[0])))
        except Exception:
            gradient_samples = []
        bursts = _burst_runs(gradient_samples)

        learning_rate, weight_decay = _optimizer_configuration(
            gfg_prefix, cut_step)
        decay_product = max(1.0e-8, learning_rate * max(weight_decay, 0.05))
        base_period = 440.0 * 0.003 / decay_product
        base_period = float(max(250.0, min(800.0, base_period)))

        margin = _margin_state(gfg_prefix, cut_step)
        reserve = _weight_reserve(gfg_prefix, cut_step)
        if reserve is None:
            reserve = 0.95

        accuracies = [float(row["validation_accuracy"]) for row in evaluations]
        recent_changes = [
            accuracies[index] - accuracies[index - 1]
            for index in range(max(1, len(accuracies) - 3), len(accuracies))
            if accuracies[index] > accuracies[index - 1]
        ]
        grid_progress = _quantile(recent_changes, 0.50) if recent_changes else 0.025
        grid_progress = float(max(0.015, min(0.080, grid_progress)))

        transition_seen = _transition_in_prefix(evaluations)
        current_accuracy = float(evaluations[-1]["validation_accuracy"])
        if transition_seen is not None:
            phase = "GENERALIZED"
        elif current_accuracy > 0.75:
            phase = "FORMATION_BURST"
        elif current_accuracy >= 0.20:
            phase = "RULE_CIRCUIT_ACCUMULATION"
        else:
            phase = "MEMORIZATION"

        return {
            "schema": "gfg-capability-formation-state-v1",
            "cut_step": cut_step,
            "evaluation_interval": interval,
            "phase": phase,
            "transition_seen_step": transition_seen,
            "train_accuracy_tail": [
                float(row["train_accuracy"]) for row in evaluations[-3:]
            ],
            "validation_accuracy_tail": accuracies[-4:],
            "validation_grid_progress": grid_progress,
            "validation_margin_median": (
                float(margin["median"]) if margin else None),
            "validation_margin_q10": (
                float(margin["q10"]) if margin else None),
            "burst_centers_tail": [
                int(row["center_step"]) for row in bursts[-3:]
            ],
            "burst_peak_tail": [
                float(row["peak_norm"]) for row in bursts[-3:]
            ],
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "asymptotic_burst_period": base_period,
            "weight_scale_reserve": float(max(0.05, min(1.25, reserve))),
            "reserve_multiplier_per_cycle": float(math.exp(
                -0.064 * (decay_product / 0.003))),
        }

    @staticmethod
    def forecast(state):
        interval = int(state["evaluation_interval"])
        cut_step = int(state["cut_step"])
        current_accuracy = float(state["validation_accuracy_tail"][-1])

        if state.get("transition_seen_step") is not None:
            formation_burst = int(state["transition_seen_step"])
            formation_period = float(state["asymptotic_burst_period"])
            transition_center = int(state["transition_seen_step"])
        elif state["phase"] == "FORMATION_BURST":
            centers = state["burst_centers_tail"]
            formation_burst = centers[-1] if centers else cut_step
            formation_period = (
                centers[-1] - centers[-2] if len(centers) >= 2
                else state["asymptotic_burst_period"])
            transition_center = _grid_up(cut_step + 1, interval)
        else:
            formation_burst, formation_period = _predict_burst(state)
            transition_center = _grid_up(formation_burst, interval) + interval

        # Margins alter the executed response, rather than being diagnostics.
        median_margin = state.get("validation_margin_median")
        if median_margin is None:
            margin_signal = 0.32
        else:
            bounded = max(-30.0, min(30.0, float(median_margin) / 3.0))
            margin_signal = 1.0 / (1.0 + math.exp(-bounded))

        post_burst_grid = _grid_up(formation_burst, interval)
        start_grid = _grid_up(cut_step + 1, interval)
        future_steps = list(range(start_grid, 10001, interval))
        progress = current_accuracy
        baseline = {}
        for step in future_steps:
            if step < post_burst_grid:
                progress = min(0.74, progress + state["validation_grid_progress"])
            elif step == post_burst_grid and state.get("transition_seen_step") is None:
                progress = 0.75 + 0.35 * progress + 0.04 * (margin_signal - 0.32)
                progress = max(0.78, min(0.895, progress))
            elif progress < 0.97:
                progress += max(0.020, 0.35 * (1.0 - progress))
            else:
                progress += 0.18 * (0.995 - progress)
            progress = max(0.0, min(0.999, progress))
            baseline[step] = progress

        pulses = _future_pulses(state, formation_burst, formation_period)
        instability = []
        pulse_by_grid = {}
        for pulse in pulses:
            event_grid = int(round(pulse["center_step"] / interval) * interval)
            reserve = pulse["reserve"]
            severe = reserve < 0.36 and pulse["cycle"] % 2 == 0
            magnitude = 0.02
            if severe:
                magnitude = min(0.48, 0.20 + (0.36 - reserve) * 3.0)
                instability.append({
                    "step_low": max(start_grid, event_grid - interval),
                    "step_high": min(10000, event_grid + interval),
                })
            pulse_by_grid[event_grid] = max(
                pulse_by_grid.get(event_grid, 0.0), magnitude)

        curve = []
        evolution = []
        reserve_now = float(state["weight_scale_reserve"])
        for step in future_steps:
            accuracy = baseline[step]
            if step in pulse_by_grid and step > transition_center:
                accuracy = max(0.45, accuracy - pulse_by_grid[step])
            curve.append({
                "optimizer_step": int(step),
                "validation_accuracy": float(round(accuracy, 8)),
            })

            if step < post_burst_grid:
                phase = "RULE_CIRCUIT_ACCUMULATION"
            elif step < transition_center:
                phase = "FORMATION_BURST"
            elif pulse_by_grid.get(step, 0.0) >= 0.20:
                phase = "TRANSIENT_DEGRADATION"
            elif pulse_by_grid.get(step - interval, 0.0) >= 0.20:
                phase = "RECOVERY"
            else:
                phase = "GENERALIZED"
            elapsed_cycles = sum(
                1 for pulse in pulses if pulse["center_step"] <= step)
            reserve_now = state["weight_scale_reserve"] * (
                state["reserve_multiplier_per_cycle"] ** elapsed_cycles)
            evolution.append({
                "optimizer_step": int(step),
                "phase": phase,
                "rule_progress": float(round(baseline[step], 8)),
                "stability_reserve": float(round(max(0.0, reserve_now), 8)),
            })

        # The transition center is generated by the curve law; scan the same
        # contract definition so interval claims and the curve cannot diverge.
        qualifying = None
        for index in range(len(curve) - 2):
            if all(row["validation_accuracy"] >= 0.90
                   for row in curve[index:index + 3]):
                qualifying = int(curve[index]["optimizer_step"])
                break
        if state.get("transition_seen_step") is not None:
            qualifying = int(state["transition_seen_step"])
        if qualifying is None:
            qualifying = int(min(10000, transition_center))

        high_low = max(0, qualifying - interval)
        high_high = min(10000, qualifying + interval)
        primary_low = max(0, qualifying - 2 * interval)
        primary_high = min(10000, qualifying + 2 * interval)
        will_transition = qualifying <= 10000

        return {
            "will_transition": bool(will_transition),
            "transition_step_low_200": int(high_low),
            "transition_step_high_200": int(high_high),
            "transition_step_low_500": int(primary_low),
            "transition_step_high_500": int(primary_high),
            "predicted_validation_curve": curve,
            "mechanism_state": {
                "initial": state,
                "predicted_evolution": evolution,
                "formation_burst_step": int(formation_burst),
            },
            "post_formation_stability": (
                "TRANSIENT_DEGRADATION_RECOVERY" if instability else "STABLE"),
            "predicted_instability_intervals": instability,
        }
