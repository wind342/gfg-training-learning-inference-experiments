"""Sealed prefix-only capability-formation mechanism.

The executable uses no run identity and performs all GFG reads in initialize.
forecast is a pure state transition and does not retain the supplied prefix.
"""

import math


def _clamp(value, low, high):
    return max(low, min(high, value))


def _quantile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(float(x) for x in values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    mix = position - lower
    return ordered[lower] * (1.0 - mix) + ordered[upper] * mix


def _median(values):
    return _quantile(values, 0.5)


def _tensor_value(gfg_prefix, row):
    value = gfg_prefix.load_tensor(row)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _flat(value):
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_flat(item))
        return result
    return [float(value)]


def _transition_already_seen(evaluations):
    earlier_low = False
    for index, row in enumerate(evaluations):
        if float(row["validation_accuracy"]) <= 0.3:
            earlier_low = True
        if not earlier_low or index + 2 >= len(evaluations):
            continue
        window = evaluations[index:index + 3]
        if all(float(x["train_accuracy"]) >= 0.99 and
               float(x["validation_accuracy"]) >= 0.9 for x in window):
            return int(row["optimizer_step"])
    return None


def _validation_margins(gfg_prefix, rows, accuracy):
    logits_row = None
    targets_row = None
    for row in rows:
        if row.get("role") == "validation_logits":
            logits_row = row
        elif row.get("role") == "evaluation_validation_targets":
            targets_row = row
    if logits_row is None or targets_row is None:
        # Conservative fallback used only if the prefix omits materialized logits.
        proxy = math.log(_clamp(accuracy, 1e-5, 1.0 - 1e-5) /
                         (1.0 - _clamp(accuracy, 1e-5, 1.0 - 1e-5)))
        return proxy, proxy, proxy
    logits = _tensor_value(gfg_prefix, logits_row)
    targets = _tensor_value(gfg_prefix, targets_row)
    margins = []
    for row_logits, row_targets in zip(logits, targets):
        target = int(row_targets[-1] if isinstance(row_targets, (list, tuple))
                     else row_targets)
        correct = float(row_logits[target])
        competitor = max(float(value) for index, value in enumerate(row_logits)
                         if index != target)
        margins.append(correct - competitor)
    return (_quantile(margins, 0.10), _quantile(margins, 0.25),
            _quantile(margins, 0.50))


def _optimizer_tension(gfg_prefix, rows):
    first_sum = 0.0
    first_count = 0
    second_sum = 0.0
    second_count = 0
    for row in rows:
        name = row.get("name", "")
        if row.get("role") != "optimizer_state" or not row.get("materialized"):
            continue
        if name.endswith(".exp_avg"):
            values = _flat(_tensor_value(gfg_prefix, row))
            first_sum += sum(value * value for value in values)
            first_count += len(values)
        elif name.endswith(".exp_avg_sq"):
            values = _flat(_tensor_value(gfg_prefix, row))
            second_sum += sum(max(0.0, value) for value in values)
            second_count += len(values)
    if not first_count or not second_count:
        return 0.0
    first_rms = math.sqrt(first_sum / first_count)
    second_rms = math.sqrt(second_sum / second_count)
    return _clamp(first_rms / max(second_rms, 1e-12), 0.0, 2.0)


def _major_bursts(clip_occurrences, clip_threshold):
    samples = []
    for occurrence in clip_occurrences:
        payload = occurrence.get("payload", {})
        if "total_norm" in payload:
            samples.append((int(occurrence["optimizer_step"]),
                            float(payload["total_norm"])))
    samples.sort()
    raw = []
    start = None
    segment = []
    prior_step = None
    for step, norm in samples:
        active = norm > clip_threshold
        if active and start is None:
            start = step
            segment = []
        if active:
            segment.append((step, norm))
        if start is not None and (not active or
                                  (prior_step is not None and step > prior_step + 1)):
            if not active and segment:
                raw.append((start, segment[-1][0],
                            max(segment, key=lambda item: item[1])))
            elif segment:
                raw.append((start, segment[-1][0],
                            max(segment, key=lambda item: item[1])))
            start = step if active else None
            segment = [(step, norm)] if active else []
        prior_step = step
    if start is not None and segment:
        raw.append((start, segment[-1][0], max(segment, key=lambda item: item[1])))

    # Short sub-bursts separated by fewer than 50 steps are one relaxation event.
    merged = []
    for begin, end, peak in raw:
        if merged and begin <= merged[-1][1] + 50:
            old = merged[-1]
            merged[-1] = (old[0], end, peak if peak[1] > old[2][1] else old[2])
        else:
            merged.append((begin, end, peak))
    major_floor = 10.0 * clip_threshold
    major = [event for event in merged if event[2][1] >= major_floor]
    if len(major) < 2:
        major = merged[-4:]
    return major, samples


def _period_state(major_bursts, cut_step):
    peaks = [event[2][0] for event in major_bursts if event[2][0] <= cut_step]
    periods = [right - left for left, right in zip(peaks, peaks[1:])]
    if periods:
        last_period = float(periods[-1])
    else:
        last_period = 300.0
    differences = [right - left for left, right in zip(periods, periods[1:])]
    delta = _clamp(_median(differences[-3:]) if differences else 0.0,
                   -50.0, 80.0)
    return peaks, last_period, delta


def _future_bursts(state, end_step=10000):
    center = float(state["last_burst_peak_step"])
    period = float(state["last_burst_period"])
    delta = float(state["burst_period_delta"])
    # If the cut lies inside a burst, the recurrence begins from the cut and
    # uses a shortened first recovery interval.  This makes phase operational,
    # rather than merely a serialized label.
    active_at_cut = state.get("burst_phase") == "ACTIVE"
    if active_at_cut:
        center = max(center, float(state["cut_step"]))
    result = []
    while center < end_step:
        period = _clamp(period + delta, 150.0, 650.0)
        if active_at_cut:
            period = max(150.0, 0.5 * period)
            active_at_cut = False
        center += period
        delta *= 0.5
        if center > state["cut_step"]:
            result.append(center)
        if len(result) > 64:
            break
    return result


class CapabilityFormationMechanism:
    """Finite burst/margin state initialized solely from a supplied prefix."""

    @staticmethod
    def initialize(gfg_prefix):
        evaluations = list(gfg_prefix.evaluations())
        if not evaluations:
            raise ValueError("PREFIX_HAS_NO_EVALUATION")
        evaluations.sort(key=lambda row: int(row["optimizer_step"]))
        latest = evaluations[-1]
        cut_step = int(latest["optimizer_step"])
        accuracy = float(latest["validation_accuracy"])
        train_accuracy = float(latest["train_accuracy"])

        # This bounded step query retrieves current state, not a stored history.
        current_rows = list(gfg_prefix.objects(
            min_step=cut_step, max_step=cut_step, materialized=True))
        q10, q25, median = _validation_margins(
            gfg_prefix, current_rows, accuracy)
        tension = _optimizer_tension(gfg_prefix, current_rows)

        clip_occurrences = list(gfg_prefix.occurrences(
            occurrence_type="gradient_clip_application"))
        major, norm_samples = _major_bursts(clip_occurrences, 1.0)
        peaks, last_period, period_delta = _period_state(major, cut_step)
        last_peak = peaks[-1] if peaks else cut_step
        last_norm = norm_samples[-1][1] if norm_samples else 0.0
        formed_step = _transition_already_seen(evaluations)

        if formed_step is not None:
            cycles_remaining = 0
        elif accuracy < 0.18:
            cycles_remaining = 4
        elif q10 < -7.0:
            cycles_remaining = 2
        else:
            cycles_remaining = 1

        recent_accuracy = [float(row["validation_accuracy"])
                           for row in evaluations[-4:]]
        return {
            "schema": "gfg-capability-burst-margin-state-v1",
            "cut_step": cut_step,
            "train_accuracy": train_accuracy,
            "validation_accuracy": accuracy,
            "rule_margin_q10": float(q10),
            "rule_margin_q25": float(q25),
            "rule_margin_median": float(median),
            "recent_validation_accuracy": recent_accuracy,
            "formation_status": "FORMED" if formed_step is not None else "MEMORIZED",
            "formed_step_if_observed": formed_step,
            "cycles_remaining": cycles_remaining,
            "clip_threshold": 1.0,
            "major_burst_count": len(peaks),
            "last_burst_peak_step": int(last_peak),
            "last_burst_period": float(last_period),
            "burst_period_delta": float(period_delta),
            "burst_phase": "ACTIVE" if last_norm > 1.0 else "QUIESCENT",
            "last_gradient_total_norm": float(last_norm),
            "optimizer_tension": float(tension),
            "forecast_horizon_step": 10000
        }

    @staticmethod
    def forecast(state):
        cut_step = int(state["cut_step"])
        future_bursts = _future_bursts(state, 10000)
        observed_transition = state.get("formed_step_if_observed")
        if observed_transition is not None:
            transition_step = int(observed_transition)
        elif len(future_bursts) >= int(state["cycles_remaining"]):
            center = future_bursts[int(state["cycles_remaining"]) - 1]
            transition_step = int(math.ceil(center / 100.0) * 100)
        else:
            transition_step = 10100
        will_transition = transition_step <= 10000

        if will_transition:
            low_200 = max(0, transition_step - 100)
            high_200 = min(10000, transition_step + 100)
            low_500 = max(0, transition_step - 200)
            high_500 = min(10000, transition_step + 200)
        else:
            low_200 = high_200 = low_500 = high_500 = "NO_TRANSITION"

        current_accuracy = _clamp(float(state["validation_accuracy"]),
                                  1e-5, 1.0 - 1e-5)
        current_logit = math.log(current_accuracy / (1.0 - current_accuracy))
        if will_transition and transition_step > cut_step:
            target_logit = math.log(0.95 / 0.05)
            growth_rate = (target_logit - current_logit) / (transition_step - cut_step)
        else:
            growth_rate = 0.0

        tension = float(state["optimizer_tension"])
        hazard_amplitude = 0.04 + min(0.04, 0.5 * tension)
        curve = []
        evolution = []
        for step in range(((cut_step // 100) + 1) * 100, 10001, 100):
            logit = _clamp(current_logit + growth_rate * (step - cut_step),
                           -50.0, 50.0)
            expected = 1.0 / (1.0 + math.exp(-logit))
            pulse = 0.0
            nearest = None
            for center in future_bursts:
                distance = abs(step - center)
                if nearest is None or distance < nearest:
                    nearest = distance
                pulse = max(pulse, math.exp(-distance / 45.0))
            if will_transition and step >= transition_step:
                expected = min(0.997, expected)
                expected -= hazard_amplitude * pulse
            expected = _clamp(expected, 0.0, 1.0)
            curve.append({"optimizer_step": step,
                          "validation_accuracy": float(expected)})

            if will_transition and step >= transition_step:
                reserve = 4.0 + 6.0 * (1.0 - math.exp(
                    -(step - transition_step) / 300.0)) - 7.0 * pulse
                status = "FORMED_VULNERABLE" if pulse > 0.35 else "FORMED_RECOVERED"
            else:
                fraction = ((step - cut_step) /
                            max(100.0, transition_step - cut_step))
                reserve = (float(state["rule_margin_q10"]) * (1.0 - fraction) +
                           1.0 * fraction)
                status = "FORMATION_BURST" if pulse > 0.35 else "MEMORIZED"
            evolution.append({
                "optimizer_step": step,
                "formation_status": status,
                "rule_margin_q10": float(reserve),
                "burst_stress": float(pulse),
                "validation_accuracy": float(expected)
            })

        intervals = []
        post_centers = [center for center in future_bursts
                        if not will_transition or center >= transition_step]
        for index, center in enumerate(post_centers):
            half = min(220, int(round(60 + 80 * tension + 5 * index)))
            low = max(transition_step if will_transition else cut_step + 1,
                      int(round(center - half)))
            high = min(10000, int(round(center + 2 * half)))
            if high < low:
                continue
            if intervals and low <= intervals[-1]["step_high"] + 1:
                intervals[-1]["step_high"] = max(intervals[-1]["step_high"], high)
            else:
                intervals.append({"step_low": low, "step_high": high})

        mechanism_forecast = {
            "initial_state": dict(state),
            "predicted_evolution": evolution,
            "update_law": "margin_growth_with_clipped_burst_hazard_v1"
        }
        stability = ("TRANSIENT_DEGRADATION_RECOVERY" if will_transition
                     else "UNDETERMINED")
        if not will_transition:
            intervals = []
        return {
            "will_transition": bool(will_transition),
            "transition_step_low_200": low_200,
            "transition_step_high_200": high_200,
            "transition_step_low_500": low_500,
            "transition_step_high_500": high_500,
            "predicted_validation_curve": curve,
            "mechanism_state": mechanism_forecast,
            "post_formation_stability": stability,
            "predicted_instability_intervals": intervals
        }
