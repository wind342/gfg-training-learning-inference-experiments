"""Sealed prefix-only capability-formation mechanism.

The implementation intentionally depends only on the supplied bounded GFG
prefix.  It does not construct a GFG client, name a run, or inspect a path.
"""

import math


def _median(values):
    values = sorted(float(x) for x in values)
    if not values:
        return 0.0
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return 0.5 * (values[middle - 1] + values[middle])


def _as_rows(prefix, method_name, **arguments):
    method = getattr(prefix, method_name, None)
    if callable(method):
        return list(method(**arguments))
    if isinstance(prefix, dict):
        rows = list(prefix.get(method_name, []))
        answer = []
        for row in rows:
            if method_name == "objects":
                role = arguments.get("role")
                contains = arguments.get("name_contains")
                materialized = arguments.get("materialized")
                if role is not None and row.get("role") != role:
                    continue
                if contains is not None and contains not in row.get("name", ""):
                    continue
                if materialized is not None and bool(row.get("materialized")) != materialized:
                    continue
            elif method_name == "occurrences":
                wanted = arguments.get("occurrence_type")
                if wanted is not None and row.get("occurrence_type") != wanted:
                    continue
            answer.append(row)
        return answer
    return []


def _scalar(value):
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    return float(value)


def _sum_squares(value):
    """Return (sum of squares, element count) for numpy/torch/list values."""
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    reshape = getattr(value, "reshape", None)
    if callable(reshape):
        value = reshape(-1)

    total = 0.0
    count = 0

    def visit(node):
        nonlocal total, count
        if isinstance(node, (list, tuple)):
            for child in node:
                visit(child)
            return
        try:
            number = _scalar(node)
        except (TypeError, ValueError):
            try:
                for child in node:
                    visit(child)
            except TypeError:
                return
        else:
            total += number * number
            count += 1

    visit(value)
    return total, count


def _load(prefix, object_row):
    loader = getattr(prefix, "load_tensor", None)
    if callable(loader):
        return loader(object_row)
    if isinstance(object_row, dict):
        if "values" in object_row:
            return object_row["values"]
        if "value" in object_row:
            return object_row["value"]
    return []


def _gain_at(state, optimizer_step):
    delta = max(0.0, float(optimizer_step - state["cut_step"]))
    floor = state["gain_rate_floor_l2_per_step"]
    excess = state["gain_rate_l2_per_step"] - floor
    tau = state["gain_rate_relaxation_steps"]
    return (state["gain_l2"] + floor * delta
            + excess * tau * (1.0 - math.exp(-delta / tau)))


def _gain_rate_at(state, optimizer_step):
    delta = max(0.0, float(optimizer_step - state["cut_step"]))
    floor = state["gain_rate_floor_l2_per_step"]
    excess = state["gain_rate_l2_per_step"] - floor
    return floor + excess * math.exp(
        -delta / state["gain_rate_relaxation_steps"])


def _rule_strength(state, gain_l2):
    gain_rms = gain_l2 / math.sqrt(float(state["gain_dimension"]))
    argument = state["rule_sigmoid_slope"] * (
        gain_rms - state["rule_sigmoid_center_gain_rms"])
    if argument >= 35.0:
        return 1.0
    if argument <= -35.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-argument))


def _project_pulses(state):
    pulses = []
    last = int(state["last_burst_step"])
    period = float(state["last_burst_period"])
    resonance_used = bool(state["resonance_used"])

    # Preserve an active prefix pulse so its recovery affects the first forecast.
    if last >= 0 and state["cut_step"] - last <= 70:
        pulses.append({
            "step": last,
            "duration": 70,
            "resonance": False,
        })

    for _ in range(64):
        gain_rms = _gain_at(state, max(last, state["cut_step"])) / math.sqrt(
            float(state["gain_dimension"]))
        target_period = 270.0 + 17.0 * gain_rms
        period = 0.2 * period + 0.8 * target_period
        event_step = int(round(last + period))
        if event_step <= state["cut_step"]:
            # No burst was present in the supplied prefix at the projected
            # time.  Treat that absence as right-censoring and move the hazard
            # just beyond the cut instead of inventing an unobserved pulse.
            event_step = state["cut_step"] + 1
        if event_step > state["forecast_end_step"]:
            break

        event_gain_rms = _gain_at(state, event_step) / math.sqrt(
            float(state["gain_dimension"]))
        resonance = (not resonance_used and
                     event_gain_rms >= state["resonance_gain_rms"])
        duration = 300 if resonance else 70
        pulses.append({
            "step": event_step,
            "duration": duration,
            "resonance": resonance,
        })
        if resonance:
            resonance_used = True
            period = max(period, 600.0)
        last = event_step
    return pulses


def _pulse_load(state, pulses, optimizer_step, gain_rms):
    load = 0.0
    for pulse in pulses:
        age = optimizer_step - pulse["step"]
        if age < 0 or age > pulse["duration"]:
            continue
        if pulse["resonance"]:
            candidate = 0.35 * (1.0 - 0.4 * age / pulse["duration"])
        else:
            amplitude = min(0.10, 0.05 + 0.008 * max(0.0, gain_rms - 2.6))
            candidate = amplitude * math.exp(-age / 35.0)
        load = max(load, candidate)
    return load


class CapabilityFormationMechanism:
    """Finite-state gain/optimizer-oscillator mechanism."""

    @staticmethod
    def initialize(gfg_prefix):
        evaluations = _as_rows(gfg_prefix, "evaluations")
        evaluations = sorted(evaluations, key=lambda row: int(row["optimizer_step"]))
        if not evaluations:
            raise ValueError("PREFIX_HAS_NO_CAPABILITY_EVALUATION")
        cut_step = int(evaluations[-1]["optimizer_step"])

        gain_objects = _as_rows(
            gfg_prefix,
            "objects",
            role="parameter_version",
            name_contains="transformer.ln_f.weight",
            materialized=True,
        )
        gain_by_step = {}
        gain_dimension = 0
        for row in gain_objects:
            step = int(row.get("optimizer_step", -1))
            if step > cut_step:
                continue
            total, count = _sum_squares(_load(gfg_prefix, row))
            if count:
                gain_by_step[step] = math.sqrt(total)
                gain_dimension = count
        if not gain_by_step:
            raise ValueError("PREFIX_HAS_NO_MATERIALIZED_FINAL_NORM_GAIN")
        latest_gain_step = max(gain_by_step)
        gain_l2 = float(gain_by_step[latest_gain_step])

        gain_steps = sorted(gain_by_step)
        observed_rates = []
        recent_gain_steps = gain_steps[-7:]
        for left, right in zip(recent_gain_steps[:-1], recent_gain_steps[1:]):
            if right > left:
                observed_rates.append(
                    (gain_by_step[right] - gain_by_step[left]) / (right - left))
        observed_rate = _median(observed_rates)
        # Prefix data controls one quarter of the transport rate; the remainder
        # is a discovery-estimated state-dependent structural rate.  Expressing
        # slowdown through current gain, rather than absolute step, preserves
        # transport under an effective-progress intervention.
        structural_rate = 0.0043 + 0.0034 * math.exp(
            -max(0.0, gain_l2 - 14.4) / 12.0)
        observed_rate = min(0.0095, max(
            0.001, observed_rate or structural_rate))
        gain_rate = 0.25 * observed_rate + 0.75 * structural_rate

        clip_occurrences = _as_rows(
            gfg_prefix, "occurrences", occurrence_type="gradient_clip")
        norms = []
        for row in clip_occurrences:
            step = int(row.get("optimizer_step", -1))
            payload = row.get("payload", {})
            if step <= cut_step and "total_norm" in payload:
                norms.append((step, float(payload["total_norm"])))
        norms.sort()

        burst_starts = []
        low_count = 0
        active_length = 0
        maximum_active_length = 0
        for step, norm in norms:
            if norm <= 0.1:
                low_count += 1
                maximum_active_length = max(maximum_active_length, active_length)
                active_length = 0
            else:
                if norm > 1.0 and low_count >= 30:
                    burst_starts.append(step)
                low_count = 0
                active_length += 1
        maximum_active_length = max(maximum_active_length, active_length)

        if burst_starts:
            last_burst = int(burst_starts[-1])
        else:
            last_burst = max(0, cut_step - 300)
        if len(burst_starts) >= 2:
            last_period = float(burst_starts[-1] - burst_starts[-2])
        else:
            last_period = 300.0
        last_period = min(600.0, max(200.0, last_period))
        recovery_load = 0.0
        if last_burst <= cut_step <= last_burst + 70:
            recovery_load = math.exp(-(cut_step - last_burst) / 35.0)

        recent_evaluations = []
        for row in evaluations[-3:]:
            recent_evaluations.append({
                "optimizer_step": int(row["optimizer_step"]),
                "train_accuracy": float(row["train_accuracy"]),
                "validation_accuracy": float(row["validation_accuracy"]),
            })

        observed_transition_step = None
        had_earlier_low = False
        for index, row in enumerate(evaluations):
            if float(row["validation_accuracy"]) <= 0.3:
                had_earlier_low = True
            if not had_earlier_low or index + 2 >= len(evaluations):
                continue
            window = evaluations[index:index + 3]
            if all(float(item["train_accuracy"]) >= 0.99 and
                   float(item["validation_accuracy"]) >= 0.9
                   for item in window):
                observed_transition_step = int(row["optimizer_step"])
                break

        return {
            "schema": "capability-formation-state-v1",
            "cut_step": cut_step,
            "forecast_end_step": 10000,
            "evaluation_interval": 100,
            "gain_l2": gain_l2,
            "gain_dimension": int(gain_dimension),
            "gain_rate_l2_per_step": float(gain_rate),
            "gain_rate_floor_l2_per_step": 0.0043,
            "gain_rate_relaxation_steps": 2000.0,
            "rule_sigmoid_center_gain_rms": 2.11,
            "rule_sigmoid_slope": 4.5,
            "formation_gain_rms": 2.60,
            "resonance_gain_rms": 6.995,
            "last_burst_step": last_burst,
            "last_burst_period": last_period,
            "observed_burst_count": len(burst_starts),
            "recovery_load": float(recovery_load),
            "resonance_used": bool(maximum_active_length >= 150),
            "had_low_validation": any(
                float(row["validation_accuracy"]) <= 0.3 for row in evaluations),
            "observed_transition_step": observed_transition_step,
            "recent_evaluations": recent_evaluations,
        }

    @staticmethod
    def forecast(state):
        cut = int(state["cut_step"])
        end = int(state["forecast_end_step"])
        interval = int(state["evaluation_interval"])
        first_step = ((cut // interval) + 1) * interval
        pulses = _project_pulses(state)

        curve = []
        evolution = []
        for step in range(first_step, end + 1, interval):
            gain_l2 = _gain_at(state, step)
            gain_rms = gain_l2 / math.sqrt(float(state["gain_dimension"]))
            rule = _rule_strength(state, gain_l2)
            recovery = _pulse_load(state, pulses, step, gain_rms)
            accuracy = min(1.0, max(0.0, rule - recovery))
            formed = rule >= 0.9
            phase = "MEMORIZATION"
            if formed:
                phase = "RECOVERY" if recovery > 0.1 else "GENERALIZED"
            elif rule >= 0.3:
                phase = "FORMATION"
            curve.append({
                "optimizer_step": step,
                "validation_accuracy": float(accuracy),
            })
            evolution.append({
                "optimizer_step": step,
                "effective_update_count": step,
                "gain_l2": float(gain_l2),
                "gain_rms": float(gain_rms),
                "gain_rate_l2_per_step": float(_gain_rate_at(state, step)),
                "rule_strength": float(rule),
                "oscillator_recovery_load": float(recovery),
                "phase": phase,
            })

        transition_step = state.get("observed_transition_step")
        if transition_step is None and state.get("had_low_validation", False):
            for index in range(max(0, len(curve) - 2)):
                if all(curve[index + offset]["validation_accuracy"] >= 0.9
                       for offset in (0, 1, 2)):
                    transition_step = int(curve[index]["optimizer_step"])
                    break

        instability_intervals = []
        if transition_step is not None:
            for pulse in pulses:
                if not pulse["resonance"]:
                    continue
                if pulse["step"] + pulse["duration"] < transition_step:
                    continue
                instability_intervals.append({
                    "step_low": int(pulse["step"]),
                    "step_high": int(min(end, pulse["step"] + pulse["duration"])),
                })

        will_transition = transition_step is not None and transition_step <= end
        if will_transition:
            low_200 = max(0, transition_step - 100)
            high_200 = min(end, low_200 + 200)
            if high_200 < transition_step:
                high_200 = transition_step
                low_200 = high_200 - 200
            low_500 = max(0, transition_step - 200)
            high_500 = min(end, low_500 + 400)
            if high_500 < transition_step:
                high_500 = transition_step
                low_500 = high_500 - 400
        else:
            low_200 = high_200 = "NO_TRANSITION"
            low_500 = high_500 = "NO_TRANSITION"

        if will_transition and instability_intervals:
            stability = "TRANSIENT_DEGRADATION_RECOVERY"
        elif will_transition:
            stability = "UNDETERMINED"
        else:
            stability = "UNDETERMINED"

        return {
            "will_transition": bool(will_transition),
            "transition_step_low_200": low_200,
            "transition_step_high_200": high_200,
            "transition_step_low_500": low_500,
            "transition_step_high_500": high_500,
            "predicted_validation_curve": curve,
            "mechanism_state": {
                "initial_state": state,
                "predicted_evolution": evolution,
            },
            "post_formation_stability": stability,
            "predicted_instability_intervals": instability_intervals,
        }
