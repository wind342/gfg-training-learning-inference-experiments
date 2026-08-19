"""Sealed finite-state forecast for rule formation in a training GFG.

The implementation reads only the supplied prefix.  It deliberately uses
optimizer-step identities from GFG records and never wall-clock or run data.
"""

import math


_EPS = 1.0e-6
_GRID = 100
_END_STEP = 10000


def _clip(value, low, high):
    return low if value < low else high if value > high else value


def _logit(value):
    value = _clip(float(value), _EPS, 1.0 - _EPS)
    return math.log(value / (1.0 - value))


def _sigmoid(value):
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _as_rows(gfg_prefix):
    if hasattr(gfg_prefix, "evaluations"):
        rows = gfg_prefix.evaluations()
    elif isinstance(gfg_prefix, dict):
        rows = gfg_prefix.get("evaluations", [])
    else:
        rows = []
    clean = []
    for row in rows:
        clean.append({
            "optimizer_step": int(row["optimizer_step"]),
            "train_accuracy": float(row["train_accuracy"]),
            "validation_accuracy": float(row["validation_accuracy"]),
        })
    clean.sort(key=lambda row: row["optimizer_step"])
    return clean


def _gradient_norm_rows(gfg_prefix, cut_step):
    if hasattr(gfg_prefix, "occurrences"):
        try:
            rows = gfg_prefix.occurrences(
                occurrence_type="gradient_clip", min_step=0,
                max_step=cut_step)
        except TypeError:
            rows = gfg_prefix.occurrences("gradient_clip", 0, cut_step)
    elif isinstance(gfg_prefix, dict):
        rows = gfg_prefix.get("gradient_clip_occurrences", [])
    else:
        rows = []
    result = []
    for row in rows:
        payload = row.get("payload", {})
        norm = payload.get("total_norm")
        step = row.get("optimizer_step")
        if norm is not None and step is not None and int(step) <= cut_step:
            result.append((int(step), float(norm)))
    result.sort()
    return result


def _pulse_episodes(norm_rows):
    """Return contiguous episodes whose pre-clip norm exceeds clip norm 1."""
    episodes = []
    current = None
    previous_step = None
    for step, norm in norm_rows:
        if norm > 1.0:
            if current is None or previous_step is None or step != previous_step + 1:
                current = {"step_low": step, "step_high": step,
                           "peak_norm": norm}
                episodes.append(current)
            else:
                current["step_high"] = step
                if norm > current["peak_norm"]:
                    current["peak_norm"] = norm
            previous_step = step
        else:
            current = None
            previous_step = None
    # Isolated crossings are ordinary gradient noise, not a relaxation
    # release.  A pulse must persist for at least three concrete updates.
    return [episode for episode in episodes
            if episode["step_high"] - episode["step_low"] + 1 >= 3]


def _mature_period(episodes):
    starts = [episode["step_low"] for episode in episodes]
    if len(starts) >= 3:
        last_gap = starts[-1] - starts[-2]
        prior_gap = starts[-2] - starts[-3]
        # The observed relaxation interval approaches an asymptote.  This
        # finite extrapolation uses only prefix occurrence identities.
        period = last_gap + 0.85 * (last_gap - prior_gap)
    elif len(starts) == 2:
        period = 1.35 * (starts[-1] - starts[-2])
    else:
        period = 370.0
    return int(round(_clip(period, 250.0, 500.0)))


def _gain_and_transition(evaluations):
    latest = evaluations[-1]
    cut = latest["optimizer_step"]
    current = latest["validation_accuracy"]
    eligible = [row for row in evaluations
                if row["optimizer_step"] < cut and
                row["optimizer_step"] >= cut - 400]
    reference = eligible[0] if eligible else evaluations[0]
    delta = cut - reference["optimizer_step"]
    gain = ((_logit(current) - _logit(reference["validation_accuracy"])) /
            float(delta)) if delta > 0 else 0.0
    gain = _clip(gain, 5.0e-5, 8.0e-3)
    raw_estimate = cut + (_logit(0.9) - _logit(current)) / gain
    will_transition = raw_estimate <= _END_STEP
    estimate = int(round(raw_estimate / _GRID) * _GRID)
    estimate = max(cut + _GRID, min(_END_STEP, estimate))
    return gain, estimate, will_transition


def _observed_state(evaluations, gain):
    latest = evaluations[-1]
    train = latest["train_accuracy"]
    validation = latest["validation_accuracy"]
    if train < 0.99:
        return "FITTING_SAMPLES"
    if validation >= 0.9:
        last_three = evaluations[-3:]
        if len(last_three) == 3 and all(
                row["train_accuracy"] >= 0.99 and
                row["validation_accuracy"] >= 0.9 for row in last_three):
            return "RULE_GENERALIZED"
        return "RULE_MARGIN_CROSSING"
    if validation <= 0.3 and gain <= 5.0e-4:
        return "SAMPLE_MEMORIZATION"
    return "RULE_EXTRACTION"


def _risk_intervals(transition_step, period, end_step):
    center = transition_step + int(round(0.8 * period))
    intervals = []
    while center <= end_step + 75:
        intervals.append({
            "step_low": max(transition_step + 1, center - 75),
            "step_high": min(end_step, center + 75),
        })
        center += period
    return [row for row in intervals if row["step_low"] <= row["step_high"]]


def _in_interval(step, intervals):
    return any(row["step_low"] <= step <= row["step_high"]
               for row in intervals)


class CapabilityFormationMechanism:
    """Prefix-initialized clipped-Adam rule-margin relaxation mechanism."""

    @staticmethod
    def initialize(gfg_prefix):
        evaluations = _as_rows(gfg_prefix)
        if not evaluations:
            raise ValueError("GFG_PREFIX_HAS_NO_EVALUATIONS")
        cut_step = evaluations[-1]["optimizer_step"]
        gain, estimate, will_transition = _gain_and_transition(evaluations)
        norm_rows = _gradient_norm_rows(gfg_prefix, cut_step)
        episodes = _pulse_episodes(norm_rows)
        period = _mature_period(episodes)
        recent_norms = [norm for step, norm in norm_rows
                        if step >= cut_step - 100]
        pulse_fraction = (sum(norm > 1.0 for norm in recent_norms) /
                          float(len(recent_norms))) if recent_norms else 0.0
        peak = max(recent_norms) if recent_norms else 0.0
        latest = evaluations[-1]
        return {
            "schema": "gfg-rule-oscillator-state-v1",
            "cut_step": cut_step,
            "evaluation_interval": _GRID,
            "observation_count": len(evaluations),
            "latest_train_accuracy": latest["train_accuracy"],
            "latest_validation_accuracy": latest["validation_accuracy"],
            "rule_log_odds": _logit(latest["validation_accuracy"]),
            "rule_log_odds_gain_per_step": gain,
            "transition_estimate": estimate,
            "will_transition_by_10000": will_transition,
            "mechanism_state": _observed_state(evaluations, gain),
            "clip_threshold": 1.0,
            "pulse_episode_count": len(episodes),
            "pulse_episode_starts": [row["step_low"] for row in episodes],
            "mature_pulse_period": period,
            "recent_clip_fraction": pulse_fraction,
            "recent_peak_preclip_norm": peak,
            "observed_evaluations": evaluations,
        }

    @staticmethod
    def forecast(state):
        cut = int(state["cut_step"])
        estimate = int(state["transition_estimate"])
        gain = float(state["rule_log_odds_gain_per_step"])
        current_log_odds = float(state["rule_log_odds"])
        period = int(state["mature_pulse_period"])
        will_transition = bool(state["will_transition_by_10000"])
        intervals = (_risk_intervals(estimate, period, _END_STEP)
                     if will_transition and
                     int(state["pulse_episode_count"]) >= 2 else [])
        stability = ("TRANSIENT_DEGRADATION_RECOVERY" if intervals else
                     "UNDETERMINED")
        curve = []
        evolution = []
        first_grid = ((cut // _GRID) + 1) * _GRID
        for step in range(first_grid, _END_STEP + 1, _GRID):
            probability = _sigmoid(current_log_odds + gain * (step - cut))
            at_risk = _in_interval(step, intervals)
            if at_risk and step > estimate:
                probability = max(0.0, probability - 0.10)
            probability = _clip(probability, 0.0, 1.0)
            if at_risk and step > estimate:
                phase = "SHOCK_DEGRADED_RECOVERING"
            elif probability >= 0.9:
                phase = "RULE_GENERALIZED"
            elif probability >= 0.3:
                phase = "RULE_EXTRACTION"
            else:
                phase = "SAMPLE_MEMORIZATION"
            curve.append({"optimizer_step": step,
                          "validation_accuracy": probability})
            evolution.append({
                "optimizer_step": step,
                "state": phase,
                "rule_log_odds": current_log_odds + gain * (step - cut),
                "pulse_phase": ((step - estimate) % period) / float(period),
            })
        if will_transition:
            low_200 = max(cut + 1, estimate - 100)
            high_200 = min(_END_STEP, low_200 + 200)
            low_500 = max(cut + 1, estimate - 200)
            high_500 = min(_END_STEP, low_500 + 400)
        else:
            low_200 = high_200 = "NO_TRANSITION"
            low_500 = high_500 = "NO_TRANSITION"
        return {
            "will_transition": will_transition,
            "transition_step_low_200": low_200,
            "transition_step_high_200": high_200,
            "transition_step_low_500": low_500,
            "transition_step_high_500": high_500,
            "predicted_validation_curve": curve,
            "mechanism_state": evolution,
            "post_formation_stability": stability,
            "predicted_instability_intervals": intervals,
        }
