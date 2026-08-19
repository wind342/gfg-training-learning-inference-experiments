"""Sealed prefix-only capability-formation mechanism.

The executable uses only information already present in the supplied GFG
prefix.  It compresses the prefix into a rule-margin state and the phase of
the weight-decay/Adam gradient-burst relaxation oscillator.  Forecasting is
then performed from that finite state; no future graph method is retained.
"""

import math


EVALUATION_GRID = 100
END_STEP = 10000
BURST_NORM_THRESHOLD = 1.0
REFERENCE_PERIOD = 380.0
RULE_LOGIT_GAIN_PER_BURST = 1.75
RULE_LOGIT_DRIFT_PER_STEP = 0.0004


def _clip(value, low, high):
    return low if value < low else high if value > high else value


def _logit(probability):
    probability = _clip(float(probability), 0.005, 0.995)
    return math.log(probability / (1.0 - probability))


def _sigmoid(value):
    if value >= 0.0:
        z = math.exp(-min(value, 60.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -60.0))
    return z / (1.0 + z)


def _median(values):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _prefix_rows(gfg_prefix, method_name, **arguments):
    """Read a prefix collection without retaining the GFG object."""
    if hasattr(gfg_prefix, method_name):
        method = getattr(gfg_prefix, method_name)
        try:
            return list(method(**arguments))
        except TypeError:
            return list(method())
    if isinstance(gfg_prefix, dict):
        value = gfg_prefix.get(method_name, [])
        if isinstance(value, dict):
            value = value.get("rows", [])
        return list(value)
    raise TypeError("gfg_prefix must expose GFG prefix collections")


def _evaluation_rows(gfg_prefix):
    rows = _prefix_rows(gfg_prefix, "evaluations")
    clean = []
    for row in rows:
        step = int(row["optimizer_step"])
        clean.append({
            "optimizer_step": step,
            "train_accuracy": float(row["train_accuracy"]),
            "validation_accuracy": float(row["validation_accuracy"]),
            "loss": float(row.get("loss", 0.0)),
        })
    clean.sort(key=lambda row: row["optimizer_step"])
    return clean


def _gradient_rows(gfg_prefix, cut_step):
    rows = _prefix_rows(
        gfg_prefix, "occurrences", occurrence_type="gradient_clip",
        min_step=0, max_step=cut_step,
    )
    clean = []
    for row in rows:
        if row.get("occurrence_type") not in (None, "gradient_clip"):
            continue
        step = int(row["optimizer_step"])
        if step > cut_step:
            continue
        payload = row.get("payload") or {}
        if "total_norm" in payload:
            clean.append((step, float(payload["total_norm"])))
    clean.sort()
    return clean


def _optimizer_constants(gfg_prefix):
    weight_decay = 1.0
    learning_rate = 0.003
    try:
        rows = _prefix_rows(
            gfg_prefix, "objects", role="optimizer_configuration",
            min_step=0, max_step=0, materialized=True,
        )
    except (KeyError, TypeError, AttributeError):
        rows = []
    for row in rows:
        literal = row.get("literal_payload") or {}
        groups = literal.get("param_groups") or []
        positive = [float(group.get("weight_decay", 0.0)) for group in groups
                    if float(group.get("weight_decay", 0.0)) > 0.0]
        rates = [float(group.get("lr", learning_rate)) for group in groups]
        if positive:
            weight_decay = max(positive)
        if rates:
            learning_rate = _median(rates)
        break
    return weight_decay, learning_rate


def _completed_bursts(gradient_rows):
    """Compress contiguous high-norm occurrences to concrete peak steps."""
    runs = []
    current = []
    previous_step = None
    for step, norm in gradient_rows:
        high = norm >= BURST_NORM_THRESHOLD
        if high:
            if current and previous_step is not None and step != previous_step + 1:
                runs.append(current)
                current = []
            current.append((step, norm))
        elif current:
            runs.append(current)
            current = []
        previous_step = step
    if current:
        runs.append(current)
    # Ignore the nonstationary initialization episode; later events are the
    # weight-decay/Adam relaxation bursts used by the transition law.
    peaks = []
    for run in runs:
        peak = max(run, key=lambda pair: pair[1])
        if peak[0] >= 100:
            peaks.append({"step": int(peak[0]), "peak_total_norm": float(peak[1])})
    # A clipped burst can briefly cross below the threshold and then rebound.
    # Merge only nearby fragments; distinct relaxation events in discovery are
    # separated by hundreds of optimizer occurrences.
    merged = []
    for peak in peaks:
        if merged and peak["step"] - merged[-1]["step"] < 80:
            if peak["peak_total_norm"] > merged[-1]["peak_total_norm"]:
                merged[-1] = peak
        else:
            merged.append(peak)
    return merged


def _next_period(bursts, weight_decay):
    steps = [item["step"] for item in bursts]
    intervals = [steps[index] - steps[index - 1]
                 for index in range(1, len(steps))]
    equilibrium = REFERENCE_PERIOD / math.sqrt(max(weight_decay, 0.05))
    if len(intervals) >= 3:
        recent = intervals[-3:]
        changes = [recent[index] - recent[index - 1]
                   for index in range(1, len(recent))]
        candidate = recent[-1] + _median(changes)
    elif len(intervals) == 2:
        candidate = intervals[-1] + 0.6 * (intervals[-1] - intervals[-2])
    elif len(intervals) == 1:
        candidate = intervals[-1] + 0.25 * (equilibrium - intervals[-1])
    else:
        candidate = equilibrium
    return _clip(candidate, 220.0, 1.15 * equilibrium)


def _historical_transition(evaluations):
    earlier_low = False
    for index, row in enumerate(evaluations):
        if row["validation_accuracy"] <= 0.3:
            earlier_low = True
        if not earlier_low or index + 2 >= len(evaluations):
            continue
        window = evaluations[index:index + 3]
        if all(item["train_accuracy"] >= 0.99 and
               item["validation_accuracy"] >= 0.9 for item in window):
            return int(row["optimizer_step"])
    return None


class CapabilityFormationMechanism:
    """Finite-state relaxation-burst mechanism compiled from primitive GFGs."""

    @staticmethod
    def initialize(gfg_prefix):
        evaluations = _evaluation_rows(gfg_prefix)
        if not evaluations:
            raise ValueError("prefix contains no capability evaluation")
        cut_step = int(evaluations[-1]["optimizer_step"])
        gradients = _gradient_rows(gfg_prefix, cut_step)
        bursts = _completed_bursts(gradients)
        weight_decay, learning_rate = _optimizer_constants(gfg_prefix)
        period = _next_period(bursts, weight_decay)
        last_burst = bursts[-1]["step"] if bursts else max(0, cut_step - period)
        elapsed = max(0.0, cut_step - last_burst)
        decay_progress = _clip(elapsed / period, 0.0, 1.5)
        current_validation = float(evaluations[-1]["validation_accuracy"])
        current_train = float(evaluations[-1]["train_accuracy"])
        historical_transition = _historical_transition(evaluations)
        formed_rows = ([row for row in evaluations
                        if historical_transition is not None and
                        row["optimizer_step"] >= historical_transition] or
                       [evaluations[-1]])
        stable_validation = max(float(row["validation_accuracy"])
                                for row in formed_rows)
        recovery_active = (historical_transition is not None and
                           current_validation < 0.9)
        return {
            "schema": "gfg-capability-state-v1",
            "cut_step": cut_step,
            "observed_evaluations": evaluations,
            "rule_logit": _logit(current_validation),
            "stable_rule_logit": _logit(stable_validation),
            "recovery_active_at_cut": bool(recovery_active),
            "memorization_strength": _clip(current_train, 0.0, 1.0),
            "gradient_burst_steps": [int(item["step"]) for item in bursts],
            "gradient_burst_peak_norms": [float(item["peak_total_norm"]) for item in bursts],
            "last_burst_step": int(last_burst),
            "estimated_next_period": float(period),
            "decay_progress": float(decay_progress),
            "positive_weight_decay": float(weight_decay),
            "learning_rate": float(learning_rate),
            "historical_transition_step": historical_transition,
            "formation_threshold": 0.9,
            "rule_logit_gain_per_burst": RULE_LOGIT_GAIN_PER_BURST,
        }

    @staticmethod
    def forecast(state):
        cut_step = int(state["cut_step"])
        observed = list(state["observed_evaluations"])
        observed_by_step = {int(row["optimizer_step"]): row for row in observed}
        weight_decay = max(float(state["positive_weight_decay"]), 0.05)
        equilibrium = REFERENCE_PERIOD / math.sqrt(weight_decay)
        period = float(state["estimated_next_period"])
        decay_progress_at_cut = float(state["decay_progress"])
        event_step = cut_step + period * max(0.0, 1.0 - decay_progress_at_cut)
        rule_gain = float(state["rule_logit_gain_per_burst"])
        formation_threshold = float(state["formation_threshold"])
        # If extrapolation falls inside the observed prefix, advance without
        # inventing a second event at the same state.
        while event_step <= cut_step:
            period = 0.75 * period + 0.25 * equilibrium
            event_step += period
        future_events = []
        while event_step <= END_STEP + equilibrium:
            future_events.append(int(round(event_step)))
            period = 0.75 * period + 0.25 * equilibrium
            event_step += period

        grid = [1] + list(range(100, END_STEP + 1, EVALUATION_GRID))
        rule_logit_at_cut = float(state["rule_logit"])
        stable_rule_logit = float(state["stable_rule_logit"])
        recovery_active_at_cut = bool(state["recovery_active_at_cut"])
        recovery_grid = ((cut_step // EVALUATION_GRID) + 1) * EVALUATION_GRID
        rows = []
        trajectory = []
        predicted_formation = state.get("historical_transition_step")
        for step in grid:
            if step in observed_by_step:
                accuracy = float(observed_by_step[step]["validation_accuracy"])
                operative_logit = _logit(accuracy)
            elif step < cut_step:
                continue
            else:
                events_seen = sum(1 for event in future_events if cut_step < event <= step)
                base_logit = (stable_rule_logit
                              if recovery_active_at_cut and step >= recovery_grid
                              else rule_logit_at_cut)
                operative_logit = (base_logit +
                                    RULE_LOGIT_DRIFT_PER_STEP * (step - cut_step) +
                                    rule_gain * events_seen)
                accuracy = _sigmoid(operative_logit)
            if (predicted_formation is None and step > cut_step and
                    accuracy >= formation_threshold):
                # The sustained-window test is performed below; this is only
                # the candidate crossing used by the recovery-risk law.
                predicted_formation = step
            next_event = next((event for event in future_events if event > step), None)
            prior_events = [event for event in future_events if event <= step]
            previous_event = prior_events[-1] if prior_events else int(state["last_burst_step"])
            local_period = (next_event - previous_event) if next_event is not None else equilibrium
            phase = _clip((step - previous_event) / max(local_period, 1.0), 0.0, 1.5)
            events_after_formation = 0
            if predicted_formation is not None:
                events_after_formation = sum(
                    1 for event in future_events
                    if predicted_formation < event <= step)
            stability_buffer = _clip(events_after_formation / 4.0, 0.0, 1.0)
            recovery_risk = bool(recovery_active_at_cut and step == cut_step)
            # The first post-formation burst is an evidenced fragile window.
            # Later mature-state collapses are emitted only for close
            # burst/evaluation resonance, avoiding a claim of a fixed clock.
            if predicted_formation is not None and step > predicted_formation:
                nearest = min(future_events, key=lambda event: abs(event - step)) if future_events else None
                if nearest is not None:
                    rank = sum(1 for event in future_events
                               if predicted_formation < event <= nearest)
                    distance = abs(nearest - step)
                    mature_resonance_width = 4.0 + 2.0 * (1.0 - stability_buffer)
                    recovery_risk = (recovery_risk or
                                     (rank == 1 and distance <= 25) or
                                     (rank > 1 and distance <= mature_resonance_width))
                    if recovery_risk and step > cut_step:
                        accuracy = min(accuracy, 0.22 if rank == 1 else 0.55)
            rows.append({
                "optimizer_step": int(step),
                "validation_accuracy": float(_clip(accuracy, 0.0, 1.0)),
            })
            trajectory.append({
                "optimizer_step": int(step),
                "rule_logit": float(operative_logit),
                "rule_strength": float(_sigmoid(operative_logit)),
                "decay_progress": float(phase),
                "burst_count_after_cut": int(sum(
                    1 for event in future_events if cut_step < event <= step)),
                "adam_recovery_active": bool(recovery_risk),
                "stability_buffer": float(stability_buffer),
            })

        # Apply the frozen three-point transition contract to the generated
        # curve, retaining the earlier low evaluation from the prefix.
        transition = state.get("historical_transition_step")
        earlier_low = False
        for index, row in enumerate(rows):
            observed_row = observed_by_step.get(row["optimizer_step"])
            train_accuracy = (float(observed_row["train_accuracy"])
                              if observed_row is not None else
                              float(state["memorization_strength"]))
            if row["validation_accuracy"] <= 0.3:
                earlier_low = True
            if transition is not None or not earlier_low or index + 2 >= len(rows):
                continue
            window = rows[index:index + 3]
            if (train_accuracy >= 0.99 and
                    all(item["validation_accuracy"] >= formation_threshold
                        for item in window)):
                transition = int(row["optimizer_step"])

        will_transition = transition is not None and transition <= END_STEP
        if will_transition:
            center = int(transition)
            low_200 = max(cut_step, center - 100)
            high_200 = low_200 + 200
            if high_200 > END_STEP:
                high_200 = END_STEP
                low_200 = max(cut_step, high_200 - 200)
            low_500 = max(cut_step, center - 200)
            high_500 = min(END_STEP, low_500 + 400)
            low_500 = max(cut_step, high_500 - 400)
        else:
            low_200 = high_200 = low_500 = high_500 = None

        instability = []
        for item in trajectory:
            if item["optimizer_step"] > cut_step and item["adam_recovery_active"]:
                instability.append({
                    "step_low": int(item["optimizer_step"]),
                    "step_high": int(item["optimizer_step"]),
                })
        return {
            "will_transition": bool(will_transition),
            "transition_step_low_200": low_200,
            "transition_step_high_200": high_200,
            "transition_step_low_500": low_500,
            "transition_step_high_500": high_500,
            "predicted_validation_curve": rows,
            "mechanism_state": {
                "schema": "gfg-capability-state-trajectory-v1",
                "initial": {
                    "cut_step": cut_step,
                    "rule_logit": float(state["rule_logit"]),
                    "stable_rule_logit": float(state["stable_rule_logit"]),
                    "recovery_active_at_cut": bool(state["recovery_active_at_cut"]),
                    "memorization_strength": float(state["memorization_strength"]),
                    "decay_progress": float(state["decay_progress"]),
                    "estimated_next_period": float(state["estimated_next_period"]),
                    "positive_weight_decay": float(state["positive_weight_decay"]),
                    "learning_rate": float(state["learning_rate"]),
                    "rule_logit_gain_per_burst": float(state["rule_logit_gain_per_burst"]),
                },
                "operative_fields": [
                    "rule_logit", "stable_rule_logit", "recovery_active_at_cut",
                    "decay_progress", "positive_weight_decay",
                    "estimated_next_period", "memorization_strength",
                    "adam_recovery_active", "stability_buffer",
                ],
                "trajectory": trajectory,
            },
            "post_formation_stability": "TRANSIENT_DEGRADATION_RECOVERY",
            "predicted_instability_intervals": instability,
        }
