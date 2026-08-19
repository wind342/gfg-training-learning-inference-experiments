"""Sealed prefix-only capability-formation mechanism.

The executable intentionally imports no filesystem, process, clock, dynamic-import,
or network modules.  initialize() is the only GFG-reading operation; forecast()
uses only its finite JSON state.
"""

import math


def _plain(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _flatten(value):
    value = _plain(value)
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_flatten(item))
        return result
    return [value]


def _rows(value, shape):
    value = _plain(value)
    if isinstance(value, list) and value and isinstance(value[0], list):
        return value
    flat = _flatten(value)
    width = int(shape[-1])
    return [flat[index:index + width] for index in range(0, len(flat), width)]


def _load(gfg_prefix, object_row):
    return gfg_prefix.load_tensor(object_row)


def _task_descriptor(gfg_prefix):
    rows = gfg_prefix.objects(role="training_task", materialized=True)
    if rows:
        return dict(rows[0].get("literal_payload", {}))
    return {}


def _training_triples(gfg_prefix):
    inputs = gfg_prefix.objects(
        role="training_batch_inputs", min_step=0, max_step=0,
        materialized=True)
    targets = gfg_prefix.objects(
        role="training_batch_targets", min_step=0, max_step=0,
        materialized=True)
    if not inputs or not targets:
        inputs = gfg_prefix.objects(role="training_batch_inputs", materialized=True)
        targets = gfg_prefix.objects(role="training_batch_targets", materialized=True)
    input_row = min(inputs, key=lambda row: int(row.get("optimizer_step", 0)))
    target_row = min(targets, key=lambda row: int(row.get("optimizer_step", 0)))
    xs = _rows(_load(gfg_prefix, input_row), input_row["shape"])
    ys = _rows(_load(gfg_prefix, target_row), target_row["shape"])
    return [[int(x[0]), int(x[-1]), int(y[-1])] for x, y in zip(xs, ys)]


def _recover_rule(triples, vocab_size, operator_token):
    tokens = [token for token in range(vocab_size) if token != operator_token]
    token_index = {token: index for index, token in enumerate(tokens)}
    modulus = len(tokens)
    matrix = []
    for left, right, outcome in triples:
        row = [0] * modulus
        row[token_index[left]] = (row[token_index[left]] + 1) % modulus
        row[token_index[right]] = (row[token_index[right]] + 1) % modulus
        row[token_index[outcome]] = (row[token_index[outcome]] - 1) % modulus
        matrix.append(row)

    rank = 0
    pivots = []
    for column in range(modulus):
        pivot = None
        for candidate in range(rank, len(matrix)):
            if matrix[candidate][column] % modulus:
                pivot = candidate
                break
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        inverse = pow(matrix[rank][column] % modulus, -1, modulus)
        matrix[rank] = [(value * inverse) % modulus for value in matrix[rank]]
        for row_index in range(len(matrix)):
            if row_index == rank or not matrix[row_index][column]:
                continue
            factor = matrix[row_index][column]
            matrix[row_index] = [
                (left - factor * right) % modulus
                for left, right in zip(matrix[row_index], matrix[rank])]
        pivots.append(column)
        rank += 1

    free = [column for column in range(modulus) if column not in pivots]
    if len(free) != 1:
        raise ValueError("TRAINING_FACTS_DO_NOT_IDENTIFY_ONE_CYCLIC_RULE_CODE")
    code_index = [0] * modulus
    code_index[free[0]] = 1
    for row_index in range(rank - 1, -1, -1):
        pivot = pivots[row_index]
        code_index[pivot] = (-sum(
            matrix[row_index][column] * code_index[column]
            for column in free)) % modulus
    if sorted(code_index) != list(range(modulus)):
        raise ValueError("RECOVERED_RULE_CODE_IS_NOT_BIJECTIVE")

    code = [None] * vocab_size
    inverse_code = [0] * modulus
    for token, index in token_index.items():
        code[token] = int(code_index[index])
        inverse_code[code_index[index]] = int(token)
    for left, right, outcome in triples:
        predicted = inverse_code[(code[left] + code[right]) % modulus]
        if predicted != outcome:
            raise ValueError("RECOVERED_RULE_CODE_FAILS_A_TRAINING_GENERATION_FACT")
    return code, inverse_code, rank


def _validation_inputs(gfg_prefix, cut_step, validation_count):
    candidates = gfg_prefix.objects(
        role="layer_input", name_contains="token_embedding.input.0",
        max_step=cut_step, materialized=True)
    candidates = [row for row in candidates
                  if "evaluation_validation" in row.get("semantic_key", "")
                  and (not validation_count or int(row["shape"][0]) == validation_count)]
    if not candidates:
        raise ValueError("NO_MATERIALIZED_VALIDATION_INPUT_IN_PREFIX")
    row = max(candidates, key=lambda item: int(item["optimizer_step"]))
    return _rows(_load(gfg_prefix, row), row["shape"])


def _quantile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _margin_history(gfg_prefix, cut_step, labels):
    objects = gfg_prefix.objects(
        role="validation_logits", max_step=cut_step, materialized=True)
    history = []
    for row in sorted(objects, key=lambda item: int(item["optimizer_step"])):
        step = int(row["optimizer_step"])
        if step < 100 or step % 100:
            continue
        logits = _rows(_load(gfg_prefix, row), row["shape"])
        if len(logits) != len(labels):
            continue
        margins = []
        for values, label in zip(logits, labels):
            correct = float(values[label])
            other = max(float(value) for index, value in enumerate(values)
                        if index != label)
            margins.append(correct - other)
        history.append({
            "optimizer_step": step,
            "rule_fraction": sum(value > 0.0 for value in margins) / len(margins),
            "mean_rule_margin": sum(margins) / len(margins),
            "q10_rule_margin": _quantile(margins, 0.10),
            "negative_margin_count": sum(value <= 0.0 for value in margins),
        })
    if not history:
        raise ValueError("NO_MATERIALIZED_VALIDATION_LOGITS_IN_PREFIX")
    return history


def _optimizer_beta2(gfg_prefix):
    rows = gfg_prefix.objects(role="optimizer_configuration", materialized=True)
    for row in rows:
        literal = row.get("literal_payload", {})
        groups = literal.get("param_groups", [])
        if groups and len(groups[0].get("betas", [])) == 2:
            return float(groups[0]["betas"][1])
    return 0.98


def _renewal_episodes(gfg_prefix, cut_step):
    try:
        rows = gfg_prefix.occurrences(
            occurrence_type="gradient_clip", min_step=0, max_step=cut_step)
    except TypeError:
        rows = gfg_prefix.occurrences("gradient_clip", 0, cut_step)
    clipped = []
    for row in rows:
        norm = float(row.get("payload", {}).get("total_norm", 0.0))
        transform = row.get("transform_reference", {})
        threshold = float(transform.get("max_norm", 1.0))
        if norm > threshold:
            clipped.append([int(row["optimizer_step"]), norm])
    clipped.sort()
    groups = []
    for item in clipped:
        if not groups or item[0] - groups[-1][-1][0] > 20:
            groups.append([])
        groups[-1].append(item)
    episodes = []
    for group in groups:
        peak = max(group, key=lambda item: item[1])
        if peak[0] < 200:
            continue
        episodes.append({
            "step_low": int(group[0][0]),
            "step_high": int(group[-1][0]),
            "peak_step": int(peak[0]),
            "peak_total_norm": float(peak[1]),
            "clipped_step_count": len(group),
        })
    return episodes


def _observed_transition(curve):
    earlier_low = False
    ordered = sorted(curve, key=lambda row: int(row["optimizer_step"]))
    for index, row in enumerate(ordered):
        if float(row["validation_accuracy"]) <= 0.30:
            earlier_low = True
        window = ordered[index:index + 3]
        if (earlier_low and len(window) == 3
                and all(float(item["train_accuracy"]) >= 0.99 for item in window)
                and all(float(item["validation_accuracy"]) >= 0.90 for item in window)):
            return int(row["optimizer_step"])
    return None


class CapabilityFormationMechanism:
    """Cyclic-rule-margin state with AdamW renewal/recovery dynamics."""

    @staticmethod
    def initialize(gfg_prefix):
        evaluations = sorted(gfg_prefix.evaluations(),
                             key=lambda row: int(row["optimizer_step"]))
        if not evaluations:
            raise ValueError("EMPTY_GFG_PREFIX")
        cut_step = int(evaluations[-1]["optimizer_step"])
        task = _task_descriptor(gfg_prefix)
        triples = _training_triples(gfg_prefix)
        operator_token = int(task.get("operator_token", max(
            max(triple) for triple in triples) + 1))
        vocab_size = int(task.get("vocab_size", operator_token + 1))
        code, inverse_code, rank = _recover_rule(
            triples, vocab_size, operator_token)
        validation_inputs = _validation_inputs(
            gfg_prefix, cut_step, int(task.get("validation_sample_count", 0)))
        modulus = len(inverse_code)
        labels = [inverse_code[(code[int(row[0])] + code[int(row[-1])]) % modulus]
                  for row in validation_inputs]
        margin_history = _margin_history(gfg_prefix, cut_step, labels)
        current = margin_history[-1]
        beta2 = _optimizer_beta2(gfg_prefix)
        episodes = _renewal_episodes(gfg_prefix, cut_step)
        observed_curve = [{
            "optimizer_step": int(row["optimizer_step"]),
            "train_accuracy": float(row["train_accuracy"]),
            "validation_accuracy": float(row["validation_accuracy"]),
        } for row in evaluations]

        return {
            "schema": "cyclic-rule-renewal-state-v1",
            "cut_step": cut_step,
            "evaluation_interval": 100,
            "rule_modulus": modulus,
            "operator_token": operator_token,
            "rule_code_by_token": code,
            "rule_code_rank": rank,
            "training_fact_count": len(triples),
            "validation_case_count": len(labels),
            "current_rule_fraction": float(current["rule_fraction"]),
            "current_mean_rule_margin": float(current["mean_rule_margin"]),
            "current_q10_rule_margin": float(current["q10_rule_margin"]),
            "negative_margin_count": int(current["negative_margin_count"]),
            "margin_history": margin_history,
            "observed_curve": observed_curve,
            "renewal_episodes": episodes,
            "adam_beta2": beta2,
            "renewal_growth": 1.0 + (1.0 - beta2) * 3.0,
            "next_interval_growth": 1.0 + (1.0 - beta2) * 4.0,
            "renewal_interval_cap": 440.0,
            "formation_drift_per_grid": 0.02,
            "formation_gain_per_renewal": 0.18,
            "formed_recovery_per_grid": 0.035,
            "observed_transition_step": _observed_transition(observed_curve),
        }

    @staticmethod
    def forecast(state):
        cut = int(state["cut_step"])
        interval = int(state["evaluation_interval"])
        observed_peaks = [int(row["peak_step"])
                          for row in state["renewal_episodes"]]
        if len(observed_peaks) >= 2:
            next_interval = min(
                float(state["renewal_interval_cap"]),
                max(300.0, (observed_peaks[-1] - observed_peaks[-2])
                    * float(state["next_interval_growth"])))
        else:
            next_interval = 320.0
        peak = float(observed_peaks[-1] if observed_peaks else cut - 80)
        future_peaks = []
        while peak < 10050:
            peak += next_interval
            future_peaks.append(peak)
            next_interval = min(
                float(state["renewal_interval_cap"]),
                next_interval * float(state["renewal_growth"]))

        event_by_grid = {}
        for event_peak in future_peaks:
            grid = int(math.ceil((event_peak + 20.0) / interval) * interval)
            if cut < grid <= 10000:
                event_by_grid[grid] = event_peak

        q = float(state["current_rule_fraction"])
        provisional_formation = state.get("observed_transition_step")
        predicted_rows = []
        state_rows = []
        instability = []
        last_peak = float(observed_peaks[-1] if observed_peaks else cut - 80)
        for step in range(((cut // interval) + 1) * interval, 10001, interval):
            event_peak = event_by_grid.get(step)
            if provisional_formation is None:
                q = min(0.995, q + float(state["formation_drift_per_grid"])
                        + (float(state["formation_gain_per_renewal"])
                           if event_peak is not None else 0.0))
                if q >= 0.90:
                    provisional_formation = step
                phase = "RULE_FORMING"
            else:
                if q < 0.90:
                    q = 0.99
                else:
                    q = min(0.995, q + float(state["formed_recovery_per_grid"]))
                phase = "FORMED"
                if event_peak is not None:
                    lag = max(0.0, step - event_peak)
                    amplitude = min(
                        0.70,
                        0.05 + 0.0001 * max(
                            0.0, event_peak - float(provisional_formation)))
                    q = max(0.05, 0.995 - amplitude * math.exp(-lag / 35.0))
                    phase = "RECOVERY"
            if event_peak is not None:
                last_peak = event_peak
                if provisional_formation is not None and event_peak > provisional_formation:
                    instability.append({
                        "step_low": max(cut + 1, int(math.floor(event_peak - 175.0))),
                        "step_high": min(10000, int(math.ceil(event_peak + 175.0))),
                    })
            age = max(0.0, step - last_peak)
            predicted_rows.append({
                "optimizer_step": step,
                "validation_accuracy": q,
            })
            state_rows.append({
                "optimizer_step": step,
                "phase": phase,
                "rule_fraction": q,
                "renewal_age": age,
                "adam_second_moment_reserve": float(state["adam_beta2"]) ** age,
            })

        combined = list(state["observed_curve"])
        combined.extend({
            "optimizer_step": row["optimizer_step"],
            "train_accuracy": 1.0,
            "validation_accuracy": row["validation_accuracy"],
        } for row in predicted_rows)
        transition = _observed_transition(combined)
        will_transition = transition is not None and transition <= 10000
        if will_transition:
            low_200 = max(0, transition - 100)
            high_200 = min(10000, transition + 100)
            low_500 = max(0, transition - 200)
            high_500 = min(10000, transition + 200)
        else:
            low_200 = high_200 = low_500 = high_500 = "NO_TRANSITION"

        observed_predictions = [{
            "optimizer_step": int(row["optimizer_step"]),
            "validation_accuracy": float(row["validation_accuracy"]),
        } for row in state["observed_curve"]
            if int(row["optimizer_step"]) % int(state["evaluation_interval"]) == 0]
        return {
            "will_transition": bool(will_transition),
            "transition_step_low_200": low_200,
            "transition_step_high_200": high_200,
            "transition_step_low_500": low_500,
            "transition_step_high_500": high_500,
            "predicted_validation_curve": observed_predictions + predicted_rows,
            "mechanism_state": {
                "schema": "cyclic-rule-renewal-evolution-v1",
                "initial": {
                    "optimizer_step": cut,
                    "rule_fraction": float(state["current_rule_fraction"]),
                    "mean_rule_margin": float(state["current_mean_rule_margin"]),
                    "q10_rule_margin": float(state["current_q10_rule_margin"]),
                    "renewal_episode_count": len(state["renewal_episodes"]),
                },
                "predicted_state_curve": state_rows,
            },
            "post_formation_stability": "TRANSIENT_DEGRADATION_RECOVERY",
            "predicted_instability_intervals": instability,
        }
