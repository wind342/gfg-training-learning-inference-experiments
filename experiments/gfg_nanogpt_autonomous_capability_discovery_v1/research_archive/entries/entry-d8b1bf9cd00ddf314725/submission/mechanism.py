"""Sealed burst/equivariance capability-formation mechanism.

The implementation consumes only the participant GFG prefix supplied by the
runner.  It does not retain run identifiers, paths, timestamps, or task answer
tables in its canonical state.
"""

import math


def _as_rows(value):
    """Convert a loaded numerical object to ordinary nested Python lists."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _median(values):
    values = sorted(float(x) for x in values)
    if not values:
        return 0.0
    n = len(values)
    if n % 2:
        return values[n // 2]
    return 0.5 * (values[n // 2 - 1] + values[n // 2])


def _method_rows(prefix, name, **kwargs):
    method = getattr(prefix, name, None)
    if method is None:
        if isinstance(prefix, dict):
            value = prefix.get(name, [])
            return list(value() if callable(value) else value)
        return []
    try:
        return list(method(**kwargs))
    except TypeError:
        # Some replay wrappers enforce their prefix bound and omit max_step.
        kwargs.pop("max_step", None)
        try:
            return list(method(**kwargs))
        except TypeError:
            return list(method())


def _load(prefix, row):
    method = getattr(prefix, "load_tensor", None)
    if method is not None:
        return _as_rows(method(row))
    if isinstance(prefix, dict):
        tensors = prefix.get("tensors", {})
        object_id = row.get("object_id")
        if object_id in tensors:
            return _as_rows(tensors[object_id])
    raise ValueError("materialized tensor loader unavailable")


def _inverse_mod(value, modulus):
    """Return a modular inverse, or None when the pivot is not a unit."""
    a, b = int(value) % modulus, modulus
    old_s, s = 1, 0
    while b:
        q = a // b
        a, b = b, a - q * b
        old_s, s = s, old_s - q * s
    if a != 1:
        return None
    return old_s % modulus


def _infer_cyclic_order(inputs, targets, vocab_size, operator_token):
    """Infer token order from atomic observed equations z = x + y.

    Only a token-to-coordinate permutation is inferred.  A completed operation
    table is neither constructed nor serialized.
    """
    tokens = [t for t in range(int(vocab_size)) if t != int(operator_token)]
    modulus = len(tokens)
    column = {token: i for i, token in enumerate(tokens)}
    equations = []
    for source, target in zip(inputs, targets):
        source = list(source)
        target = list(target)
        if len(source) < 3 or len(target) < 1:
            continue
        a, b, z = int(source[0]), int(source[-1]), int(target[-1])
        if a not in column or b not in column or z not in column:
            continue
        row = [0] * modulus
        row[column[a]] = (row[column[a]] + 1) % modulus
        row[column[b]] = (row[column[b]] + 1) % modulus
        row[column[z]] = (row[column[z]] - 1) % modulus
        equations.append(row)
    if len(equations) < modulus:
        return tokens

    rank = 0
    pivots = []
    for col in range(modulus):
        pivot = None
        inverse = None
        for row_index in range(rank, len(equations)):
            inverse = _inverse_mod(equations[row_index][col], modulus)
            if inverse is not None:
                pivot = row_index
                break
        if pivot is None:
            continue
        equations[rank], equations[pivot] = equations[pivot], equations[rank]
        equations[rank] = [(x * inverse) % modulus for x in equations[rank]]
        for row_index in range(len(equations)):
            if row_index == rank:
                continue
            multiplier = equations[row_index][col]
            if multiplier:
                equations[row_index] = [
                    (x - multiplier * y) % modulus
                    for x, y in zip(equations[row_index], equations[rank])
                ]
        pivots.append(col)
        rank += 1

    free = [i for i in range(modulus) if i not in pivots]
    if len(free) != 1 or rank != modulus - 1:
        return tokens
    coordinates = [0] * modulus
    coordinates[free[0]] = 1
    for row_index in range(rank - 1, -1, -1):
        col = pivots[row_index]
        coordinates[col] = (-sum(
            equations[row_index][j] * coordinates[j]
            for j in range(col + 1, modulus)
        )) % modulus
    if len(set(coordinates)) != modulus:
        return tokens
    for row in equations:
        if sum(x * y for x, y in zip(row, coordinates)) % modulus:
            return tokens
    by_coordinate = [None] * modulus
    for token, coordinate in zip(tokens, coordinates):
        by_coordinate[coordinate] = token
    return by_coordinate if all(x is not None for x in by_coordinate) else tokens


def _translation_equivariance(weight, token_order):
    """Off-diagonal Gram variance explained by cyclic displacement."""
    weight = _as_rows(weight)
    if not weight or not token_order:
        return None
    rows = [list(weight[int(token)]) for token in token_order]
    width = len(rows[0])
    count = len(rows)
    if count < 3 or width < 1:
        return None
    means = [sum(rows[i][j] for i in range(count)) / count
             for j in range(width)]
    centered = [[rows[i][j] - means[j] for j in range(width)]
                for i in range(count)]
    gram = [[sum(centered[i][j] * centered[k][j] for j in range(width))
             for k in range(count)] for i in range(count)]
    off_mean = sum(gram[i][k] for i in range(count) for k in range(count)
                   if i != k) / (count * (count - 1))
    displacement_mean = [0.0]
    for displacement in range(1, count):
        displacement_mean.append(sum(
            gram[i][(i + displacement) % count] for i in range(count)
        ) / count)
    total = sum((gram[i][k] - off_mean) ** 2
                for i in range(count) for k in range(count) if i != k)
    if total <= 0.0:
        return 0.0
    residual = sum(
        (gram[i][k] - displacement_mean[(k - i) % count]) ** 2
        for i in range(count) for k in range(count) if i != k
    )
    return max(-1.0, min(1.0, 1.0 - residual / total))


def _burst_clusters(occurrences, cut_step):
    high = []
    for occurrence in occurrences:
        step = int(occurrence.get("optimizer_step", -1))
        payload = occurrence.get("payload", {}) or {}
        norm = float(payload.get("total_norm", 0.0))
        if 0 <= step <= cut_step and norm > 1.0:
            high.append((step, norm))
    high.sort()
    clusters = []
    for point in high:
        if not clusters or point[0] - clusters[-1][-1][0] > 20:
            clusters.append([])
        clusters[-1].append(point)
    peaks = []
    peak_norms = []
    for cluster in clusters:
        peak_step, peak_norm = max(cluster, key=lambda item: item[1])
        # The initialization transient is not a circuit-rewrite episode.
        if peak_step >= 100:
            peaks.append(int(peak_step))
            peak_norms.append(float(peak_norm))
    return peaks, peak_norms


def _predict_bursts(peaks, needed, cut_step):
    if len(peaks) >= 4:
        intervals = [peaks[i] - peaks[i - 1] for i in range(1, len(peaks))]
        changes = [intervals[i] - intervals[i - 1]
                   for i in range(1, len(intervals))]
        growth = _median(changes[-3:]) if changes else 20.0
        growth = max(-25.0, min(50.0, growth))
        gap = float(intervals[-1])
        cursor = float(peaks[-1])
        result = []
        for _ in range(max(1, needed)):
            gap = max(100.0, min(600.0, gap + growth))
            cursor += gap
            result.append(int(round(cursor)))
        return result
    # Prefix selection normally exposes four episodes.  This fallback remains
    # relative to the unseen prefix and contains no discovery-run identity.
    spacing = 350
    return [cut_step + spacing * (i + 1) for i in range(max(1, needed))]


class CapabilityFormationMechanism:
    """Finite-state, task-relative grokking mechanism."""

    @staticmethod
    def initialize(gfg_prefix):
        evaluations = _method_rows(gfg_prefix, "evaluations")
        evaluations = sorted(evaluations, key=lambda row: int(row["optimizer_step"]))
        if not evaluations:
            raise ValueError("complete GFG prefix has no capability evaluations")
        current = evaluations[-1]
        cut_step = int(current["optimizer_step"])

        task_rows = _method_rows(
            gfg_prefix, "objects", role="training_task", max_step=0)
        task = (task_rows[0].get("literal_payload", {}) if task_rows else {})
        vocab_size = int(task.get("vocab_size", 0))
        operator_token = int(task.get("operator_token", max(0, vocab_size - 1)))

        token_order = [t for t in range(vocab_size) if t != operator_token]
        try:
            input_rows = _method_rows(
                gfg_prefix, "objects", role="training_batch_inputs",
                materialized=True, max_step=0)
            target_rows = _method_rows(
                gfg_prefix, "objects", role="training_batch_targets",
                materialized=True, max_step=0)
            if input_rows and target_rows:
                inputs = _load(gfg_prefix, input_rows[0])
                targets = _load(gfg_prefix, target_rows[0])
                token_order = _infer_cyclic_order(
                    inputs, targets, vocab_size, operator_token)
        except (KeyError, TypeError, ValueError, IndexError):
            pass

        evaluation_steps = {int(row["optimizer_step"]) for row in evaluations}
        score_history = []
        try:
            parameter_rows = _method_rows(
                gfg_prefix, "objects", role="parameter_version",
                name_contains="transformer.wte.weight", materialized=True,
                max_step=cut_step)
            parameter_rows = [row for row in parameter_rows
                              if row.get("name") == "transformer.wte.weight"
                              and int(row.get("optimizer_step", -1)) in evaluation_steps]
            parameter_rows.sort(key=lambda row: int(row["optimizer_step"]))
            for row in parameter_rows[-6:]:
                score = _translation_equivariance(_load(gfg_prefix, row), token_order)
                if score is not None:
                    score_history.append({
                        "optimizer_step": int(row["optimizer_step"]),
                        "translation_equivariance": round(float(score), 6),
                    })
        except (KeyError, TypeError, ValueError, IndexError):
            score_history = []

        gradient_occurrences = _method_rows(
            gfg_prefix, "occurrences", occurrence_type="gradient_clip",
            max_step=cut_step)
        peaks, peak_norms = _burst_clusters(gradient_occurrences, cut_step)
        current_score = (score_history[-1]["translation_equivariance"]
                         if score_history else None)
        current_validation = float(current["validation_accuracy"])
        if current_score is not None:
            bursts_needed = 2 if current_score < 0.80 else 1
        else:
            bursts_needed = 2 if current_validation < 0.45 else 1
        predicted_centers = _predict_bursts(peaks, bursts_needed, cut_step)
        predicted_evaluations = [
            int(math.ceil(center / 100.0) * 100) for center in predicted_centers
        ]
        transition_step = predicted_evaluations[-1]

        return {
            "schema": "burst-equivariance-state-v1",
            "cut_step": cut_step,
            "evaluation_interval": 100,
            "current_train_accuracy": float(current["train_accuracy"]),
            "current_validation_accuracy": current_validation,
            "recent_validation_accuracy": [
                {"optimizer_step": int(row["optimizer_step"]),
                 "validation_accuracy": float(row["validation_accuracy"])}
                for row in evaluations[-4:]
            ],
            "gradient_burst_peaks": peaks,
            "gradient_burst_peak_norms": [round(x, 6) for x in peak_norms],
            "translation_equivariance_history": score_history,
            "current_translation_equivariance": current_score,
            "bursts_needed": int(bursts_needed),
            "predicted_burst_centers": predicted_centers,
            "predicted_burst_evaluation_steps": predicted_evaluations,
            "predicted_transition_step": int(transition_step),
            "phase": "SYMMETRY_NUCLEATION",
        }

    @staticmethod
    def forecast(state):
        cut = int(state["cut_step"])
        transition = int(state["predicted_transition_step"])
        burst_evaluations = list(state["predicted_burst_evaluation_steps"])
        first_rewrite = int(burst_evaluations[0])
        current = float(state["current_validation_accuracy"])
        rule_level = 0.97
        gap = max(0.0, rule_level - current)
        late_instability = transition + 6100
        curve = []
        evolution = []
        first_future_grid = ((cut // 100) + 1) * 100
        for step in range(first_future_grid, 10001, 100):
            if step < first_rewrite:
                accuracy = current
                phase = "SYMMETRY_NUCLEATION"
                order = state.get("current_translation_equivariance")
            elif step < transition:
                if step == first_rewrite:
                    accuracy = current + 0.40 * gap
                    order = 0.83
                else:
                    accuracy = current + 0.72 * gap
                    order = 0.85
                phase = "RULE_CIRCUIT_CONSOLIDATING"
            elif step < late_instability:
                accuracy = rule_level
                phase = "RULE_GENERALIZATION"
                order = 0.89
            else:
                accuracy = 0.60
                phase = "REGULARIZATION_LIMIT_CYCLE"
                order = 0.78
            accuracy = max(0.0, min(1.0, float(accuracy)))
            curve.append({
                "optimizer_step": step,
                "validation_accuracy": round(accuracy, 6),
            })
            evolution.append({
                "optimizer_step": step,
                "phase": phase,
                "translation_equivariance": (
                    None if order is None else round(float(order), 6)),
                "burst_count_relative_to_cut": sum(
                    center <= step for center in state["predicted_burst_centers"]),
            })

        will_transition = transition <= 10000
        return {
            "will_transition": bool(will_transition),
            "transition_step_low_200": max(first_future_grid, transition - 100),
            "transition_step_high_200": min(10000, transition + 100),
            "transition_step_low_500": max(first_future_grid, transition - 200),
            "transition_step_high_500": min(10000, transition + 200),
            "predicted_validation_curve": curve,
            "mechanism_state": {
                "schema": "burst-equivariance-forecast-state-v1",
                "initial_phase": state["phase"],
                "cut_step": cut,
                "predicted_transition_step": transition,
                "predicted_evolution": evolution,
            },
        }
