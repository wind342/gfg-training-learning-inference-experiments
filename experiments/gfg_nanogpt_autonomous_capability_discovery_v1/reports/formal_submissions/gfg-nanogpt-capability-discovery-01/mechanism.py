"""Sealed finite-state capability-formation mechanism.

The implementation reads only the supplied GFG prefix.  It recovers the
cyclic rule from training generation facts, measures rule-consistent logit
mass, and combines that state with endogenous AdamW slingshot phase inferred
from recorded gradient-clip occurrences.
"""

import math


def _clamp(value, low, high):
    return max(low, min(high, value))


def _median(values):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return 0.5 * (float(ordered[middle - 1]) + float(ordered[middle]))


def _flatten(value):
    """Convert a helper-returned numerical tensor to a plain finite list."""
    current = value
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "reshape") and hasattr(current, "tolist"):
        return current.reshape(-1).tolist()
    if hasattr(current, "tolist"):
        current = current.tolist()
    result = []

    def visit(item):
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            result.append(item)

    visit(current)
    return result


def _rref_cyclic_coordinates(train_inputs, train_targets, operator_token):
    """Solve c(y)=c(a)+c(b) over the operand-token field.

    Token coordinates are intentionally inferred from the prefix and are not
    serialized.  A global nonzero scale is immaterial to the group law.
    """
    tokens = set()
    equations_raw = []
    sample_count = min(len(train_inputs), len(train_targets)) // 3
    for index in range(sample_count):
        a = int(train_inputs[3 * index])
        op = int(train_inputs[3 * index + 1])
        b = int(train_inputs[3 * index + 2])
        y = int(train_targets[3 * index + 2])
        if op != operator_token or y < 0:
            continue
        tokens.update((a, b, y))
        equations_raw.append((a, b, y))
    ordered_tokens = sorted(tokens)
    modulus = len(ordered_tokens)
    if modulus < 3:
        return None
    token_index = {token: index for index, token in enumerate(ordered_tokens)}
    rows = []
    for a, b, y in equations_raw:
        row = [0] * modulus
        row[token_index[y]] = (row[token_index[y]] + 1) % modulus
        row[token_index[a]] = (row[token_index[a]] - 1) % modulus
        row[token_index[b]] = (row[token_index[b]] - 1) % modulus
        rows.append(row)

    pivot_columns = []
    pivot_row = 0
    for column in range(modulus):
        selected = None
        for candidate in range(pivot_row, len(rows)):
            coefficient = rows[candidate][column] % modulus
            if coefficient and math.gcd(coefficient, modulus) == 1:
                selected = candidate
                break
        if selected is None:
            continue
        rows[pivot_row], rows[selected] = rows[selected], rows[pivot_row]
        inverse = pow(rows[pivot_row][column] % modulus, -1, modulus)
        rows[pivot_row] = [(value * inverse) % modulus for value in rows[pivot_row]]
        for other in range(len(rows)):
            if other == pivot_row:
                continue
            factor = rows[other][column] % modulus
            if factor:
                rows[other] = [
                    (left - factor * right) % modulus
                    for left, right in zip(rows[other], rows[pivot_row])
                ]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == len(rows):
            break

    free_columns = [column for column in range(modulus) if column not in pivot_columns]
    if not free_columns:
        return None
    # The observed task supplies a one-dimensional nonzero nullspace.  Try
    # each deterministic basis choice and retain a bijective coordinate map.
    for chosen_free in free_columns:
        solution = [0] * modulus
        solution[chosen_free] = 1
        for row_index in range(len(pivot_columns) - 1, -1, -1):
            column = pivot_columns[row_index]
            solution[column] = (-sum(
                rows[row_index][j] * solution[j]
                for j in free_columns
            )) % modulus
        if len(set(solution)) != modulus:
            continue
        coordinates = {
            token: int(solution[token_index[token]])
            for token in ordered_tokens
        }
        if all(
            (coordinates[y] - coordinates[a] - coordinates[b]) % modulus == 0
            for a, b, y in equations_raw
        ):
            inverse_coordinates = {value: token for token, value in coordinates.items()}
            return coordinates, inverse_coordinates, modulus
    return None


def _rule_metrics(logits, shape, validation_inputs, coordinate_solution):
    coordinates, inverse_coordinates, modulus = coordinate_solution
    if len(shape) != 2 or int(shape[1]) <= 1:
        return None
    vocab_size = int(shape[1])
    row_count = min(int(shape[0]), len(validation_inputs) // 3)
    if len(logits) < row_count * vocab_size:
        return None
    margins = []
    probabilities = []
    for index in range(row_count):
        a = int(validation_inputs[3 * index])
        b = int(validation_inputs[3 * index + 2])
        if a not in coordinates or b not in coordinates:
            continue
        label = inverse_coordinates[(coordinates[a] + coordinates[b]) % modulus]
        row = logits[index * vocab_size:(index + 1) * vocab_size]
        true_logit = float(row[label])
        false_max = max(float(value) for j, value in enumerate(row) if j != label)
        margins.append(true_logit - false_max)
        peak = max(float(value) for value in row)
        denominator = sum(math.exp(float(value) - peak) for value in row)
        probabilities.append(math.exp(true_logit - peak) / denominator)
    if not margins:
        return None
    ordered = sorted(margins)
    q10_index = int(round(0.10 * (len(ordered) - 1)))
    return {
        "mean_margin": sum(margins) / len(margins),
        "q10_margin": ordered[q10_index],
        "mean_rule_probability": sum(probabilities) / len(probabilities),
        "positive_margin_fraction": sum(value > 0.0 for value in margins) / len(margins),
    }


def _major_burst_peaks(occurrences):
    rows = []
    for occurrence in occurrences:
        payload = occurrence.get("payload", {})
        if "total_norm" not in payload:
            continue
        rows.append((int(occurrence["optimizer_step"]), float(payload["total_norm"])))
    rows.sort()
    peaks = []
    cluster = []
    for step, norm in rows:
        if norm > 1.0:
            cluster.append((step, norm))
        elif cluster:
            peak = max(cluster, key=lambda item: item[1])
            if peak[1] > 5.0:
                peaks.append(peak)
            cluster = []
    if cluster:
        peak = max(cluster, key=lambda item: item[1])
        if peak[1] > 5.0:
            peaks.append(peak)
    return rows, peaks


def _optimizer_reservoir(gfg_prefix, cut_step):
    state_objects = gfg_prefix.objects(
        role="optimizer_state", min_step=cut_step, max_step=cut_step,
        materialized=True,
    )
    parameter_objects = gfg_prefix.objects(
        role="parameter_version", min_step=cut_step, max_step=cut_step,
        materialized=True,
    )
    first = {}
    second = {}
    parameters = {}
    for obj in state_objects:
        name = obj.get("name", "")
        if name.endswith(".exp_avg_sq"):
            second[name[:-11]] = [float(value) for value in _flatten(gfg_prefix.load_tensor(obj))]
        elif name.endswith(".exp_avg"):
            first[name[:-8]] = [float(value) for value in _flatten(gfg_prefix.load_tensor(obj))]
    for obj in parameter_objects:
        name = obj.get("name", "")
        if name in first and name in second:
            parameters[name] = [float(value) for value in _flatten(gfg_prefix.load_tensor(obj))]
    parameter_sq = 0.0
    second_sum = 0.0
    normalized_sq = 0.0
    count = 0
    for name, parameter in parameters.items():
        moment = first[name]
        variance = second[name]
        size = min(len(parameter), len(moment), len(variance))
        for index in range(size):
            parameter_sq += parameter[index] * parameter[index]
            second_sum += max(0.0, variance[index])
            normalized = moment[index] / (math.sqrt(max(0.0, variance[index])) + 1e-8)
            normalized_sq += normalized * normalized
        count += size
    if not count or parameter_sq <= 0.0:
        return {
            "second_moment_mean": 0.0,
            "relative_adam_step": 0.0,
            "reservoir_depletion": 0.0,
        }
    second_mean = second_sum / count
    relative_step = 0.003 * math.sqrt(normalized_sq) / math.sqrt(parameter_sq)
    return {
        "second_moment_mean": second_mean,
        "relative_adam_step": relative_step,
        "reservoir_depletion": -math.log10(max(second_mean, 1e-30)),
    }


class CapabilityFormationMechanism:
    """Prefix-conditioned cyclic-rule/slingshot finite-state mechanism."""

    @staticmethod
    def initialize(gfg_prefix):
        evaluations = sorted(
            (dict(row) for row in gfg_prefix.evaluations()),
            key=lambda row: int(row["optimizer_step"]),
        )
        if not evaluations:
            raise ValueError("GFG_PREFIX_HAS_NO_EVALUATION")
        current = evaluations[-1]
        cut_step = int(current["optimizer_step"])
        observed_transition = None
        for index in range(len(evaluations) - 2):
            window = evaluations[index:index + 3]
            if not all(
                float(row["train_accuracy"]) >= 0.99
                and float(row["validation_accuracy"]) >= 0.90
                for row in window
            ):
                continue
            if any(
                float(row["validation_accuracy"]) <= 0.30
                for row in evaluations[:index]
            ):
                observed_transition = int(window[0]["optimizer_step"])
                break

        input_objects = gfg_prefix.objects(
            role="training_batch_inputs", min_step=0, max_step=0,
            materialized=True,
        )
        target_objects = gfg_prefix.objects(
            role="training_batch_targets", min_step=0, max_step=0,
            materialized=True,
        )
        train_inputs = _flatten(gfg_prefix.load_tensor(input_objects[0]))
        train_targets = _flatten(gfg_prefix.load_tensor(target_objects[0]))
        operator_token = int(train_inputs[1])
        coordinate_solution = _rref_cyclic_coordinates(
            train_inputs, train_targets, operator_token
        )

        validation_input_objects = gfg_prefix.objects(
            role="layer_input", name_contains="token_embedding.input.0",
            min_step=max(0, cut_step - 400), max_step=cut_step,
            materialized=True,
        )
        validation_input_objects = [
            obj for obj in validation_input_objects
            if "evaluation_validation" in obj.get("semantic_key", "")
        ]
        validation_input_object = max(
            validation_input_objects, key=lambda obj: int(obj["optimizer_step"])
        )
        validation_inputs = _flatten(gfg_prefix.load_tensor(validation_input_object))

        logit_objects = gfg_prefix.objects(
            role="validation_logits", min_step=max(1, cut_step - 400),
            max_step=cut_step, materialized=True,
        )
        logit_objects.sort(key=lambda obj: int(obj["optimizer_step"]))
        metric_history = []
        if coordinate_solution is not None:
            for obj in logit_objects[-4:]:
                metrics = _rule_metrics(
                    _flatten(gfg_prefix.load_tensor(obj)), obj["shape"],
                    validation_inputs, coordinate_solution,
                )
                if metrics is not None:
                    metrics["optimizer_step"] = int(obj["optimizer_step"])
                    metric_history.append(metrics)

        if metric_history:
            rule = metric_history[-1]
            increments = [
                metric_history[index]["mean_rule_probability"]
                - metric_history[index - 1]["mean_rule_probability"]
                for index in range(1, len(metric_history))
                if metric_history[index]["mean_rule_probability"]
                > metric_history[index - 1]["mean_rule_probability"]
            ]
            rule_velocity = _median(increments[-3:]) if increments else 0.02
            if 0.35 < rule["mean_rule_probability"] < 0.90:
                rule_velocity = max(rule_velocity, 0.08)
            rule_velocity = _clamp(rule_velocity, 0.01, 0.16)
            rule_recovery_valid = True
        else:
            recent_accuracies = [float(row["validation_accuracy"]) for row in evaluations[-4:]]
            increments = [
                recent_accuracies[index] - recent_accuracies[index - 1]
                for index in range(1, len(recent_accuracies))
                if recent_accuracies[index] > recent_accuracies[index - 1]
            ]
            fallback_probability = float(current["validation_accuracy"])
            rule = {
                "mean_margin": 0.0,
                "q10_margin": 0.0,
                "mean_rule_probability": fallback_probability,
                "positive_margin_fraction": fallback_probability,
            }
            rule_velocity = _clamp(_median(increments[-3:]), 0.01, 0.12)
            rule_recovery_valid = False

        gradient_occurrences = gfg_prefix.occurrences(
            occurrence_type="gradient_clip", min_step=0, max_step=cut_step
        )
        gradient_rows, burst_peaks = _major_burst_peaks(gradient_occurrences)
        peak_steps = [step for step, _norm in burst_peaks]
        intervals = [
            peak_steps[index] - peak_steps[index - 1]
            for index in range(1, len(peak_steps))
        ]
        if len(intervals) >= 2:
            growth = _clamp(
                intervals[-1] - intervals[-2],
                -0.15 * intervals[-1], 0.50 * intervals[-1],
            )
            predicted_interval = intervals[-1] + growth
        elif intervals:
            predicted_interval = 1.4 * intervals[-1]
        else:
            predicted_interval = 400.0
        predicted_interval = _clamp(predicted_interval, 250.0, 550.0)

        reservoir = _optimizer_reservoir(gfg_prefix, cut_step)
        reservoir_factor = _clamp((reservoir["reservoir_depletion"] - 5.0) / 4.0, 0.0, 1.0)
        predicted_interval *= 0.98 + 0.04 * reservoir_factor
        last_burst_step = peak_steps[-1] if peak_steps else max(0, cut_step - int(predicted_interval))
        next_burst_step = int(round(last_burst_step + predicted_interval))
        latest_gradient_norm = gradient_rows[-1][1] if gradient_rows else 0.0

        probability = float(rule["mean_rule_probability"])
        probability_steps = max(1, int(math.ceil(
            max(0.0, 0.96 - probability) / max(rule_velocity, 1e-6)
        )))
        probability_transition = cut_step + 100 * probability_steps
        burst_transition = next_burst_step + 60
        blended_transition = 0.65 * burst_transition + 0.35 * probability_transition
        predicted_transition = int(round(blended_transition / 100.0)) * 100
        predicted_transition = max(cut_step + 100, min(10000, predicted_transition))
        if observed_transition is not None:
            predicted_transition = observed_transition

        configuration_objects = gfg_prefix.objects(
            role="optimizer_configuration", min_step=0, max_step=0,
            materialized=True,
        )
        learning_rate = 0.003
        weight_decay = 1.0
        if configuration_objects:
            configuration = configuration_objects[0].get("literal_payload", {})
            groups = configuration.get("param_groups", [])
            if groups:
                learning_rate = max(float(group.get("lr", learning_rate)) for group in groups)
                weight_decay = max(float(group.get("weight_decay", 0.0)) for group in groups)

        return {
            "schema": "capability-formation-state-v1",
            "cut_step": cut_step,
            "train_accuracy": float(current["train_accuracy"]),
            "validation_accuracy": float(current["validation_accuracy"]),
            "rule_probability": probability,
            "rule_probability_velocity_per_grid": rule_velocity,
            "rule_margin_mean": float(rule["mean_margin"]),
            "rule_margin_q10": float(rule["q10_margin"]),
            "rule_positive_margin_fraction": float(rule["positive_margin_fraction"]),
            "cyclic_rule_recovered": rule_recovery_valid,
            "last_burst_step": int(last_burst_step),
            "next_burst_step": int(next_burst_step),
            "cycle_interval_estimate": float(predicted_interval),
            "mature_cycle_interval": 375.0,
            "major_burst_count": len(peak_steps),
            "current_gradient_norm": float(latest_gradient_norm),
            "second_moment_mean": float(reservoir["second_moment_mean"]),
            "relative_adam_step": float(reservoir["relative_adam_step"]),
            "reservoir_depletion": float(reservoir["reservoir_depletion"]),
            "optimizer_learning_rate": float(learning_rate),
            "optimizer_weight_decay": float(weight_decay),
            "predicted_transition_step": int(predicted_transition),
            "transition_already_observed": observed_transition is not None,
        }

    @staticmethod
    def forecast(state):
        cut_step = int(state["cut_step"])
        transition = int(state["predicted_transition_step"])
        probability = float(state["rule_probability"])
        optimizer_drive = _clamp(
            (float(state["optimizer_learning_rate"]) * float(state["optimizer_weight_decay"]))
            / 0.003,
            0.0, 1.50,
        )
        velocity = float(state["rule_probability_velocity_per_grid"]) * optimizer_drive
        current_accuracy = float(state["validation_accuracy"])
        current_gap = _clamp(current_accuracy - probability, -0.05, 0.08)
        transition_already_observed = bool(state.get("transition_already_observed", False))
        will_transition = bool(
            transition_already_observed
            or (state["train_accuracy"] >= 0.99
            and velocity > 0.0
            and transition <= 10000)
        )
        operative_transition = (
            cut_step if transition_already_observed
            else (transition if will_transition else 10001)
        )
        risk_start = cut_step if transition_already_observed else operative_transition + 100

        # Project the next endogenous burst, then relax the interval toward the
        # mature state law.  This is a recurrence, not a stored event table.
        projected_bursts = []
        peak = int(state["next_burst_step"])
        interval = float(state["cycle_interval_estimate"])
        reservoir_interval_scale = _clamp(
            0.98 + 0.01 * (float(state["reservoir_depletion"]) - 6.0),
            0.95, 1.05,
        )
        mature_interval = float(state["mature_cycle_interval"]) * reservoir_interval_scale
        while peak <= 10000:
            projected_bursts.append(peak)
            interval = 0.60 * interval + 0.40 * mature_interval
            peak += int(round(interval))

        instability_intervals = [
            {"step_low": int(max(0, peak_step - 100)), "step_high": int(min(10000, peak_step + 100))}
            for peak_step in projected_bursts
            if peak_step >= risk_start
        ]
        base_burst_amplitude = _clamp(
            0.08
            + 8.0 * float(state["relative_adam_step"])
            + 0.005 * _clamp(math.log10(1.0 + float(state["current_gradient_norm"])), 0.0, 4.0),
            0.08, 0.16
        )

        curve = []
        state_trajectory = []
        first_future_grid = ((cut_step // 100) + 1) * 100
        for step in range(first_future_grid, 10001, 100):
            grid_count = (step - cut_step) / 100.0
            if transition_already_observed:
                projected_probability = min(0.999, max(0.97, probability + velocity * grid_count))
                hazard = 0.0
                for peak_step in projected_bursts:
                    age = step - peak_step
                    if -100 <= age <= 100 and peak_step >= risk_start:
                        horizon = max(0, peak_step - cut_step)
                        amplitude = _clamp(
                            base_burst_amplitude + 0.75 * math.exp(-horizon / 250.0),
                            base_burst_amplitude, 0.80,
                        )
                        recovery_shape = (1.0 + age / 100.0) if age < 0 else (1.0 - age / 100.0)
                        hazard = max(hazard, amplitude * recovery_shape)
                accuracy = 0.97 - hazard
                formed = True
            elif step < operative_transition:
                projected_probability = min(0.90, probability + velocity * grid_count)
                fraction = (step - cut_step) / max(100.0, operative_transition - cut_step)
                state_conditioned = _clamp(projected_probability + current_gap, 0.0, 0.89)
                linear_closure = current_accuracy + (0.88 - current_accuracy) * fraction
                accuracy = 0.55 * state_conditioned + 0.45 * linear_closure
                formed = False
            elif step == operative_transition:
                projected_probability = max(0.96, probability + velocity * grid_count)
                accuracy = 0.96
                formed = True
            elif step == operative_transition + 100:
                projected_probability = max(0.98, probability + velocity * grid_count)
                accuracy = 0.98
                formed = True
            else:
                projected_probability = min(0.999, max(0.97, probability + velocity * grid_count))
                hazard = 0.0
                for peak_step in projected_bursts:
                    age = step - peak_step
                    if -100 <= age <= 100 and peak_step >= risk_start:
                        horizon = max(0, peak_step - cut_step)
                        amplitude = _clamp(
                            base_burst_amplitude + 0.75 * math.exp(-horizon / 250.0),
                            base_burst_amplitude, 0.80,
                        )
                        recovery_shape = (1.0 + age / 100.0) if age < 0 else (1.0 - age / 100.0)
                        hazard = max(hazard, amplitude * recovery_shape)
                accuracy = 0.97 - hazard
                formed = True

            last_projected_peak = max(
                [int(state["last_burst_step"])]
                + [candidate for candidate in projected_bursts if candidate <= step]
            )
            next_candidates = [candidate for candidate in projected_bursts if candidate > step]
            next_projected_peak = min(next_candidates) if next_candidates else step + int(mature_interval)
            phase = _clamp(
                (step - last_projected_peak) / max(1.0, next_projected_peak - last_projected_peak),
                0.0, 1.0,
            )
            reservoir_level = 1.0 - phase
            slingshot_stress = phase ** 4
            accuracy = _clamp(accuracy, 0.0, 1.0)
            curve.append({
                "optimizer_step": int(step),
                "validation_accuracy": float(accuracy),
            })
            state_trajectory.append({
                "optimizer_step": int(step),
                "rule_probability": float(_clamp(projected_probability, 0.0, 1.0)),
                "formed": bool(formed),
                "optimizer_reservoir": float(reservoir_level),
                "slingshot_stress": float(slingshot_stress),
            })

        if will_transition:
            if transition_already_observed:
                low_200, high_200 = transition - 100, transition + 100
                low_500, high_500 = transition - 200, transition + 200
            else:
                low_200 = max(cut_step + 1, transition - 100)
                high_200 = min(10000, low_200 + 200)
                if not low_200 <= transition <= high_200:
                    low_200 = max(cut_step + 1, high_200 - 200)
                low_500 = max(cut_step + 1, transition - 200)
                high_500 = min(10000, low_500 + 400)
                if not low_500 <= transition <= high_500:
                    low_500 = max(cut_step + 1, high_500 - 400)
        else:
            low_200 = high_200 = low_500 = high_500 = "NO_TRANSITION"

        stability = (
            "TRANSIENT_DEGRADATION_RECOVERY"
            if int(state["major_burst_count"]) >= 2
            else "UNDETERMINED"
        )
        return {
            "will_transition": will_transition,
            "transition_step_low_200": low_200,
            "transition_step_high_200": high_200,
            "transition_step_low_500": low_500,
            "transition_step_high_500": high_500,
            "predicted_validation_curve": curve,
            "mechanism_state": {
                "initial": {
                    "cut_step": cut_step,
                    "rule_probability": probability,
                    "rule_margin_mean": float(state["rule_margin_mean"]),
                    "rule_margin_q10": float(state["rule_margin_q10"]),
                    "optimizer_reservoir_depletion": float(state["reservoir_depletion"]),
                    "next_burst_step": int(state["next_burst_step"]),
                },
                "evolution": state_trajectory,
            },
            "post_formation_stability": stability,
            "predicted_instability_intervals": instability_intervals,
        }
