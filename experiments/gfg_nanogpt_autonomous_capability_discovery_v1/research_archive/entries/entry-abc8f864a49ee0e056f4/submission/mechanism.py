"""Finite cyclic-composition mechanism for an opaque-token training GFG."""


def _as_rows(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return [value]
    return value


def _call(prefix, name, **kwargs):
    member = getattr(prefix, name, None)
    if callable(member):
        return member(**kwargs)
    if isinstance(prefix, dict):
        rows = prefix.get(name, [])
        if callable(rows):
            return rows(**kwargs)
        if name == "objects":
            result = []
            for row in rows:
                if kwargs.get("materialized") is not None:
                    if bool(row.get("materialized")) != kwargs["materialized"]:
                        continue
                step = int(row.get("optimizer_step", 0))
                if kwargs.get("min_step") is not None and step < kwargs["min_step"]:
                    continue
                if kwargs.get("max_step") is not None and step > kwargs["max_step"]:
                    continue
                result.append(row)
            return result
        return list(rows)
    raise TypeError("gfg_prefix does not expose " + name)


def _load(prefix, row):
    loader = getattr(prefix, "load_tensor", None)
    if callable(loader):
        return _as_rows(loader(row))
    for key in ("value", "tensor", "data"):
        if key in row:
            return _as_rows(row[key])
    raise TypeError("materialized object has no tensor loader")


def _mode_interval(evaluations):
    counts = {}
    prior = None
    for row in evaluations:
        step = int(row["optimizer_step"])
        if prior is not None:
            delta = step - prior
            if delta > 1:
                counts[delta] = counts.get(delta, 0) + 1
        prior = step
    if not counts:
        return 100
    return min(counts, key=lambda value: (-counts[value], value))


def _observed_transition(evaluations):
    earlier_low = False
    for index, row in enumerate(evaluations):
        if float(row["validation_accuracy"]) <= 0.3:
            earlier_low = True
        if not earlier_low or index + 2 >= len(evaluations):
            continue
        window = evaluations[index:index + 3]
        if all(float(item["train_accuracy"]) >= 0.99 and
               float(item["validation_accuracy"]) >= 0.9
               for item in window):
            return int(row["optimizer_step"])
    return None


def _is_prime(value):
    if value < 2:
        return False
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 1
    return True


def _infer_coordinates(pairs, answers):
    symbols = sorted(set(
        [token for pair in pairs for token in pair] + list(answers)))
    modulus = len(symbols)
    if not _is_prime(modulus) or modulus < 3:
        return None
    index = {token: position for position, token in enumerate(symbols)}
    matrix = []
    for pair, answer in zip(pairs, answers):
        row = [0] * modulus
        for token, coefficient in (
                (pair[0], 1), (pair[1], 1), (answer, -1)):
            if token not in index:
                return None
            position = index[token]
            row[position] = (row[position] + coefficient) % modulus
        matrix.append(row)
    pivot_columns = []
    pivot_row = 0
    for column in range(modulus):
        selected = None
        for candidate in range(pivot_row, len(matrix)):
            if matrix[candidate][column] % modulus:
                selected = candidate
                break
        if selected is None:
            continue
        matrix[pivot_row], matrix[selected] = (
            matrix[selected], matrix[pivot_row])
        inverse = pow(matrix[pivot_row][column], -1, modulus)
        matrix[pivot_row] = [
            value * inverse % modulus for value in matrix[pivot_row]]
        for other in range(len(matrix)):
            if other == pivot_row or not matrix[other][column] % modulus:
                continue
            factor = matrix[other][column]
            matrix[other] = [
                (left - factor * right) % modulus
                for left, right in zip(matrix[other], matrix[pivot_row])]
        pivot_columns.append(column)
        pivot_row += 1
    free = [
        column for column in range(modulus)
        if column not in pivot_columns]
    if len(free) != 1 or len(pivot_columns) != modulus - 1:
        return None
    coordinate = [0] * modulus
    coordinate[free[0]] = 1
    for row_number, column in enumerate(pivot_columns):
        coordinate[column] = (-matrix[row_number][free[0]]) % modulus
    if len(set(coordinate)) != modulus:
        return None
    return {
        "modulus": modulus,
        "phi": {
            token: coordinate[index[token]] for token in symbols},
        "symbols": symbols,
        "constraint_rank": len(pivot_columns),
    }


def _select_eval_input(rows, split, count):
    marker = "stage:evaluation_" + split
    for row in rows:
        if (row.get("role") == "layer_input" and
                row.get("name") == "token_embedding.input.0" and
                marker in row.get("semantic_key", "")):
            return row
    for row in rows:
        shape = row.get("shape", [])
        if (row.get("role") == "layer_input" and
                row.get("name") == "token_embedding.input.0" and
                shape and int(shape[0]) == count):
            return row
    return None


def _arg_positions(input_rows):
    width = len(input_rows[0])
    diversity = []
    for column in range(width):
        diversity.append(len(set(row[column] for row in input_rows)))
    operator_position = min(
        range(width), key=lambda column: (diversity[column], column))
    arguments = [
        column for column in range(width) if column != operator_position]
    if len(arguments) < 2:
        return None
    return arguments[0], arguments[-1]


def _circulant_metrics(logit_rows, pairs, coordinate):
    symbols = coordinate["symbols"]
    phi = coordinate["phi"]
    modulus = coordinate["modulus"]
    total_sum = 0.0
    total_square = 0.0
    residual_sum = [0.0] * modulus
    residual_count = [0] * modulus
    margins = []
    predictions = []
    inverse = {value: token for token, value in phi.items()}
    for logits, pair in zip(logit_rows, pairs):
        prediction = max(
            range(len(logits)), key=lambda column: (logits[column], -column))
        predictions.append(prediction)
        expected = inverse[(phi[pair[0]] + phi[pair[1]]) % modulus]
        alternatives = []
        for token in symbols:
            value = float(logits[token])
            total_sum += value
            total_square += value * value
            residual = (
                phi[token] - phi[pair[0]] - phi[pair[1]]) % modulus
            residual_sum[residual] += value
            residual_count[residual] += 1
            if token != expected:
                alternatives.append(value)
        margins.append(float(logits[expected]) - max(alternatives))
    count = len(logit_rows) * modulus
    mean = total_sum / count
    total_variance = total_square - count * mean * mean
    between = 0.0
    for residual in range(modulus):
        group_mean = residual_sum[residual] / residual_count[residual]
        between += residual_count[residual] * (group_mean - mean) ** 2
    eta = between / total_variance if total_variance > 0.0 else 0.0
    scale = (total_variance / count) ** 0.5
    margin_z = (
        sum(margins) / len(margins) / scale if scale > 0.0 else 0.0)
    return eta, margin_z, predictions


def _axiom_metrics(predictions, pairs, symbols):
    table = {pair: answer for pair, answer in zip(pairs, predictions)}
    if len(table) != len(symbols) * len(symbols):
        return 0.0, 0.0, 0.0
    symbol_set = set(symbols)
    commutative = sum(
        table[(left, right)] == table[(right, left)]
        for left, right in pairs) / len(pairs)
    latin_total = 0
    for left in symbols:
        latin_total += len(set(table[(left, right)] for right in symbols))
    for right in symbols:
        latin_total += len(set(table[(left, right)] for left in symbols))
    latin = latin_total / (2.0 * len(symbols) * len(symbols))
    associative_hits = 0
    associative_total = len(symbols) ** 3
    for left in symbols:
        for middle in symbols:
            left_middle = table[(left, middle)]
            for right in symbols:
                middle_right = table[(middle, right)]
                if (left_middle in symbol_set and
                        middle_right in symbol_set and
                        table[(left_middle, right)] ==
                        table[(left, middle_right)]):
                    associative_hits += 1
    return (
        commutative,
        latin,
        associative_hits / associative_total,
    )


def _structure_from_prefix(prefix, objects, cut_step):
    current = [
        row for row in objects
        if int(row.get("optimizer_step", -1)) == cut_step]
    batch_inputs = next(
        row for row in objects
        if row.get("role") == "training_batch_inputs")
    batch_targets = next(
        row for row in objects
        if row.get("role") == "training_batch_targets")
    train_logits_object = next(
        row for row in current if row.get("role") == "train_logits")
    validation_logits_object = next(
        row for row in current if row.get("role") == "validation_logits")
    input_rows = _as_rows(_load(prefix, batch_inputs))
    target_rows = _as_rows(_load(prefix, batch_targets))
    train_logits = _as_rows(_load(prefix, train_logits_object))
    validation_logits = _as_rows(_load(prefix, validation_logits_object))
    train_input_object = _select_eval_input(
        current, "train", len(train_logits))
    validation_input_object = _select_eval_input(
        current, "validation", len(validation_logits))
    if train_input_object is None or validation_input_object is None:
        raise ValueError("evaluation token inputs unavailable")
    eval_train_inputs = _as_rows(_load(prefix, train_input_object))
    eval_validation_inputs = _as_rows(_load(
        prefix, validation_input_object))
    positions = _arg_positions(input_rows)
    if positions is None:
        raise ValueError("operator position unavailable")
    left_position, right_position = positions
    constraint_pairs = [
        (row[left_position], row[right_position]) for row in input_rows]
    answers = []
    for row in target_rows:
        candidates = [value for value in row if int(value) >= 0]
        if not candidates:
            raise ValueError("training target unavailable")
        answers.append(candidates[-1])
    coordinate = _infer_coordinates(constraint_pairs, answers)
    if coordinate is None:
        raise ValueError("cyclic coordinate constraints not identifiable")
    all_inputs = eval_train_inputs + eval_validation_inputs
    pairs = [
        (row[left_position], row[right_position]) for row in all_inputs]
    if any(
            pair[0] not in coordinate["phi"] or
            pair[1] not in coordinate["phi"] for pair in pairs):
        raise ValueError("evaluation symbols outside inferred coordinate")
    logits = train_logits + validation_logits
    eta, margin_z, predictions = _circulant_metrics(
        logits, pairs, coordinate)
    commutative, latin, associative = _axiom_metrics(
        predictions, pairs, coordinate["symbols"])
    return {
        "cyclic_constraint_rank": coordinate["constraint_rank"],
        "cyclic_symbol_count": coordinate["modulus"],
        "rule_logit_circulant_eta": eta,
        "rule_margin_z": margin_z,
        "predicted_table_commutativity": commutative,
        "predicted_table_latin_fraction": latin,
        "predicted_table_associativity": associative,
    }


class CapabilityFormationMechanism:
    @staticmethod
    def initialize(gfg_prefix):
        evaluations = sorted(
            _call(gfg_prefix, "evaluations"),
            key=lambda row: int(row["optimizer_step"]))
        if not evaluations:
            raise ValueError("complete prefix has no evaluation")
        cut_step = int(evaluations[-1]["optimizer_step"])
        interval = _mode_interval(evaluations)
        objects = _call(
            gfg_prefix, "objects", min_step=0, max_step=cut_step,
            materialized=True)
        structure = {
            "cyclic_constraint_rank": 0,
            "cyclic_symbol_count": 0,
            "rule_logit_circulant_eta": 0.0,
            "rule_margin_z": 0.0,
            "predicted_table_commutativity": 0.0,
            "predicted_table_latin_fraction": 0.0,
            "predicted_table_associativity": 0.0,
        }
        try:
            structure.update(_structure_from_prefix(
                gfg_prefix, objects, cut_step))
        except (KeyError, TypeError, ValueError, StopIteration, IndexError):
            pass
        last = evaluations[-1]
        transition = _observed_transition(evaluations)
        train_accuracy = float(last["train_accuracy"])
        validation_accuracy = float(last["validation_accuracy"])
        eta = float(structure["rule_logit_circulant_eta"])
        if transition is not None:
            phase = "RULE_LOCKED"
        elif train_accuracy >= 0.99 and validation_accuracy <= 0.3:
            phase = "MEMORIZATION"
        elif validation_accuracy > 0.3 or eta >= 0.55:
            phase = "RULE_ASSEMBLY"
        else:
            phase = "FITTING"
        progress = min(1.0, max(0.0,
            0.55 * eta / 0.85 +
            0.25 * float(structure["predicted_table_associativity"]) / 0.94 +
            0.20 * validation_accuracy / 0.90))
        state = {
            "schema": "cyclic-composition-state-v1",
            "phase": phase,
            "cut_step": cut_step,
            "evaluation_interval": interval,
            "train_accuracy": round(train_accuracy, 8),
            "validation_accuracy": round(validation_accuracy, 8),
            "formation_progress": round(progress, 8),
            "transition_already_observed": transition is not None,
            "observed_transition_step": transition,
        }
        for key, value in structure.items():
            if isinstance(value, float):
                state[key] = round(value, 8)
            else:
                state[key] = value
        return state

    @staticmethod
    def forecast(state):
        cut_step = int(state["cut_step"])
        interval = int(state.get("evaluation_interval", 100))
        endpoint = 10000
        current_validation = float(state["validation_accuracy"])
        gap = max(0.0, 0.92 - current_validation)
        remaining_points = max(2, min(10, int(gap / 0.10 + 0.999999)))
        eta = float(state.get("rule_logit_circulant_eta", 0.0))
        if eta >= 0.80:
            remaining_points = max(2, remaining_points - 1)
        elif eta < 0.55:
            remaining_points = min(10, remaining_points + 1)
        center = cut_step + remaining_points * interval
        will_transition = center + 2 * interval <= endpoint
        if will_transition:
            transition_low = center - interval
            transition_high = center + interval
        else:
            transition_low = "NO_TRANSITION"
            transition_high = "NO_TRANSITION"
        curve = []
        trajectory = []
        current_progress = float(state.get("formation_progress", 0.0))
        for step in range(cut_step + interval, endpoint + 1, interval):
            if step <= center:
                fraction = (
                    (step - cut_step) / float(max(interval, center - cut_step)))
                predicted = current_validation + (
                    0.93 - current_validation) * fraction * fraction
                progress = current_progress + (
                    1.0 - current_progress) * fraction
            else:
                distance = (step - center) / float(2 * interval)
                predicted = 0.93 + 0.065 * (1.0 - 0.5 ** distance)
                progress = 1.0
            predicted = min(0.995, max(0.0, predicted))
            if step < center:
                phase = "RULE_ASSEMBLY"
            elif step < center + 2 * interval:
                phase = "RULE_EXPRESSED"
            else:
                phase = "RULE_LOCKED"
            curve.append({
                "optimizer_step": step,
                "validation_accuracy": round(predicted, 8),
            })
            trajectory.append({
                "optimizer_step": step,
                "phase": phase,
                "formation_progress": round(min(1.0, progress), 8),
            })
        return {
            "schema": "cyclic-composition-forecast-v1",
            "forecast_cut_step": cut_step,
            "will_transition": will_transition,
            "transition_step_low": transition_low,
            "transition_step_high": transition_high,
            "predicted_validation_curve": curve,
            "mechanism_state": {
                "current": state,
                "trajectory": trajectory,
            },
        }
