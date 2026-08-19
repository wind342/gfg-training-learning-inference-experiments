"""Sealed prefix-only capability-formation mechanism.

The executable deliberately reads the GFG only in initialize().  forecast()
is a closed finite-state update law and cannot inspect a future graph.
"""

import math


class CapabilityFormationMechanism:
    SCHEMA = "gfg-capability-state-v1"
    GRID = 100
    END_STEP = 10000
    NOMINAL_LR = 0.003
    NOMINAL_WEIGHT_DECAY = 1.0
    ANCHOR_TO_TRANSITION = 600
    STRESS_PERIOD = 400
    FIRST_STRESS_OFFSET = 300

    @staticmethod
    def _get(row, key, default=None):
        try:
            if isinstance(row, dict):
                return row.get(key, default)
            return row[key]
        except Exception:
            return getattr(row, key, default)

    @staticmethod
    def _as_rows(gfg_prefix):
        rows = []
        try:
            if hasattr(gfg_prefix, "evaluations"):
                source = gfg_prefix.evaluations()
            elif isinstance(gfg_prefix, dict):
                source = gfg_prefix.get("evaluations", [])
            else:
                source = []
            for row in source:
                step = int(CapabilityFormationMechanism._get(
                    row, "optimizer_step"))
                rows.append({
                    "optimizer_step": step,
                    "train_accuracy": float(CapabilityFormationMechanism._get(
                        row, "train_accuracy")),
                    "validation_accuracy": float(
                        CapabilityFormationMechanism._get(
                            row, "validation_accuracy")),
                    "loss": float(CapabilityFormationMechanism._get(
                        row, "loss")),
                })
        except Exception:
            rows = []
        rows.sort(key=lambda item: item["optimizer_step"])
        return rows

    @staticmethod
    def _objects(gfg_prefix, **kwargs):
        if not hasattr(gfg_prefix, "objects"):
            return []
        try:
            return list(gfg_prefix.objects(**kwargs))
        except Exception:
            return []

    @staticmethod
    def _fact_blocks(gfg_prefix, **kwargs):
        if not hasattr(gfg_prefix, "fact_blocks"):
            return []
        try:
            return list(gfg_prefix.fact_blocks(**kwargs))
        except Exception:
            return []

    @staticmethod
    def _flat(tensor):
        try:
            return [float(value) for value in tensor.reshape(-1)]
        except Exception:
            result = []

            def visit(value):
                if isinstance(value, (list, tuple)):
                    for child in value:
                        visit(child)
                else:
                    result.append(float(value))

            visit(tensor)
            return result

    @staticmethod
    def _load(gfg_prefix, row):
        if not hasattr(gfg_prefix, "load_tensor"):
            return []
        return CapabilityFormationMechanism._flat(
            gfg_prefix.load_tensor(row))

    @staticmethod
    def _transition(rows):
        earlier_low = False
        for index, row in enumerate(rows):
            if row["validation_accuracy"] <= 0.3:
                earlier_low = True
            if not earlier_low or index + 2 >= len(rows):
                continue
            window = rows[index:index + 3]
            if (window[1]["optimizer_step"] - window[0]["optimizer_step"]
                    != 100 or
                    window[2]["optimizer_step"] - window[1]["optimizer_step"]
                    != 100):
                continue
            if all(item["train_accuracy"] >= 0.99 and
                   item["validation_accuracy"] >= 0.9
                   for item in window):
                return row["optimizer_step"]
        return None

    @staticmethod
    def _optimizer_configuration(gfg_prefix):
        lr = CapabilityFormationMechanism.NOMINAL_LR
        weight_decay = CapabilityFormationMechanism.NOMINAL_WEIGHT_DECAY
        rows = CapabilityFormationMechanism._objects(
            gfg_prefix, role="optimizer_configuration", min_step=0,
            max_step=0, materialized=True)
        for row in rows:
            literal = CapabilityFormationMechanism._get(
                row, "literal_payload", {}) or {}
            groups = literal.get("param_groups", []) if isinstance(
                literal, dict) else []
            decayed = [group for group in groups
                       if float(group.get("weight_decay", 0.0)) > 0.0]
            if decayed:
                lr = float(decayed[0].get("lr", lr))
                weight_decay = float(decayed[0].get(
                    "weight_decay", weight_decay))
                break
        return lr, weight_decay

    @staticmethod
    def _latent_addition_map(inputs, targets, width):
        """Infer token -> Z_p coordinates solely from training facts."""
        labels = [int(targets[index + width - 1])
                  for index in range(0, len(targets), width)
                  if int(targets[index + width - 1]) >= 0]
        if not labels:
            return None
        prime = max(labels) + 1
        if prime < 3:
            return None
        equations = []
        for index in range(0, min(len(inputs), len(targets)), width):
            left = int(inputs[index])
            right = int(inputs[index + width - 1])
            outcome = int(targets[index + width - 1])
            if not (0 <= left < prime and 0 <= right < prime and
                    0 <= outcome < prime):
                continue
            row = [0] * prime
            row[left] = (row[left] + 1) % prime
            row[right] = (row[right] + 1) % prime
            row[outcome] = (row[outcome] - 1) % prime
            equations.append(row)
        rank = 0
        pivots = []
        for column in range(prime):
            pivot = None
            for candidate in range(rank, len(equations)):
                if equations[candidate][column] % prime:
                    pivot = candidate
                    break
            if pivot is None:
                continue
            equations[rank], equations[pivot] = (
                equations[pivot], equations[rank])
            try:
                inverse = pow(equations[rank][column], -1, prime)
            except Exception:
                return None
            equations[rank] = [
                (value * inverse) % prime for value in equations[rank]]
            for candidate in range(len(equations)):
                if candidate == rank or not equations[candidate][column]:
                    continue
                factor = equations[candidate][column]
                equations[candidate] = [
                    (left - factor * right) % prime
                    for left, right in zip(equations[candidate],
                                           equations[rank])]
            pivots.append(column)
            rank += 1
        free = [column for column in range(prime)
                if column not in pivots]
        if len(free) != 1 or rank != prime - 1:
            return None
        coordinates = [0] * prime
        coordinates[free[0]] = 1
        for equation_index, column in reversed(list(enumerate(pivots))):
            coordinates[column] = (-sum(
                equations[equation_index][free_column] *
                coordinates[free_column] for free_column in free)) % prime
        if len(set(coordinates)) != prime:
            return None
        if not all((coordinates[int(inputs[index])] +
                    coordinates[int(inputs[index + width - 1])] -
                    coordinates[int(targets[index + width - 1])]) % prime == 0
                   for index in range(0, min(len(inputs), len(targets)), width)
                   if int(targets[index + width - 1]) >= 0):
            return None
        return coordinates

    @staticmethod
    def _tensor_features(gfg_prefix, step):
        result = {
            "task_structure_rank": None,
            "task_structure_size": None,
            "algebraic_spectral_concentration": None,
            "rule_margin_q10": None,
            "gradient_norm": None,
            "clipped_gradient_norm": None,
            "layernorm_scale": None,
            "adam_first_moment_norm": None,
            "adam_sqrt_second_moment_norm": None,
        }
        try:
            batch_inputs = CapabilityFormationMechanism._objects(
                gfg_prefix, role="training_batch_inputs", min_step=0,
                max_step=0, materialized=True)
            batch_targets = CapabilityFormationMechanism._objects(
                gfg_prefix, role="training_batch_targets", min_step=0,
                max_step=0, materialized=True)
            if not batch_inputs or not batch_targets:
                return result
            inputs = CapabilityFormationMechanism._load(
                gfg_prefix, batch_inputs[0])
            targets = CapabilityFormationMechanism._load(
                gfg_prefix, batch_targets[0])
            shape = CapabilityFormationMechanism._get(
                batch_inputs[0], "shape", [0, 3])
            width = int(shape[-1])
            coordinates = CapabilityFormationMechanism._latent_addition_map(
                inputs, targets, width)
            if coordinates is None:
                return result
            prime = len(coordinates)
            result["task_structure_rank"] = prime - 1
            result["task_structure_size"] = prime

            parameters = CapabilityFormationMechanism._objects(
                gfg_prefix, role="parameter_version", min_step=step,
                max_step=step, materialized=True)
            validation_logits = CapabilityFormationMechanism._objects(
                gfg_prefix, role="validation_logits", min_step=step,
                max_step=step, materialized=True)
            evaluated_sources = set()
            if validation_logits:
                logits_id = CapabilityFormationMechanism._get(
                    validation_logits[-1], "object_id")
                for fact_block in CapabilityFormationMechanism._fact_blocks(
                        gfg_prefix, min_step=step, max_step=step):
                    outcomes = CapabilityFormationMechanism._get(
                        fact_block, "outcomes", []) or []
                    if not any(CapabilityFormationMechanism._get(
                            outcome, "object_id") == logits_id
                               for outcome in outcomes):
                        continue
                    for source in CapabilityFormationMechanism._get(
                            fact_block, "sources", []) or []:
                        if CapabilityFormationMechanism._get(
                                source, "relation_role") == (
                                    "evaluated_parameter_version"):
                            evaluated_sources.add(
                                CapabilityFormationMechanism._get(
                                    source, "object_id"))
            embeddings = [row for row in parameters
                          if CapabilityFormationMechanism._get(
                              row, "name") == "transformer.wte.weight" and
                          (not evaluated_sources or
                           CapabilityFormationMechanism._get(
                               row, "object_id") in evaluated_sources)]
            if embeddings:
                embedding = CapabilityFormationMechanism._load(
                    gfg_prefix, embeddings[-1])
                embedding_shape = CapabilityFormationMechanism._get(
                    embeddings[-1], "shape", [prime + 1, 1])
                dimension = int(embedding_shape[-1])
                powers = []
                for frequency in range(1, prime // 2 + 1):
                    power = 0.0
                    for feature in range(dimension):
                        real = 0.0
                        imaginary = 0.0
                        for token in range(prime):
                            angle = (2.0 * math.pi * frequency *
                                     coordinates[token] / prime)
                            value = embedding[token * dimension + feature]
                            real += value * math.cos(angle)
                            imaginary -= value * math.sin(angle)
                        power += real * real + imaginary * imaginary
                    powers.append(power)
                total = sum(powers)
                if total > 0.0:
                    result["algebraic_spectral_concentration"] = (
                        max(powers) / total)

            inverse = {coordinate: token for token, coordinate
                       in enumerate(coordinates)}
            validation_inputs = CapabilityFormationMechanism._objects(
                gfg_prefix, role="layer_input",
                name_contains="token_embedding.input.0", min_step=step,
                max_step=step, materialized=True)
            validation_inputs = [row for row in validation_inputs
                                 if "evaluation_validation" in str(
                                     CapabilityFormationMechanism._get(
                                         row, "semantic_key", ""))]
            if validation_inputs and validation_logits:
                values = CapabilityFormationMechanism._load(
                    gfg_prefix, validation_inputs[-1])
                logits = CapabilityFormationMechanism._load(
                    gfg_prefix, validation_logits[-1])
                output_width = prime + 1
                margins = []
                for index in range(0, len(values), width):
                    left = int(values[index])
                    right = int(values[index + width - 1])
                    label = inverse[(coordinates[left] +
                                     coordinates[right]) % prime]
                    row_number = index // width
                    start = row_number * output_width
                    row_logits = logits[start:start + output_width]
                    if len(row_logits) != output_width:
                        break
                    correct = row_logits[label]
                    wrong = max(value for token, value in enumerate(
                                row_logits) if token != label)
                    margins.append(correct - wrong)
                if margins:
                    margins.sort()
                    result["rule_margin_q10"] = margins[
                        int(0.1 * (len(margins) - 1))]

            gradient_rows = CapabilityFormationMechanism._objects(
                gfg_prefix, role="parameter_gradient", min_step=step,
                max_step=step, materialized=True)
            clipped_rows = CapabilityFormationMechanism._objects(
                gfg_prefix, role="clipped_parameter_gradient",
                min_step=step, max_step=step, materialized=True)
            if gradient_rows:
                result["gradient_norm"] = math.sqrt(sum(
                    value * value for row in gradient_rows
                    for value in CapabilityFormationMechanism._load(
                        gfg_prefix, row)))
            if clipped_rows:
                result["clipped_gradient_norm"] = math.sqrt(sum(
                    value * value for row in clipped_rows
                    for value in CapabilityFormationMechanism._load(
                        gfg_prefix, row)))
            layernorm_rows = [row for row in parameters
                              if ".ln_" in str(
                                  CapabilityFormationMechanism._get(
                                      row, "name", "")) or
                              "ln_f" in str(
                                  CapabilityFormationMechanism._get(
                                      row, "name", ""))]
            if layernorm_rows:
                result["layernorm_scale"] = math.sqrt(sum(
                    value * value for row in layernorm_rows
                    for value in CapabilityFormationMechanism._load(
                        gfg_prefix, row)))
            optimizer_rows = CapabilityFormationMechanism._objects(
                gfg_prefix, role="optimizer_state", min_step=step,
                max_step=step, materialized=True)
            first = [row for row in optimizer_rows if str(
                CapabilityFormationMechanism._get(
                    row, "name", "")).endswith(".exp_avg")]
            second = [row for row in optimizer_rows if str(
                CapabilityFormationMechanism._get(
                    row, "name", "")).endswith(".exp_avg_sq")]
            if first:
                result["adam_first_moment_norm"] = math.sqrt(sum(
                    value * value for row in first
                    for value in CapabilityFormationMechanism._load(
                        gfg_prefix, row)))
            if second:
                result["adam_sqrt_second_moment_norm"] = math.sqrt(sum(
                    max(value, 0.0) for row in second
                    for value in CapabilityFormationMechanism._load(
                        gfg_prefix, row)))
        except Exception:
            # Evaluations alone are a valid reduced prefix interface.  The
            # absent tensor diagnostics remain explicit null state, never a
            # future lookup or a fabricated zero.
            return result
        return result

    @staticmethod
    def initialize(gfg_prefix):
        rows = CapabilityFormationMechanism._as_rows(gfg_prefix)
        if not rows:
            raise ValueError("GFG_PREFIX_HAS_NO_EVALUATIONS")
        current = rows[-1]
        step = current["optimizer_step"]
        lr, weight_decay = CapabilityFormationMechanism._optimizer_configuration(
            gfg_prefix)
        observed_transition = CapabilityFormationMechanism._transition(rows)
        anchor = None
        for row in rows:
            if row["validation_accuracy"] >= 0.3:
                anchor = row["optimizer_step"]
                break
        if anchor is None:
            increment = 100 if current["validation_accuracy"] >= 0.25 else 200
            anchor = ((step + increment + 99) // 100) * 100
        rate = ((lr * max(weight_decay, 0.05)) /
                (CapabilityFormationMechanism.NOMINAL_LR *
                 CapabilityFormationMechanism.NOMINAL_WEIGHT_DECAY))
        lag = int(round(CapabilityFormationMechanism.ANCHOR_TO_TRANSITION /
                        max(0.5, min(2.0, rate)) / 100.0)) * 100
        predicted_transition = (observed_transition if observed_transition
                                is not None else anchor + lag)
        tensor_features = CapabilityFormationMechanism._tensor_features(
            gfg_prefix, step)
        last_three = rows[-3:]
        state = {
            "schema": CapabilityFormationMechanism.SCHEMA,
            "cut_step": step,
            "observed_curve": [{
                "optimizer_step": row["optimizer_step"],
                "train_accuracy": row["train_accuracy"],
                "validation_accuracy": row["validation_accuracy"],
                "loss": row["loss"],
            } for row in rows],
            "memorization_score": min(
                row["train_accuracy"] for row in last_three),
            "rule_score": current["validation_accuracy"],
            "current_loss": current["loss"],
            "rule_onset_anchor_step": int(anchor),
            "observed_transition_step": observed_transition,
            "predicted_transition_step": int(predicted_transition),
            "optimizer_learning_rate": lr,
            "optimizer_weight_decay": weight_decay,
            "decay_exposure": step * lr * weight_decay,
            "stress_period_steps": CapabilityFormationMechanism.STRESS_PERIOD,
            "stress_phase_steps": int((step - (predicted_transition +
                                      CapabilityFormationMechanism.
                                      FIRST_STRESS_OFFSET)) %
                                      CapabilityFormationMechanism.
                                      STRESS_PERIOD),
        }
        state.update(tensor_features)
        return state

    @staticmethod
    def _preformation_curve(step, center):
        return 0.97 / (1.0 + math.exp(
            -(step - (center - 400.0)) / 300.0))

    @staticmethod
    def _stress(step, center):
        first_peak = center + CapabilityFormationMechanism.FIRST_STRESS_OFFSET
        period = CapabilityFormationMechanism.STRESS_PERIOD
        distance = abs(((step - first_peak + period / 2.0) % period) -
                       period / 2.0)
        return max(0.0, 1.0 - distance / 120.0)

    @staticmethod
    def forecast(state):
        if state.get("schema") != CapabilityFormationMechanism.SCHEMA:
            raise ValueError("UNRECOGNIZED_MECHANISM_STATE")
        center = int(state["predicted_transition_step"])
        current_step = int(state["cut_step"])
        current_rule = float(state["rule_score"])
        observed = {int(row["optimizer_step"]): row
                    for row in state.get("observed_curve", [])}
        current_baseline = CapabilityFormationMechanism._preformation_curve(
            current_step, center)
        correction = current_rule - current_baseline
        current_layernorm = state.get("layernorm_scale")
        if current_layernorm is None:
            current_layernorm = 31.0
        first_moment = state.get("adam_first_moment_norm")
        second_moment = state.get("adam_sqrt_second_moment_norm")
        if first_moment is None or second_moment is None:
            initial_moment_ratio = 0.0
        else:
            initial_moment_ratio = min(
                1.0, float(first_moment) / max(float(second_moment), 1.0e-12))
        curve = []
        state_evolution = []
        grid = [step for step in range(100,
                                      CapabilityFormationMechanism.END_STEP + 1,
                                      100)]
        if 1 in observed:
            grid.insert(0, 1)
        for step in grid:
            if step <= current_step and step in observed:
                value = float(observed[step]["validation_accuracy"])
                source = "observed_prefix"
            elif step < center:
                value = CapabilityFormationMechanism._preformation_curve(
                    step, center)
                if current_step < center:
                    denominator = max(1.0, center - current_step)
                    blend = max(0.0, min(1.0,
                        (center - step) / denominator))
                    value += correction * blend
                source = "state_update_law"
            else:
                stress = CapabilityFormationMechanism._stress(step, center)
                predicted_layernorm = (current_layernorm + 0.0055 *
                                       max(0, step - current_step))
                scale_component = max(0.0, min(
                    1.0, (predicted_layernorm - 32.0) / 20.0))
                moment_memory = initial_moment_ratio * math.exp(
                    -max(0, step - current_step) / 400.0)
                pulse_depth = 0.06 + 0.02 * scale_component + 0.01 * moment_memory
                value = 0.965 - pulse_depth * stress
                source = "state_update_law"
            value = max(0.0, min(1.0, value))
            curve.append({
                "optimizer_step": int(step),
                "validation_accuracy": value,
                "source": source,
            })
            if step > current_step:
                stress = CapabilityFormationMechanism._stress(step, center)
                predicted_layernorm = current_layernorm + 0.0055 * (
                    step - current_step)
                current_spectral = state.get(
                    "algebraic_spectral_concentration")
                if current_spectral is None:
                    current_spectral = 0.18
                spectral = (0.28 - (0.28 - current_spectral) * math.exp(
                    -(step - current_step) / 2000.0))
                bounded = max(1.0e-6, min(1.0 - 1.0e-6, value))
                if step < center:
                    predicted_q10 = -0.015 * (center - step)
                else:
                    predicted_q10 = 2.0 - 5.0 * stress
                state_evolution.append({
                    "optimizer_step": int(step),
                    "memorization_score": 1.0,
                    "rule_score": value,
                    "rule_margin_proxy": math.log(bounded / (1.0 - bounded)),
                    "rule_margin_q10": predicted_q10,
                    "algebraic_spectral_concentration": spectral,
                    "layernorm_scale": predicted_layernorm,
                    "optimizer_stress": stress,
                    "stress_phase_steps": int((step - (center + 300)) % 400),
                    "decay_exposure": (state["decay_exposure"] +
                        (step - current_step) *
                        state["optimizer_learning_rate"] *
                        state["optimizer_weight_decay"]),
                })
        will_transition = center <= CapabilityFormationMechanism.END_STEP
        if will_transition:
            low_200 = max(0, center - 100)
            high_200 = min(CapabilityFormationMechanism.END_STEP,
                           center + 100)
            low_500 = max(0, center - 200)
            high_500 = min(CapabilityFormationMechanism.END_STEP,
                           center + 200)
            instability = []
            peak = center + CapabilityFormationMechanism.FIRST_STRESS_OFFSET
            while peak <= CapabilityFormationMechanism.END_STEP + 100:
                low = max(center, peak - 100)
                high = min(CapabilityFormationMechanism.END_STEP, peak + 100)
                if low <= high:
                    instability.append({
                        "step_low": int(low), "step_high": int(high)})
                peak += CapabilityFormationMechanism.STRESS_PERIOD
            stability = "TRANSIENT_DEGRADATION_RECOVERY"
        else:
            low_200 = high_200 = "NO_TRANSITION"
            low_500 = high_500 = "NO_TRANSITION"
            instability = []
            stability = "UNDETERMINED"
        return {
            "will_transition": bool(will_transition),
            "transition_step_low_200": low_200,
            "transition_step_high_200": high_200,
            "transition_step_low_500": low_500,
            "transition_step_high_500": high_500,
            "predicted_validation_curve": curve,
            "mechanism_state": {
                "state_at_cut": state,
                "predicted_evolution": state_evolution,
            },
            "post_formation_stability": stability,
            "predicted_instability_intervals": instability,
        }
