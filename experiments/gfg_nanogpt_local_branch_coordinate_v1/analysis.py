from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from experiments.gfg_nanogpt_native_prebranch_left_history_v1.analysis import (
    COMPONENTS,
    FACTOR_RECORDS,
    IDENTITY_MATERIAL,
    RESPONSE_ROOT,
    STEPWISE_ROOT,
    PayloadLoader,
    RobustSpace,
    binary_metrics,
    canonical_json,
    compile_dataset,
    component_layout_names,
    cosine,
    file_sha256,
    graph_objects,
    layout_slice,
    pr_auc,
    query_neighbors,
    response_metrics,
    risk_from_neighbors,
    select_temporal_object,
    weighted_values,
    weights_from_distance,
)


REPOSITORY_ROOT = Path(__file__).parents[2]
ARCHIVE_ENTRY_ROOT = (
    REPOSITORY_ROOT
    / "experiments"
    / "gfg_nanogpt_autonomous_capability_discovery_v1"
    / "research_archive"
    / "entries"
)
INVENTORY_PATH = RESPONSE_ROOT / "RESOLVED_INVENTORY.json"
DEVELOPMENT_RUNS = (
    "entry-abc8f864a49ee0e056f4",
    "entry-54ba9566f731754a0e3f",
    "entry-7a22f51938059541de98",
    "entry-f34e7e61444c90976b36",
    "entry-7c1a3094f8acf9cf4bb0",
    "entry-5bb1186bc27eb82111fb",
    "entry-8fa6576fc7128f93a228",
    "entry-d8b1bf9cd00ddf314725",
)
CONFIRMATION_RUNS = (
    "entry-4ed462761347d6b87e61",
    "entry-d5b80ca9b9cd18fa343f",
    "entry-786d0a3628f6f791399f",
    "entry-481b86f81d58d496a687",
)
CANDIDATES = (
    "action_support_alignment",
    "action_support_necessity_alignment",
    "action_support_velocity_alignment",
    "adam_receiver_alignment",
    "batch_target_advantage",
    "exact_context_target_advantage",
    "preconditioned_support_velocity_alignment",
    "update_receiver_alignment",
)
K = 64


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_verified_descriptor(loader: PayloadLoader, entry_id: str, descriptor: dict[str, Any]) -> np.ndarray:
    return np.asarray(loader.load(entry_id, descriptor), dtype=np.float64)


def named_tensor(value: np.ndarray, descriptor: dict[str, Any], name: str) -> np.ndarray:
    matches = [row for row in descriptor["layout"] if str(row["name"]) == name]
    require(len(matches) == 1, f"NAMED_TENSOR_CARDINALITY:{name}:{len(matches)}")
    row = matches[0]
    offset = int(row["offset"])
    count = int(row["element_count"])
    return np.asarray(value[offset : offset + count], dtype=np.float64).reshape(row["shape"])


def component_shares(value: np.ndarray, descriptor: dict[str, Any]) -> np.ndarray:
    norms = []
    for component in COMPONENTS:
        names = component_layout_names(descriptor, component)
        norms.append(float(np.linalg.norm(layout_slice(value, descriptor, names))))
    values = np.asarray(norms, dtype=np.float64)
    total = float(np.sum(values))
    return values / total if total > 1e-20 else np.full(4, 0.25, dtype=np.float64)


def archive_tensor_path(entry_id: str, descriptor: dict[str, Any]) -> Path:
    manifest = read_json(ARCHIVE_ENTRY_ROOT / entry_id / "gfg_manifest.json")
    bundle_root = REPOSITORY_ROOT / str(manifest["local_repository_path"])
    locator = str(descriptor["locator"])
    require(locator.startswith("objects://"), f"UNSUPPORTED_ARCHIVE_LOCATOR:{locator}")
    return bundle_root / "tensor-objects" / locator.removeprefix("objects://")


def load_archive_tensor(entry_id: str, descriptor: dict[str, Any]) -> tuple[np.ndarray, Path, str]:
    path = archive_tensor_path(entry_id, descriptor)
    require(path.is_file(), f"ARCHIVE_TENSOR_MISSING:{path}")
    value = np.asarray(np.load(path, allow_pickle=False))
    expected = str(descriptor["content_sha256"])
    actual = sha256_array(value)
    require(actual == expected, f"ARCHIVE_TENSOR_CONTENT_MISMATCH:{path}")
    return value, path, file_sha256(path)


def finite_cos(left: np.ndarray, right: np.ndarray) -> float | None:
    value = cosine(np.asarray(left, dtype=np.float64).ravel(), np.asarray(right, dtype=np.float64).ravel())
    return float(value) if value is not None and math.isfinite(value) else None


def _section_inventory() -> dict[str, dict[str, Any]]:
    material = read_json(INVENTORY_PATH)
    return {str(row["section_id"]): row for row in material["sections"]}


def _coordinate_rows(
    compiled: dict[str, Any],
) -> tuple[list[dict[str, float | None]], list[dict[str, Any]], dict[str, Any]]:
    records = compiled["records"]
    identities = read_json(IDENTITY_MATERIAL)["entries"]
    inventory = _section_inventory()
    by_section: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        by_section[str(row["section_id"])].append(index)
    loader = PayloadLoader(STEPWISE_ROOT)
    rows: list[dict[str, float | None]] = [{} for _ in records]
    source_ledger: list[dict[str, Any]] = []
    availability = Counter()

    for section_id, indices in sorted(by_section.items()):
        first = records[indices[0]]
        entry_id = str(first["entry_id"])
        step = int(first["optimizer_step"])
        transition_path = Path(str(first["source_refs"]["transition"]))
        transition = read_json(transition_path)
        require(int(transition["optimizer_step"]) == step, f"TRANSITION_STEP_MISMATCH:{section_id}")
        state_path = transition_path.parent.parent / "states" / f"step-{step:05d}.json"
        state = read_json(state_path)["state"]

        update_desc = transition["step"]["parameter_update"]
        preconditioned_desc = transition["step"]["optimizer_deltas"]["post_preconditioned_direction"]
        parameter_desc = state["parameters"]
        exp_avg_desc = state["optimizer_exp_avg"]
        update = load_verified_descriptor(loader, entry_id, update_desc)
        preconditioned = load_verified_descriptor(loader, entry_id, preconditioned_desc)
        parameters = load_verified_descriptor(loader, entry_id, parameter_desc)
        exp_avg = load_verified_descriptor(loader, entry_id, exp_avg_desc)
        update_wte = named_tensor(update, update_desc, "transformer.wte.weight")
        parameter_wte = named_tensor(parameters, parameter_desc, "transformer.wte.weight")
        exp_avg_wte = named_tensor(exp_avg, exp_avg_desc, "transformer.wte.weight")
        preconditioned_shares = component_shares(preconditioned, preconditioned_desc)

        batch_objects = transition["batch"]["source_training_gfg_objects"]
        batch_inputs, batch_input_path, batch_input_file_sha = load_archive_tensor(
            entry_id, batch_objects["training_batch_inputs"]
        )
        batch_targets, batch_target_path, batch_target_file_sha = load_archive_tensor(
            entry_id, batch_objects["training_batch_targets"]
        )
        require(batch_inputs.shape == batch_targets.shape and batch_inputs.ndim == 2, f"BATCH_SHAPE_INVALID:{section_id}")
        valid_targets = batch_targets[batch_targets >= 0].astype(np.int64)
        require(len(valid_targets) > 0, f"NO_VALID_BATCH_TARGETS:{section_id}")

        inv = inventory[section_id]
        evaluation_path = Path(str(inv["evaluation_input_path"]))
        evaluation_inputs = np.asarray(np.load(evaluation_path, allow_pickle=False))
        require(
            sha256_array(evaluation_inputs) == str(inv["evaluation_input_sha256"]),
            f"EVAL_INPUT_CONTENT_MISMATCH:{entry_id}",
        )
        identity_index = {str(row["evaluation_unit_id"]): idx for idx, row in enumerate(identities[entry_id])}
        require(evaluation_inputs.shape[0] == len(identity_index), f"EVAL_IDENTITY_COUNT_MISMATCH:{entry_id}")

        section_path = RESPONSE_ROOT / "sections" / f"{section_id}.npz"
        require(file_sha256(section_path) == str(first["source_refs"]["section_npz_sha256"]), f"SECTION_HASH_MISMATCH:{section_id}")
        with np.load(section_path, allow_pickle=False) as section:
            current_allocation = np.asarray(section["support_allocation"], dtype=np.float64)[1]
        objects = graph_objects(entry_id, step)
        left_allocation_obj = select_temporal_object(
            objects,
            "input_available_at_cut:finite_difference_left:support_allocation",
            [step - 1, step],
        )
        left_allocation = load_verified_descriptor(loader, entry_id, left_allocation_obj["payload"])
        require(left_allocation.shape == (4, 23), f"LEFT_ALLOCATION_SHAPE_INVALID:{section_id}")

        availability["sections"] += 1
        availability["formed_update"] += 1
        availability["pre_update_parameter_and_adam"] += 1
        availability["native_training_batch"] += 1
        availability["identity_aligned_evaluation_input"] += 1
        availability["pre_response_support_and_left_velocity"] += 1
        source_ledger.append(
            {
                "section_id": section_id,
                "entry_id": entry_id,
                "optimizer_step": step,
                "transition_path": str(transition_path),
                "transition_sha256": file_sha256(transition_path),
                "state_path": str(state_path),
                "state_sha256": file_sha256(state_path),
                "response_section_path": str(section_path),
                "response_section_sha256": file_sha256(section_path),
                "evaluation_input_path": str(evaluation_path),
                "evaluation_input_file_sha256": file_sha256(evaluation_path),
                "evaluation_input_content_sha256": sha256_array(evaluation_inputs),
                "batch_input_path": str(batch_input_path),
                "batch_input_file_sha256": batch_input_file_sha,
                "batch_input_content_sha256": sha256_array(batch_inputs),
                "batch_target_path": str(batch_target_path),
                "batch_target_file_sha256": batch_target_file_sha,
                "batch_target_content_sha256": sha256_array(batch_targets),
                "parameter_payload_sha256": str(parameter_desc["file_sha256"]),
                "exp_avg_payload_sha256": str(exp_avg_desc["file_sha256"]),
                "update_payload_sha256": str(update_desc["file_sha256"]),
                "preconditioned_payload_sha256": str(preconditioned_desc["file_sha256"]),
                "left_allocation_object_id": str(left_allocation_obj["object_id"]),
                "left_allocation_payload_sha256": str(left_allocation_obj["payload"]["file_sha256"]),
            }
        )

        for index in indices:
            record = records[index]
            target = int(record["target_group"])
            competitor = int(compiled["meta"][index]["current_competitor_identity"])
            unit_row = identity_index[str(record["evaluation_unit_id"])]
            evaluation_input = np.asarray(evaluation_inputs[unit_row])
            exact = np.all(batch_inputs == evaluation_input[None, :], axis=1)
            exact_target_values = batch_targets[exact]
            target_count = int(np.sum(valid_targets == target))
            competitor_count = int(np.sum(valid_targets == competitor))
            exact_target_count = int(np.sum(exact_target_values == target))
            exact_competitor_count = int(np.sum(exact_target_values == competitor))

            receiver_difference = parameter_wte[target] - parameter_wte[competitor]
            update_difference = update_wte[target] - update_wte[competitor]
            adam_difference = exp_avg_wte[target] - exp_avg_wte[competitor]
            f3 = record["features"]["F3"]["numeric"]
            update_shares = np.asarray(
                [float(f3[f"{component.replace('.', '_')}_update_share"]) for component in COMPONENTS],
                dtype=np.float64,
            )
            allocation = np.asarray(current_allocation[:, target], dtype=np.float64)
            allocation_velocity = np.asarray(left_allocation[:, target], dtype=np.float64)
            necessity = np.asarray(
                [float(record["features"]["F4"]["numeric"][f"necessity_{component.replace('.', '_')}"]) for component in COMPONENTS],
                dtype=np.float64,
            )
            necessity = np.maximum(necessity, 0.0)
            necessity_sum = float(np.sum(necessity))
            normalized_necessity = necessity / necessity_sum if necessity_sum > 1e-20 else np.full(4, 0.25)
            rows[index] = {
                "batch_target_advantage": float((target_count - competitor_count) / len(valid_targets)),
                "exact_context_target_advantage": float(
                    (exact_target_count - exact_competitor_count) / (int(np.sum(exact)) + 1)
                ),
                "update_receiver_alignment": finite_cos(receiver_difference, update_difference),
                "adam_receiver_alignment": finite_cos(receiver_difference, adam_difference),
                "action_support_alignment": float(np.dot(update_shares, allocation)),
                "action_support_velocity_alignment": float(np.dot(update_shares, allocation_velocity)),
                "action_support_necessity_alignment": float(np.dot(update_shares, normalized_necessity)),
                "preconditioned_support_velocity_alignment": float(
                    np.dot(preconditioned_shares, allocation_velocity)
                ),
            }

    require(all(sorted(row) == list(CANDIDATES) for row in rows), "CANDIDATE_SCHEMA_MISMATCH")
    audit = {
        "schema": "nanogpt-local-branch-coordinate-availability-v1",
        "status": "PASS",
        "record_count": len(rows),
        "section_count": len(by_section),
        "run_count": len(set(compiled["entries"].tolist())),
        "candidate_names": list(CANDIDATES),
        "coverage_counts": dict(availability),
        "verified_stepwise_payload_file_count": len(loader.verified_files),
        "post_response_inputs_used": False,
        "run_or_step_identity_used_as_feature": False,
        "current_alpha_positive_probe_used_as_feature": False,
        "candidate_semantics_frozen_before_outcome_evaluation": True,
    }
    return rows, source_ledger, audit


def _candidate_matrix(rows: list[dict[str, float | None]]) -> np.ndarray:
    return np.asarray(
        [[np.nan if row[name] is None else float(row[name]) for name in CANDIDATES] for row in rows],
        dtype=np.float64,
    )


def compile_coordinate_dataset() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    compiled, upstream_audit, _ = compile_dataset()
    rows, source_ledger, audit = _coordinate_rows(compiled)
    candidate_matrix = _candidate_matrix(rows)
    require(candidate_matrix.shape == (15264, len(CANDIDATES)), "CANDIDATE_MATRIX_SHAPE_INVALID")
    entries = set(compiled["unique_entries"])
    require(entries == set(DEVELOPMENT_RUNS) | set(CONFIRMATION_RUNS), "FROZEN_SPLIT_RUN_SET_MISMATCH")
    audit["upstream_availability_status"] = upstream_audit["status"]
    audit["finite_counts"] = {
        name: int(np.sum(np.isfinite(candidate_matrix[:, index]))) for index, name in enumerate(CANDIDATES)
    }
    audit["source_ledger_sha256"] = hashlib.sha256(canonical_json(source_ledger).encode("utf-8")).hexdigest()
    compiled["candidate_rows"] = rows
    compiled["candidate_matrix"] = candidate_matrix
    return compiled, audit, source_ledger


def _space_fit(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = RobustSpace().fit(train)
    train_z = scaler.transform(train)
    test_z = scaler.transform(test)
    dimension = max(train_z.shape[1], 1)
    return train_z / math.sqrt(dimension), test_z / math.sqrt(dimension)


def _q_fit(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    median = float(np.nanmedian(train))
    scale = float(np.nanpercentile(train, 75) - np.nanpercentile(train, 25))
    # A frozen candidate may be constant in one training fold. In that fold it
    # contributes no coordinate information; treating it as a zero block is the
    # exact no-information result rather than a reason to drop or reroll it.
    if not math.isfinite(median) or not math.isfinite(scale) or scale < 1e-12:
        return np.zeros((len(train), 1), dtype=np.float64), np.zeros((len(test), 1), dtype=np.float64)
    train_filled = np.where(np.isfinite(train), train, median)
    test_filled = np.where(np.isfinite(test), test, median)
    return ((train_filled - median) / scale)[:, None], ((test_filled - median) / scale)[:, None]


def _fold_prediction(
    x3: np.ndarray,
    q: np.ndarray | None,
    labels: np.ndarray,
    curves: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> dict[str, Any]:
    train_x, test_x = _space_fit(x3[train_mask], x3[test_mask])
    if q is not None:
        train_q, test_q = _q_fit(q[train_mask], q[test_mask])
        train_x = np.concatenate([train_x, train_q], axis=1)
        test_x = np.concatenate([test_x, test_q], axis=1)
    distances, neighbors_local = query_neighbors(train_x, test_x, K)
    weights = weights_from_distance(distances)
    train_indices = np.flatnonzero(train_mask)
    neighbors = train_indices[neighbors_local]
    return {
        "test_indices": np.flatnonzero(test_mask),
        "neighbors": neighbors,
        "weights": weights,
        "risk": risk_from_neighbors(labels[train_mask], neighbors_local, weights),
        "curve": weighted_values(curves[train_mask], neighbors_local, weights),
        "distance": distances,
    }


def _pooled_binary(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    return binary_metrics(labels.astype(bool), scores, 0.5)


def _development_candidate_scores(compiled: dict[str, Any]) -> tuple[dict[str, Any], str, dict[str, Any]]:
    entries = np.asarray(compiled["entries"], dtype=object)
    dev_mask = np.isin(entries, DEVELOPMENT_RUNS)
    labels = compiled["labels"]["severe_conflict"].astype(bool)
    x3 = np.asarray(compiled["spaces"]["X3"], dtype=np.float64)
    curves = np.asarray(compiled["curves"], dtype=np.float64)
    candidate_matrix = np.asarray(compiled["candidate_matrix"], dtype=np.float64)
    scores: dict[str, Any] = {}
    runwise: dict[str, Any] = {}
    for candidate_index, candidate in enumerate(CANDIDATES):
        prediction = np.full(len(entries), np.nan, dtype=np.float64)
        run_values: dict[str, Any] = {}
        for held_out in DEVELOPMENT_RUNS:
            train_mask = dev_mask & (entries != held_out)
            test_mask = entries == held_out
            fold = _fold_prediction(
                x3,
                candidate_matrix[:, candidate_index],
                labels,
                curves,
                train_mask,
                test_mask,
            )
            prediction[fold["test_indices"]] = fold["risk"]
            run_values[held_out] = _pooled_binary(labels[test_mask], fold["risk"])
        require(np.all(np.isfinite(prediction[dev_mask])), f"DEV_PREDICTION_INCOMPLETE:{candidate}")
        pooled = _pooled_binary(labels[dev_mask], prediction[dev_mask])
        scores[candidate] = pooled
        runwise[candidate] = run_values
    ranking = sorted(CANDIDATES, key=lambda name: (-float(scores[name]["pr_auc"]), name))
    selected = ranking[0]
    selection = {
        "schema": "nanogpt-local-branch-coordinate-selection-v1",
        "status": "SEALED",
        "selection_set": "DEVELOPMENT_RUNS_ONLY",
        "selection_metric": "complete_leave_one_development_run_out_severe_conflict_pr_auc",
        "selected_coordinate": selected,
        "candidate_ranking": ranking,
        "candidate_metrics": scores,
        "confirmation_results_seen_during_selection": False,
        "reroll_allowed": False,
    }
    return selection, selected, {"runwise": runwise}


def _oracle_curves(
    compiled: dict[str, Any], selected_q: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray
) -> np.ndarray:
    x3 = np.asarray(compiled["spaces"]["X3"], dtype=np.float64)
    labels = compiled["labels"]["severe_conflict"].astype(bool)
    curves = np.asarray(compiled["curves"], dtype=np.float64)
    train_x, test_x = _space_fit(x3[train_mask], x3[test_mask])
    train_q, test_q = _q_fit(selected_q[train_mask], selected_q[test_mask])
    train_x = np.concatenate([train_x, train_q], axis=1)
    test_x = np.concatenate([test_x, test_q], axis=1)
    output = np.empty((int(np.sum(test_mask)), curves.shape[1]), dtype=np.float64)
    train_labels = labels[train_mask]
    test_labels = labels[test_mask]
    for branch in (False, True):
        branch_train = train_labels == branch
        branch_test = test_labels == branch
        distances, neighbors = query_neighbors(train_x[branch_train], test_x[branch_test], K)
        output[branch_test] = weighted_values(curves[train_mask][branch_train], neighbors, weights_from_distance(distances))
    return output


def _confirmation_evaluation(compiled: dict[str, Any], selected: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    entries = np.asarray(compiled["entries"], dtype=object)
    train_mask = np.isin(entries, DEVELOPMENT_RUNS)
    test_mask = np.isin(entries, CONFIRMATION_RUNS)
    labels = compiled["labels"]["severe_conflict"].astype(bool)
    curves = np.asarray(compiled["curves"], dtype=np.float64)
    margin0 = np.asarray(compiled["margin0"], dtype=np.float64)
    x3 = np.asarray(compiled["spaces"]["X3"], dtype=np.float64)
    selected_index = CANDIDATES.index(selected)
    q = np.asarray(compiled["candidate_matrix"][:, selected_index], dtype=np.float64)
    baseline = _fold_prediction(x3, None, labels, curves, train_mask, test_mask)
    augmented = _fold_prediction(x3, q, labels, curves, train_mask, test_mask)
    test_indices = augmented["test_indices"]
    require(np.array_equal(test_indices, baseline["test_indices"]), "CONFIRMATION_INDEX_DRIFT")
    oracle_curve = _oracle_curves(compiled, q, train_mask, test_mask)
    baseline_binary = _pooled_binary(labels[test_mask], baseline["risk"])
    augmented_binary = _pooled_binary(labels[test_mask], augmented["risk"])
    runwise: dict[str, Any] = {}
    local_position = {int(value): pos for pos, value in enumerate(test_indices.tolist())}
    for entry in CONFIRMATION_RUNS:
        global_indices = np.flatnonzero(entries == entry)
        positions = np.asarray([local_position[int(value)] for value in global_indices], dtype=np.int64)
        runwise[entry] = {
            "X3": _pooled_binary(labels[global_indices], baseline["risk"][positions]),
            "X3_plus_q": _pooled_binary(labels[global_indices], augmented["risk"][positions]),
        }
    margin_test = margin0[test_mask]
    severe_test = labels[test_mask]
    response = {
        "X3": response_metrics(curves[test_mask], baseline["curve"], margin_test, severe_test),
        "X3_plus_q": response_metrics(curves[test_mask], augmented["curve"], margin_test, severe_test),
        "oracle_same_true_branch_X3_plus_q": response_metrics(curves[test_mask], oracle_curve, margin_test, severe_test),
    }

    # Matched diagnostic: every confirmation severe record is matched to the nearest
    # development ordinary record in X3, before q is consulted.
    train_x, test_x = _space_fit(x3[train_mask & (~labels)], x3[test_mask & labels])
    distances, neighbors_local = query_neighbors(train_x, test_x, 1)
    normal_indices = np.flatnonzero(train_mask & (~labels))[neighbors_local[:, 0]]
    severe_indices = np.flatnonzero(test_mask & labels)
    q_scaler = RobustSpace().fit(q[train_mask, None])
    q_all = q_scaler.transform(q[:, None])[:, 0]
    matches = []
    for severe_index, normal_index, distance in zip(severe_indices, normal_indices, distances[:, 0]):
        matches.append(
            {
                "severe_record_id": compiled["meta"][int(severe_index)]["record_id"],
                "severe_entry_id": str(entries[int(severe_index)]),
                "normal_record_id": compiled["meta"][int(normal_index)]["record_id"],
                "normal_entry_id": str(entries[int(normal_index)]),
                "X3_distance": float(distance),
                "q_severe": float(q_all[int(severe_index)]),
                "q_normal": float(q_all[int(normal_index)]),
                "absolute_q_separation": float(abs(q_all[int(severe_index)] - q_all[int(normal_index)])),
            }
        )

    ledger: list[dict[str, Any]] = []
    baseline_neighbor_map = baseline["neighbors"]
    augmented_neighbor_map = augmented["neighbors"]
    for position, index in enumerate(test_indices):
        ledger.append(
            {
                "record_id": compiled["meta"][int(index)]["record_id"],
                "entry_id": str(entries[int(index)]),
                "optimizer_step": int(compiled["meta"][int(index)]["optimizer_step"]),
                "true_severe_conflict": bool(labels[int(index)]),
                "selected_q_name": selected,
                "selected_q_value": None if not math.isfinite(float(q[int(index)])) else float(q[int(index)]),
                "all_candidate_values": compiled["candidate_rows"][int(index)],
                "X3_risk": float(baseline["risk"][position]),
                "X3_plus_q_risk": float(augmented["risk"][position]),
                "true_response_curve": curves[int(index)].tolist(),
                "X3_response_prediction": baseline["curve"][position].tolist(),
                "X3_plus_q_response_prediction": augmented["curve"][position].tolist(),
                "oracle_response_prediction": oracle_curve[position].tolist(),
                "margin0": float(margin0[int(index)]),
                "X3_neighbor_record_ids": [compiled["meta"][int(value)]["record_id"] for value in baseline_neighbor_map[position]],
                "X3_plus_q_neighbor_record_ids": [compiled["meta"][int(value)]["record_id"] for value in augmented_neighbor_map[position]],
                "neighbor_entries_are_development_only": True,
            }
        )
    results = {
        "schema": "nanogpt-local-branch-coordinate-results-v1",
        "status": "PASS",
        "selected_coordinate": selected,
        "confirmation_branch_risk": {"X3": baseline_binary, "X3_plus_q": augmented_binary},
        "confirmation_runwise": runwise,
        "confirmation_response": response,
        "matched_pair_summary": {
            "pair_count": len(matches),
            "median_X3_distance": float(np.median([row["X3_distance"] for row in matches])),
            "median_absolute_q_separation": float(np.median([row["absolute_q_separation"] for row in matches])),
            "normal_control_reuse_count": len(matches) - len(set(row["normal_record_id"] for row in matches)),
        },
        "matched_pairs": matches,
        "confirmation_labels_used_as_inputs": False,
        "oracle_is_diagnostic_only": True,
    }
    return results, ledger


def evaluate_coordinate(compiled: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selection, selected, development = _development_candidate_scores(compiled)
    confirmation, ledger = _confirmation_evaluation(compiled, selected)
    runwise_nonnegative = sum(
        1
        for value in confirmation["confirmation_runwise"].values()
        if value["X3"]["pr_auc"] is not None
        and value["X3_plus_q"]["pr_auc"] is not None
        and float(value["X3_plus_q"]["pr_auc"]) >= float(value["X3"]["pr_auc"])
    )
    baseline = confirmation["confirmation_branch_risk"]["X3"]
    augmented = confirmation["confirmation_branch_risk"]["X3_plus_q"]
    base_response = confirmation["confirmation_response"]["X3"]
    aug_response = confirmation["confirmation_response"]["X3_plus_q"]
    severe_before = float(base_response["severe_conflict"]["curve_rmse"])
    severe_after = float(aug_response["severe_conflict"]["curve_rmse"])
    severe_improvement = (severe_before - severe_after) / severe_before if severe_before else 0.0
    pr_improved = float(augmented["pr_auc"]) > float(baseline["pr_auc"])
    response_gate = severe_improvement >= 0.03
    boundary_gate = (
        float(aug_response["overall"]["boundary_accuracy"])
        >= float(base_response["overall"]["boundary_accuracy"]) - 0.01
    )
    if pr_improved and runwise_nonnegative >= 3 and response_gate and boundary_gate:
        verdict = "SUPPORTED"
    elif pr_improved or confirmation["matched_pair_summary"]["median_absolute_q_separation"] >= 0.5:
        verdict = "PARTIALLY_SUPPORTED"
    else:
        verdict = "NOT_SUPPORTED"
    decision = {
        "schema": "nanogpt-local-branch-coordinate-decision-v1",
        "status": "PASS",
        "verdict": verdict,
        "selected_coordinate": selected,
        "confirmation_pr_auc_delta": float(augmented["pr_auc"] - baseline["pr_auc"]),
        "confirmation_runs_nonnegative_pr_auc_delta": runwise_nonnegative,
        "severe_curve_rmse_relative_improvement": severe_improvement,
        "overall_boundary_accuracy_delta": float(
            aug_response["overall"]["boundary_accuracy"] - base_response["overall"]["boundary_accuracy"]
        ),
        "gates": {
            "confirmation_pr_auc_improved": pr_improved,
            "at_least_three_confirmation_runs_nonnegative": runwise_nonnegative >= 3,
            "severe_curve_rmse_improved_at_least_3pct": response_gate,
            "overall_boundary_drop_at_most_1pp": boundary_gate,
            "independent_and_gfg_validation_pending": True,
        },
        "no_candidate_reroll_performed": True,
    }
    return {
        "selection": selection,
        "development": development,
        "confirmation": confirmation,
        "decision": decision,
    }, ledger


__all__ = [
    "CANDIDATES",
    "CONFIRMATION_RUNS",
    "DEVELOPMENT_RUNS",
    "compile_coordinate_dataset",
    "evaluate_coordinate",
    "file_sha256",
    "response_metrics",
    "binary_metrics",
]
