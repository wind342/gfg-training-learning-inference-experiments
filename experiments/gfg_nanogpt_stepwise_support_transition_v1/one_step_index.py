from __future__ import annotations

from collections import defaultdict, deque
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    file_sha256,
    payload_sha256,
    read_json,
    require,
    write_json,
)


SCHEMA = "nanogpt-one-step-lightweight-reference-index-v1"
AUDIT_SCHEMA = "nanogpt-one-step-capability-screening-audit-v1"
SELECTION_SCHEMA = "nanogpt-one-step-development-pilot-selection-v1"
DEFAULT_HELD_OUT_ENTRY_ID = "entry-362ded584a953f360aec"
CAPABILITY_DENOMINATOR = 212
CAPABILITY_ZERO_BAND = 1.0 / (2.0 * CAPABILITY_DENOMINATOR)
DECLINE_DELTA_THRESHOLD = -0.02
RIDGE_ALPHA = 1.0e-3


def _jsonl_write(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def _jsonl_read(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                row = json.loads(line)
                require(isinstance(row, dict), f"ONE_STEP_INDEX_ROW_NOT_OBJECT:{line_number}")
                rows.append(row)
    return rows


def _flatten_numeric(value: Any, *, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, bool):
        result[prefix] = float(value)
    elif isinstance(value, (int, float)) and math.isfinite(float(value)):
        result[prefix] = float(value)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            key = f"{prefix}[{index}]"
            result.update(_flatten_numeric(child, prefix=key))
    elif isinstance(value, dict):
        for name in sorted(value):
            key = f"{prefix}.{name}" if prefix else str(name)
            result.update(_flatten_numeric(value[name], prefix=key))
    return result


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _probe_map(entry_root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    probe_root = entry_root / "probe-observations"
    for path in sorted(probe_root.rglob("*.json")):
        probe = read_json(path)
        require(probe.get("status") == "PASS", f"ONE_STEP_PROBE_NOT_PASS:{path}")
        state_id = str(probe["observed_state_id"])
        row = {"path": path, "payload": probe}
        if state_id in result:
            require(
                result[state_id]["payload"]["result_sha256"] == probe["result_sha256"],
                f"ONE_STEP_CONFLICTING_PROBE:{state_id}",
            )
        else:
            result[state_id] = row
    return result


def _state_map(window_root: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for path in sorted((window_root / "states").glob("step-*.json")):
        payload = read_json(path)
        step = int(payload["optimizer_step"])
        require(step not in result, f"ONE_STEP_DUPLICATE_STATE_STEP:{window_root.name}:{step}")
        result[step] = {"path": path, "payload": payload}
    return result


def _transition_refs(transition: dict[str, Any]) -> dict[str, Any]:
    step = transition["step"]
    batch = transition["batch"]

    def tensor_ref(reference: dict[str, Any] | None) -> dict[str, Any] | None:
        if reference is None:
            return None
        return {
            "locator": reference["locator"],
            "file_sha256": reference["file_sha256"],
            "raw_tensor_sha256": reference["raw_tensor_sha256"],
            "representation": reference["representation"],
            "dtype": reference["dtype"],
            "shape": reference["shape"],
            "named_tensor_count": len(reference["canonical_name_order"]),
        }

    source_objects = {
        name: {
            "object_id": reference["object_id"],
            "content_sha256": reference["content_sha256"],
            "locator": reference["locator"],
            "semantic_key": reference["semantic_key"],
        }
        for name, reference in sorted(batch.get("source_training_gfg_objects", {}).items())
    }
    return {
        "batch_evidence_sha256": batch["batch_evidence_sha256"],
        "source_training_gfg_objects": source_objects,
        "step_evidence_sha256": step["step_evidence_sha256"],
        "parameter_update": tensor_ref(step.get("parameter_update")),
        "raw_gradients": tensor_ref(step.get("raw_gradients")),
        "clipped_gradients": tensor_ref(step.get("clipped_gradients")),
        "optimizer_deltas": {
            name: tensor_ref(reference)
            for name, reference in sorted(step.get("optimizer_deltas", {}).items())
        },
    }


def _screen_features(
    pre_state: dict[str, Any],
    post_state: dict[str, Any],
    pre_probe: dict[str, Any],
    transition: dict[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    x_features = {
        "capability_accuracy": float(pre_probe["capability_accuracy"]),
        **{
            f"state.{name}": value
            for name, value in _flatten_numeric(pre_state["state_summary"]).items()
        },
        **{
            f"probe_load.{name}": value
            for name, value in _flatten_numeric(pre_probe["component_loads"]).items()
        },
    }
    step = transition["step"]
    u_features = {
        "loss": float(step["loss"]),
        "total_gradient_norm": float(step["total_gradient_norm"]),
        "execute_optimizer": float(bool(step["execute_optimizer"])),
        **{
            f"optimizer_config.{name}": value
            for name, value in _flatten_numeric(step.get("optimizer_config", {})).items()
        },
    }
    pre_summary = _flatten_numeric(pre_state["state_summary"])
    post_summary = _flatten_numeric(post_state["state_summary"])
    for name in sorted(set(pre_summary) & set(post_summary)):
        u_features[f"realized_state_delta.{name}"] = post_summary[name] - pre_summary[name]
    return x_features, u_features


def build_lightweight_index(
    *,
    formal_root: Path,
    output_root: Path,
    held_out_entry_id: str = DEFAULT_HELD_OUT_ENTRY_ID,
) -> dict[str, Any]:
    formal_root = formal_root.resolve()
    output_root = output_root.resolve()
    entry_roots = sorted(path for path in formal_root.iterdir() if path.is_dir() and path.name.startswith("entry-"))
    require(len(entry_roots) == 13, f"ONE_STEP_ENTRY_COUNT_NOT_13:{len(entry_roots)}")
    require(any(path.name == held_out_entry_id for path in entry_roots), "ONE_STEP_HELD_OUT_ENTRY_MISSING")

    rows: list[dict[str, Any]] = []
    entry_counts: dict[str, int] = {}
    for entry_root in entry_roots:
        manifest_path = entry_root / "stepwise_support_transition_gfg_manifest.json"
        validation_path = entry_root / "stepwise_support_transition_gfg_validation.json"
        manifest = read_json(manifest_path)
        validation = read_json(validation_path)
        require(manifest["entry_id"] == entry_root.name, f"ONE_STEP_ENTRY_ID_MISMATCH:{entry_root.name}")
        require(validation.get("status") == "PASS", f"ONE_STEP_GFG_NOT_VALIDATED:{entry_root.name}")
        probes = _probe_map(entry_root)
        count_before = len(rows)
        for window_root in sorted((entry_root / "windows").iterdir()):
            if not window_root.is_dir():
                continue
            states = _state_map(window_root)
            for transition_path in sorted((window_root / "transitions").glob("step-*-to-*.json")):
                transition = read_json(transition_path)
                step = int(transition["optimizer_step"])
                require(step in states and step + 1 in states, f"ONE_STEP_ADJACENT_STATE_MISSING:{entry_root.name}:{window_root.name}:{step}")
                pre_record = states[step]
                post_record = states[step + 1]
                pre_state = pre_record["payload"]
                post_state = post_record["payload"]
                pre_state_id = str(pre_state["state"]["state_id"])
                post_state_id = str(post_state["state"]["state_id"])
                require(pre_state_id in probes and post_state_id in probes, f"ONE_STEP_PROBE_MISSING:{entry_root.name}:{window_root.name}:{step}")
                pre_probe_row = probes[pre_state_id]
                post_probe_row = probes[post_state_id]
                pre_probe = pre_probe_row["payload"]
                post_probe = post_probe_row["payload"]
                require(
                    transition["from_state_sha256"] == pre_state["state"]["commitment"]["state_sha256"],
                    "ONE_STEP_FROM_STATE_COMMITMENT_MISMATCH",
                )
                require(
                    transition["to_state_sha256"] == post_state["state"]["commitment"]["state_sha256"],
                    "ONE_STEP_TO_STATE_COMMITMENT_MISMATCH",
                )
                x_features, u_features = _screen_features(pre_state, post_state, pre_probe, transition)
                identity = {
                    "entry_id": entry_root.name,
                    "run_id": manifest["source_bundle_id"],
                    "window_id": window_root.name,
                    "optimizer_step": step,
                    "transition_id": transition["transition_id"],
                }
                sample_id = f"one-step-{payload_sha256(identity)[:32]}"
                state_catalog = manifest["state_catalog"]
                row_without_sha = {
                    "schema": SCHEMA,
                    "sample_id": sample_id,
                    **identity,
                    "partition": "HELD_OUT" if entry_root.name == held_out_entry_id else "DEVELOPMENT",
                    "event_boundary": "ACTUAL_UPDATE_FORMED_BEFORE_TARGET_CSRG_PROBE",
                    "x_features": x_features,
                    "u_features": u_features,
                    "targets": {
                        "capability_accuracy_t": float(pre_probe["capability_accuracy"]),
                        "capability_accuracy_t_plus_1": float(post_probe["capability_accuracy"]),
                        "capability_delta": float(post_probe["capability_accuracy"]) - float(pre_probe["capability_accuracy"]),
                    },
                    "references": {
                        "entry_root": str(entry_root),
                        "gfg_database": "stepwise_support_transition_gfg.sqlite3",
                        "gfg_database_sha256": manifest["database_sha256"],
                        "gfg_validation_path": _relative(validation_path, entry_root),
                        "gfg_manifest_path": _relative(manifest_path, entry_root),
                        "gfg_manifest_sha256": manifest["manifest_sha256"],
                        "pre_state": {
                            "path": _relative(pre_record["path"], entry_root),
                            "state_id": pre_state_id,
                            "result_sha256": pre_state["result_sha256"],
                            "gfg_object_id": state_catalog[f"{window_root.name}:{pre_state_id}"],
                        },
                        "current_action": {
                            "path": _relative(transition_path, entry_root),
                            "transition_id": transition["transition_id"],
                            "result_sha256": transition["result_sha256"],
                            "tensor_and_batch_refs": _transition_refs(transition),
                        },
                        "post_state": {
                            "path": _relative(post_record["path"], entry_root),
                            "state_id": post_state_id,
                            "result_sha256": post_state["result_sha256"],
                            "gfg_object_id": state_catalog[f"{window_root.name}:{post_state_id}"],
                        },
                        "pre_probe": {
                            "path": _relative(pre_probe_row["path"], entry_root),
                            "probe_observation_id": pre_probe["probe_observation_id"],
                            "result_sha256": pre_probe["result_sha256"],
                        },
                        "target_probe": {
                            "path": _relative(post_probe_row["path"], entry_root),
                            "probe_observation_id": post_probe["probe_observation_id"],
                            "result_sha256": post_probe["result_sha256"],
                        },
                    },
                }
                rows.append({**row_without_sha, "index_record_sha256": payload_sha256(row_without_sha)})
        entry_counts[entry_root.name] = len(rows) - count_before

    require(len(rows) == 17_270, f"ONE_STEP_TRANSITION_COUNT_NOT_17270:{len(rows)}")
    require(len({row["sample_id"] for row in rows}) == len(rows), "ONE_STEP_SAMPLE_ID_NOT_UNIQUE")
    rows.sort(key=lambda row: (row["entry_id"], row["window_id"], row["optimizer_step"]))
    index_path = output_root / "one_step_reference_index.jsonl"
    written = _jsonl_write(index_path, rows)
    require(written == len(rows), "ONE_STEP_INDEX_WRITE_COUNT_MISMATCH")
    manifest_material = {
        "schema": f"{SCHEMA}-manifest",
        "status": "PASS",
        "scientific_scope": "CAPABILITY_SCALAR_SCREENING_ONLY_NOT_FULL_STATE_COUNTEREXAMPLE_ADJUDICATION",
        "formal_root": str(formal_root),
        "index_file": index_path.name,
        "index_sha256": file_sha256(index_path),
        "index_size_bytes": index_path.stat().st_size,
        "record_count": len(rows),
        "entry_count": len(entry_roots),
        "entry_counts": entry_counts,
        "held_out_entry_id": held_out_entry_id,
        "development_record_count": sum(row["partition"] == "DEVELOPMENT" for row in rows),
        "held_out_record_count": sum(row["partition"] == "HELD_OUT" for row in rows),
        "tensor_payloads_copied": 0,
        "new_gfg_constructed": False,
    }
    manifest = {**manifest_material, "manifest_sha256": payload_sha256(manifest_material)}
    write_json(output_root / "index_manifest.json", manifest)
    return manifest


def _feature_names(rows: list[dict[str, Any]]) -> list[str]:
    require(rows, "ONE_STEP_NO_ROWS")
    common = set(rows[0]["x_features"]) | {f"u.{name}" for name in rows[0]["u_features"]}
    for row in rows[1:]:
        names = set(row["x_features"]) | {f"u.{name}" for name in row["u_features"]}
        common &= names
    require(common, "ONE_STEP_NO_COMMON_FEATURES")
    return sorted(common)


def _matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    result = np.empty((len(rows), len(feature_names)), dtype=np.float64)
    for row_index, row in enumerate(rows):
        for column_index, name in enumerate(feature_names):
            value = row["u_features"][name[2:]] if name.startswith("u.") else row["x_features"][name]
            result[row_index, column_index] = float(value)
    require(bool(np.isfinite(result).all()), "ONE_STEP_NONFINITE_FEATURE")
    return result


def _fit_ridge(train_x: np.ndarray, train_y: np.ndarray) -> dict[str, np.ndarray | float]:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale == 0.0] = 1.0
    normalized = (train_x - mean) / scale
    target_mean = float(train_y.mean())
    gram = normalized.T @ normalized
    weights = np.linalg.solve(gram + RIDGE_ALPHA * np.eye(gram.shape[0]), normalized.T @ (train_y - target_mean))
    return {"mean": mean, "scale": scale, "weights": weights, "target_mean": target_mean}


def _predict(model: dict[str, np.ndarray | float], x: np.ndarray) -> np.ndarray:
    return ((x - model["mean"]) / model["scale"]) @ model["weights"] + model["target_mean"]


def _direction(delta: np.ndarray) -> np.ndarray:
    return np.where(delta > CAPABILITY_ZERO_BAND, 1, np.where(delta < -CAPABILITY_ZERO_BAND, -1, 0))


def _metrics(rows: list[dict[str, Any]], predictions: np.ndarray, tolerance: float) -> dict[str, Any]:
    truth = np.asarray([row["targets"]["capability_accuracy_t_plus_1"] for row in rows], dtype=np.float64)
    current = np.asarray([row["targets"]["capability_accuracy_t"] for row in rows], dtype=np.float64)
    residual = predictions - truth
    truth_delta = truth - current
    predicted_delta = predictions - current
    continuous_error = np.abs(residual) > tolerance
    direction_error = _direction(truth_delta) != _direction(predicted_delta)
    decline_error = (truth_delta <= DECLINE_DELTA_THRESHOLD) != (predicted_delta <= DECLINE_DELTA_THRESHOLD)
    any_error = continuous_error | direction_error | decline_error
    return {
        "record_count": len(rows),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "continuous_error_count": int(continuous_error.sum()),
        "direction_error_count": int(direction_error.sum()),
        "decline_label_error_count": int(decline_error.sum()),
        "any_error_count": int(any_error.sum()),
        "rows": [
            {
                "sample_id": row["sample_id"],
                "entry_id": row["entry_id"],
                "run_id": row["run_id"],
                "window_id": row["window_id"],
                "optimizer_step": row["optimizer_step"],
                "truth": float(truth[index]),
                "prediction": float(predictions[index]),
                "current": float(current[index]),
                "absolute_error": float(abs(residual[index])),
                "continuous_error": bool(continuous_error[index]),
                "direction_error": bool(direction_error[index]),
                "decline_label_error": bool(decline_error[index]),
                "baseline_error_candidate": bool(any_error[index]),
            }
            for index, row in enumerate(rows)
        ],
    }


def _stratified_selection(error_rows: list[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    buckets: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for row in error_rows:
        signature = "".join(
            (
                "C" if row["continuous_error"] else "-",
                "D" if row["direction_error"] else "-",
                "L" if row["decline_label_error"] else "-",
            )
        )
        key = f"{row['entry_id']}|{row['window_id']}|{signature}"
        buckets[key].append(row)
    for key in tuple(buckets):
        buckets[key] = deque(sorted(buckets[key], key=lambda row: (-row["absolute_error"], row["sample_id"])))
    selected: list[dict[str, Any]] = []
    active = sorted(buckets)
    while active and len(selected) < maximum:
        next_active: list[str] = []
        for key in active:
            if len(selected) >= maximum:
                break
            bucket = buckets[key]
            if bucket:
                selected.append(bucket.popleft())
            if bucket:
                next_active.append(key)
        active = next_active
    return selected


def audit_capability_screening(
    *,
    index_path: Path,
    output_root: Path,
    held_out_entry_id: str = DEFAULT_HELD_OUT_ENTRY_ID,
    maximum_selection: int = 500,
) -> dict[str, Any]:
    rows = _jsonl_read(index_path)
    require(len(rows) == 17_270, f"ONE_STEP_AUDIT_INDEX_COUNT_NOT_17270:{len(rows)}")
    development = [row for row in rows if row["entry_id"] != held_out_entry_id]
    held_out = [row for row in rows if row["entry_id"] == held_out_entry_id]
    require(development and held_out, "ONE_STEP_AUDIT_PARTITION_EMPTY")
    feature_names = _feature_names(development)
    development_x = _matrix(development, feature_names)
    development_y = np.asarray([row["targets"]["capability_accuracy_t_plus_1"] for row in development], dtype=np.float64)
    run_ids = np.asarray([row["run_id"] for row in development])

    oof_predictions = np.empty(len(development), dtype=np.float64)
    for run_id in sorted(set(run_ids)):
        evaluation_mask = run_ids == run_id
        training_mask = ~evaluation_mask
        model = _fit_ridge(development_x[training_mask], development_y[training_mask])
        oof_predictions[evaluation_mask] = _predict(model, development_x[evaluation_mask])
    oof_absolute_error = np.abs(oof_predictions - development_y)
    tolerance = max(CAPABILITY_ZERO_BAND, float(np.quantile(oof_absolute_error, 0.95, method="higher")))
    development_metrics = _metrics(development, oof_predictions, tolerance)

    final_model = _fit_ridge(development_x, development_y)
    held_out_x = _matrix(held_out, feature_names)
    held_out_predictions = _predict(final_model, held_out_x)
    held_out_metrics = _metrics(held_out, held_out_predictions, tolerance)

    persistence_development = np.asarray(
        [row["targets"]["capability_accuracy_t"] for row in development], dtype=np.float64
    )
    persistence_held_out = np.asarray(
        [row["targets"]["capability_accuracy_t"] for row in held_out], dtype=np.float64
    )
    persistence_metrics = {
        "development_oof_partition": _metrics(development, persistence_development, tolerance),
        "held_out": _metrics(held_out, persistence_held_out, tolerance),
    }

    development_errors = [row for row in development_metrics.pop("rows") if row["baseline_error_candidate"]]
    held_out_rows = held_out_metrics.pop("rows")
    persistence_metrics["development_oof_partition"].pop("rows")
    persistence_metrics["held_out"].pop("rows")
    selected = _stratified_selection(development_errors, maximum_selection)
    selection_rows = [
        {
            **row,
            "selection_rank": rank,
            "selection_scope": "DEVELOPMENT_ONLY_FULL_STATE_FOLLOWUP_CANDIDATE",
        }
        for rank, row in enumerate(selected, start=1)
    ]
    selection_path = output_root / "development_pilot_selection.jsonl"
    _jsonl_write(selection_path, selection_rows)
    held_out_path = output_root / "held_out_capability_screening_rows.jsonl"
    _jsonl_write(held_out_path, held_out_rows)

    report_material = {
        "schema": AUDIT_SCHEMA,
        "status": "PASS",
        "scientific_scope": "CAPABILITY_SCALAR_SCREENING_ONLY_NOT_FULL_STATE_COUNTEREXAMPLE_ADJUDICATION",
        "index_sha256": file_sha256(index_path),
        "held_out_entry_id": held_out_entry_id,
        "held_out_used_for_model_fitting": False,
        "held_out_used_for_scaling_or_thresholds": False,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "ridge_alpha": RIDGE_ALPHA,
        "capability_zero_band": CAPABILITY_ZERO_BAND,
        "decline_delta_threshold": DECLINE_DELTA_THRESHOLD,
        "development_frozen_absolute_error_tolerance": tolerance,
        "development_oof_metrics": development_metrics,
        "held_out_metrics": held_out_metrics,
        "persistence_metrics": persistence_metrics,
        "development_error_candidate_count": len(development_errors),
        "held_out_error_candidate_count": sum(row["baseline_error_candidate"] for row in held_out_rows),
        "selection": {
            "schema": SELECTION_SCHEMA,
            "maximum": maximum_selection,
            "selected_count": len(selection_rows),
            "source_population": "development baseline-error candidates only",
            "method": "deterministic round-robin by entry, window and error signature; severity then sample identity within bucket",
            "file": selection_path.name,
            "sha256": file_sha256(selection_path),
        },
        "held_out_rows_file": held_out_path.name,
        "held_out_rows_sha256": file_sha256(held_out_path),
        "gpu_execution": False,
        "persistent_ai_session": False,
        "new_gfg_constructed": False,
        "tensor_payloads_copied": 0,
    }
    report = {**report_material, "audit_sha256": payload_sha256(report_material)}
    write_json(output_root / "capability_screening_audit.json", report)
    return report


def run_lightweight_pilot(
    *,
    formal_root: Path,
    output_root: Path,
    held_out_entry_id: str = DEFAULT_HELD_OUT_ENTRY_ID,
    maximum_selection: int = 500,
) -> dict[str, Any]:
    index_manifest = build_lightweight_index(
        formal_root=formal_root,
        output_root=output_root,
        held_out_entry_id=held_out_entry_id,
    )
    audit = audit_capability_screening(
        index_path=output_root / index_manifest["index_file"],
        output_root=output_root,
        held_out_entry_id=held_out_entry_id,
        maximum_selection=maximum_selection,
    )
    result_material = {
        "schema": "nanogpt-one-step-lightweight-index-pilot-result-v1",
        "status": "PASS",
        "index_manifest_sha256": index_manifest["manifest_sha256"],
        "audit_sha256": audit["audit_sha256"],
        "record_count": index_manifest["record_count"],
        "development_error_candidate_count": audit["development_error_candidate_count"],
        "held_out_error_candidate_count": audit["held_out_error_candidate_count"],
        "selected_development_record_count": audit["selection"]["selected_count"],
        "full_state_counterexample_count_established": False,
        "next_stage": "materialize full CSRG summaries only for the frozen development selection before defining formal one-step counterexamples",
    }
    result = {**result_material, "result_sha256": payload_sha256(result_material)}
    write_json(output_root / "RESULT.json", result)
    return result


__all__ = [
    "AUDIT_SCHEMA",
    "CAPABILITY_ZERO_BAND",
    "DECLINE_DELTA_THRESHOLD",
    "DEFAULT_HELD_OUT_ENTRY_ID",
    "SCHEMA",
    "audit_capability_screening",
    "build_lightweight_index",
    "run_lightweight_pilot",
]
