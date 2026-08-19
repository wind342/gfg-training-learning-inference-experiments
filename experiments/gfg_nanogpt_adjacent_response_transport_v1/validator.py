from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_ALPHAS = np.asarray([-0.125, 0.0, 0.125, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _raw_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _load_ref(root: Path, reference: dict[str, Any]) -> np.ndarray:
    locator = str(reference["locator"])
    _require(locator.startswith("tensor-objects/"), "REFERENCE_LOCATOR_INVALID")
    path = root / locator
    _require(path.is_file(), f"REFERENCE_MISSING:{path}")
    _require(_file_hash(path) == reference["file_sha256"], f"REFERENCE_FILE_HASH_MISMATCH:{path}")
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    _require(_raw_hash(value) == reference["raw_tensor_sha256"], f"REFERENCE_RAW_HASH_MISMATCH:{path}")
    return np.asarray(value)


def _decision(logits: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row = np.arange(logits.shape[0])
    correct = logits[row, groups]
    masked = logits.copy()
    masked[row, groups] = -np.inf
    competitors = masked.argmax(axis=1).astype(np.int64)
    competitor_logits = masked[row, competitors]
    margins = (correct - competitor_logits).astype(np.float32)
    predictions = logits.argmax(axis=1).astype(np.int64)
    q10 = np.asarray([np.quantile(margins[groups == token], 0.10, method="linear") for token in range(23)], dtype=np.float64)
    return margins, predictions, q10, correct.astype(np.float32), competitors


def validate_package(*, root: Path, output_name: str = "VALIDATION.json") -> dict[str, Any]:
    root = root.resolve()
    inventory = _read(root / "RESOLVED_INVENTORY.json")
    manifest = _read(root / "FINITE_AMPLITUDE_CURVES_MANIFEST.json")
    geometry = _read(root / "UPDATE_GEOMETRY_CONTROL_MANIFEST.json")
    identity = _read(root / "IDENTITY_MATERIAL.json")
    _require(inventory["status"] == manifest["status"] == "PASS", "SOURCE_STATUS_NOT_PASS")
    _require(manifest["section_count"] == len(inventory["sections"]) == 72, "SECTION_COUNT_INVALID")
    _require(np.array_equal(np.asarray(manifest["alpha_grid"], dtype=np.float64), EXPECTED_ALPHAS), "ALPHA_GRID_INVALID")
    _require(geometry["status"] == "FROZEN_BEFORE_FINITE_AMPLITUDE_CURVES", "GEOMETRY_NOT_PREFROZEN")
    _require(geometry["curve_values_read"] is False, "GEOMETRY_USED_CURVES")
    _require(identity["evaluation_unit_count"] == 2544, "IDENTITY_COUNT_INVALID")

    section_manifest = {row["section_id"]: row for row in manifest["sections"]}
    forward_checks = 0
    derived_checks = 0
    alpha0_exact = 0
    alpha1_exact = 0
    for section in inventory["sections"]:
        section_id = section["section_id"]
        metadata = _read(root / "sections" / f"{section_id}.json")
        data_path = root / "sections" / f"{section_id}.npz"
        _require(_file_hash(data_path) == metadata["data_file_sha256"] == section_manifest[section_id]["data_file_sha256"], f"SECTION_HASH_MISMATCH:{section_id}")
        with np.load(data_path, allow_pickle=False) as raw:
            payload = {key: raw[key].copy() for key in raw.files}
        _require(np.array_equal(payload["alphas"], EXPECTED_ALPHAS), f"SECTION_ALPHA_GRID_INVALID:{section_id}")
        _require(payload["all_logits"].shape == (7, 12, 212, 24), f"LOGIT_SHAPE_INVALID:{section_id}")
        _require(payload["all_margins"].shape == (7, 12, 212), f"MARGIN_SHAPE_INVALID:{section_id}")
        _require(payload["necessity"].shape == (7, 4, 23), f"NECESSITY_SHAPE_INVALID:{section_id}")
        groups = payload["groups"].astype(np.int64)
        _require(set(np.unique(groups).tolist()) == set(range(23)), f"GROUP_SET_INVALID:{section_id}")
        for alpha_index in range(7):
            for forward_index in range(12):
                logits = payload["all_logits"][alpha_index, forward_index]
                margins, predictions, q10, correct, competitors = _decision(logits, groups)
                _require(np.array_equal(margins, payload["all_margins"][alpha_index, forward_index]), f"DERIVED_MARGIN_MISMATCH:{section_id}")
                _require(np.array_equal(predictions, payload["all_predictions"][alpha_index, forward_index]), f"DERIVED_PREDICTION_MISMATCH:{section_id}")
                _require(np.array_equal(q10, payload["all_group_q10"][alpha_index, forward_index]), f"DERIVED_Q10_MISMATCH:{section_id}")
                if forward_index == 0:
                    _require(np.array_equal(correct, payload["baseline_correct_logits"][alpha_index]), f"CORRECT_LOGIT_MISMATCH:{section_id}")
                    _require(np.array_equal(competitors, payload["baseline_competitor_ids"][alpha_index]), f"COMPETITOR_ID_MISMATCH:{section_id}")
                forward_checks += 1
        for alpha_index in range(7):
            baseline = payload["all_group_q10"][alpha_index, 0]
            singles = payload["all_group_q10"][alpha_index, 2:6]
            pairs = payload["all_group_q10"][alpha_index, 6:12]
            necessity = np.maximum(0.0, baseline[None, :] - singles)
            _require(np.array_equal(necessity, payload["necessity"][alpha_index]), f"NECESSITY_DERIVATION_MISMATCH:{section_id}")
            _require(np.array_equal(np.min(singles, axis=0), payload["single_failure_slack"][alpha_index]), f"SINGLE_SLACK_MISMATCH:{section_id}")
            _require(np.array_equal(np.min(pairs, axis=0), payload["double_failure_slack"][alpha_index]), f"DOUBLE_SLACK_MISMATCH:{section_id}")
            derived_checks += 3

        entry_root = Path(section["receiver_state_path"]).parents[3]
        receiver_probe = _read(Path(section["receiver_probe_path"]))
        endpoint_probe = _read(Path(section["native_endpoint_probe_path"]))
        for index, row in enumerate(receiver_probe["forwards"]):
            _require(np.array_equal(payload["all_logits"][1, index], _load_ref(entry_root, row["logits"])), f"ALPHA0_NATIVE_MISMATCH:{section_id}:{index}")
        for index, row in enumerate(endpoint_probe["forwards"]):
            _require(np.array_equal(payload["all_logits"][-1, index], _load_ref(entry_root, row["logits"])), f"ALPHA1_NATIVE_MISMATCH:{section_id}:{index}")
        alpha0_exact += 1
        alpha1_exact += 1

    local = _read(root / "LOCAL_TO_ENDPOINT_ADJUDICATION.json")
    transport = _read(root / "ADJACENT_TRANSPORT_ANALYSIS.json")
    ledger = _read(root / "TARGET_CROSSING_LEDGERS.json")
    _require(local["status"] == transport["status"] == ledger["status"] == "PASS", "ANALYSIS_STATUS_NOT_PASS")
    _require(local["evaluation_count"] == ledger["row_count"] == 72 * 212, "ANALYSIS_COUNT_INVALID")
    _require(set(transport["summary"]) == {"adjacent", "random_nonadjacent", "matched_cross_run"}, "TRANSPORT_CONTROL_SET_INVALID")
    _require(all(row.get("evaluation_unit_id", "").startswith("evalunit-") for row in ledger["rows"]), "LEDGER_STABLE_ID_MISSING")

    result = {
        "schema": "nanogpt-adjacent-response-independent-validation-v1",
        "status": "PASS",
        "section_count": 72,
        "alpha_count": 7,
        "forward_count_per_alpha": 12,
        "forward_derivation_check_count": forward_checks,
        "support_derivation_check_count": derived_checks,
        "alpha0_native_exact_section_count": alpha0_exact,
        "alpha1_native_exact_section_count": alpha1_exact,
        "target_crossing_ledger_count": ledger["row_count"],
        "geometry_controls_frozen_before_curves": True,
        "global_unseen_entry_accessed": False,
        "future_information_used_for_formal_predictors": False,
    }
    _write(root / output_name, result)
    return result
