from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import GLOBAL_UNSEEN_ENTRY, file_sha256, read_json, require, write_json


REQUIRED_ANALYSIS_FILES = (
    "FACTOR_ANALYSIS_CONTRACT.md",
    "ANALYSIS_FREEZE.json",
    "RECOVERY_AUDIT.json",
    "BOUNDARY_VIOLATION.json",
    "FACTOR_SCHEMA.json",
    "PRETARGET_FEATURE_AVAILABILITY.json",
    "MATCHING_PROTOCOL.json",
    "SINGLE_FACTOR_RESULTS.json",
    "INCREMENTAL_CONDITIONING_RESULTS.json",
    "LEAVE_ONE_FACTOR_OUT_RESULTS.json",
    "HISTORY_INCREMENT_RESULTS.json",
    "RESPONSE_TYPE_CONDITIONING.json",
    "UNCHANGED_TARGET_ANALYSIS.json",
    "IDENTITY_RESIDUAL_RESULTS.json",
    "SURVIVING_CONDITIONAL_COUNTEREXAMPLES.json",
    "ROBUSTNESS_AND_SENSITIVITY.json",
    "SCIENTIFIC_ASSESSMENT.md",
    "PRETARGET_FACTOR_RECORDS.jsonl.gz",
    "MATCH_LEDGER.jsonl.gz",
    "MATCH_LEDGER_MANIFEST.json",
    "MANIFEST.json",
)


def _read_jsonl_gzip(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_factor_analysis(
    *,
    analysis_root: Path,
    response_root: Path,
    output_name: str = "VALIDATION.json",
) -> dict[str, Any]:
    checks = 0
    for name in REQUIRED_ANALYSIS_FILES:
        require((analysis_root / name).is_file(), f"REQUIRED_FILE_MISSING:{name}")
        checks += 1
    freeze = read_json(analysis_root / "ANALYSIS_FREEZE.json")
    manifest = read_json(analysis_root / "MANIFEST.json")
    availability = read_json(analysis_root / "PRETARGET_FEATURE_AVAILABILITY.json")
    boundary = read_json(analysis_root / "BOUNDARY_VIOLATION.json")
    require(freeze["status"] == "FROZEN_BEFORE_FACTOR_RESULTS", "FREEZE_STATUS_INVALID")
    require(freeze["global_unseen_entry_accessed"] is False, "FREEZE_UNSEEN_FLAG_INVALID")
    require(manifest["global_unseen_entry_accessed"] is True, "MANIFEST_DISCLOSED_UNSEEN_ACCESS_MISSING")
    require(boundary["strict_unseen_claim_remains_valid"] is False, "BOUNDARY_VIOLATION_NOT_DISCLOSED")
    require(boundary["used_in_factor_analysis"] is False, "BOUNDARY_ENTRY_USED_IN_ANALYSIS")
    require(manifest["prediction_model_trained"] is False, "PREDICTION_MODEL_WAS_TRAINED")
    require(availability["global_unseen_entry_accessed"] is False, "AVAILABILITY_UNSEEN_FLAG_INVALID")
    checks += 7
    for name, expected in freeze["source_hashes"].items():
        require(file_sha256(response_root / name) == expected, f"SOURCE_HASH_MISMATCH:{name}")
        checks += 1

    records = _read_jsonl_gzip(analysis_root / "PRETARGET_FACTOR_RECORDS.jsonl.gz")
    require(len(records) == 15264, "RECORD_COUNT_INVALID")
    by_id = {row["record_id"]: row for row in records}
    require(len(by_id) == len(records), "RECORD_ID_NOT_UNIQUE")
    entries = {row["entry_id"] for row in records}
    require(len(entries) == 12 and GLOBAL_UNSEEN_ENTRY not in entries, "ENTRY_SET_INVALID")
    f7_count = sum(row["features"]["F7"]["categorical"].get("availability") == "AVAILABLE" for row in records)
    require(f7_count == 7632, "F7_COUNT_INVALID")
    checks += 4
    forbidden_feature_fragments = ("current_alpha_positive", "alpha_0.125", "alpha_0.25", "alpha_0.5", "alpha_0.75", "alpha_1")
    for row in records:
        require(row["entry_id"] != GLOBAL_UNSEEN_ENTRY, "GLOBAL_UNSEEN_RECORD")
        for block, values in row["features"].items():
            if block == "F7":
                continue
            names = [*values["numeric"], *values["categorical"]]
            require(not any(fragment in name for name in names for fragment in forbidden_feature_fragments), f"CURRENT_PROBE_FEATURE_PRESENT:{row['record_id']}:{block}")
        checks += 1

    identity = read_json(response_root / "IDENTITY_MATERIAL.json")
    index_by_entry = {
        entry: {row["evaluation_unit_id"]: int(row["element_index_audit_only"]) for row in rows}
        for entry, rows in identity["entries"].items()
    }
    section_cache: dict[str, dict[str, np.ndarray]] = {}
    spot = sorted(records, key=lambda row: row["record_id"])[:256]
    for row in spot:
        section_id = row["section_id"]
        if section_id not in section_cache:
            with np.load(response_root / "sections" / f"{section_id}.npz", allow_pickle=False) as data:
                section_cache[section_id] = {
                    "margins": np.asarray(data["all_margins"], dtype=np.float64),
                    "groups": np.asarray(data["groups"], dtype=np.int64),
                }
        index = index_by_entry[row["entry_id"]][row["evaluation_unit_id"]]
        source = section_cache[section_id]
        expected_curve = source["margins"][:, 0, index]
        require(np.array_equal(expected_curve, np.asarray(row["response"]["margin_curve"], dtype=np.float64)), f"RESPONSE_CURVE_REPLAY_MISMATCH:{row['record_id']}")
        require(int(source["groups"][index]) == int(row["target_group"]), f"TARGET_GROUP_REPLAY_MISMATCH:{row['record_id']}")
        require(float(expected_curve[1]) == float(row["features"]["F1"]["numeric"]["margin"]), f"PRETARGET_MARGIN_REPLAY_MISMATCH:{row['record_id']}")
        checks += 3

    match_rows = _read_jsonl_gzip(analysis_root / "MATCH_LEDGER.jsonl.gz")
    match_manifest = read_json(analysis_root / "MATCH_LEDGER_MANIFEST.json")
    require(file_sha256(analysis_root / "MATCH_LEDGER.jsonl.gz") == match_manifest["sha256"], "MATCH_LEDGER_HASH_MISMATCH")
    require(len(match_rows) == match_manifest["row_count"], "MATCH_LEDGER_COUNT_MISMATCH")
    checks += 2
    config_shape: dict[str, list[float]] = {}
    for row in match_rows:
        query = by_id[row["query_record_id"]]
        require(query["entry_id"] == row["query_entry_id"], "MATCH_QUERY_ENTRY_MISMATCH")
        for reference_id, reference_entry in zip(row["reference_record_ids"], row["reference_entry_ids"], strict=True):
            reference = by_id[reference_id]
            require(reference["entry_id"] == reference_entry, "MATCH_REFERENCE_ENTRY_MISMATCH")
            require(reference["entry_id"] != query["entry_id"], "WITHIN_RUN_MATCH_PRESENT")
            if row["configuration"] != "C0_random_cross_run" and row["configuration"] != "full_current_different_identity":
                require(reference["semantic_target_key"] == query["semantic_target_key"], f"SEMANTIC_MATCH_VIOLATION:{row['configuration']}")
        require(all(float(value) >= 0 for value in row["distances"]), "NEGATIVE_MATCH_DISTANCE")
        shape = row["metrics"].get("normalized_shape_correlation")
        if shape is not None:
            config_shape.setdefault(row["configuration"], []).append(float(shape))
        checks += 4

    lofo = read_json(analysis_root / "LEAVE_ONE_FACTOR_OUT_RESULTS.json")
    full_name = lofo["full_current_summary"]["name"]
    expected = lofo["full_current_summary"]["metrics"]["normalized_shape_correlation"]["estimate"]
    actual = float(np.mean(config_shape[full_name]))
    require(math.isclose(float(expected), actual, rel_tol=0, abs_tol=1e-12), "FULL_CURRENT_SHAPE_SUMMARY_MISMATCH")
    checks += 1

    result = {
        "schema": "nanogpt-response-factor-analysis-independent-validation-v1",
        "status": "PASS_WITH_DISCLOSED_BOUNDARY_VIOLATION",
        "check_count": checks,
        "record_count": len(records),
        "match_row_count": len(match_rows),
        "spot_replay_count": len(spot),
        "entry_count": len(entries),
        "prior_curve_record_count": f7_count,
        "global_unseen_entry_accessed": True,
        "global_unseen_entry_used_in_factor_records": False,
        "current_step_alpha_positive_used_as_condition": False,
        "prediction_model_trained": False,
        "manifest_sha256": file_sha256(analysis_root / "MANIFEST.json"),
    }
    write_json(analysis_root / output_name, result)
    return result


__all__ = ["validate_factor_analysis"]
