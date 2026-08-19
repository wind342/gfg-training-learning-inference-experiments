from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.gfg_nanogpt_state_conditioned_response_v1.dataset import build_dataset, load_records
from experiments.gfg_nanogpt_state_conditioned_response_v1.metrics import evaluate_by_run
from experiments.gfg_nanogpt_state_conditioned_response_v1.model import RandomFeatureCurveRegressor, file_sha256, require, stable_seed
from experiments.gfg_nanogpt_state_conditioned_response_v1.runner import DEFAULT_OUTPUT, SOURCE_ROOT, write_json


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_prediction_ledger(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def check(output_root: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    manifest = read_json(output_root / "MANIFEST.json")
    checks: list[str] = []
    for name, expected in manifest["deliverables"].items():
        path = output_root / name
        require(path.is_file(), f"DELIVERABLE_MISSING:{name}")
        require(file_sha256(path) == expected["sha256"], f"DELIVERABLE_HASH_MISMATCH:{name}")
        checks.append(f"hash:{name}")

    feature_manifest = read_json(output_root / "FEATURE_MANIFEST.json")
    require(feature_manifest["current_step_functional_probe_used"] is False, "FUNCTIONAL_PROBE_INPUT_VIOLATION")
    require(feature_manifest["future_result_used"] is False, "FUTURE_INPUT_VIOLATION")
    require(feature_manifest["record_count"] == 15264, "RECORD_COUNT_MISMATCH")
    checks.extend(["no_functional_probe_input", "no_future_input", "record_count"])

    dataset = build_dataset(load_records(SOURCE_ROOT / "PRETARGET_FACTOR_RECORDS.jsonl.gz"))
    ledger = read_prediction_ledger(output_root / "RESPONSE_CURVE_PREDICTIONS.jsonl.gz")
    require(len(ledger) == len(dataset["records"]), "PREDICTION_LEDGER_COUNT_MISMATCH")
    ledger_ids = [row["record_id"] for row in ledger]
    require(ledger_ids == [row["record_id"] for row in dataset["records"]], "PREDICTION_LEDGER_ORDER_MISMATCH")
    predictions = np.asarray([row["predictions"]["M4"]["displacement"] for row in ledger], dtype=np.float64)
    normalizers = np.asarray([row["outer_normalization_scale"] for row in ledger], dtype=np.float64)
    recomputed = evaluate_by_run(
        dataset["displacement"], predictions, dataset["margin0"], normalizers, dataset["entries"]
    )
    reported = read_json(output_root / "NONLINEAR_RESPONSE_MODEL_RESULTS.json")["primary_metrics"]
    for path in (
        ("overall", "displacement", "mae"),
        ("overall", "displacement", "rmse"),
        ("overall", "displacement", "nrmse"),
        ("overall", "normalized_shape", "rmse"),
        ("overall", "endpoint", "rmse"),
        ("overall", "unchanged_target", "false_crossing_rate"),
    ):
        actual: Any = recomputed
        expected: Any = reported
        for key in path:
            actual = actual[key]
            expected = expected[key]
        require(abs(float(actual) - float(expected)) <= 1e-12, f"METRIC_REPLAY_MISMATCH:{'.'.join(path)}")
        checks.append(f"metric:{'.'.join(path)}")

    splits = read_json(output_root / "TRAINING_SPLIT_MANIFEST.json")
    first_fold = splits["outer_folds"][0]
    held = first_fold["outer_entry"]
    scheme = first_fold["selected_scheme"]
    test = dataset["entries"] == held
    train = ~test
    model = RandomFeatureCurveRegressor(
        scheme=scheme,
        seed=stable_seed(f"outer:{held}:M4:{scheme}"),
        feature_names=dataset["feature_names"]["M4"],
    ).fit(dataset["matrices"]["M4"][train], dataset["displacement"][train])
    replay = model.predict(dataset["matrices"]["M4"][test])
    recorded = predictions[test]
    maximum_difference = float(np.max(np.abs(replay - recorded)))
    require(maximum_difference <= 1e-12, f"OUTER_FOLD_REPLAY_MISMATCH:{maximum_difference}")
    checks.append("first_outer_fold_exact_refit")

    final_model = RandomFeatureCurveRegressor.load(
        output_root / "FINAL_M4_MODEL.npz", output_root / "FINAL_M4_MODEL_METADATA.json"
    )
    final_prediction = final_model.predict(dataset["matrices"]["M4"][:256])
    require(final_prediction.shape == (256, 5), "FINAL_MODEL_OUTPUT_SHAPE_MISMATCH")
    require(np.all(np.isfinite(final_prediction)), "FINAL_MODEL_NONFINITE_OUTPUT")
    checks.append("final_model_load_and_predict")

    result = {
        "schema": "nanogpt-state-conditioned-response-independent-check-v1",
        "status": "PASS",
        "check_count": len(checks),
        "checks": checks,
        "first_outer_fold": held,
        "first_outer_fold_max_abs_replay_difference": maximum_difference,
        "strict_global_unseen_claim_valid": False,
    }
    write_json(output_root / "INDEPENDENT_CHECK.json", result)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    value = check(arguments.output_root)
    print(json.dumps({"status": value["status"], "check_count": value["check_count"]}, sort_keys=True))
