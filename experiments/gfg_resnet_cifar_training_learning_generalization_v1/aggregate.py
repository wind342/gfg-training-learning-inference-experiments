from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from .numeric import sha256_file, write_json


TRANSITIONS = (
    "MAINTAIN_CORRECT",
    "CORRECT_TO_WRONG",
    "MAINTAIN_WRONG",
    "WRONG_TO_CORRECT",
)


def _transition(before: bool, after: bool) -> str:
    if before and after:
        return "MAINTAIN_CORRECT"
    if before and not after:
        return "CORRECT_TO_WRONG"
    if not before and after:
        return "WRONG_TO_CORRECT"
    return "MAINTAIN_WRONG"


def metrics(rows: list[dict[str, Any]], predicted: list[bool]) -> dict[str, Any]:
    truth = [bool(row["post_correct"]) for row in rows]
    correct = sum(left == right for left, right in zip(truth, predicted))
    binary_recalls = []
    for label in (False, True):
        indices = [index for index, value in enumerate(truth) if value == label]
        binary_recalls.append(
            sum(predicted[index] == label for index in indices) / len(indices)
            if indices
            else None
        )
    predicted_transitions = [
        _transition(bool(row["pre_correct"]), value)
        for row, value in zip(rows, predicted)
    ]
    recalls: dict[str, float | None] = {}
    for label in TRANSITIONS:
        indices = [index for index, row in enumerate(rows) if row["transition"] == label]
        recalls[label] = (
            sum(predicted_transitions[index] == label for index in indices) / len(indices)
            if indices
            else None
        )
    observed = [value for value in recalls.values() if value is not None]
    return {
        "count": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "balanced_accuracy": sum(value for value in binary_recalls if value is not None)
        / sum(value is not None for value in binary_recalls),
        "four_way_macro_recall": sum(observed) / len(observed),
        "transition_recall": recalls,
        "transition_counts": dict(Counter(row["transition"] for row in rows)),
    }


def _load(root: Path):
    runs: dict[int, dict[str, Any]] = {}
    for path in sorted(root.glob("seed_*/EVENTS.json")):
        seed = int(path.parent.name.removeprefix("seed_"))
        events = json.loads(path.read_text(encoding="utf-8"))
        rows = []
        for event in events:
            for row in event["analysis"]["records"]:
                rows.append({"seed": seed, "event": event["event_index"], **row})
        summary = json.loads((path.parent / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
        exchanges = json.loads((path.parent / "EXCHANGES.json").read_text(encoding="utf-8"))
        runs[seed] = {"rows": rows, "events": events, "summary": summary, "exchanges": exchanges}
    if not runs:
        raise RuntimeError("NO_RUNS_FOUND")
    return runs


def _family_indices(row: dict[str, Any], model: str) -> list[int]:
    spans = row["feature_families"]
    f1 = list(range(*spans["F1"]))
    f3 = list(range(*spans["F3"]))
    f5 = list(range(*spans["F5"]))
    return {
        "F1": f1,
        "F1_F3": f1 + f3,
        "F1_F5": f1 + f5,
        "F1_F3_F5": f1 + f3 + f5,
    }[model]


def _knn_predict(train: list[dict[str, Any]], test: list[dict[str, Any]], model: str):
    indices = _family_indices(train[0], model)
    train_x = torch.tensor(
        [[row["features"][index] for index in indices] for row in train],
        dtype=torch.float64,
    )
    test_x = torch.tensor(
        [[row["features"][index] for index in indices] for row in test],
        dtype=torch.float64,
    )
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0).clamp_min(1e-12)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std
    distances = torch.cdist(test_x, train_x)
    k = min(32, len(train))
    nearest_distance, nearest_index = distances.topk(k, largest=False, dim=1)
    delta = torch.tensor(
        [row["post_margin"] - row["pre_margin"] for row in train],
        dtype=torch.float64,
    )
    weights = 1.0 / nearest_distance.clamp_min(1e-9)
    predicted_delta = (weights * delta[nearest_index]).sum(dim=1) / weights.sum(dim=1)
    pre_margin = torch.tensor([row["pre_margin"] for row in test], dtype=torch.float64)
    return (pre_margin + predicted_delta >= 0).tolist()


def _permuted_family(
    rows: list[dict[str, Any]], family: str, seed: int
) -> list[dict[str, Any]]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(rows), generator=generator).tolist()
    start, stop = rows[0]["feature_families"][family]
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        copied = {**row, "features": list(row["features"])}
        copied["features"][start:stop] = rows[permutation[index]]["features"][
            start:stop
        ]
        result.append(copied)
    return result


def _permuted_training_outcomes(
    rows: list[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(rows), generator=generator).tolist()
    deltas = [row["post_margin"] - row["pre_margin"] for row in rows]
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        copied = dict(row)
        copied["post_margin"] = row["pre_margin"] + deltas[permutation[index]]
        result.append(copied)
    return result


def aggregate(root: Path) -> dict[str, Any]:
    runs = _load(root)
    all_rows = [row for run in runs.values() for row in run["rows"]]
    direct = {}
    for method in ("unchanged", "linear", "quadratic"):
        direct[method] = metrics(
            all_rows, [bool(row["predictions"][method]) for row in all_rows]
        )
    knn_predictions: dict[str, list[tuple[dict[str, Any], bool]]] = defaultdict(list)
    control_predictions: dict[str, list[tuple[dict[str, Any], bool]]] = defaultdict(list)
    per_run_knn: dict[str, dict[str, Any]] = {}
    per_run_controls: dict[str, dict[str, Any]] = {}
    if len(runs) >= 3:
        for test_seed, test_run in runs.items():
            train = [
                row
                for seed, run in runs.items()
                if seed != test_seed
                for row in run["rows"]
            ]
            test = test_run["rows"]
            per_run_knn[str(test_seed)] = {}
            per_run_controls[str(test_seed)] = {}
            for model in ("F1", "F1_F3", "F1_F5", "F1_F3_F5"):
                prediction = _knn_predict(train, test, model)
                per_run_knn[str(test_seed)][model] = metrics(test, prediction)
                knn_predictions[model].extend(zip(test, prediction))
            label_prediction = _knn_predict(
                _permuted_training_outcomes(train, test_seed + 101),
                test,
                "F1_F3_F5",
            )
            update_prediction = _knn_predict(
                _permuted_family(train, "F3", test_seed + 201),
                _permuted_family(test, "F3", test_seed + 202),
                "F1_F3_F5",
            )
            momentum_prediction = _knn_predict(
                _permuted_family(train, "F5", test_seed + 301),
                _permuted_family(test, "F5", test_seed + 302),
                "F1_F3_F5",
            )
            for control, prediction in (
                ("target_outcome_permutation", label_prediction),
                ("F3_update_geometry_permutation", update_prediction),
                ("F5_momentum_state_permutation", momentum_prediction),
            ):
                per_run_controls[str(test_seed)][control] = metrics(test, prediction)
                control_predictions[control].extend(zip(test, prediction))
    knn = {
        model: metrics(
            [row for row, _ in pairs], [prediction for _, prediction in pairs]
        )
        for model, pairs in knn_predictions.items()
    }
    negative_controls = {
        control: metrics(
            [row for row, _ in pairs], [prediction for _, prediction in pairs]
        )
        for control, pairs in control_predictions.items()
    }
    integrity = {
        "all_gfg_valid": all(
            run["summary"]["integrity"]["gfg_validation"]["status"] == "PASS"
            for run in runs.values()
        ),
        "max_sgd_formula_error": max(
            run["summary"]["integrity"]["sgd_formula_max_abs_error"]
            for run in runs.values()
        ),
        "max_alpha0_error": max(
            run["summary"]["integrity"]["alpha0_max_abs_error"]
            for run in runs.values()
        ),
        "max_alpha1_error": max(
            run["summary"]["integrity"]["alpha1_native_max_abs_error"]
            for run in runs.values()
        ),
        "max_repeat_error": max(
            run["summary"]["integrity"]["support_repeat_max_abs_error"]
            for run in runs.values()
        ),
    }
    response_rows = all_rows
    nonlinear = [
        row
        for row in response_rows
        if row["normalized_chord_deviation"] > 0.1
        and row["max_chord_deviation"] > 0.001
    ]
    morphologies = Counter(row["morphology"] for row in response_rows)
    support_records = [
        (event["analysis"]["records"], event["analysis"]["support"])
        for run in runs.values()
        for event in run["events"]
    ]
    active_counts = [
        value for _, support in support_records for value in support["distributed_active_component_count"]
    ]
    reallocations = [
        value for _, support in support_records for value in support["reallocation_l1_normalized"]
    ]
    switches = [
        value for _, support in support_records for value in support["primary_support_switch"]
    ]
    exchange_rows = [
        exchange for run in runs.values() for exchange in run["exchanges"]
    ]
    state_nrmse = [
        value["response_nrmse"]
        for exchange in exchange_rows
        for key, value in exchange.items()
        if key.startswith("update_")
    ]
    momentum_nrmse = [
        exchange["momentum_receiving_state_exchange"]["response_nrmse"]
        for exchange in exchange_rows
    ]
    mechanism = {
        "nonlinear_count": len(nonlinear),
        "response_count": len(response_rows),
        "nonlinear_fraction": len(nonlinear) / len(response_rows),
        "morphology_counts": dict(morphologies),
        "distributed_support_fraction": sum(value >= 2 for value in active_counts)
        / len(active_counts),
        "support_reallocation_gt_0_01_fraction": sum(value > 0.01 for value in reallocations)
        / len(reallocations),
        "primary_support_switch_fraction": sum(bool(value) for value in switches)
        / len(switches),
        "receiving_state_exchange_nrmse": state_nrmse,
        "momentum_exchange_nrmse": momentum_nrmse,
    }
    integrity_pass = (
        integrity["all_gfg_valid"]
        and integrity["max_sgd_formula_error"] <= 1e-6
        and integrity["max_alpha0_error"] <= 1e-5
        and integrity["max_alpha1_error"] <= 1e-5
        and integrity["max_repeat_error"] <= 1e-6
    )
    state_pass = bool(state_nrmse) and sum(value > 0.05 for value in state_nrmse) / len(state_nrmse) >= 0.75
    momentum_pass = bool(momentum_nrmse) and sum(value > 0.05 for value in momentum_nrmse) / len(momentum_nrmse) >= 0.75
    nonlinear_pass = mechanism["nonlinear_fraction"] >= 0.1
    support_pass = (
        mechanism["distributed_support_fraction"] >= 0.5
        and mechanism["support_reallocation_gt_0_01_fraction"] >= 0.1
    )
    prediction_pass = bool(knn) and (
        knn["F1_F3_F5"]["four_way_macro_recall"]
        >= knn["F1"]["four_way_macro_recall"] + 0.02
    )
    tests = {
        "integrity": integrity_pass,
        "receiving_parameter_state": state_pass,
        "receiving_optimizer_memory": momentum_pass,
        "finite_amplitude_nonlinearity": nonlinear_pass,
        "distributed_support_reorganization": support_pass,
        "held_out_coordinate_prediction": prediction_pass,
    }
    if not integrity_pass:
        verdict = "INTEGRITY_FAILURE"
    elif all(tests.values()):
        verdict = "CROSS_SYSTEM_GENERALIZATION_SUPPORTED"
    elif sum(tests.values()) >= 4:
        verdict = "CROSS_SYSTEM_GENERALIZATION_PARTIALLY_SUPPORTED"
    else:
        verdict = "CROSS_SYSTEM_GENERALIZATION_NOT_SUPPORTED"
    return {
        "schema": "gfg-resnet-cifar-generalization-results-v1",
        "verdict": verdict,
        "run_count": len(runs),
        "seeds": sorted(runs),
        "integrity": integrity,
        "tests": tests,
        "mechanism": mechanism,
        "direct_boundary_prediction": direct,
        "held_out_knn_coordinate_ablation": knn,
        "per_run_knn": per_run_knn,
        "negative_controls": negative_controls,
        "per_run_negative_controls": per_run_controls,
        "final_test_accuracy": {
            str(seed): run["summary"]["final_test"]["accuracy"]
            for seed, run in runs.items()
        },
    }


def _assessment(result: dict[str, Any]) -> str:
    tests = result["tests"]
    lines = [
        "# Scientific assessment",
        "",
        f"**Verdict: `{result['verdict']}`.**",
        "",
        "This experiment changes architecture, data modality, task and optimizer from nanoGPT/Adam/text to ResNet-18/SGD-momentum/images. It is a cross-system falsification test, not an unconditional universality proof.",
        "",
        "## Frozen tests",
        "",
    ]
    for name, passed in tests.items():
        lines.append(f"- {name}: `{'PASS' if passed else 'FAIL'}`")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A supported result establishes that the same formation coordinates and mechanism chain transport to this executed ResNet/CIFAR-100/SGD-momentum system. A partial or negative result localizes which relation failed without deleting counterexamples. No result from this experiment alone licenses a claim about every neural architecture or environment.",
            "",
            "The result file also reports the frozen target-outcome, F3 update-geometry and F5 momentum-state permutation controls. These controls are audits of identity and coordinate pairing; they do not replace the native causal exchanges.",
            "",
        ]
    )
    return "\n".join(lines)


def _final_manifest(root: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    excluded = {"AGGREGATE_STDOUT.txt", "FINAL_MANIFEST.json"}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "schema": "gfg-resnet-cifar-final-manifest-v1",
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = aggregate(args.root.resolve())
    write_json(args.root / "GENERALIZATION_RESULTS.json", result)
    (args.root / "SCIENTIFIC_ASSESSMENT.md").write_text(
        _assessment(result), encoding="utf-8", newline="\n"
    )
    write_json(args.root / "FINAL_MANIFEST.json", _final_manifest(args.root))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"] != "INTEGRITY_FAILURE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
