from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from .numeric import sha256_file, write_json


PACKAGE = Path(__file__).resolve().parent
CONTRACT = json.loads((PACKAGE / "MODEL_CONTRACT.json").read_text(encoding="utf-8"))
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
    binary_recalls: list[float] = []
    for label in (False, True):
        indices = [index for index, value in enumerate(truth) if value == label]
        if indices:
            binary_recalls.append(sum(predicted[index] == label for index in indices) / len(indices))
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
        "balanced_accuracy": sum(binary_recalls) / len(binary_recalls),
        "four_way_macro_recall": sum(observed) / len(observed),
        "transition_recall": recalls,
        "transition_counts": dict(Counter(row["transition"] for row in rows)),
    }


def _load(root: Path) -> dict[int, dict[str, Any]]:
    runs: dict[int, dict[str, Any]] = {}
    for path in sorted(root.glob("seed_*/EVENTS.json")):
        seed = int(path.parent.name.removeprefix("seed_"))
        events = json.loads(path.read_text(encoding="utf-8"))
        rows = [
            {"seed": seed, "event": event["event_index"], **row}
            for event in events
            for row in event["analysis"]["records"]
        ]
        runs[seed] = {
            "rows": rows,
            "events": events,
            "summary": json.loads((path.parent / "RUN_SUMMARY.json").read_text(encoding="utf-8")),
            "exchanges": json.loads((path.parent / "EXCHANGES.json").read_text(encoding="utf-8")),
            "inference": json.loads((path.parent / "INFERENCE_RESULTS.json").read_text(encoding="utf-8")),
        }
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


def _knn_predict(train: list[dict[str, Any]], test: list[dict[str, Any]], model: str) -> list[bool]:
    indices = _family_indices(train[0], model)
    train_x = torch.tensor([[row["features"][index] for index in indices] for row in train], dtype=torch.float64)
    test_x = torch.tensor([[row["features"][index] for index in indices] for row in test], dtype=torch.float64)
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0).clamp_min(1e-12)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std
    distances = torch.cdist(test_x, train_x)
    k = min(int(CONTRACT["evaluation"]["knn_neighbors"]), len(train))
    nearest_distance, nearest_index = distances.topk(k, largest=False, dim=1)
    deltas = torch.tensor([row["post_margin"] - row["pre_margin"] for row in train], dtype=torch.float64)
    weights = 1.0 / nearest_distance.clamp_min(1e-9)
    predicted_delta = (weights * deltas[nearest_index]).sum(dim=1) / weights.sum(dim=1)
    pre_margin = torch.tensor([row["pre_margin"] for row in test], dtype=torch.float64)
    return (pre_margin + predicted_delta >= 0).tolist()


def _permuted_family(rows: list[dict[str, Any]], family: str, seed: int) -> list[dict[str, Any]]:
    permutation = torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed)).tolist()
    start, stop = rows[0]["feature_families"][family]
    result = []
    for index, row in enumerate(rows):
        copied = {**row, "features": list(row["features"])}
        copied["features"][start:stop] = rows[permutation[index]]["features"][start:stop]
        result.append(copied)
    return result


def _permuted_outcomes(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    permutation = torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed)).tolist()
    deltas = [row["post_margin"] - row["pre_margin"] for row in rows]
    return [
        {**row, "post_margin": row["pre_margin"] + deltas[permutation[index]]}
        for index, row in enumerate(rows)
    ]


def aggregate(root: Path) -> dict[str, Any]:
    runs = _load(root)
    rows = [row for run in runs.values() for row in run["rows"]]
    direct = {
        name: metrics(rows, [bool(row["predictions"][name]) for row in rows])
        for name in ("unchanged", "linear", "quadratic")
    }
    combined: dict[str, list[tuple[dict[str, Any], bool]]] = defaultdict(list)
    controls: dict[str, list[tuple[dict[str, Any], bool]]] = defaultdict(list)
    per_seed: dict[str, Any] = {}
    if len(runs) >= 3:
        for test_seed, test_run in runs.items():
            train = [row for seed, run in runs.items() if seed != test_seed for row in run["rows"]]
            test = test_run["rows"]
            per_seed[str(test_seed)] = {"models": {}, "controls": {}}
            for name in ("F1", "F1_F3", "F1_F5", "F1_F3_F5"):
                prediction = _knn_predict(train, test, name)
                per_seed[str(test_seed)]["models"][name] = metrics(test, prediction)
                combined[name].extend(zip(test, prediction))
            candidates = {
                "target_outcome_permutation": _knn_predict(_permuted_outcomes(train, test_seed + 101), test, "F1_F3_F5"),
                "F3_update_geometry_permutation": _knn_predict(_permuted_family(train, "F3", test_seed + 201), _permuted_family(test, "F3", test_seed + 202), "F1_F3_F5"),
                "F5_adam_state_permutation": _knn_predict(_permuted_family(train, "F5", test_seed + 301), _permuted_family(test, "F5", test_seed + 302), "F1_F3_F5"),
            }
            for name, prediction in candidates.items():
                per_seed[str(test_seed)]["controls"][name] = metrics(test, prediction)
                controls[name].extend(zip(test, prediction))
    knn = {
        name: metrics([row for row, _ in pairs], [prediction for _, prediction in pairs])
        for name, pairs in combined.items()
    }
    negative_controls = {
        name: metrics([row for row, _ in pairs], [prediction for _, prediction in pairs])
        for name, pairs in controls.items()
    }
    integrity = {
        "all_gfg_valid": all(run["summary"]["integrity"]["gfg_validation"]["status"] == "PASS" for run in runs.values()),
        "max_adamw_formula_error": max(run["summary"]["integrity"]["adamw_formula_max_abs_error"] for run in runs.values()),
        "max_alpha0_error": max(run["summary"]["integrity"]["alpha0_max_abs_error"] for run in runs.values()),
        "max_alpha1_error": max(run["summary"]["integrity"]["alpha1_native_max_abs_error"] for run in runs.values()),
        "max_repeat_error": max(run["summary"]["integrity"]["support_repeat_max_abs_error"] for run in runs.values()),
        "all_persistent_states_frozen": all(run["summary"]["integrity"]["persistent_model_state_frozen"] and run["summary"]["integrity"]["persistent_optimizer_state_frozen"] for run in runs.values()),
    }
    nonlinear = [
        row for row in rows
        if row["normalized_chord_deviation"] > float(CONTRACT["evaluation"]["nonlinear_normalized_chord_deviation"])
        and row["max_chord_deviation"] > float(CONTRACT["evaluation"]["nonlinear_absolute_chord_deviation"])
    ]
    support = [event["analysis"]["support"] for run in runs.values() for event in run["events"]]
    active = [value for item in support for value in item["distributed_active_component_count"]]
    reallocations = [value for item in support for value in item["reallocation_l1_normalized"]]
    switches = [value for item in support for value in item["primary_support_switch"]]
    routing = [event["analysis"]["routing"] for run in runs.values() for event in run["events"]]
    exchange_rows = [item for run in runs.values() for item in run["exchanges"]]
    state_nrmse = [
        value["response_nrmse"]
        for item in exchange_rows
        for name, value in item.items()
        if name.startswith("update_")
    ]
    memory_nrmse = [item["adam_receiving_state_exchange"]["response_nrmse"] for item in exchange_rows]
    mechanism = {
        "response_count": len(rows),
        "nonlinear_count": len(nonlinear),
        "nonlinear_fraction": len(nonlinear) / len(rows),
        "morphology_counts": dict(Counter(row["morphology"] for row in rows)),
        "distributed_support_fraction": sum(value >= 2 for value in active) / len(active),
        "support_reallocation_gt_threshold_fraction": sum(value > float(CONTRACT["frozen_tests"]["support_reallocation_threshold"]) for value in reallocations) / len(reallocations),
        "primary_support_switch_fraction": sum(bool(value) for value in switches) / len(switches),
        "receiving_state_exchange_nrmse": state_nrmse,
        "adam_memory_exchange_nrmse": memory_nrmse,
        "routing": {
            "all_top2": all(item["minimum_nonzero_experts_per_token"] == 2 and item["maximum_nonzero_experts_per_token"] == 2 for item in routing),
            "maximum_weight_sum_error": max(item["maximum_weight_sum_error"] for item in routing),
            "maximum_query_profile_l1": max(item["maximum_query_profile_l1"] for item in routing),
            "mean_profile_reallocation_l1": sum(item["mean_profile_reallocation_l1"] for item in routing) / len(routing),
            "mean_top2_identity_change_fraction": sum(item["top2_identity_change_fraction"] for item in routing) / len(routing),
        },
    }
    threshold = CONTRACT["frozen_tests"]
    integrity_pass = (
        integrity["all_gfg_valid"]
        and integrity["max_adamw_formula_error"] <= 1e-6
        and integrity["max_alpha0_error"] <= 1e-5
        and integrity["max_alpha1_error"] <= 1e-5
        and integrity["max_repeat_error"] <= 1e-6
        and integrity["all_persistent_states_frozen"]
    )
    state_pass = bool(state_nrmse) and sum(value > float(threshold["receiving_state_nrmse_minimum"]) for value in state_nrmse) / len(state_nrmse) >= float(threshold["receiving_state_passing_fraction"])
    memory_pass = bool(memory_nrmse) and sum(value > float(threshold["optimizer_memory_nrmse_minimum"]) for value in memory_nrmse) / len(memory_nrmse) >= float(threshold["optimizer_memory_passing_fraction"])
    prediction_pass = bool(knn) and knn["F1_F3_F5"]["four_way_macro_recall"] >= knn["F1"]["four_way_macro_recall"] + float(threshold["coordinate_prediction_macro_recall_gain_minimum"])
    tl_tests = {
        "integrity": integrity_pass,
        "receiving_parameter_state": state_pass,
        "receiving_optimizer_memory": memory_pass,
        "finite_amplitude_nonlinearity": mechanism["nonlinear_fraction"] >= float(threshold["nonlinear_response_fraction_minimum"]),
        "distributed_support_reorganization": mechanism["distributed_support_fraction"] >= float(threshold["distributed_support_fraction_minimum"]) and mechanism["support_reallocation_gt_threshold_fraction"] >= float(threshold["support_reallocation_fraction_minimum"]),
        "held_out_coordinate_prediction": prediction_pass,
        "routing_integrity": mechanism["routing"]["all_top2"] and mechanism["routing"]["maximum_weight_sum_error"] <= float(threshold["router_topk_tolerance"]),
    }
    if not integrity_pass:
        tl_verdict = "INTEGRITY_FAILURE"
    elif all(tl_tests.values()):
        tl_verdict = "MOE_TRAINING_LEARNING_GENERALIZATION_SUPPORTED"
    elif sum(tl_tests.values()) >= 5:
        tl_verdict = "MOE_TRAINING_LEARNING_GENERALIZATION_PARTIALLY_SUPPORTED"
    else:
        tl_verdict = "MOE_TRAINING_LEARNING_GENERALIZATION_NOT_SUPPORTED"
    inference_runs = {str(seed): run["inference"] for seed, run in runs.items()}
    inference_supported = all(item["pass"] for item in inference_runs.values())
    inference_verdict = "MOE_FROZEN_INFERENCE_PROJECTION_SUPPORTED" if inference_supported else "MOE_FROZEN_INFERENCE_PROJECTION_NOT_SUPPORTED"
    joint = (
        "MOE_UNIFIED_DYNAMICS_SUPPORTED"
        if tl_verdict == "MOE_TRAINING_LEARNING_GENERALIZATION_SUPPORTED" and inference_supported
        else "MOE_UNIFIED_DYNAMICS_NOT_FULLY_SUPPORTED"
    )
    return {
        "schema": "gfg-moe-cifar-generalization-results-v1",
        "joint_verdict": joint,
        "run_count": len(runs),
        "seeds": sorted(runs),
        "integrity": integrity,
        "training_learning": {"verdict": tl_verdict, "tests": tl_tests, "mechanism": mechanism},
        "inference": {"verdict": inference_verdict, "all_seeds_pass": inference_supported, "runs": inference_runs},
        "direct_boundary_prediction": direct,
        "held_out_knn_coordinate_ablation": knn,
        "per_seed_held_out": per_seed,
        "negative_controls": negative_controls,
        "final_test_accuracy": {str(seed): run["summary"]["final_test"]["accuracy"] for seed, run in runs.items()},
    }


def _results_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Formal results",
        "",
        f"**Joint verdict: `{result['joint_verdict']}`.**",
        "",
        f"- TL-G03: `{result['training_learning']['verdict']}`",
        f"- INF-G02: `{result['inference']['verdict']}`",
        "",
        "## Frozen TL-G03 tests",
        "",
    ]
    for name, passed in result["training_learning"]["tests"].items():
        lines.append(f"- {name}: `{'PASS' if passed else 'FAIL'}`")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This is a falsification-first validation in the executed deterministic top-2 Vision-MoE/CIFAR-100/AdamW system. It changes architecture and adds routed expert organization; it does not prove a claim about every MoE implementation or every neural system. Failed frozen tests remain counterevidence.",
            "",
        ]
    )
    return "\n".join(lines)


def _final_manifest(root: Path) -> dict[str, Any]:
    excluded = {"FINAL_MANIFEST.json", "AGGREGATE_STDOUT.txt"}
    files = {
        path.relative_to(root).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path.relative_to(root).as_posix() not in excluded
    }
    return {"schema": "gfg-moe-cifar-final-manifest-v1", "files": files}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    result = aggregate(root)
    write_json(root / "GENERALIZATION_RESULTS.json", result)
    (root / "RESULTS.md").write_text(_results_markdown(result), encoding="utf-8", newline="\n")
    write_json(root / "FINAL_MANIFEST.json", _final_manifest(root))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result["training_learning"]["verdict"] == "INTEGRITY_FAILURE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
