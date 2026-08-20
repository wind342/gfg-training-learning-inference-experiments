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
    binary_recalls: list[float] = []
    for label in (False, True):
        indices = [index for index, value in enumerate(truth) if value == label]
        if indices:
            binary_recalls.append(
                sum(predicted[index] == label for index in indices) / len(indices)
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
            "summary": json.loads(
                (path.parent / "RUN_SUMMARY.json").read_text(encoding="utf-8")
            ),
            "exchanges": json.loads(
                (path.parent / "EXCHANGES.json").read_text(encoding="utf-8")
            ),
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


def _knn_predict(
    train: list[dict[str, Any]], test: list[dict[str, Any]], model: str
) -> list[bool]:
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
    permutation = torch.randperm(
        len(rows), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    start, stop = rows[0]["feature_families"][family]
    result = []
    for index, row in enumerate(rows):
        copied = {**row, "features": list(row["features"])}
        copied["features"][start:stop] = rows[permutation[index]]["features"][start:stop]
        result.append(copied)
    return result


def _permuted_training_outcomes(
    rows: list[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    permutation = torch.randperm(
        len(rows), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    deltas = [row["post_margin"] - row["pre_margin"] for row in rows]
    result = []
    for index, row in enumerate(rows):
        copied = dict(row)
        copied["post_margin"] = row["pre_margin"] + deltas[permutation[index]]
        result.append(copied)
    return result


def aggregate(root: Path) -> dict[str, Any]:
    runs = _load(root)
    all_rows = [row for run in runs.values() for row in run["rows"]]
    direct = {
        method: metrics(all_rows, [bool(row["predictions"][method]) for row in all_rows])
        for method in ("unchanged", "linear", "quadratic")
    }
    pooled: dict[str, list[tuple[dict[str, Any], bool]]] = defaultdict(list)
    pooled_controls: dict[str, list[tuple[dict[str, Any], bool]]] = defaultdict(list)
    per_run: dict[str, dict[str, Any]] = {}
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
            per_run[str(test_seed)] = {}
            per_run_controls[str(test_seed)] = {}
            for model in ("F1", "F1_F3", "F1_F5", "F1_F3_F5"):
                prediction = _knn_predict(train, test, model)
                per_run[str(test_seed)][model] = metrics(test, prediction)
                pooled[model].extend(zip(test, prediction))
            controls = {
                "target_outcome_permutation": _knn_predict(
                    _permuted_training_outcomes(train, test_seed + 101),
                    test,
                    "F1_F3_F5",
                ),
                "F3_update_geometry_permutation": _knn_predict(
                    _permuted_family(train, "F3", test_seed + 201),
                    _permuted_family(test, "F3", test_seed + 202),
                    "F1_F3_F5",
                ),
                "F5_adamw_state_permutation": _knn_predict(
                    _permuted_family(train, "F5", test_seed + 301),
                    _permuted_family(test, "F5", test_seed + 302),
                    "F1_F3_F5",
                ),
            }
            for name, prediction in controls.items():
                per_run_controls[str(test_seed)][name] = metrics(test, prediction)
                pooled_controls[name].extend(zip(test, prediction))
    knn = {
        model: metrics([row for row, _ in pairs], [value for _, value in pairs])
        for model, pairs in pooled.items()
    }
    controls = {
        name: metrics([row for row, _ in pairs], [value for _, value in pairs])
        for name, pairs in pooled_controls.items()
    }
    integrity = {
        "all_gfg_valid": all(
            run["summary"]["integrity"]["gfg_validation"]["status"] == "PASS"
            for run in runs.values()
        ),
        "max_adamw_formula_error": max(
            run["summary"]["integrity"]["adamw_formula_max_abs_error"]
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
    morphology_counts = Counter(row["morphology"] for row in all_rows)
    nonlinear_count = sum(row["morphology"] != "NEAR_LINEAR" for row in all_rows)
    active = [
        row["support"]["distributed_active_component_count"] for row in all_rows
    ]
    reallocation = [
        row["support"]["reallocation_l1_normalized"] for row in all_rows
    ]
    switches = [row["support"]["primary_support_switch"] for row in all_rows]
    exchange_rows = [row for run in runs.values() for row in run["exchanges"]]
    state_nrmse = [row["receiving_state"]["response_nrmse"] for row in exchange_rows]
    memory_nrmse = [row["adam_memory"]["response_nrmse"] for row in exchange_rows]
    mechanism = {
        "response_count": len(all_rows),
        "nonlinear_count": nonlinear_count,
        "nonlinear_fraction": nonlinear_count / len(all_rows),
        "morphology_counts": dict(morphology_counts),
        "transition_counts": dict(Counter(row["transition"] for row in all_rows)),
        "distributed_support_fraction": sum(value >= 2 for value in active) / len(active),
        "support_reallocation_gt_0_01_fraction": sum(value > 0.01 for value in reallocation)
        / len(reallocation),
        "primary_support_switch_fraction": sum(bool(value) for value in switches)
        / len(switches),
        "receiving_state_exchange_nrmse": state_nrmse,
        "adamw_memory_exchange_nrmse": memory_nrmse,
    }
    integrity_pass = bool(
        integrity["all_gfg_valid"]
        and integrity["max_adamw_formula_error"] <= 2e-6
        and integrity["max_alpha0_error"] <= 1e-6
        and integrity["max_alpha1_error"] <= 1e-6
        and integrity["max_repeat_error"] <= 1e-7
    )
    state_pass = bool(state_nrmse) and sum(value > 0.05 for value in state_nrmse) / len(state_nrmse) >= 0.75
    memory_pass = bool(memory_nrmse) and sum(value > 0.05 for value in memory_nrmse) / len(memory_nrmse) >= 0.60
    nonlinear_pass = mechanism["nonlinear_fraction"] >= 0.10
    support_pass = (
        mechanism["distributed_support_fraction"] >= 0.50
        and mechanism["support_reallocation_gt_0_01_fraction"] >= 0.25
    )
    prediction_pass = False
    control_pass = False
    if knn:
        prediction_pass = (
            knn["F1_F3_F5"]["four_way_macro_recall"]
            >= knn["F1"]["four_way_macro_recall"] + 0.02
            and all(
                rows["F1_F3_F5"]["four_way_macro_recall"]
                > rows["F1"]["four_way_macro_recall"]
                for rows in per_run.values()
            )
        )
        main_score = knn["F1_F3_F5"]["four_way_macro_recall"]
        control_pass = all(
            main_score > row["four_way_macro_recall"] for row in controls.values()
        )
    tests = {
        "integrity": integrity_pass,
        "receiving_parameter_state": state_pass,
        "receiving_optimizer_memory": memory_pass,
        "finite_amplitude_nonlinearity": nonlinear_pass,
        "distributed_support_reorganization": support_pass,
        "held_out_coordinate_prediction": prediction_pass,
        "negative_controls": control_pass,
    }
    if not integrity_pass:
        verdict = "INTEGRITY_FAILURE"
    elif all(tests.values()):
        verdict = "CROSS_SYSTEM_GENERALIZATION_SUPPORTED"
    elif sum(tests.values()) >= 5:
        verdict = "CROSS_SYSTEM_GENERALIZATION_PARTIALLY_SUPPORTED"
    else:
        verdict = "CROSS_SYSTEM_GENERALIZATION_NOT_SUPPORTED"
    return {
        "schema": "gfg-ddpm-cifar-generalization-results-v1",
        "verdict": verdict,
        "run_count": len(runs),
        "seeds": sorted(runs),
        "integrity": integrity,
        "tests": tests,
        "mechanism": mechanism,
        "direct_boundary_prediction": direct,
        "held_out_knn_coordinate_ablation": knn,
        "per_run_knn": per_run,
        "negative_controls": controls,
        "per_run_negative_controls": per_run_controls,
        "final_test_epsilon_mse": {
            str(seed): run["summary"]["final_test"]["epsilon_mse"]
            for seed, run in runs.items()
        },
    }


def _assessment(result: dict[str, Any]) -> str:
    lines = [
        "# Scientific assessment",
        "",
        f"**Verdict: `{result['verdict']}`.**",
        "",
        "This experiment changes the studied system to a generative diffusion objective with a time-conditioned U-Net and AdamW. The readout target is an identified image--timestep--noise occurrence, not a class label.",
        "",
        "## Frozen tests",
        "",
    ]
    for name, passed in result["tests"].items():
        lines.append(f"- {name}: `{'PASS' if passed else 'FAIL'}`")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A supported result transports the tested training--learning relations to this executed diffusion system. It does not by itself prove an unconditional law for every neural architecture, optimizer or data-generating process. All counterexamples and failed frozen tests remain in the result package.",
            "",
        ]
    )
    return "\n".join(lines)


def _final_manifest(root: Path) -> dict[str, Any]:
    excluded = {"AGGREGATE_STDOUT.txt", "FINAL_MANIFEST.json"}
    files = {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path.relative_to(root).as_posix() not in excluded
    }
    return {"schema": "gfg-ddpm-cifar-final-manifest-v1", "files": files}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    result = aggregate(root)
    write_json(root / "GENERALIZATION_RESULTS.json", result)
    (root / "SCIENTIFIC_ASSESSMENT.md").write_text(
        _assessment(result), encoding="utf-8", newline="\n"
    )
    write_json(root / "FINAL_MANIFEST.json", _final_manifest(root))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"] != "INTEGRITY_FAILURE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
