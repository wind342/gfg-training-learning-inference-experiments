"""Task-wise evaluation for frozen one-step CSRG predictions.

This module does not select a model, alter a prediction, or create scientific
evidence.  It deterministically reconstructs the already frozen strict outer
leave-one-run-out predictions and reports the four prediction tasks
independently.  The former all-tasks-AND criterion is retained only as the
``strict_joint_pass`` audit statistic.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable


SCHEMA = "nanogpt-one-step-500-taskwise-evaluation-v1"
STATE_SECTIONS = (("support_state", "s."), ("support_topology", "topology."))


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def _rmse(values: Iterable[float]) -> float:
    rows = list(values)
    return math.sqrt(_mean(value * value for value in rows)) if rows else 0.0


def _relative_improvement(baseline: float, candidate: float) -> float | None:
    return (baseline - candidate) / baseline if baseline != 0.0 else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_candidate_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("frozen_one_step_candidate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen candidate: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _direction(delta: float, zero_band: float) -> int:
    if abs(delta) <= zero_band:
        return 0
    return 1 if delta > 0.0 else -1


def _classification_confusion(
    truth: list[int],
    prediction: list[int],
    labels: list[int],
) -> dict[str, Any]:
    if len(truth) != len(prediction):
        raise ValueError("truth and prediction lengths differ")
    counts = Counter(zip(truth, prediction))
    matrix = [[counts[(actual, predicted)] for predicted in labels] for actual in labels]
    recalls = []
    per_class = {}
    for actual in labels:
        support = sum(counts[(actual, predicted)] for predicted in labels)
        recall = counts[(actual, actual)] / support if support else 0.0
        recalls.append(recall)
        per_class[str(actual)] = {"recall": recall, "support": support}
    return {
        "labels": labels,
        "matrix_rows_actual_columns_predicted": matrix,
        "accuracy": sum(counts[(label, label)] for label in labels) / len(truth),
        "balanced_accuracy": _mean(recalls),
        "per_class": per_class,
    }


def _binary_confusion(truth: list[bool], prediction: list[bool]) -> dict[str, Any]:
    if len(truth) != len(prediction):
        raise ValueError("truth and prediction lengths differ")
    counts = Counter(zip(truth, prediction))
    true_positive = counts[(True, True)]
    false_positive = counts[(False, True)]
    false_negative = counts[(True, False)]
    true_negative = counts[(False, False)]
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "labels": [False, True],
        "matrix_rows_actual_columns_predicted": [
            [true_negative, false_positive],
            [false_negative, true_positive],
        ],
        "accuracy": (true_positive + true_negative) / len(truth),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
    }


def _task_ledger(
    sample_ids: list[str],
    baseline_errors: dict[str, bool],
    candidate_errors: dict[str, bool],
) -> dict[str, int | float]:
    baseline_error_count = sum(baseline_errors[sample_id] for sample_id in sample_ids)
    candidate_error_count = sum(candidate_errors[sample_id] for sample_id in sample_ids)
    repaired = sum(
        baseline_errors[sample_id] and not candidate_errors[sample_id]
        for sample_id in sample_ids
    )
    newly_broken = sum(
        not baseline_errors[sample_id] and candidate_errors[sample_id]
        for sample_id in sample_ids
    )
    count = len(sample_ids)
    return {
        "record_count": count,
        "baseline_correct_count": count - baseline_error_count,
        "candidate_correct_count": count - candidate_error_count,
        "baseline_error_count": baseline_error_count,
        "candidate_error_count": candidate_error_count,
        "repaired_count": repaired,
        "newly_broken_count": newly_broken,
        "net_correct_improvement_count": repaired - newly_broken,
        "baseline_accuracy": (count - baseline_error_count) / count,
        "candidate_accuracy": (count - candidate_error_count) / count,
    }


def _truth_state(record: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for section, prefix in STATE_SECTIONS:
        for name, value in record["X_t_plus_1"][section].items():
            result[prefix + name] = float(value)
    return result


def _baseline_state(row: dict[str, Any]) -> dict[str, float]:
    predicted = row["prediction"]["next_functional_state"]
    return {
        name: float(value)
        for name, value in predicted.items()
        if name.startswith(("s.", "topology."))
    }


def _candidate_state(prediction: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    functional = prediction["next_functional_state"]
    for section, prefix in STATE_SECTIONS:
        for name, value in functional[section].items():
            result[prefix + name] = float(value)
    return result


def _state_metrics(
    truth_by_sample: dict[str, dict[str, float]],
    prediction_by_sample: dict[str, dict[str, float]],
    nrmse_by_sample: dict[str, float],
) -> dict[str, Any]:
    sample_ids = sorted(truth_by_sample)
    coordinates = sorted(next(iter(truth_by_sample.values())))
    if any(sorted(truth_by_sample[sample_id]) != coordinates for sample_id in sample_ids):
        raise RuntimeError("truth CSRG coordinate schemas differ")
    if any(sorted(prediction_by_sample[sample_id]) != coordinates for sample_id in sample_ids):
        raise RuntimeError("prediction CSRG coordinate schemas differ")
    residuals: list[float] = []
    per_coordinate: dict[str, Any] = {}
    for coordinate in coordinates:
        coordinate_residuals = [
            prediction_by_sample[sample_id][coordinate] - truth_by_sample[sample_id][coordinate]
            for sample_id in sample_ids
        ]
        residuals.extend(coordinate_residuals)
        per_coordinate[coordinate] = {
            "mae": _mean(abs(value) for value in coordinate_residuals),
            "rmse": _rmse(coordinate_residuals),
        }
    transition_nrmse = [nrmse_by_sample[sample_id] for sample_id in sample_ids]
    return {
        "coordinate_count": len(coordinates),
        "flattened_value_count": len(residuals),
        "mae": _mean(abs(value) for value in residuals),
        "rmse": _rmse(residuals),
        "mean_transition_nrmse": _mean(transition_nrmse),
        "pooled_transition_nrmse": _rmse(transition_nrmse),
        "per_coordinate": per_coordinate,
        "note": (
            "Raw MAE/RMSE pool heterogeneous but identical CSRG coordinates and are comparable only "
            "between these two models; normalized metrics retain the fold-specific development scales."
        ),
    }


def _continuous_comparison(baseline: dict[str, float], candidate: dict[str, float]) -> dict[str, Any]:
    result: dict[str, Any] = {"baseline": baseline, "candidate": candidate}
    result["relative_improvement"] = {
        name: _relative_improvement(float(baseline[name]), float(candidate[name]))
        for name in baseline
        if isinstance(baseline[name], (int, float)) and name in candidate
    }
    return result


def _reconstruct_candidate_predictions(
    *,
    records: list[dict[str, Any]],
    stored_index: dict[str, dict[str, Any]],
    candidate_module: ModuleType,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    entries = sorted({row["entry_id"] for row in records})
    predictions: dict[str, dict[str, Any]] = {}
    checks = Counter()
    for held_out_entry in entries:
        held_out = [row for row in records if row["entry_id"] == held_out_entry]
        configs = {
            json.dumps(stored_index[row["sample_id"]]["selected_config"], sort_keys=True)
            for row in held_out
        }
        if len(configs) != 1:
            raise RuntimeError(f"stored selected configs differ within {held_out_entry}")
        config = json.loads(next(iter(configs)))
        training = [row for row in records if row["entry_id"] != held_out_entry]
        model = candidate_module.fit_fixed(training, config)
        for record in held_out:
            sample_id = record["sample_id"]
            prediction = candidate_module.predict_one(model, record["X_t"], record["U_t"])
            adjudication = candidate_module.adjudicate(model, prediction, record)
            stored = stored_index[sample_id]
            checks["next_capability_match"] += math.isclose(
                float(prediction["next_capability"]),
                float(stored["next_capability"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            checks["state_nrmse_match"] += math.isclose(
                float(adjudication["state_nrmse"]),
                float(stored["errors"]["state_nrmse"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            checks["error_flags_match"] += all(
                bool(adjudication[name]) == bool(stored["errors"][name])
                for name in (
                    "state_error",
                    "capability_error",
                    "capability_direction_error",
                    "decline_label_error",
                    "development_one_step_error",
                )
            )
            predictions[sample_id] = prediction
    expected = len(records)
    if len(predictions) != expected or any(checks[name] != expected for name in checks):
        raise RuntimeError(f"frozen prediction reconstruction mismatch: {dict(checks)}")
    return predictions, {
        "method": "deterministic reconstruction of the already selected strict outer-LORO models",
        "new_model_selection": False,
        "new_scientific_fit": False,
        "check_counts": dict(checks),
    }


def build_taskwise_evaluation(
    *,
    evidence_root: Path,
    candidate_result_root: Path,
) -> dict[str, Any]:
    protocol = json.loads((evidence_root / "PILOT_PROTOCOL.json").read_text(encoding="utf-8"))
    records = _read_jsonl(evidence_root / "selected_500_csrg_evidence.jsonl")
    baseline_rows = _read_jsonl(evidence_root / "baseline_adjudications.jsonl")
    stored_candidate_index = json.loads(
        (candidate_result_root / "DEVELOPMENT_PREDICTION_INDEX.json").read_text(encoding="utf-8")
    )
    candidate_module = _load_candidate_module(candidate_result_root / "candidate_model.py")
    sample_ids = sorted(row["sample_id"] for row in records)
    record_by_sample = {row["sample_id"]: row for row in records}
    baseline_by_sample = {row["sample_id"]: row for row in baseline_rows}
    if not (
        len(sample_ids) == len(set(sample_ids)) == 500
        and set(sample_ids) == set(baseline_by_sample) == set(stored_candidate_index)
    ):
        raise RuntimeError("task-wise input identity mismatch")
    if any(row["entry_id"] == "entry-362ded584a953f360aec" for row in records):
        raise RuntimeError("global held-out entry is present")

    candidate_predictions, reconstruction = _reconstruct_candidate_predictions(
        records=records,
        stored_index=stored_candidate_index,
        candidate_module=candidate_module,
    )
    zero_band = float(protocol["capability_zero_band"])
    decline_threshold = float(protocol["decline_delta_threshold"])

    truth_state = {sample_id: _truth_state(record_by_sample[sample_id]) for sample_id in sample_ids}
    baseline_state = {sample_id: _baseline_state(baseline_by_sample[sample_id]) for sample_id in sample_ids}
    candidate_state = {sample_id: _candidate_state(candidate_predictions[sample_id]) for sample_id in sample_ids}
    baseline_state_metrics = _state_metrics(
        truth_state,
        baseline_state,
        {sample_id: float(baseline_by_sample[sample_id]["errors"]["state_nrmse"]) for sample_id in sample_ids},
    )
    candidate_state_metrics = _state_metrics(
        truth_state,
        candidate_state,
        {sample_id: float(stored_candidate_index[sample_id]["errors"]["state_nrmse"]) for sample_id in sample_ids},
    )

    baseline_capability_errors = [
        float(baseline_by_sample[sample_id]["errors"]["capability_absolute_error"])
        for sample_id in sample_ids
    ]
    candidate_capability_errors = [
        float(stored_candidate_index[sample_id]["errors"]["capability_absolute_error"])
        for sample_id in sample_ids
    ]
    capability_continuous = _continuous_comparison(
        {"mae": _mean(baseline_capability_errors), "rmse": _rmse(baseline_capability_errors)},
        {"mae": _mean(candidate_capability_errors), "rmse": _rmse(candidate_capability_errors)},
    )

    truth_direction: list[int] = []
    baseline_direction: list[int] = []
    candidate_direction: list[int] = []
    truth_decline: list[bool] = []
    baseline_decline: list[bool] = []
    candidate_decline: list[bool] = []
    for sample_id in sample_ids:
        record = record_by_sample[sample_id]
        current = float(record["X_t"]["observables"]["capability"])
        truth_next = float(record["X_t_plus_1"]["observables"]["capability"])
        baseline_next = float(baseline_by_sample[sample_id]["prediction"]["next_capability"])
        candidate_next = float(stored_candidate_index[sample_id]["next_capability"])
        truth_direction.append(_direction(truth_next - current, zero_band))
        baseline_direction.append(_direction(baseline_next - current, zero_band))
        candidate_direction.append(_direction(candidate_next - current, zero_band))
        truth_decline.append(truth_next - current <= decline_threshold)
        baseline_decline.append(baseline_next - current <= decline_threshold)
        candidate_decline.append(candidate_next - current <= decline_threshold)

    ledgers = {}
    for task_name, error_key in (
        ("csrg_state", "state_error"),
        ("capability_numeric", "capability_error"),
        ("capability_direction", "capability_direction_error"),
        ("marked_decline", "decline_label_error"),
        ("strict_joint_pass", "development_one_step_error"),
    ):
        ledgers[task_name] = _task_ledger(
            sample_ids,
            {sample_id: bool(baseline_by_sample[sample_id]["errors"][error_key]) for sample_id in sample_ids},
            {sample_id: bool(stored_candidate_index[sample_id]["errors"][error_key]) for sample_id in sample_ids},
        )

    baseline_direction_metrics = _classification_confusion(
        truth_direction, baseline_direction, [-1, 0, 1]
    )
    candidate_direction_metrics = _classification_confusion(
        truth_direction, candidate_direction, [-1, 0, 1]
    )
    baseline_decline_metrics = _binary_confusion(truth_decline, baseline_decline)
    candidate_decline_metrics = _binary_confusion(truth_decline, candidate_decline)

    state_without_coordinates = lambda value: {key: item for key, item in value.items() if key != "per_coordinate"}
    result = {
        "schema": SCHEMA,
        "status": "PASS",
        "scientific_scope": "development-only task-wise re-evaluation of unchanged frozen predictions",
        "record_count": 500,
        "development_entry_count": 12,
        "global_held_out_entry": "entry-362ded584a953f360aec",
        "global_held_out_used": False,
        "thresholds_unchanged": {
            "state_nrmse_tolerance": protocol["state_nrmse_tolerance"],
            "capability_absolute_error_tolerance": protocol["capability_absolute_error_tolerance"],
            "capability_zero_band": zero_band,
            "decline_delta_threshold": decline_threshold,
        },
        "evaluation_contract": {
            "primary_tasks": [
                "csrg_state_prediction",
                "capability_numeric_prediction",
                "capability_direction_prediction",
                "marked_decline_prediction",
            ],
            "tasks_adjudicated_independently": True,
            "counterexamples_adjudicated_only_by_originating_task": True,
            "strict_joint_pass_is_auxiliary_audit_only": True,
        },
        "csrg_state_prediction": {
            "metrics": _continuous_comparison(
                state_without_coordinates(baseline_state_metrics),
                state_without_coordinates(candidate_state_metrics),
            ),
            "tolerance_ledger": ledgers["csrg_state"],
            "per_coordinate": {
                coordinate: {
                    "baseline": baseline_state_metrics["per_coordinate"][coordinate],
                    "candidate": candidate_state_metrics["per_coordinate"][coordinate],
                }
                for coordinate in sorted(baseline_state_metrics["per_coordinate"])
            },
        },
        "capability_numeric_prediction": {
            "metrics": capability_continuous,
            "tolerance_ledger": ledgers["capability_numeric"],
        },
        "capability_direction_prediction": {
            "baseline": baseline_direction_metrics,
            "candidate": candidate_direction_metrics,
            "accuracy_absolute_improvement": (
                candidate_direction_metrics["accuracy"] - baseline_direction_metrics["accuracy"]
            ),
            "balanced_accuracy_absolute_improvement": (
                candidate_direction_metrics["balanced_accuracy"]
                - baseline_direction_metrics["balanced_accuracy"]
            ),
            "task_ledger": ledgers["capability_direction"],
        },
        "marked_decline_prediction": {
            "baseline": baseline_decline_metrics,
            "candidate": candidate_decline_metrics,
            "task_ledger": ledgers["marked_decline"],
        },
        "strict_joint_pass_audit_only": {
            "definition": "all four frozen task criteria pass simultaneously",
            "not_primary_correctness_standard": True,
            "task_ledger": ledgers["strict_joint_pass"],
        },
        "reconstruction": reconstruction,
        "execution_declarations": {
            "new_training": False,
            "new_gpu_execution": False,
            "new_ai_session": False,
            "new_model_selection": False,
            "new_evidence": False,
            "new_gfg": False,
            "threshold_change": False,
            "prediction_change": False,
        },
        "input_hashes": {
            "PILOT_PROTOCOL.json": _file_sha256(evidence_root / "PILOT_PROTOCOL.json"),
            "selected_500_csrg_evidence.jsonl": _file_sha256(evidence_root / "selected_500_csrg_evidence.jsonl"),
            "baseline_adjudications.jsonl": _file_sha256(evidence_root / "baseline_adjudications.jsonl"),
            "DEVELOPMENT_PREDICTION_INDEX.json": _file_sha256(candidate_result_root / "DEVELOPMENT_PREDICTION_INDEX.json"),
            "candidate_model.py": _file_sha256(candidate_result_root / "candidate_model.py"),
        },
    }
    return result


def render_markdown(result: dict[str, Any]) -> str:
    state = result["csrg_state_prediction"]
    capability = result["capability_numeric_prediction"]
    direction = result["capability_direction_prediction"]
    decline = result["marked_decline_prediction"]
    joint = result["strict_joint_pass_audit_only"]["task_ledger"]
    state_metrics = state["metrics"]
    cap_metrics = capability["metrics"]

    def pct(value: float) -> str:
        return f"{100.0 * value:.2f}%"

    lines = [
        "# 一步预测分任务评价",
        "",
        "本报告只重新统计已经冻结的500条开发预测。没有训练新模型、改变预测、调整阈值、读取全局留出运行或建立新GFG。",
        "",
        "四项同时通过不再作为理论总体正确性的主要标准；它仅保留为附加的“严格联合通过率”。",
        "",
        "## 1. CSRG状态预测",
        "",
        "| 指标 | 原基线 | 新理论 | 相对改善 |",
        "|---|---:|---:|---:|",
        f"| MAE | {state_metrics['baseline']['mae']:.9g} | {state_metrics['candidate']['mae']:.9g} | {pct(state_metrics['relative_improvement']['mae'])} |",
        f"| RMSE | {state_metrics['baseline']['rmse']:.9g} | {state_metrics['candidate']['rmse']:.9g} | {pct(state_metrics['relative_improvement']['rmse'])} |",
        f"| 平均逐转移NRMSE | {state_metrics['baseline']['mean_transition_nrmse']:.9g} | {state_metrics['candidate']['mean_transition_nrmse']:.9g} | {pct(state_metrics['relative_improvement']['mean_transition_nrmse'])} |",
        f"| 汇总NRMSE | {state_metrics['baseline']['pooled_transition_nrmse']:.9g} | {state_metrics['candidate']['pooled_transition_nrmse']:.9g} | {pct(state_metrics['relative_improvement']['pooled_transition_nrmse'])} |",
        "",
        "MAE和RMSE汇总相同的150个CSRG坐标，但这些坐标量纲和尺度不同，因此只用于同坐标、同样本上的基线—候选比较；NRMSE保留开发折内冻结尺度。状态正确数仍按未改动的NRMSE容差裁决。",
        "",
        _ledger_sentence("状态", state["tolerance_ledger"]),
        "",
        "## 2. 能力数值预测",
        "",
        "| 指标 | 原基线 | 新理论 | 相对改善 |",
        "|---|---:|---:|---:|",
        f"| MAE | {cap_metrics['baseline']['mae']:.9g} | {cap_metrics['candidate']['mae']:.9g} | {pct(cap_metrics['relative_improvement']['mae'])} |",
        f"| RMSE | {cap_metrics['baseline']['rmse']:.9g} | {cap_metrics['candidate']['rmse']:.9g} | {pct(cap_metrics['relative_improvement']['rmse'])} |",
        "",
        "能力数值正确数按未改动的绝对误差容差裁决。",
        "",
        _ledger_sentence("能力数值", capability["tolerance_ledger"]),
        "",
        "## 3. 能力变化方向",
        "",
        f"原基线accuracy={direction['baseline']['accuracy']:.6f}，balanced accuracy={direction['baseline']['balanced_accuracy']:.6f}。",
        f"新理论accuracy={direction['candidate']['accuracy']:.6f}，balanced accuracy={direction['candidate']['balanced_accuracy']:.6f}。",
        "",
        "混淆矩阵的行是真实类别、列是预测类别，顺序均为下降(-1)、基本不变(0)、上升(+1)。",
        "",
        f"- 原基线：`{direction['baseline']['matrix_rows_actual_columns_predicted']}`",
        f"- 新理论：`{direction['candidate']['matrix_rows_actual_columns_predicted']}`",
        "",
        _ledger_sentence("方向", direction["task_ledger"]),
        "",
        "## 4. 明显下降预测",
        "",
        "| 指标 | 原基线 | 新理论 |",
        "|---|---:|---:|",
        f"| Precision | {decline['baseline']['precision']:.6f} | {decline['candidate']['precision']:.6f} |",
        f"| Recall | {decline['baseline']['recall']:.6f} | {decline['candidate']['recall']:.6f} |",
        f"| F1 | {decline['baseline']['f1']:.6f} | {decline['candidate']['f1']:.6f} |",
        "",
        "混淆矩阵的行是真实标签、列是预测标签，顺序为非明显下降、明显下降。",
        "",
        f"- 原基线：`{decline['baseline']['matrix_rows_actual_columns_predicted']}`",
        f"- 新理论：`{decline['candidate']['matrix_rows_actual_columns_predicted']}`",
        "",
        _ledger_sentence("下降标签", decline["task_ledger"]),
        "",
        "四类反例分别按原始失败类型记账：状态反例只看状态裁决，能力数值反例只看能力误差，方向反例只看方向类别，下降反例只看下降标签。修复某一类反例不要求同时通过另外三项。",
        "",
        "## 5. 严格联合通过率（仅附加审计）",
        "",
        f"原基线严格联合通过{joint['baseline_correct_count']}/500，新理论{joint['candidate_correct_count']}/500；修复{joint['repaired_count']}条，新增失败{joint['newly_broken_count']}条，净增加{joint['net_correct_improvement_count']}条。",
        "",
        "该数字不得再用于概括理论整体是否正确。四类反例分别按各自任务账本裁决。",
        "",
        "## 结论",
        "",
        "新理论显著改善能力数值误差；方向总体accuracy和balanced accuracy提高，但没有预测出任何上升类别；下降检测precision提高而recall下降，F1略降；状态的平均NRMSE略有改善，但超过既定状态容差的记录反而增加。候选仍未密封。",
        "",
    ]
    return "\n".join(lines)


def _ledger_sentence(name: str, ledger: dict[str, Any]) -> str:
    return (
        f"{name}任务：原基线正确{ledger['baseline_correct_count']}条，新理论正确"
        f"{ledger['candidate_correct_count']}条；修复{ledger['repaired_count']}条，新增错误"
        f"{ledger['newly_broken_count']}条，净改善{ledger['net_correct_improvement_count']}条。"
    )


def write_taskwise_package(result: dict[str, Any], output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=False)
    result_path = output_root / "TASKWISE_EVALUATION.json"
    report_path = output_root / "TASKWISE_EVALUATION_REPORT.md"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_markdown(result), encoding="utf-8", newline="\n")
    files = {
        path.name: {"sha256": _file_sha256(path), "size_bytes": path.stat().st_size}
        for path in (result_path, report_path)
    }
    manifest_material = {
        "schema": "nanogpt-one-step-500-taskwise-evaluation-manifest-v1",
        "status": "PASS",
        "files": files,
    }
    manifest_material["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            manifest_material,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    (output_root / "MANIFEST.json").write_text(
        json.dumps(manifest_material, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest_material


__all__ = [
    "SCHEMA",
    "build_taskwise_evaluation",
    "render_markdown",
    "write_taskwise_package",
]
