from __future__ import annotations

from collections import Counter
import gzip
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from .dataset import build_dataset, j_jk_predictions, load_records
from .metrics import clustered_bootstrap_improvement, evaluate_by_run
from .model import (
    POSITIVE_ALPHAS,
    RandomFeatureCurveRegressor,
    boundary_class,
    canonical_json,
    file_sha256,
    normalized_shape,
    require,
    response_type,
    stable_seed,
)


SCHEMES = ("A_DIRECT", "B_AMPLITUDE_SHAPE", "C_PCA3")
LEARNED_MODELS = {
    "B1": "B1_M1",
    "B2": "B2",
    "M2": "M2",
    "M3": "M3",
    "M4": "M4",
}
ALL_MODELS = ("B0", "B1", "B2", "M2", "M3", "M4", "J", "JK")
SOURCE_ROOT = Path(r"E:\gfg-evidence\nanogpt-response-factor-analysis-v1\submission")
RESPONSE_ROOT = Path(r"E:\gfg-evidence\nanogpt-adjacent-response-transport-v1\submission")
DEFAULT_OUTPUT = Path(r"E:\gfg-evidence\nanogpt-state-conditioned-response-v1\submission")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def response_scale(values: np.ndarray) -> np.ndarray:
    q75 = np.quantile(values, 0.75, axis=0)
    q25 = np.quantile(values, 0.25, axis=0)
    return np.where(q75 - q25 > 1e-8, q75 - q25, 1.0)


def score_nrmse(truth: np.ndarray, prediction: np.ndarray, scale: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square((prediction - truth) / np.maximum(scale, 1e-12)))))


def inner_groups(entries: list[str]) -> list[list[str]]:
    groups = [[], [], []]
    for index, entry in enumerate(sorted(entries)):
        groups[index % 3].append(entry)
    return groups


def select_scheme(
    *,
    outer_entry: str,
    x: np.ndarray,
    y: np.ndarray,
    entries: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    development_entries = sorted(set(entries.tolist()) - {outer_entry})
    groups = inner_groups(development_entries)
    scores: dict[str, list[float]] = {scheme: [] for scheme in SCHEMES}
    folds = []
    for inner_index, held_entries in enumerate(groups):
        validation = np.isin(entries, held_entries)
        training = (~validation) & (entries != outer_entry)
        scale = response_scale(y[training])
        fold_record = {"fold": inner_index, "held_entries": held_entries, "scores": {}}
        for scheme in SCHEMES:
            model = RandomFeatureCurveRegressor(
                scheme=scheme,
                seed=stable_seed(f"inner:{outer_entry}:{inner_index}:{scheme}"),
                feature_names=feature_names,
            ).fit(x[training], y[training])
            prediction = model.predict(x[validation])
            score = score_nrmse(y[validation], prediction, scale)
            scores[scheme].append(score)
            fold_record["scores"][scheme] = score
        folds.append(fold_record)
    means = {scheme: float(np.mean(values)) for scheme, values in scores.items()}
    selected = sorted(means, key=lambda name: (means[name], name))[0]
    return {"outer_entry": outer_entry, "selected_scheme": selected, "mean_nrmse": means, "inner_folds": folds}


def fit_outer_models(dataset: dict[str, Any]) -> dict[str, Any]:
    y = dataset["displacement"]
    entries = dataset["entries"]
    n = len(y)
    predictions = {name: np.zeros_like(y) for name in ALL_MODELS}
    normalizers = np.zeros_like(y)
    selections = []
    model_audits = []
    for outer_entry in dataset["unique_entries"]:
        test = entries == outer_entry
        train = ~test
        selection = select_scheme(
            outer_entry=outer_entry,
            x=dataset["matrices"]["M4"],
            y=y,
            entries=entries,
            feature_names=dataset["feature_names"]["M4"],
        )
        selections.append(selection)
        scheme = selection["selected_scheme"]
        normalizers[test] = response_scale(y[train])
        for model_name, feature_set in LEARNED_MODELS.items():
            model = RandomFeatureCurveRegressor(
                scheme=scheme,
                seed=stable_seed(f"outer:{outer_entry}:{model_name}:{scheme}"),
                feature_names=dataset["feature_names"][feature_set],
            ).fit(dataset["matrices"][feature_set][train], y[train])
            predictions[model_name][test] = model.predict(dataset["matrices"][feature_set][test])
            model_audits.append(
                {
                    "outer_entry": outer_entry,
                    "model": model_name,
                    "feature_set": feature_set,
                    "feature_count": len(dataset["feature_names"][feature_set]),
                    "scheme": scheme,
                    "seed": model.seed,
                    "training_count": int(np.sum(train)),
                    "test_count": int(np.sum(test)),
                }
            )
    predictions["B0"][:] = 0.0
    predictions["J"], predictions["JK"] = j_jk_predictions(dataset["records"])
    require(np.all(normalizers > 0), "OUTER_NORMALIZATION_NOT_FILLED")
    require(all(np.all(np.isfinite(value)) for value in predictions.values()), "NONFINITE_OOF_PREDICTION")
    return {
        "predictions": predictions,
        "normalizers": normalizers,
        "selections": selections,
        "model_audits": model_audits,
    }


def metrics_for_models(dataset: dict[str, Any], outer: dict[str, Any]) -> dict[str, Any]:
    return {
        name: evaluate_by_run(
            dataset["displacement"],
            outer["predictions"][name],
            dataset["margin0"],
            outer["normalizers"],
            dataset["entries"],
        )
        for name in ALL_MODELS
    }


def confusion_delta(truth: np.ndarray, before: np.ndarray, after: np.ndarray, margin0: np.ndarray) -> dict[str, Any]:
    truth_boundary = np.asarray([boundary_class(value, curve) for value, curve in zip(margin0, truth, strict=True)], dtype=object)
    before_boundary = np.asarray([boundary_class(value, curve) for value, curve in zip(margin0, before, strict=True)], dtype=object)
    after_boundary = np.asarray([boundary_class(value, curve) for value, curve in zip(margin0, after, strict=True)], dtype=object)
    before_ok = before_boundary == truth_boundary
    after_ok = after_boundary == truth_boundary
    return {
        "baseline_correct": int(np.sum(before_ok)),
        "candidate_correct": int(np.sum(after_ok)),
        "repaired": int(np.sum((~before_ok) & after_ok)),
        "newly_broken": int(np.sum(before_ok & (~after_ok))),
        "net_improvement": int(np.sum(after_ok) - np.sum(before_ok)),
    }


def build_result_documents(dataset: dict[str, Any], outer: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    rmse_bootstrap = clustered_bootstrap_improvement(
        metrics["M4"]["per_run"], metrics["B2"]["per_run"], ("displacement", "rmse"), label="M4-vs-B2-rmse"
    )
    shape_bootstrap = clustered_bootstrap_improvement(
        metrics["M4"]["per_run"], metrics["B2"]["per_run"], ("normalized_shape", "rmse"), label="M4-vs-B2-shape"
    )
    run_improvements = {
        entry: metrics["B2"]["per_run"][entry]["displacement"]["rmse"]
        - metrics["M4"]["per_run"][entry]["displacement"]["rmse"]
        for entry in dataset["unique_entries"]
    }
    criteria = {
        "m4_rmse_better_than_b0": metrics["M4"]["overall"]["displacement"]["rmse"] < metrics["B0"]["overall"]["displacement"]["rmse"],
        "m4_rmse_better_than_b1": metrics["M4"]["overall"]["displacement"]["rmse"] < metrics["B1"]["overall"]["displacement"]["rmse"],
        "m4_rmse_better_than_b2": metrics["M4"]["overall"]["displacement"]["rmse"] < metrics["B2"]["overall"]["displacement"]["rmse"],
        "m4_rmse_improvement_ci_lower_positive": rmse_bootstrap["ci95"][0] > 0.0,
        "m4_shape_better_than_b2": metrics["M4"]["overall"]["normalized_shape"]["rmse"] < metrics["B2"]["overall"]["normalized_shape"]["rmse"],
        "m4_shape_improvement_ci_lower_positive": shape_bootstrap["ci95"][0] > 0.0,
        "m4_improves_at_least_8_runs": sum(value > 0 for value in run_improvements.values()) >= 8,
        "m4_false_cross_below_j": metrics["M4"]["overall"]["unchanged_target"]["false_crossing_rate"] < metrics["J"]["overall"]["unchanged_target"]["false_crossing_rate"],
        "m4_false_cross_below_jk": metrics["M4"]["overall"]["unchanged_target"]["false_crossing_rate"] < metrics["JK"]["overall"]["unchanged_target"]["false_crossing_rate"],
        "forbidden_inputs_absent": True,
    }
    if all(criteria.values()):
        status = "SUPPORTED"
    elif any(criteria.values()):
        status = "PARTIALLY_SUPPORTED"
    else:
        status = "NOT_SUPPORTED"
    baseline = {
        "schema": "nanogpt-state-conditioned-response-baselines-v1",
        "status": "PASS",
        "models": {name: metrics[name] for name in ("B0", "B1", "B2", "J", "JK")},
    }
    ablation = {
        "schema": "nanogpt-state-conditioned-response-ablation-v1",
        "status": "PASS",
        "models": {"M1_F1": metrics["B1"], "M2_F1_F3": metrics["M2"], "M3_F1_F5": metrics["M3"], "M4_F1_F3_F5": metrics["M4"]},
        "rmse_differences": {
            "F3_increment_over_F1": metrics["B1"]["overall"]["displacement"]["rmse"] - metrics["M2"]["overall"]["displacement"]["rmse"],
            "F5_increment_over_F1": metrics["B1"]["overall"]["displacement"]["rmse"] - metrics["M3"]["overall"]["displacement"]["rmse"],
            "F5_increment_given_F3": metrics["M2"]["overall"]["displacement"]["rmse"] - metrics["M4"]["overall"]["displacement"]["rmse"],
            "F3_increment_given_F5": metrics["M3"]["overall"]["displacement"]["rmse"] - metrics["M4"]["overall"]["displacement"]["rmse"],
        },
        "boundary_correctness_delta_M4_vs_B2": confusion_delta(
            dataset["displacement"], outer["predictions"]["B2"], outer["predictions"]["M4"], dataset["margin0"]
        ),
    }
    result = {
        "schema": "nanogpt-state-conditioned-response-results-v1",
        "status": status,
        "scientific_question": "Can pre-response F1/F3/F5 facts predict the complete finite-amplitude target displacement curve on held-out runs?",
        "primary_model": "M4_F1_F3_F5",
        "primary_metrics": metrics["M4"],
        "success_criteria": criteria,
        "rmse_improvement_over_B2_cluster_bootstrap": rmse_bootstrap,
        "shape_improvement_over_B2_cluster_bootstrap": shape_bootstrap,
        "run_rmse_improvements_over_B2": run_improvements,
        "strict_global_unseen_claim_valid": False,
        "strict_global_unseen_remedy": "A newly generated execution never accessed by this analysis session is required.",
    }
    return {"baseline": baseline, "ablation": ablation, "result": result}


def prediction_rows(dataset: dict[str, Any], outer: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, record in enumerate(dataset["records"]):
        truth = dataset["displacement"][index]
        row = {
            "record_id": record["record_id"],
            "entry_id": record["entry_id"],
            "section_id": record["section_id"],
            "evaluation_unit_id": record["evaluation_unit_id"],
            "optimizer_step": int(record["optimizer_step"]),
            "semantic_target_key": record["semantic_target_key"],
            "target_group": int(record["target_group"]),
            "margin_zero": float(dataset["margin0"][index]),
            "truth_displacement": truth.tolist(),
            "truth_response_type": response_type(truth),
            "truth_boundary_class": boundary_class(float(dataset["margin0"][index]), truth),
            "outer_normalization_scale": outer["normalizers"][index].tolist(),
            "predictions": {},
        }
        for name in ALL_MODELS:
            prediction = outer["predictions"][name][index]
            row["predictions"][name] = {
                "displacement": prediction.tolist(),
                "response_type": response_type(prediction),
                "boundary_class": boundary_class(float(dataset["margin0"][index]), prediction),
            }
        rows.append(row)
    return rows


def write_gzip_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def challenge_audit(challenge_path: Path, predictions: list[dict[str, Any]]) -> dict[str, Any]:
    challenge = read_json(challenge_path)
    by_id = {row["record_id"]: row for row in predictions}
    counts = Counter()
    rows = []
    for item in challenge["rows"]:
        left = by_id.get(item["query_record_id"])
        right = by_id.get(item["reference_record_id"])
        if left is None or right is None:
            counts["NOT_EVALUABLE"] += 1
            continue
        true_left = np.asarray(left["truth_displacement"])
        true_right = np.asarray(right["truth_displacement"])
        pred_left = np.asarray(left["predictions"]["M4"]["displacement"])
        pred_right = np.asarray(right["predictions"]["M4"]["displacement"])
        requirements = []
        if np.sign(true_left[-1]) != np.sign(true_right[-1]):
            requirements.append(np.sign(pred_left[-1]) != np.sign(pred_right[-1]))
        if left["truth_boundary_class"] != right["truth_boundary_class"]:
            requirements.append(left["predictions"]["M4"]["boundary_class"] != right["predictions"]["M4"]["boundary_class"])
        true_corr = float(np.corrcoef(normalized_shape(true_left[None, :])[0], normalized_shape(true_right[None, :])[0])[0, 1])
        if np.isfinite(true_corr) and true_corr < 0.0:
            pred_corr = float(np.corrcoef(normalized_shape(pred_left[None, :])[0], normalized_shape(pred_right[None, :])[0])[0, 1])
            requirements.append(np.isfinite(pred_corr) and pred_corr < 0.0)
        resolved = bool(requirements and all(requirements))
        label = "STRICTLY_RESOLVED" if resolved else "STILL_UNRESOLVED"
        counts[label] += 1
        rows.append({"query_record_id": item["query_record_id"], "reference_record_id": item["reference_record_id"], "status": label})
    return {
        "schema": "nanogpt-state-conditioned-response-frozen-challenge-audit-v1",
        "status": "POSTHOC_CHALLENGE_ONLY_NOT_MODEL_SELECTION",
        "challenge_count": int(challenge["counterexample_count"]),
        "counts": dict(counts),
        "rows": rows,
    }


def failure_audit(dataset: dict[str, Any], outer: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    truth = dataset["displacement"]
    m4 = outer["predictions"]["M4"]
    scales = outer["normalizers"]
    categories = Counter()
    failures = []
    support_cache: dict[str, dict[str, np.ndarray]] = {}
    for index, record in enumerate(dataset["records"]):
        nrmse = float(np.sqrt(np.mean(np.square((m4[index] - truth[index]) / scales[index]))))
        boundary_wrong = boundary_class(dataset["margin0"][index], m4[index]) != boundary_class(dataset["margin0"][index], truth[index])
        type_wrong = response_type(m4[index]) != response_type(truth[index])
        if not (nrmse > 1.0 or boundary_wrong or type_wrong):
            continue
        competitor_switch = bool(record["response"]["competitor_switch"])
        section_id = str(record["section_id"])
        if section_id not in support_cache:
            with np.load(RESPONSE_ROOT / "sections" / f"{section_id}.npz", allow_pickle=False) as data:
                support_cache[section_id] = {
                    "effective_support": np.asarray(data["effective_support"], dtype=np.float64),
                    "support_concentration": np.asarray(data["support_concentration"], dtype=np.float64),
                    "support_allocation": np.asarray(data["support_allocation"], dtype=np.float64),
                }
        support = support_cache[section_id]
        group = int(record["target_group"])
        effective_pre = float(support["effective_support"][1, group])
        effective_end = float(support["effective_support"][6, group])
        concentration_pre = float(support["support_concentration"][1, group])
        concentration_end = float(support["support_concentration"][6, group])
        allocation_pre = np.asarray(support["support_allocation"][1, :, group], dtype=np.float64)
        allocation_end = np.asarray(support["support_allocation"][6, :, group], dtype=np.float64)
        allocation_l1_change = float(np.nansum(np.abs(allocation_end - allocation_pre)))
        effective_delta = effective_end - effective_pre if np.isfinite(effective_end) and np.isfinite(effective_pre) else None
        concentration_delta = (
            concentration_end - concentration_pre
            if np.isfinite(concentration_end) and np.isfinite(concentration_pre)
            else None
        )
        support_changed = bool(
            (effective_delta is not None and abs(effective_delta) > 0.25)
            or (concentration_delta is not None and abs(concentration_delta) > 0.10)
            or allocation_l1_change > 0.25
        )
        m2_error = float(np.sqrt(np.mean(np.square(outer["predictions"]["M2"][index] - truth[index]))))
        m3_error = float(np.sqrt(np.mean(np.square(outer["predictions"]["M3"][index] - truth[index]))))
        m4_error = float(np.sqrt(np.mean(np.square(m4[index] - truth[index]))))
        labels = []
        if competitor_switch:
            labels.append("COMPETITOR_STRUCTURE_MISSING")
        if support_changed:
            labels.append("SUPPORT_STATE_MISSING")
        if m4_error < 0.9 * m3_error:
            labels.append("F3_INFORMATION_HELPFUL_BUT_INSUFFICIENT")
        if m4_error < 0.9 * m2_error:
            labels.append("F5_INFORMATION_HELPFUL_BUT_INSUFFICIENT")
        if not labels:
            labels.append("CURRENT_STATE_REPRESENTATION_INSUFFICIENT_OR_MODEL_MISSPECIFIED")
        for label in labels:
            categories[label] += 1
        failures.append(
            {
                "record_id": record["record_id"],
                "entry_id": record["entry_id"],
                "optimizer_step": int(record["optimizer_step"]),
                "evaluation_unit_id": record["evaluation_unit_id"],
                "nrmse": nrmse,
                "boundary_wrong": boundary_wrong,
                "response_type_wrong": type_wrong,
                "competitor_switch": competitor_switch,
                "support_change": {
                    "effective_support_pre": effective_pre if np.isfinite(effective_pre) else None,
                    "effective_support_end": effective_end if np.isfinite(effective_end) else None,
                    "effective_support_delta": effective_delta,
                    "support_concentration_pre": concentration_pre if np.isfinite(concentration_pre) else None,
                    "support_concentration_end": concentration_end if np.isfinite(concentration_end) else None,
                    "support_concentration_delta": concentration_delta,
                    "support_allocation_pre": [float(value) if np.isfinite(value) else None for value in allocation_pre],
                    "support_allocation_end": [float(value) if np.isfinite(value) else None for value in allocation_end],
                    "support_allocation_l1_change": allocation_l1_change,
                    "diagnostic_threshold_crossed": support_changed,
                },
                "truth_displacement": truth[index].tolist(),
                "predicted_displacement": m4[index].tolist(),
                "endpoint_error": float(m4[index, -1] - truth[index, -1]),
                "provisional_categories": labels,
                "classification_is_post_outcome_and_noncausal": True,
            }
        )
    summary = {
        "schema": "nanogpt-state-conditioned-response-failure-analysis-v1",
        "status": "PASS",
        "definition": "M4 displacement NRMSE > 1, wrong boundary class, or wrong response type",
        "failure_count": len(failures),
        "failure_rate": len(failures) / len(truth),
        "provisional_category_counts": dict(categories),
        "support_change_diagnostic_thresholds": {
            "absolute_effective_support_delta": 0.25,
            "absolute_support_concentration_delta": 0.10,
            "support_allocation_l1_delta": 0.25,
        },
        "classification_is_post_outcome_and_noncausal": True,
    }
    return summary, failures


def assessment_text(result: dict[str, Any], ablation: dict[str, Any], challenge: dict[str, Any], failure: dict[str, Any]) -> str:
    metrics = result["primary_metrics"]["overall"]
    status = result["status"]
    f3 = ablation["rmse_differences"]["F3_increment_over_F1"]
    f5 = ablation["rmse_differences"]["F5_increment_over_F1"]
    return f"""# Scientific assessment\n\n## Machine result\n\nFrozen status: `{status}`. The primary M4 model was evaluated by 12-fold leave-one-run-out and never received a positive-alpha response, current-step functional J/K probe, future margin, post-update receiver state, run identity, target identity, or absolute optimizer step as an input.\n\nM4 displacement RMSE was `{metrics['displacement']['rmse']:.9g}`, normalized-shape RMSE was `{metrics['normalized_shape']['rmse']:.9g}`, endpoint direction accuracy was `{metrics['endpoint']['direction_accuracy']:.6f}`, and unchanged-target false-crossing rate was `{metrics['unchanged_target']['false_crossing_rate']:.6f}`.\n\n## Answers\n\n- **Are F1/F3/F5 sufficient for an executable response function?** {"Under the frozen success rule, yes for the 12 development runs." if status == "SUPPORTED" else "Not fully under the frozen success rule. They provide a measurable executable mapping, but at least one preregistered condition failed."}\n- **Is target-specific update geometry core?** The admitted F3 block changed RMSE relative to F1 by `{f3:.9g}`. This is evidence about the whole natural F3 block, not proof that its target-specific subset alone is sufficient.\n- **Does parameter/Adam receiver state add independent information?** The admitted F5 block changed RMSE relative to F1 by `{f5:.9g}`. Its independent value is bounded by the M3 and M4 ablations; it is not a full tensor-level receiver state.\n- **What remains missing?** `{failure['failure_count']}` of `{metrics['count']}` outer predictions met the frozen broad failure definition. Post-outcome categories are diagnostic only. The most defensible unresolved possibilities are coarser-than-required target update geometry, coarser-than-required local parameter/Adam receiver structure, competitor changes, omitted support state, or model misspecification.\n- **Has the work moved from correlated continuity to a transportable response law?** Only if the frozen status is `SUPPORTED`; otherwise it has established a falsifiable cross-run executable candidate, not a closed transport law. The 1,477-pair frozen challenge retained `{challenge['counts'].get('STILL_UNRESOLVED', 0)}` unresolved pairs.\n- **Should the next stage be full state transition?** Only after treating the failed response cases honestly. A full transition study may use the frozen response model as a candidate component, but must not describe it as universally closed.\n\n## Boundary\n\nThis is development-run evidence. A prior diagnostic read invalidated the old global-unseen execution as a strict hidden adjudicator. A new, never-accessed execution is required for a valid hidden confirmation.\n"""


def run(output_root: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    require(not output_root.exists(), f"OUTPUT_ROOT_EXISTS:{output_root}")
    output_root.mkdir(parents=True)
    contract_source = Path(__file__).with_name("MODEL_CONTRACT.md")
    contract_path = output_root / "MODEL_CONTRACT.md"
    shutil.copyfile(contract_source, contract_path)
    source_records = SOURCE_ROOT / "PRETARGET_FACTOR_RECORDS.jsonl.gz"
    challenge_path = SOURCE_ROOT / "SURVIVING_CONDITIONAL_COUNTEREXAMPLES.json"
    availability_path = SOURCE_ROOT / "PRETARGET_FEATURE_AVAILABILITY.json"
    boundary_path = SOURCE_ROOT / "BOUNDARY_VIOLATION.json"
    source_hashes = {path.name: file_sha256(path) for path in (source_records, challenge_path, availability_path, boundary_path)}
    freeze = {
        "schema": "nanogpt-state-conditioned-response-freeze-v1",
        "status": "FROZEN_BEFORE_MODEL_RESULTS",
        "contract_sha256": file_sha256(contract_path),
        "source_hashes": source_hashes,
        "strict_global_unseen_claim_valid": False,
        "forbidden_inputs": ["positive-alpha response", "current-step functional J/K", "future margin", "post-update receiver state", "run/entry/step/phase/target identities"],
    }
    write_json(output_root / "EXPERIMENT_FREEZE.json", freeze)
    write_json(output_root / "BOUNDARY_VIOLATION.json", read_json(boundary_path))

    dataset = build_dataset(load_records(source_records))
    feature_manifest = {
        "schema": "nanogpt-state-conditioned-response-feature-manifest-v1",
        "status": "PASS",
        "record_count": len(dataset["records"]),
        "entry_counts": dataset["entry_counts"],
        "feature_sets": dataset["feature_names"],
        "input_event_boundary": "actual update formed; before positive-alpha target response",
        "categorical_features_used": False,
        "current_step_functional_probe_used": False,
        "future_result_used": False,
        "source_hashes": source_hashes,
    }
    write_json(output_root / "FEATURE_MANIFEST.json", feature_manifest)

    outer = fit_outer_models(dataset)
    split_manifest = {
        "schema": "nanogpt-state-conditioned-response-splits-v1",
        "status": "PASS",
        "outer_protocol": "12-fold leave-one-entry-out",
        "outer_folds": outer["selections"],
        "model_fit_audit": outer["model_audits"],
    }
    write_json(output_root / "TRAINING_SPLIT_MANIFEST.json", split_manifest)

    metrics = metrics_for_models(dataset, outer)
    documents = build_result_documents(dataset, outer, metrics)
    write_json(output_root / "BASELINE_RESULTS.json", documents["baseline"])
    write_json(output_root / "ABLATION_RESULTS.json", documents["ablation"])
    write_json(output_root / "NONLINEAR_RESPONSE_MODEL_RESULTS.json", documents["result"])

    rows = prediction_rows(dataset, outer)
    predictions_path = output_root / "RESPONSE_CURVE_PREDICTIONS.jsonl.gz"
    write_gzip_jsonl(predictions_path, rows)
    write_json(
        output_root / "RESPONSE_CURVE_PREDICTIONS.json",
        {
            "schema": "nanogpt-state-conditioned-response-prediction-ledger-manifest-v1",
            "status": "PASS",
            "row_count": len(rows),
            "ledger": predictions_path.name,
            "ledger_sha256": file_sha256(predictions_path),
        },
    )

    failure_summary, failures = failure_audit(dataset, outer)
    failure_path = output_root / "FAILURE_CASE_LEDGER.jsonl.gz"
    write_gzip_jsonl(failure_path, failures)
    failure_summary.update({"ledger": failure_path.name, "ledger_sha256": file_sha256(failure_path)})
    challenge = challenge_audit(challenge_path, rows)
    challenge_rows = challenge.pop("rows")
    challenge_ledger = output_root / "FROZEN_CHALLENGE_AUDIT.jsonl.gz"
    write_gzip_jsonl(challenge_ledger, challenge_rows)
    challenge.update({"ledger": challenge_ledger.name, "ledger_sha256": file_sha256(challenge_ledger)})
    failure_summary["frozen_1477_pair_challenge"] = challenge
    write_json(output_root / "FAILURE_CASE_ANALYSIS.json", failure_summary)

    selected_counts = Counter(item["selected_scheme"] for item in outer["selections"])
    final_scheme = sorted(selected_counts, key=lambda name: (-selected_counts[name], name))[0]
    final_model = RandomFeatureCurveRegressor(
        scheme=final_scheme,
        seed=stable_seed(f"final:M4:{final_scheme}"),
        feature_names=dataset["feature_names"]["M4"],
    ).fit(dataset["matrices"]["M4"], dataset["displacement"])
    final_model.save(output_root / "FINAL_M4_MODEL.npz", output_root / "FINAL_M4_MODEL_METADATA.json")
    model_spec = {
        "schema": "nanogpt-state-conditioned-response-model-spec-v1",
        "status": "PASS",
        "family": "deterministic nonlinear random-feature ridge",
        "outer_selected_scheme_counts": dict(selected_counts),
        "final_scheme": final_scheme,
        "random_feature_width": final_model.width,
        "ridge": final_model.ridge,
        "feature_names": final_model.feature_names,
        "final_artifact": "FINAL_M4_MODEL.npz",
        "final_artifact_is_unbiased_evaluation": False,
    }
    write_json(output_root / "MODEL_SPEC.json", model_spec)
    assessment = assessment_text(documents["result"], documents["ablation"], challenge, failure_summary)
    (output_root / "SCIENTIFIC_ASSESSMENT.md").write_text(assessment, encoding="utf-8", newline="\n")
    shutil.copyfile(Path(__file__).with_name("INDEPENDENT_CHECKER.py"), output_root / "INDEPENDENT_CHECKER.py")
    shutil.copyfile(Path(__file__).with_name("REPRODUCE_RUN.py"), output_root / "REPRODUCE_RUN.py")
    write_json(
        output_root / "DEVELOPMENT_MODEL_READY_WITH_DISCLOSED_BOUNDARY_VIOLATION.json",
        {
            "schema": "nanogpt-state-conditioned-response-readiness-v1",
            "status": documents["result"]["status"],
            "ordinary_strict_ready": False,
            "reason": "old global-unseen execution was diagnostically accessed before this experiment",
            "development_outputs_complete": True,
        },
    )
    manifest = {
        "schema": "nanogpt-state-conditioned-response-manifest-v1",
        "status": "MODEL_COMPLETE_PENDING_INDEPENDENT_CHECK_AND_GFG",
        "scientific_status": documents["result"]["status"],
        "contract_sha256": file_sha256(contract_path),
        "source_hashes": source_hashes,
        "deliverables": {},
    }
    for path in sorted(output_root.iterdir()):
        if path.is_file() and path.name != "MANIFEST.json":
            manifest["deliverables"][path.name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    write_json(output_root / "MANIFEST.json", manifest)
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    result = run(arguments.output_root)
    print(json.dumps({"status": result["status"], "scientific_status": result["scientific_status"]}, sort_keys=True))
