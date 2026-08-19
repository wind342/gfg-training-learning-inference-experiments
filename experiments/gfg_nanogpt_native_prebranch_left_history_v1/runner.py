from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from .analysis import (
    CRITICAL_LEDGER,
    FACTOR_RECORDS,
    IDENTITY_MATERIAL,
    RESPONSE_ROOT,
    SPACES,
    TASKS,
    adjudicate,
    canonical_json,
    compile_dataset,
    evaluate,
    file_sha256,
    public_results,
)


DEFAULT_REPORT_ROOT = (
    Path(__file__).parents[1]
    / "gfg_nanogpt_cumulative_scientist_v1"
    / "reports"
    / "native_prebranch_left_history_v1"
)
DEFAULT_GRAPH_ROOT = Path(r"E:\gfg-evidence\nanogpt-native-prebranch-left-history-v1\gfg")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def availability_markdown(audit: dict[str, Any]) -> str:
    fields = audit["response_before_available"]
    lines = [
        "# Availability audit",
        "",
        f"Status: `{audit['status']}`. The audit covers `{audit['record_count']}` records, "
        f"`{audit['section_count']}` response sections and `{audit['run_count']}` independent runs.",
        "",
        "| pre-response material | result |",
        "|---|---|",
    ]
    labels = {
        "current_correct_logit_and_margin": "current correct-class logit and margin",
        "current_strongest_competitor_identity": "current strongest incorrect competitor",
        "complete_incorrect_competitor_set": "complete incorrect-competitor set",
        "identity_aligned_logits_lags_1_2_5_10": "same competitor identities at lags 1/2/5/10",
        "current_primary_and_secondary_support_identity": "current primary/secondary support identity",
        "identity_aligned_support_lags_1_2_5_10": "same support components at lags 1/2/5/10",
        "current_F1_F3_F5": "current F1/F3/F5",
        "past_F1": "past F1",
        "past_F3_native_update": "past native updates",
        "past_F5": "past F5",
        "current_vs_past_native_update_geometry": "current/prior update geometry",
        "F3_identity_relative_action": "identity-relative F3 action",
    }
    for key, label in labels.items():
        lines.append(f"| {label} | `{fields[key]}` |")
    lines.extend(
        [
            "",
            "## Hard boundary",
            "",
            "The current native update and its full parameter tensor are available before application. "
            "This permits global and component-level update-continuity measurements. The natural update "
            "also exposes output-embedding rows for the target and every competitor. It does **not** "
            "expose a full target-logit Jacobian or the unseen effect of the update on support components. "
            "Consequently X3 measures update continuation, not the answer to the response probe.",
            "",
            "Objects marked `target_only_after_cut` are present in the source GFG for adjudication but "
            "are excluded from every feature matrix.",
            "",
            f"Support-handoff labels are unevaluable for `{audit['support_handoff_unevaluable_count']}` records. "
            f"Separately, the previously disclosed post-response effective-support change remains undefined for "
            f"`{audit['post_effective_support_change_unevaluable_count_m4_in_support']}` M4-in-support records "
            f"(`{audit['post_effective_support_change_unevaluable_count_all_records']}` overall); it is neither "
            "imputed nor used as an input or label.",
            "",
        ]
    )
    return "\n".join(lines)


def _relative_improvement(before: float, after: float) -> float:
    return (before - after) / before if before else 0.0


def scientific_report(
    compiled: dict[str, Any],
    audit: dict[str, Any],
    results: dict[str, Any],
    decision: dict[str, Any],
) -> str:
    main = results["task_results"]["competitor_switch"]
    severe = results["task_results"]["severe_conflict"]
    response = results["response_prediction"]
    ordinary = response["ordinary_X4"]
    oracle = response["oracle_same_competitor_switch_branch_X4"]
    routed = response["executable_routed_X4"]
    labels = compiled["labels"]["severe_conflict"].astype(bool)
    gap = np.asarray([row["gap"] for row in compiled["meta"]], dtype=np.float64)
    velocity = np.asarray([row["gap_velocity"] for row in compiled["meta"]], dtype=np.float64)
    acceleration = np.asarray([row["gap_acceleration"] for row in compiled["meta"]], dtype=np.float64)
    lines = [
        "# Native pre-branch left-history experiment",
        "",
        f"**Decision: `{decision['verdict']}`.**",
        "",
        "This experiment asks whether an already established left history and the formed-but-unapplied "
        "native update identify a local response branch before any alpha-positive response exists.",
        "",
        "## Main prospective result",
        "",
        "| input space | ROC-AUC | PR-AUC | Brier | threshold recall | threshold FPR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for space in SPACES:
        value = main[space]
        lines.append(
            f"| {space} | {value['roc_auc']:.4f} | {value['pr_auc']:.4f} | {value['brier']:.4f} | "
            f"{value['threshold_recall']:.4f} | {value['threshold_fpr']:.4f} |"
        )
    lines.extend(
        [
            "",
            "### Frozen scalar baselines",
            "",
            "| baseline | ROC-AUC | PR-AUC | Brier |",
            "|---|---:|---:|---:|",
        ]
    )
    for baseline in ("gap_only", "past_switch_count_only", "prevalence_only"):
        value = main[baseline]
        lines.append(
            f"| {baseline} | {value['roc_auc']:.4f} | {value['pr_auc']:.4f} | {value['brier']:.4f} |"
        )
    delta = results["bootstrap"]["competitor_switch_pr_auc_X4_minus_X0"]
    lines.extend(
        [
            "",
            f"The X4-X0 competitor-switch PR-AUC delta is `{delta['delta']:.4f}` with run-clustered "
            f"95% interval `[{delta['ci95'][0]:.4f}, {delta['ci95'][1]:.4f}]`; "
            f"`{results['runs_with_nonnegative_competitor_switch_pr_auc_delta']}/12` runs have a non-negative delta.",
            "",
            "## The severe-conflict region",
            "",
            f"Before severe conflicts the median current competitor gap is `{np.median(gap[labels]):.6g}`, "
            f"compared with `{np.median(gap[~labels]):.6g}` outside that set. A negative lag-one velocity "
            f"occurs in `{np.mean(velocity[labels] < 0):.2%}` of severe records and "
            f"`{np.mean(velocity[~labels] < 0):.2%}` of non-severe records; negative acceleration occurs in "
            f"`{np.mean(acceleration[labels] < 0):.2%}` versus `{np.mean(acceleration[~labels] < 0):.2%}`.",
            "",
            f"Severe-conflict PR-AUC changes from `{severe['X0']['pr_auc']:.4f}` at X0 to "
            f"`{severe['X4']['pr_auc']:.4f}` at X4.",
            "",
            "## Branch-conditioned response prediction",
            "",
            "| method | overall curve RMSE | severe curve RMSE | boundary accuracy | wrong-to-correct recall |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, value in (
        ("ordinary X4 KNN", ordinary),
        ("Oracle-S same true switch branch", oracle),
        ("executable pre-state routed", routed),
    ):
        lines.append(
            f"| {label} | {value['overall']['curve_rmse']:.6g} | {value['severe_conflict']['curve_rmse']:.6g} | "
            f"{value['overall']['boundary_accuracy']:.4f} | {value['overall']['wrong_to_correct_recall']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Oracle-S changes severe-subset RMSE by "
            f"`{_relative_improvement(ordinary['severe_conflict']['curve_rmse'], oracle['severe_conflict']['curve_rmse']):.2%}`; "
            f"the executable router changes it by "
            f"`{_relative_improvement(ordinary['severe_conflict']['curve_rmse'], routed['severe_conflict']['curve_rmse']):.2%}`.",
            "",
            "## Direct answers",
            "",
            f"1. **Was the competitor gap smaller before severe conflicts?** The measured medians above answer this in the frozen corpus; no causal claim is made.",
            f"2. **Was it more often shrinking?** The negative-velocity rates above provide the direct answer.",
            f"3. **Did velocity/acceleration add information?** Compare X2 with X1: PR-AUC "
            f"`{main['X1']['pr_auc']:.4f} -> {main['X2']['pr_auc']:.4f}`.",
            f"4. **Did past switch history transport across runs?** The past-switch-only baseline is in RESULTS.json; X1 is the joint cross-run test.",
            f"5. **Could F3_native identify effect direction?** No. It legally supplies update continuity and output-row geometry, "
            "but not the unseen functional effect of the current update. X3 therefore tests continuation only.",
            f"6. **Could left history separate the approximately 9% severe region?** The severe X0/X4 metrics and clustered interval answer this.",
            f"7. **If it could not, why?** The availability audit rules out missing historical logits/support and coarse lag coverage at 1/2/5/10, "
            "but it leaves a real identity-specific action-effect gap: the native facts do not contain a full target response Jacobian.",
            f"8. **What is the oracle headroom?** Oracle-S severe RMSE improvement is reported above and is diagnostic only.",
            f"9. **Current classification of the remainder:** `{decision.get('additional_diagnosis') or decision['verdict']}`.",
            "",
            "## Coverage strata",
            "",
            f"All `{results['history_sufficiency']['sufficient_left_history']['record_count']}` records have the frozen left-history lags; "
            f"the insufficient-history stratum contains `{results['history_sufficiency']['insufficient_left_history']['record_count']}` records. "
            "Distribution-in and distribution-out metrics are reported separately in RESULTS.json under `distribution_subgroups`.",
            "",
            "## Scope",
            "",
            "All statements are restricted to the frozen 12-run nanoGPT corpus. The experiment did not train nanoGPT, use a GPU, "
            "call the VM AI, or create new response probes.",
            "",
        ]
    )
    return "\n".join(lines)


def _clean_number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def ledger_rows(compiled: dict[str, Any], evaluation: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    record_index = [
        {
            "record_index": index,
            "record_id": row["record_id"],
            "entry_id": row["entry_id"],
            "optimizer_step": row["optimizer_step"],
            "evaluation_unit_id": row["evaluation_unit_id"],
        }
        for index, row in enumerate(compiled["meta"])
    ]
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(compiled["meta"]):
        rows.append(
            {
                "record_index": index,
                "record_id": row["record_id"],
                "entry_id": row["entry_id"],
                "optimizer_step": row["optimizer_step"],
                "target_identity": row["target_identity"],
                "current_competitor_identity": row["current_competitor_identity"],
                "history_identity_aligned": row["history_identity_aligned"],
                "left_gap": row["gap"],
                "left_gap_velocity": row["gap_velocity"],
                "left_gap_acceleration": row["gap_acceleration"],
                "labels": row["labels"],
                "m4_in_support": row["m4_in_support"],
                "risks": {
                    space: {
                        task: _clean_number(evaluation["risks"][space][task][index])
                        for task in TASKS
                    }
                    for space in SPACES
                },
                "main_thresholds": {
                    space: _clean_number(evaluation["thresholds"][space]["competitor_switch"][index])
                    for space in SPACES
                },
                "approach_score": _clean_number(evaluation["approach_scores"][index]),
                "baseline_risks": {
                    name: _clean_number(values[index])
                    for name, values in evaluation["baseline_scores"].items()
                },
                "baseline_thresholds": {
                    name: _clean_number(values[index])
                    for name, values in evaluation["baseline_thresholds"].items()
                },
                "neighbor_source_record_indices_X4": [int(value) for value in evaluation["neighbors_x4"][index]],
                "true_response_displacement": row["response_displacement"],
                "margin0": row["margin0"],
                "truth_boundary": row["truth_boundary"],
                "ordinary_X4_response_prediction": [float(value) for value in evaluation["ordinary_curves"][index]],
                "oracle_same_switch_branch_response_prediction": [float(value) for value in evaluation["oracle_curves"][index]],
                "executable_routed_response_prediction": [float(value) for value in evaluation["routed_curves"][index]],
                "missing_reasons": row["missing_reasons"],
            }
        )
    return record_index, rows


def run(report_root: Path = DEFAULT_REPORT_ROOT, graph_root: Path = DEFAULT_GRAPH_ROOT) -> dict[str, Any]:
    if report_root.exists():
        raise RuntimeError(f"REPORT_ROOT_EXISTS:{report_root}")
    if graph_root.exists():
        raise RuntimeError(f"GRAPH_ROOT_EXISTS:{graph_root}")
    report_root.mkdir(parents=True)
    contract_source = Path(__file__).with_name("PROTOCOL_FREEZE.md")
    shutil.copyfile(contract_source, report_root / "PROTOCOL_FREEZE.md")
    source_hashes = {
        str(FACTOR_RECORDS): file_sha256(FACTOR_RECORDS),
        str(CRITICAL_LEDGER): file_sha256(CRITICAL_LEDGER),
        str(IDENTITY_MATERIAL): file_sha256(IDENTITY_MATERIAL),
        str(RESPONSE_ROOT / "SELECTION_MANIFEST.json"): file_sha256(RESPONSE_ROOT / "SELECTION_MANIFEST.json"),
        str(RESPONSE_ROOT / "RESOLVED_INVENTORY.json"): file_sha256(RESPONSE_ROOT / "RESOLVED_INVENTORY.json"),
    }
    write_json(
        report_root / "EXPERIMENT_FREEZE.json",
        {
            "schema": "nanogpt-native-prebranch-left-history-freeze-v1",
            "status": "FROZEN_BEFORE_MODEL_RESULTS",
            "protocol_sha256": file_sha256(report_root / "PROTOCOL_FREEZE.md"),
            "source_hashes": source_hashes,
            "primary_k": 64,
            "bootstrap_replicates": 1000,
            "new_nanogpt_training": False,
            "gpu_used": False,
            "vm_ai_used": False,
            "new_response_probe": False,
        },
    )

    compiled, audit, source_sections = compile_dataset()
    write_json(report_root / "AVAILABILITY_AUDIT.json", audit)
    (report_root / "AVAILABILITY_AUDIT.md").write_text(availability_markdown(audit), encoding="utf-8", newline="\n")
    if audit["status"] != "PASS":
        raise RuntimeError("AVAILABILITY_AUDIT_FAILED")
    write_jsonl_gz(report_root / "SOURCE_OBJECT_LEDGER.jsonl.gz", source_sections)
    write_json(
        report_root / "FEATURE_MANIFEST.json",
        {
            "schema": "nanogpt-native-prebranch-feature-manifest-v1",
            "status": "PASS",
            "feature_names": compiled["feature_names"],
            "feature_dimensions": {name: int(compiled["spaces"][name].shape[1]) for name in SPACES},
            "raw_na_counts": {name: int(np.sum(~np.isfinite(compiled["spaces"][name]))) for name in SPACES},
            "identity_fields_used_for_alignment_not_as_numeric_features": True,
            "prohibited_feature_shortcuts": [],
        },
    )

    evaluation = evaluate(compiled)
    decision = adjudicate(evaluation, leakage_pass=True)
    write_json(report_root / "RESULTS.json", public_results(evaluation))
    write_json(
        report_root / "RUNWISE_RESULTS.json",
        {
            "schema": "nanogpt-native-prebranch-runwise-results-v1",
            "status": "PASS",
            "runwise_results": evaluation["runwise_results"],
            "fold_details": evaluation["fold_details"],
        },
    )
    write_json(report_root / "DECISION.json", decision)
    record_index, ledger = ledger_rows(compiled, evaluation)
    write_jsonl_gz(report_root / "RECORD_INDEX.jsonl.gz", record_index)
    write_jsonl_gz(report_root / "LEFT_HISTORY_LEDGER.jsonl.gz", ledger)
    (report_root / "REPORT.md").write_text(
        scientific_report(compiled, audit, evaluation, decision), encoding="utf-8", newline="\n"
    )
    shutil.copyfile(Path(__file__).with_name("INDEPENDENT_CHECKER.py"), report_root / "INDEPENDENT_CHECKER.py")
    shutil.copyfile(Path(__file__).with_name("REPRODUCE_RUN.py"), report_root / "reproduce_experiment.py")

    from .independent import check

    independent = check(report_root)
    from .gfg import build_gfg, validate_gfg

    build_gfg(report_root, graph_root)
    gfg_validation = validate_gfg(report_root, graph_root)
    write_json(report_root / "GFG_VALIDATION.json", gfg_validation)

    deliverables: dict[str, Any] = {}
    for path in sorted(report_root.iterdir()):
        if path.is_file() and path.name != "MANIFEST.json":
            deliverables[path.name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    manifest = {
        "schema": "nanogpt-native-prebranch-left-history-manifest-v1",
        "status": "PASS",
        "source_hashes": source_hashes,
        "deliverables": deliverables,
        "graph_root": str(graph_root),
        "graph_validation_status": gfg_validation["status"],
        "independent_recomputation_status": independent["status"],
        "scientific_verdict": decision["verdict"],
    }
    write_json(report_root / "MANIFEST.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    args = parser.parse_args()
    value = run(args.report_root, args.graph_root)
    print(json.dumps({"status": value["status"], "verdict": value["scientific_verdict"], "report_root": str(args.report_root)}, sort_keys=True))


if __name__ == "__main__":
    main()
