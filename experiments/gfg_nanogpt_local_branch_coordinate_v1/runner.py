from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil
from typing import Any

from .analysis import (
    CANDIDATES,
    CONFIRMATION_RUNS,
    DEVELOPMENT_RUNS,
    FACTOR_RECORDS,
    IDENTITY_MATERIAL,
    INVENTORY_PATH,
    compile_coordinate_dataset,
    evaluate_coordinate,
    file_sha256,
)
from .gfg import build_gfg, validate_gfg
from .independent import check


DEFAULT_REPORT_ROOT = Path(__file__).parents[1] / "gfg_nanogpt_cumulative_scientist_v1" / "reports" / "local_branch_coordinate_v1"
DEFAULT_GRAPH_ROOT = Path(r"E:\gfg-evidence\nanogpt-local-branch-coordinate-v1\gfg")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def report(results: dict[str, Any], audit: dict[str, Any]) -> str:
    selection = results["selection"]
    confirm = results["confirmation"]
    decision = results["decision"]
    baseline = confirm["confirmation_branch_risk"]["X3"]
    augmented = confirm["confirmation_branch_risk"]["X3_plus_q"]
    response = confirm["confirmation_response"]
    lines = [
        "# Local branch coordinate discovery experiment",
        "",
        f"**Decision: `{decision['verdict']}`.**",
        "",
        "The experiment froze eight pre-response relational coordinates, selected exactly one on eight development runs, and evaluated it without rerolling on four untouched confirmation runs.",
        "",
        f"Selected coordinate: `{selection['selected_coordinate']}`.",
        "",
        "## Development-only selection",
        "",
        "| candidate coordinate | LORO severe-conflict PR-AUC | ROC-AUC | Brier |",
        "|---|---:|---:|---:|",
    ]
    for name in selection["candidate_ranking"]:
        value = selection["candidate_metrics"][name]
        lines.append(f"| {name} | {value['pr_auc']:.4f} | {value['roc_auc']:.4f} | {value['brier']:.4f} |")
    lines.extend([
        "",
        "## Untouched confirmation runs",
        "",
        "| coordinate space | severe-conflict PR-AUC | ROC-AUC | Brier |",
        "|---|---:|---:|---:|",
        f"| X3 | {baseline['pr_auc']:.4f} | {baseline['roc_auc']:.4f} | {baseline['brier']:.4f} |",
        f"| X3 + q | {augmented['pr_auc']:.4f} | {augmented['roc_auc']:.4f} | {augmented['brier']:.4f} |",
        "",
        f"PR-AUC delta: `{decision['confirmation_pr_auc_delta']:.4f}`; non-negative per-run deltas: `{decision['confirmation_runs_nonnegative_pr_auc_delta']}/4`.",
        "",
        "## Does the coordinate help the same KNN transport response curves?",
        "",
        "| method | overall curve RMSE | severe curve RMSE | overall boundary accuracy | severe boundary accuracy |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in ("X3", "X3_plus_q", "oracle_same_true_branch_X3_plus_q"):
        value = response[name]
        lines.append(f"| {name} | {value['overall']['curve_rmse']:.6f} | {value['severe_conflict']['curve_rmse']:.6f} | {value['overall']['boundary_accuracy']:.4f} | {value['severe_conflict']['boundary_accuracy']:.4f} |")
    lines.extend([
        "",
        f"Relative severe-curve RMSE change: `{decision['severe_curve_rmse_relative_improvement']:.2%}`. Overall boundary-accuracy delta: `{decision['overall_boundary_accuracy_delta']:.4f}`.",
        "",
        "## Matched local comparison",
        "",
        f"Every confirmation severe record was matched to its nearest development ordinary record in X3 before q was examined. The `{confirm['matched_pair_summary']['pair_count']}` pairs have median X3 distance `{confirm['matched_pair_summary']['median_X3_distance']:.4f}` and median absolute standardized q separation `{confirm['matched_pair_summary']['median_absolute_q_separation']:.4f}`.",
        "",
        "## Interpretation boundary",
        "",
        "A positive result would establish a pre-response local coordinate for this executed witness family, not a universal training coordinate. A negative result would mean that these eight semantically frozen relations do not yet distinguish the local branch; it would not erase the established F1/F3/F5 response space. The same KNN and fixed k=64 are used throughout. No new training, VM-AI inference, response probing or confirmation-driven coordinate search occurred.",
        "",
        f"Availability: `{audit['status']}` across `{audit['record_count']}` records and `{audit['section_count']}` response sections.",
        "",
    ])
    return "\n".join(lines)


def run(report_root: Path, graph_root: Path) -> None:
    if report_root.exists():
        raise RuntimeError(f"REPORT_ROOT_EXISTS:{report_root}")
    if graph_root.exists():
        raise RuntimeError(f"GRAPH_ROOT_EXISTS:{graph_root}")
    report_root.mkdir(parents=True)
    protocol_source = Path(__file__).with_name("PROTOCOL_FREEZE.md")
    shutil.copy2(protocol_source, report_root / "PROTOCOL_FREEZE.md")
    source_paths = [FACTOR_RECORDS, IDENTITY_MATERIAL, INVENTORY_PATH, protocol_source]
    freeze = {
        "schema": "nanogpt-local-branch-coordinate-experiment-freeze-v1",
        "status": "FROZEN_BEFORE_OUTCOME_EVALUATION",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_runs": list(DEVELOPMENT_RUNS),
        "confirmation_runs": list(CONFIRMATION_RUNS),
        "candidate_coordinates": list(CANDIDATES),
        "selection_metric": "development-only complete LORO severe-conflict PR-AUC",
        "knn_k": 64,
        "new_training": False,
        "new_response_probe": False,
        "vm_ai_used": False,
        "source_hashes": {str(path): file_sha256(path) for path in source_paths},
    }
    write_json(report_root / "EXPERIMENT_FREEZE.json", freeze)

    compiled, audit, source_ledger = compile_coordinate_dataset()
    write_json(report_root / "AVAILABILITY_AUDIT.json", audit)
    write_rows(report_root / "SOURCE_OBJECT_LEDGER.jsonl.gz", source_ledger)
    write_json(report_root / "FEATURE_MANIFEST.json", {
        "schema": "nanogpt-local-branch-coordinate-feature-manifest-v1",
        "status": "PASS",
        "known_coordinate_space": "X3",
        "candidate_coordinates": list(CANDIDATES),
        "distance": "sqrt(mean(robust_scaled_X3_delta^2)+robust_scaled_q_delta^2)",
        "post_response_input_count": 0,
        "identity_input_count": 0,
        "candidate_finite_counts": audit["finite_counts"],
    })

    evaluation, ledger = evaluate_coordinate(compiled)
    selection = evaluation["selection"]
    write_json(report_root / "CANDIDATE_SELECTION.json", selection)
    write_json(report_root / "DEVELOPMENT_RESULTS.json", {
        "schema": "nanogpt-local-branch-coordinate-development-results-v1",
        "status": "PASS",
        "selected_coordinate": selection["selected_coordinate"],
        "runwise": evaluation["development"]["runwise"],
        "confirmation_results_used": False,
    })
    confirmation = evaluation["confirmation"]
    matches = confirmation.pop("matched_pairs")
    write_json(report_root / "MATCHED_PAIRS.json", {"schema": "nanogpt-local-branch-coordinate-matched-pairs-v1", "status": "PASS", "pairs": matches})
    write_json(report_root / "RESULTS.json", confirmation)
    write_rows(report_root / "CONFIRMATION_LEDGER.jsonl.gz", ledger)
    write_json(report_root / "DECISION.json", evaluation["decision"])
    (report_root / "REPORT.md").write_text(report(evaluation, audit), encoding="utf-8", newline="\n")
    check(report_root)
    decision = read_json(report_root / "DECISION.json")
    decision["gates"].pop("independent_and_gfg_validation_pending", None)
    decision["gates"]["independent_recomputation_pass"] = True
    decision["gfg_validation_recorded_separately"] = True
    write_json(report_root / "DECISION.json", decision)
    manifest = build_gfg(report_root, graph_root)
    validation = validate_gfg(report_root, graph_root)
    write_json(report_root / "FINAL_MANIFEST.json", {
        "schema": "nanogpt-local-branch-coordinate-final-manifest-v1",
        "status": "PASS",
        "verdict": decision["verdict"],
        "selected_coordinate": decision["selected_coordinate"],
        "decision_sha256": file_sha256(report_root / "DECISION.json"),
        "independent_sha256": file_sha256(report_root / "INDEPENDENT_RECOMPUTATION.json"),
        "gfg_manifest_sha256": file_sha256(graph_root / "GFG_MANIFEST.json"),
        "gfg_validation_sha256": file_sha256(graph_root / "GFG_VALIDATION.json"),
        "gfg_validation_status": validation["status"],
    })
    (report_root / "READY").write_text("READY\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    args = parser.parse_args()
    run(args.report_root.resolve(), args.graph_root.resolve())


if __name__ == "__main__":
    main()
