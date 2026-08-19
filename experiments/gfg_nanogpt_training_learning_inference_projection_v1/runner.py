from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil
from typing import Any

import torch

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    file_sha256,
    payload_sha256,
)

from .analysis import analyse_run, summarize
from .gfg import aggregate_manifest, validate_run_gfg, write_run_gfg
from .independent import check
from .runtime import execute_run, select_phases, validated_accuracy_curve


PACKAGE = Path(__file__).parent
REPOSITORY = PACKAGE.parents[1]
DEFAULT_REPORT_ROOT = (
    REPOSITORY
    / "experiments"
    / "gfg_nanogpt_cumulative_scientist_v1"
    / "reports"
    / "training_learning_inference_projection_v1"
)
DEFAULT_GRAPH_ROOT = Path(r"E:\gfg-evidence\nanogpt-training-learning-inference-projection-v1\gfg")
DEFAULT_SOURCE_ARCHIVE = REPOSITORY / "data_private" / "gfg_nanogpt_mechanism_discovery_archive" / "gfg"
DEFAULT_SUPPORT_ARCHIVE = REPOSITORY / "data_private" / "gfg_nanogpt_support_redundancy_v1" / "formal-v1"
DEFAULT_TRAINER_ROOT = Path(r"E:\gfg-downloads\nanoGPT-3adf61e")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def _report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Training-learning-inference support projection experiment",
        "",
        f"**Machine decision: `{summary['status']}`.**",
        "",
        "All thirteen validated nanoGPT runs were evaluated. No new training was performed. Every observation was a real CUDA inference from an exact historical parameter checkpoint on held-out validation rows.",
        "",
        "## Primary gates",
        "",
        "| gate | result |",
        "|---|---|",
    ]
    for name, value in summary["gates"].items():
        lines.append(f"| {name} | {'PASS' if value else 'FAIL'} |")
    lines.extend(
        [
            "",
            "## Run-level execution",
            "",
            "| run | pre/formed step | decline step and accuracy | recovery step and accuracy | formed group profiles | non-additive pair/group cells | rollback components changing logits |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        phases = row["phases"]
        recovery_key = "recovered" if "recovered" in phases else "post_decline_not_recovered"
        rollback_changed = sum(value["rollback_changed_logits"] for value in row["rollback"].values())
        lines.append(
            f"| {row['entry_id']} | {phases['pre_formation']['optimizer_step']}/{phases['formed']['optimizer_step']} | "
            f"{phases['decline']['optimizer_step']} ({phases['decline']['validation_accuracy']:.4f}) | "
            f"{phases[recovery_key]['optimizer_step']} ({phases[recovery_key]['validation_accuracy']:.4f}) | "
            f"{phases['formed']['distinct_target_group_support_profiles']} | "
            f"{phases['formed']['nonadditive_pair_group_count_at_1e_6']} | {rollback_changed}/4 |"
        )
    lines.extend(
        [
            "",
            "## Result",
            "",
            summary["interpretation"],
            "",
            "The evidence distinguishes three objects: training-established parameter versions, the learned distributed support measured by target-group causal gates, and new concrete inference occurrences that call and combine that support for held-out queries. The component-version rollback is explicitly a hybrid causal intervention; it is not represented as a natural training state.",
            "",
            f"Observed {summary['native_component_call_count']} native component calls across {summary['phase_observation_count']} phase checkpoints. The minimum number of distinct target-group support profiles in any formed run was {summary['formed_distinct_group_profile_minimum']}; the minimum number of non-additive component-pair/group cells was {summary['formed_nonadditive_pair_group_minimum']}.",
            "",
            "## Claim boundary",
            "",
            summary["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def run(
    *,
    report_root: Path,
    graph_root: Path,
    source_archive_root: Path,
    support_archive_root: Path,
    trainer_root: Path,
) -> None:
    require(not report_root.exists(), f"TLI_REPORT_ROOT_EXISTS:{report_root}")
    require(not graph_root.exists(), f"TLI_GRAPH_ROOT_EXISTS:{graph_root}")
    require((trainer_root / "model.py").exists(), "TLI_TRAINER_MODEL_MISSING")
    archive = read_json(support_archive_root / "archive_manifest.json")
    require(archive["status"] == "PASS" and len(archive["support_bundles"]) == 13, "TLI_SUPPORT_ARCHIVE_INVALID")
    items = sorted(archive["support_bundles"], key=lambda value: value["entry_id"])

    phase_selections: dict[str, Any] = {}
    for item in items:
        curve = validated_accuracy_curve(support_archive_root / item["gfg_bundle_id"])
        phase_selections[item["entry_id"]] = select_phases(curve)

    report_root.mkdir(parents=True)
    graph_root.mkdir(parents=True)
    protocol = PACKAGE / "PROTOCOL_FREEZE.md"
    shutil.copy2(protocol, report_root / "PROTOCOL_FREEZE.md")
    source_files = sorted(PACKAGE.glob("*.py")) + [protocol, support_archive_root / "archive_manifest.json", trainer_root / "model.py"]
    freeze = {
        "schema": "nanogpt-training-learning-inference-projection-freeze-v1",
        "status": "FROZEN_BEFORE_NEW_NATIVE_INFERENCE",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_commit": "3adf61e154c3fe3fca428ad6bc3818b27a3b8291",
        "entry_count": 13,
        "entries": [item["entry_id"] for item in items],
        "phase_selections": phase_selections,
        "phase_selection_rule": "first_accuracy_ge_0.90; largest_postformation_adjacent_drop; first_later_accuracy_ge_0.90",
        "component_registry": ["h0.attn", "h0.mlp", "h1.attn", "h1.mlp"],
        "single_gate_count_per_phase": 4,
        "pair_gate_count_per_phase": 6,
        "rollback_count_per_run": 4,
        "new_training": False,
        "native_cuda_inference": True,
        "held_out_validation_queries": True,
        "source_hashes": {str(path.resolve()): file_sha256(path) for path in source_files},
    }
    freeze["freeze_sha256"] = payload_sha256(freeze)
    write_json(report_root / "EXPERIMENT_FREEZE.json", freeze)

    rows: list[dict[str, Any]] = []
    gfg_validations: list[dict[str, Any]] = []
    for item in items:
        entry_id = item["entry_id"]
        bundle_id = item["gfg_bundle_id"]
        record = execute_run(
            entry_id=entry_id,
            source_bundle_id=bundle_id,
            source_bundle=source_archive_root / bundle_id,
            trainer_root=trainer_root,
            phase_selection=phase_selections[entry_id],
        )
        ledger, phase_arrays = analyse_run(record)
        write_run_gfg(
            record=record,
            ledger=ledger,
            phase_arrays=phase_arrays,
            graph_root=graph_root,
            protocol_sha256=file_sha256(protocol),
        )
        gfg_validations.append(validate_run_gfg(graph_root / entry_id))
        rows.append(ledger)
        del record
        torch.cuda.empty_cache()

    archive_manifest = aggregate_manifest(graph_root, rows)
    write_rows(report_root / "OBSERVATION_LEDGER.jsonl.gz", rows)
    write_json(report_root / "PHASE_SELECTION.json", {
        "schema": "nanogpt-training-learning-inference-phase-selection-v1",
        "status": "SEALED_BEFORE_NATIVE_INFERENCE",
        "selections": phase_selections,
    })
    summary = summarize(rows)
    write_json(report_root / "RESULTS.json", summary)
    write_json(report_root / "GFG_VALIDATION_SUMMARY.json", {
        "schema": "nanogpt-training-learning-inference-gfg-validation-summary-v1",
        "status": "PASS" if all(value["status"] == "PASS" for value in gfg_validations) else "FAIL",
        "entry_count": len(gfg_validations),
        "validations": gfg_validations,
        "archive_manifest_sha256": file_sha256(graph_root / "ARCHIVE_MANIFEST.json"),
    })
    (report_root / "REPORT.md").write_text(_report(summary, rows), encoding="utf-8", newline="\n")
    independent = check(
        report_root=report_root,
        graph_root=graph_root,
        source_archive_root=source_archive_root,
        trainer_root=trainer_root,
    )
    final = {
        "schema": "nanogpt-training-learning-inference-projection-final-manifest-v1",
        "status": "PASS" if summary["status"] == independent["status"] == archive_manifest["status"] == "PASS" else "FAIL",
        "result_sha256": file_sha256(report_root / "RESULTS.json"),
        "observation_ledger_sha256": file_sha256(report_root / "OBSERVATION_LEDGER.jsonl.gz"),
        "independent_check_sha256": file_sha256(report_root / "INDEPENDENT_CHECK.json"),
        "gfg_archive_manifest_sha256": file_sha256(graph_root / "ARCHIVE_MANIFEST.json"),
        "new_training": False,
        "native_cuda_inference": True,
        "run_count": 13,
    }
    write_json(report_root / "FINAL_MANIFEST.json", final)
    (report_root / ("READY" if final["status"] == "PASS" else "FAILED")).write_text(final["status"] + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    parser.add_argument("--source-archive-root", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--support-archive-root", type=Path, default=DEFAULT_SUPPORT_ARCHIVE)
    parser.add_argument("--trainer-root", type=Path, default=DEFAULT_TRAINER_ROOT)
    args = parser.parse_args()
    run(
        report_root=args.report_root.resolve(),
        graph_root=args.graph_root.resolve(),
        source_archive_root=args.source_archive_root.resolve(),
        support_archive_root=args.support_archive_root.resolve(),
        trainer_root=args.trainer_root.resolve(),
    )


if __name__ == "__main__":
    main()
