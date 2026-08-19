from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import file_sha256

from .analysis import analyse_run
from .runtime import execute_run


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _compare_entry(expected: dict[str, Any], observed: dict[str, Any]) -> list[str]:
    checks: list[str] = []
    require(expected["phase_selection"] == observed["phase_selection"], "TLI_REPLAY_PHASE_SELECTION")
    for phase in expected["phases"]:
        left = expected["phases"][phase]
        right = observed["phases"][phase]
        require(left["baseline_logits_sha256"] == right["baseline_logits_sha256"], f"TLI_REPLAY_BASELINE:{phase}")
        require(left["single_gate_logits_sha256"] == right["single_gate_logits_sha256"], f"TLI_REPLAY_SINGLE_GATE:{phase}")
        require(left["pair_gate_logits_sha256"] == right["pair_gate_logits_sha256"], f"TLI_REPLAY_PAIR_GATE:{phase}")
        require(left["support_profile_sha256"] == right["support_profile_sha256"], f"TLI_REPLAY_SUPPORT_PROFILE:{phase}")
    checks.append("all_phase_native_baseline_and_gate_hashes")
    for component in expected["rollback"]:
        require(
            expected["rollback"][component]["rollback_logits_sha256"]
            == observed["rollback"][component]["rollback_logits_sha256"],
            f"TLI_REPLAY_ROLLBACK:{component}",
        )
        require(observed["rollback"][component]["restore_byte_exact"], f"TLI_REPLAY_RESTORE:{component}")
    checks.append("all_formed_component_rollbacks_and_restorations")
    return checks


def check(
    *,
    report_root: Path,
    graph_root: Path,
    source_archive_root: Path,
    trainer_root: Path,
) -> dict[str, Any]:
    freeze = read_json(report_root / "EXPERIMENT_FREEZE.json")
    summary = read_json(report_root / "RESULTS.json")
    rows = read_rows(report_root / "OBSERVATION_LEDGER.jsonl.gz")
    archive = read_json(graph_root / "ARCHIVE_MANIFEST.json")
    checks: list[str] = []
    require(len(rows) == 13 == summary["run_count"], "TLI_INDEPENDENT_RUN_COUNT")
    require(archive["status"] == "PASS" and archive["entry_count"] == 13, "TLI_INDEPENDENT_GFG_ARCHIVE")
    require(all(entry["validation_status"] == "PASS" for entry in archive["entries"]), "TLI_INDEPENDENT_GFG_ENTRY")
    checks.append("all_13_run_gfg_validations")
    for path, digest in freeze["source_hashes"].items():
        require(file_sha256(Path(path)) == digest, f"TLI_INDEPENDENT_SOURCE_HASH:{path}")
    checks.append("frozen_source_hashes")
    require(all(summary["gates"].values()), "TLI_INDEPENDENT_REPORTED_GATE")
    require(
        all(
            phase["baseline_repeat_exact"] and phase["parameter_identity_exact"]
            for row in rows
            for phase in row["phases"].values()
        ),
        "TLI_INDEPENDENT_IDENTITY_OR_REPEAT",
    )
    checks.append("identity_and_repeat_ledgers")
    require(
        all(
            row["at_least_one_rollback_changed_logits"] and row["all_rollbacks_restored_exactly"]
            for row in rows
        ),
        "TLI_INDEPENDENT_ROLLBACK_LEDGER",
    )
    checks.append("rollback_change_and_exact_restore_ledgers")

    row_by_entry = {row["entry_id"]: row for row in rows}
    replay_entries = [sorted(row_by_entry)[0], sorted(row_by_entry)[-1]]
    replay_rows = []
    for entry_id in replay_entries:
        expected = row_by_entry[entry_id]
        bundle_id = expected["source_bundle_id"]
        replay = execute_run(
            entry_id=entry_id,
            source_bundle_id=bundle_id,
            source_bundle=source_archive_root / bundle_id,
            trainer_root=trainer_root,
            phase_selection=expected["phase_selection"],
        )
        observed, _arrays = analyse_run(replay)
        replay_checks = _compare_entry(expected, observed)
        replay_rows.append({"entry_id": entry_id, "status": "PASS", "checks": replay_checks})
    checks.append("two_run_independent_native_replay")
    result = {
        "schema": "nanogpt-training-learning-inference-independent-check-v1",
        "status": "PASS",
        "check_count": len(checks),
        "checks": checks,
        "replay_entries": replay_rows,
        "future_information_used": False,
        "new_training_performed": False,
        "native_cuda_inference_replayed": True,
    }
    write_json(report_root / "INDEPENDENT_CHECK.json", result)
    return result


__all__ = ["check"]

