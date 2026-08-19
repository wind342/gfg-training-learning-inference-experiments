from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

from .common import (
    BASELINE_COMMIT,
    NANOGPT_COMMIT,
    file_sha256,
    payload_sha256,
    read_json,
    write_json,
)


BASELINE_BRANCH = "codex/datafusion-nanogpt-optimization-experiments-v1"
FINAL_COMMIT_SENTINEL = "CONTAINING_COMMIT_IS_FINAL_COMMIT"


def _tree_manifest(root: Path) -> tuple[list[dict[str, Any]], str]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == "external_archive_manifest.json":
            continue
        rows.append(
            {
                "byte_count": path.stat().st_size,
                "path": relative,
                "sha256": file_sha256(path),
            }
        )
    return rows, payload_sha256(rows)


def _changed_paths(repository: Path) -> tuple[list[str], list[str]]:
    committed = subprocess.check_output(
        [
            "git",
            "diff",
            "--name-only",
            BASELINE_COMMIT,
            "--",
        ],
        cwd=repository,
        text=True,
    ).splitlines()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=repository,
        text=True,
    ).splitlines()
    untracked = [line[3:] for line in status if line.startswith("?? ")]
    changed = sorted(
        {
            path.replace("\\", "/")
            for path in [*committed, *untracked]
        }
    )
    allowed = (
        "experiments/gfg_nanogpt_autonomous_capability_discovery_v1/",
        "tests/experiments/gfg_nanogpt_autonomous_capability_discovery_v1/",
    )
    protected = [
        path for path in changed if not path.startswith(allowed)
    ]
    return changed, protected


def build_public_summary(
    *,
    repository: Path,
    private_root: Path,
    trainer_root: Path,
    aggregate: dict[str, Any],
    instance_details: list[dict[str, Any]],
    negative_controls: dict[str, Any],
    test_summary: dict[str, Any],
) -> dict[str, Any]:
    experiment_root = (
        repository
        / "experiments"
        / "gfg_nanogpt_autonomous_capability_discovery_v1"
    )
    contracts = experiment_root / "contracts"
    commitments = read_json(contracts / "formal_instance_commitments.json")
    freeze = read_json(
        experiment_root / "reports" / "protocol_freeze_manifest.json"
    )
    archive_rows, archive_sha = _tree_manifest(private_root)
    write_json(
        private_root / "external_archive_manifest.json",
        {
            "archive_sha256": archive_sha,
            "file_count": len(archive_rows),
            "files": archive_rows,
            "schema": "external-private-archive-manifest-v1",
        },
    )
    changed, protected = _changed_paths(repository)
    public_instances = []
    for commitment, row in zip(
        commitments["instances"], instance_details, strict=True
    ):
        public_instances.append(
            {
                "ai_session_attestation_sha256": row["session"][
                    "attestation_sha256"
                ],
                "baseline_transition_step": row["causal_validation"][
                    "baseline_transition_step"
                ],
                "candidate_seal_sha256": row["candidate_seal"][
                    "candidate_seal_sha256"
                ],
                "causal_validation": row["causal_validation"],
                "discovery_gfg_counts": row["discovery_counts"],
                "discovery_task_commitment": commitment[
                    "discovery_task_commitment"
                ],
                "discovery_training_sha256": row[
                    "discovery_training_sha256"
                ],
                "forecast_seal_sha256": row["sealed_forecast"][
                    "forecast_seal_sha256"
                ],
                "forecast_validation": row["forecast_validation"],
                "instance_attestation_sha256": row[
                    "instance_attestation_sha256"
                ],
                "instance_id": row["instance_id"],
                "intervention_transition_step": row[
                    "causal_validation"
                ]["intervention_transition_step"],
                "orientation_receipt_sha256": row["session"][
                    "orientation_validation"
                ]["receipt_sha256"],
                "participant_gfg_id": row["participant_gfg_id"],
                "participant_gfg_validation_sha256": row[
                    "participant_gfg_validation_sha256"
                ],
                "prediction_cut_step": row["sealed_forecast"][
                    "prediction_cut_step"
                ],
                "query_log_sha256": row["query_log_sha256"],
                "status": row["status"],
                "validation_task_commitment": commitment[
                    "validation_task_commitment"
                ],
            }
        )
    gfg_schema_sha = payload_sha256(
        {
            "capture": file_sha256(experiment_root / "training_capture.py"),
            "query_validation": file_sha256(
                experiment_root / "training_gfg.py"
            ),
        }
    )
    material = {
        "baseline_branch": BASELINE_BRANCH,
        "baseline_commit": BASELINE_COMMIT,
        "candidate_interface_sha256": file_sha256(
            contracts / "candidate_interface.json"
        ),
        "capability_transition_contract_sha256": file_sha256(
            contracts / "capability_transition.json"
        ),
        "capture_protocol_sha256": file_sha256(
            contracts / "capture_protocol.json"
        ),
        "changed_files": changed,
        "external_archive_sha256": archive_sha,
        "final_commit": FINAL_COMMIT_SENTINEL,
        "final_commit_resolution": (
            "The Git commit containing this summary is the final commit; "
            "a literal self-SHA cannot be embedded in its own content."
        ),
        "formal_instance_commitments_sha256": commitments[
            "commitments_sha256"
        ],
        "gfg_orientation_sha256": file_sha256(
            experiment_root / "orientation" / "GFG_MACHINE_SEMANTICS.md"
        ),
        "mechanism_discovery_guide_sha256": file_sha256(
            experiment_root
            / "orientation"
            / "EXECUTABLE_MECHANISM_DISCOVERY_GUIDE.md"
        ),
        "gfg_schema_sha256": gfg_schema_sha,
        "instances": public_instances,
        "intervention_api_sha256": file_sha256(
            contracts / "intervention_api.json"
        ),
        "nanoGPT_code_commit": NANOGPT_COMMIT,
        "nanoGPT_model_py_sha256": file_sha256(trainer_root / "model.py"),
        "negative_controls": negative_controls,
        "overall_status": aggregate["status"],
        "protected_paths_changed_files": protected,
        "protocol_freeze_commit": aggregate["protocol_freeze"][
            "protocol_freeze_commit"
        ],
        "protocol_freeze_sha256": freeze["protocol_freeze_sha256"],
        "schema": "gfg-nanogpt-autonomous-discovery-public-summary-v1",
        "task_family_sha256": file_sha256(
            contracts / "task_family.json"
        ),
        "test_summary": test_summary,
        "trained_new_research_ai": False,
        "human_modified_candidate": False,
        "human_supplied_mechanism": False,
        "added_experiments_after_failure": False,
        "scientific_input_outside_participant_gfg": False,
        "training_time_alignment_sha256": file_sha256(
            contracts / "training_time_alignment.json"
        ),
    }
    material["public_summary_sha256"] = payload_sha256(material)
    return material
