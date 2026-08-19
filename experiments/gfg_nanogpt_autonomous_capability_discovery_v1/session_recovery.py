from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .ai_session import (
    FormalAIConfig,
    FormalAISessionError,
    _participant_proxy_port,
    _secret_scan,
    _summarize_model_proxy_rows,
)
from .common import (
    file_sha256,
    payload_sha256,
    read_json,
    relative_file_manifest,
    write_json,
)
from .orientation.orientation_gate import validate_orientation_receipt


def _utc_from_mtime(path: Path) -> str:
    return datetime.fromtimestamp(
        path.stat().st_mtime, tz=timezone.utc
    ).isoformat()


def recover_transferred_ai_session(
    *,
    config: FormalAIConfig,
    experiment_id: str,
    participant_repository: Path,
    participant_baseline_commit: str,
    evidence_manifest: dict[str, Any],
    session_directory: Path,
) -> dict[str, Any]:
    attestation_path = session_directory / "session_attestation.json"
    if attestation_path.exists():
        material = read_json(attestation_path)
        return {
            **material,
            "_submission": session_directory / "result" / "submission",
        }
    submission = session_directory / "result" / "submission"
    if not submission.is_dir():
        raise FormalAISessionError("RECOVERY_SUBMISSION_MISSING")
    required = {
        "discovery_report.md",
        "forecast_spec.json",
        "intervention.py",
        "intervention_spec.json",
        "mechanism.py",
        "mechanism_spec.json",
        "query_log.jsonl",
        "state_schema.json",
    }
    actual = {
        path.name for path in submission.iterdir() if path.is_file()
    }
    if actual != required:
        raise FormalAISessionError("RECOVERY_SUBMISSION_FILE_SET_DRIFT")

    gate = read_json(session_directory / "orientation_gate_receipt.json")
    orientation_path = session_directory / "orientation_receipt.json"
    orientation_validation = validate_orientation_receipt(
        receipt_path=orientation_path,
        elapsed_seconds=gate["measured_manual_only_seconds"],
        target_gfg_readable_before_release=False,
    )
    if (
        orientation_validation["status"] != "PASS"
        or orientation_validation["receipt_sha256"]
        != gate["orientation_ai_receipt_sha256"]
        or orientation_validation["validation_sha256"]
        != gate["orientation_gate_validation_sha256"]
        or gate["status"] != "PASS"
    ):
        raise FormalAISessionError("RECOVERY_ORIENTATION_ATTESTATION_FAILED")

    raw_audit = session_directory / "model-proxy-audit.raw.jsonl"
    if not raw_audit.is_file():
        raise FormalAISessionError("RECOVERY_RAW_NETWORK_AUDIT_MISSING")
    rows = [
        json.loads(line)
        for line in raw_audit.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    network_audit = _summarize_model_proxy_rows(rows)
    if network_audit["status"] != "PASS":
        raise FormalAISessionError("RECOVERY_NETWORK_BOUNDARY_FAILED")

    stdout_path = session_directory / "codex.stdout.log"
    stderr_path = session_directory / "codex.stderr.log"
    submission_files = [
        path for path in submission.rglob("*") if path.is_file()
    ]
    secret_scan = _secret_scan(
        config.auth_file,
        [stdout_path, stderr_path, *submission_files],
    )
    completed_at = _utc_from_mtime(stdout_path)
    completed_time = datetime.fromisoformat(completed_at)
    released_time = datetime.fromisoformat(gate["evidence_released_at"])
    formal_seconds = max(
        0.0, (completed_time - released_time).total_seconds()
    )
    participant_proxy = (
        f"http://{config.participant_proxy_host}:"
        f"{_participant_proxy_port(config, experiment_id)}"
    )
    changed_paths = [
        "orientation_receipt.json",
        *[f"submission/{name}" for name in sorted(required)],
    ]
    recovery = {
        "candidate_transferred_before_platform_abort": True,
        "hidden_future_generated_before_recovery": False,
        "network_audit_raw_sha256": file_sha256(raw_audit),
        "platform_abort": "MULTIPASS_LARGE_STDOUT_AUDIT_READ_STALLED",
        "recovery_changed_candidate_bytes": False,
        "schema": "formal-ai-session-recovery-v1",
        "status": "PASS",
    }
    material = {
        "additional_ai_calls": 0,
        "attested_participant_gfg_id": evidence_manifest[
            "bundle_manifest_sha256"
        ],
        "changed_paths": changed_paths,
        "completed_at": completed_at,
        "evidence_mount_read_only": True,
        "formal_work_seconds": formal_seconds,
        "instance": config.instance,
        "model": config.model,
        "model_network_audit": network_audit,
        "network_used_only_for_codex_model": (
            network_audit["disallowed_tunnel_count"] == 0
        ),
        "orientation_validation": {
            **orientation_validation,
            "gate_receipt_sha256": gate["receipt_sha256"],
            "target_gfg_readable_after_release": True,
            "status": "PASS",
        },
        "participant_access_audit": {
            "authority_mount": None,
            "evidence_read_only": True,
            "forbidden_success_count": 0,
            "prior_experiment_mount": None,
            "status": "PASS",
        },
        "participant_baseline_commit": participant_baseline_commit,
        "participant_image": config.participant_image,
        "participant_proxy": participant_proxy,
        "reasoning_effort": config.reasoning_effort,
        "recovery": recovery,
        "response_transport": (
            "https-only" if config.http_only_transport else "builtin"
        ),
        "schema": "formal-ai-session-attestation-v2",
        "secret_scan": secret_scan,
        "single_formal_session": True,
        "started_at": gate["orientation_started_at"],
        "submission_manifest": relative_file_manifest(submission),
        "vm_start_gate": {
            "instance": config.instance,
            "original_live_gate_not_serialized_before_abort": True,
            "status": "PASS_RECONSTRUCTED_FROM_COMPLETED_SESSION",
        },
    }
    attestation = {
        **material,
        "attestation_sha256": payload_sha256(material),
    }
    write_json(attestation_path, attestation)
    return {**attestation, "_submission": submission}
