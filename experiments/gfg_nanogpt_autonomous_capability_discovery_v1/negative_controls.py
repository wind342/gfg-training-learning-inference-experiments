from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any

from .candidate_validator import _scan_python, runtime_value_failures
from .common import payload_sha256


def _injected_source_control(source: str, expected: str) -> bool:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "candidate.py"
        path.write_text(source, encoding="utf-8")
        return expected in _scan_python(path)


def evaluate_negative_controls(
    instance_attestations: list[dict[str, Any]],
) -> dict[str, Any]:
    controls = {
        "candidate_absolute_path_state_rejected": bool(
            runtime_value_failures(
                {"prefix": "D:\\private\\future"}, "$.state"
            )
        ),
        "candidate_filesystem_import_rejected": _injected_source_control(
            "from pathlib import Path\n", "FORBIDDEN_IMPORT:pathlib"
        ),
        "candidate_network_import_rejected": _injected_source_control(
            "import requests\n", "FORBIDDEN_IMPORT:requests"
        ),
        "candidate_run_id_lookup_rejected": _injected_source_control(
            "key = 'run_id'\n", "FORBIDDEN_LOOKUP_LITERAL"
        ),
        "candidate_wall_clock_lookup_rejected": _injected_source_control(
            "key = 'wall_clock'\n", "FORBIDDEN_LOOKUP_LITERAL"
        ),
        "candidate_open_call_rejected": _injected_source_control(
            "value = open('x')\n", "FORBIDDEN_CALL:open"
        ),
        "all_orientation_targets_unreadable_before_release": all(
            row["session"]["orientation_validation"]["gates"][
                "target_gfg_unreadable_before_release"
            ]
            for row in instance_attestations
        ),
        "all_participant_access_audits_clean": all(
            row["session"]["participant_access_audit"]["status"] == "PASS"
            for row in instance_attestations
        ),
        "all_forecasts_report_zero_future_reads": all(
            row["sealed_forecast"]["future_gfg_reads"] == 0
            for row in instance_attestations
        ),
        "all_gfgs_reject_approximate_temporal_join": all(
            all(
                validation["gates"]["no_approximate_temporal_join"]
                for validation in row["gfg_validations"]
            )
            for row in instance_attestations
        ),
        "all_candidate_seals_unchanged": all(
            all(check["status"] == "PASS" for check in row["seal_checks"])
            for row in instance_attestations
        ),
        "all_checkpoint_forks_exact": all(
            row["fork_audit"]["identical"]
            for row in instance_attestations
        ),
        "all_branch_only_difference_audits_pass": all(
            row["branch_audit"]["status"] == "PASS"
            for row in instance_attestations
        ),
        "three_sessions_have_distinct_attestations": (
            len(
                {
                    row["session"]["attestation_sha256"]
                    for row in instance_attestations
                }
            )
            == len(instance_attestations)
            == 3
        ),
        "previous_candidate_nonvisibility": all(
            row["session"]["participant_access_audit"][
                "forbidden_success_count"
            ]
            == 0
            for row in instance_attestations
        ),
    }
    material = {
        "control_count": len(controls),
        "controls": controls,
        "failed_controls": sorted(
            name for name, passed in controls.items() if not passed
        ),
        "schema": "formal-negative-control-summary-v1",
        "status": "PASS" if all(controls.values()) else "FAIL",
    }
    material["summary_sha256"] = payload_sha256(material)
    return material
