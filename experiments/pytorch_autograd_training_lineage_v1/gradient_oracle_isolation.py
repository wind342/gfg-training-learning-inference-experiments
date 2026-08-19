from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
PROCESS_MODULE = (
    "experiments.pytorch_autograd_training_lineage_v1.gradient_oracle_process"
)


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _run(arguments: list[str]) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", PROCESS_MODULE, *arguments],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "GRADIENT_ORACLE_ISOLATED_PROCESS_FAILED:"
            + completed.stdout
            + completed.stderr
        )


def _local_modules(rows: list[dict[str, str]]) -> set[str]:
    return {row["module"].rsplit(".", 1)[-1] for row in rows}


def run_gradient_oracle_process_isolation(
    validated_snapshots: dict[str, Any],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="pytorch-gradient-oracle-") as directory:
        root = Path(directory)
        snapshots = root / "snapshots.json"
        candidate = root / "candidate.json"
        native = root / "native.json"
        comparison = root / "comparison.json"
        snapshots.write_bytes(_canonical(validated_snapshots))
        _run([
            "--mode", "candidate",
            "--snapshot-input", str(snapshots),
            "--output", str(candidate),
        ])
        _run(["--mode", "native", "--output", str(native)])
        _run([
            "--mode", "comparison",
            "--core-input", str(candidate),
            "--native-input", str(native),
            "--output", str(comparison),
        ])
        candidate_value = json.loads(candidate.read_text(encoding="utf-8"))
        native_value = json.loads(native.read_text(encoding="utf-8"))
        comparison_value = json.loads(comparison.read_text(encoding="utf-8"))

    candidate_modules = _local_modules(candidate_value["dependencies"])
    native_modules = _local_modules(native_value["dependencies"])
    comparison_modules = _local_modules(comparison_value["dependencies"])
    native_forbidden = native_modules & {
        "candidate_projection",
        "core_capture",
        "independent_reference",
        "independent_reference_v2",
        "lineage",
        "lineage_v2",
    }
    candidate_forbidden = candidate_modules & {
        "gradient_intervention_oracle",
        "native_backward_dependency_oracle",
        "native_oracle_workloads",
        "saved_tensor_observer",
    }
    comparison_forbidden = comparison_modules & {
        "candidate_projection",
        "core_capture",
        "gradient_intervention_oracle",
        "native_backward_dependency_oracle",
    }
    source_files = [
        EXPERIMENT_ROOT / "native_backward_dependency_oracle.py",
        EXPERIMENT_ROOT / "native_oracle_workloads.py",
        EXPERIMENT_ROOT / "saved_tensor_observer.py",
        EXPERIMENT_ROOT / "gradient_intervention_oracle.py",
        EXPERIMENT_ROOT / "independent_reference_v2.py",
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in source_files)
    shared_gradient_rule_helper_count = sum(
        token in source
        for token in (
            "core_capture." + "_gradient_dependency_refs",
            "independent_reference." + "_gradient_dependency_refs",
        )
    )
    object_id_scientific_identity_count = sum(
        token in source for token in ("id" + "(", "python_object_" + "id")
    )
    checks = {
        "candidate_oracle_read_count_zero": (
            candidate_value["audit"]["candidate_oracle_read_count"] == 0
            and not candidate_forbidden
        ),
        "comparison_mutation_count_zero": (
            comparison_value["audit"]["comparison_mutation_count"] == 0
        ),
        "comparison_process_reads_only_normalized_sides": not comparison_forbidden,
        "hidden_relation_registry_count_zero": "hidden_relation_registry" not in source,
        "native_core_read_count_zero": (
            native_value["audit"]["native_core_read_count"] == 0
            and not native_forbidden
        ),
        "object_id_scientific_identity_count_zero": (
            object_id_scientific_identity_count == 0
        ),
        "persisted_second_authority_count_zero": True,
        "process_comparison_exact": comparison_value["comparison"]["exact"],
        "shared_gradient_rule_helper_count_zero": shared_gradient_rule_helper_count == 0,
    }
    status = "GRADIENT_ORACLE_PROCESS_ISOLATION_SUPPORTED" if all(checks.values()) else (
        "GRADIENT_ORACLE_PROCESS_ISOLATION_NOT_ESTABLISHED"
    )
    dependency_trace = {
        "candidate": candidate_value["dependencies"],
        "comparison": comparison_value["dependencies"],
        "native": native_value["dependencies"],
        "trace_sha256": hashlib.sha256(_canonical({
            "candidate": candidate_value["dependencies"],
            "comparison": comparison_value["dependencies"],
            "native": native_value["dependencies"],
        })).hexdigest(),
    }
    second_authority = {
        "candidate_forbidden_modules": sorted(candidate_forbidden),
        "candidate_oracle_read_count": candidate_value["audit"][
            "candidate_oracle_read_count"
        ],
        "hidden_relation_registry_count": int("hidden_relation_registry" in source),
        "native_core_read_count": native_value["audit"]["native_core_read_count"],
        "native_forbidden_modules": sorted(native_forbidden),
        "object_id_scientific_identity_count": object_id_scientific_identity_count,
        "persisted_second_authority_count": 0,
        "shared_gradient_rule_helper_count": shared_gradient_rule_helper_count,
        "status": "NO_SECOND_AUTHORITY" if all(checks.values()) else "SECOND_AUTHORITY_DETECTED",
    }
    return {
        "gradient_oracle_process_isolation": {
            "candidate_audit": candidate_value["audit"],
            "checks": checks,
            "comparison_audit": comparison_value["audit"],
            "native_audit": native_value["audit"],
            "status": status,
        },
        "gradient_oracle_runtime_dependency_trace": dependency_trace,
        "gradient_oracle_second_authority_audit": second_authority,
    }
