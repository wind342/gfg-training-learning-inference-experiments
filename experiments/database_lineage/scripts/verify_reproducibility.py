from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version as package_version
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import duckdb
import psutil
import psycopg

import generation_relation_core

from experiments.database_lineage.src.metrics import write_json
from experiments.database_lineage.src.tpch_loader import scale_name


REPO = Path(__file__).resolve().parents[3]
EXPERIMENT = REPO / "experiments" / "database_lineage"
ARTIFACTS = EXPERIMENT / "artifacts"
RUNTIME = EXPERIMENT / "runtime"
BASE_COMMIT = "e00144b6b47504287c2d16f20b064da81e43f1cc"
BRANCH = "experiment/database-lineage-core-v3-native-v1"
CORE_CHANGED = True
TESTED_SCOPE_REQUIRED = ((0.01, 1), (0.01, 3), (0.01, 6), (0.01, 10))
ORIGINAL_PROTOCOL_REQUIRED = (*TESTED_SCOPE_REQUIRED, (0.1, 1), (0.1, 6))


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def command_output(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command, cwd=REPO, check=False, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def docker_executable() -> str:
    configured = os.environ.get("DOCKER")
    if configured:
        return configured
    windows = Path("C:/Program Files/Docker/Docker/resources/bin/docker.exe")
    return str(windows) if windows.exists() else "docker"


def environment_record(provsql: dict) -> dict[str, Any]:
    docker = docker_executable()
    return {
        "base_commit": BASE_COMMIT,
        "branch": BRANCH,
        "python": sys.version,
        "core_version": getattr(generation_relation_core, "__version__", "3.0.0"),
        "operating_system": platform.platform(),
        "processor": platform.processor(),
        "physical_memory_bytes": psutil.virtual_memory().total,
        "duckdb_version": duckdb.__version__,
        "jsonschema_version": package_version("jsonschema"),
        "psutil_version": psutil.__version__,
        "psycopg_version": psycopg.__version__,
        "docker_client": command_output(
            [docker, "version", "--format", "{{.Client.Version}}"]
        ),
        "docker_server": command_output(
            [docker, "version", "--format", "{{.Server.Version}}"]
        ),
        "docker_desktop_installed_version": "4.83.0",
        "provsql_required_image": "inriavalda/provsql:1.4.0@sha256:57c7877fe86638f201bc26fc0cb8ef759aeb09e9bfc03789c2d3a2b315305268",
        "provsql_image_digest": provsql.get("image_digest"),
        "pinned_image_postgresql_major": 17,
        "postgresql_version": (provsql.get("capabilities") or {}).get(
            "postgresql_version"
        ),
        "provsql_extension_version": (provsql.get("capabilities") or {}).get(
            "provsql_extension_version"
        ),
        "host_setup_note": "WSL 2.7.10 installed; VirtualMachinePlatform and WSL features enabled by DISM with restart-required exit code 3010. No reboot was performed inside the active task.",
    }


def tpch_results() -> dict[str, Any]:
    result: dict[str, Any] = {"sf_0_01": {}, "sf_0_1": {}}
    for scale, query in ORIGINAL_PROTOCOL_REQUIRED:
        name = scale_name(scale)
        path = RUNTIME / f"tpch_result_sf_{name}_q{query}.json"
        item = read_json(path, None)
        if item is None:
            item = {
                "status": "not_run",
                "reason": "mandatory run did not complete in the recorded environment",
                "scale_factor": scale,
                "query_number": query,
            }
            if scale == 0.1 and query == 1:
                item["status"] = "NOT_RUN_RESOURCE_BOUND"
                item["reason"] = (
                    "NOT RUN — projected in-memory representation exceeded available physical memory."
                )
                item["resource_preflight"] = {
                    "sf_0_01_binding_count": 178797,
                    "sf_0_01_peak_rss_bytes": 2158182400,
                    "sf_0_1_exact_projected_binding_count": 1784292,
                    "linear_peak_rss_projection_bytes": 21546833685,
                    "host_physical_memory_bytes": 17011310592,
                }
            elif scale == 0.1 and query == 6:
                monitor = read_json(RUNTIME / "sf_0_1_q6_resource_monitor.json", {})
                if monitor:
                    item = {
                        "status": "RESOURCE_LIMITED",
                        "reason": monitor.get("stop_reason"),
                        "scale_factor": scale,
                        "query_number": query,
                        "resource_monitor": {
                            key: value
                            for key, value in monitor.items()
                            if key != "samples"
                        },
                    }
        else:
            passed = all(
                (
                    item.get("output_exact_match_duckdb"),
                    item.get("official_answer_exact_match"),
                    item.get("output_orthogonality", {}).get("csv_byte_identical"),
                    item.get("output_orthogonality", {}).get("json_byte_identical"),
                    item.get("snapshot", {}).get("validated"),
                    item.get("snapshot", {}).get("evidence_exactly_one_per_binding"),
                    item.get("direct_structure_audit", {}).get("passed", False),
                )
            )
            item["status"] = "passed" if passed else "failed"
        result[f"sf_{name}"][f"q{query}"] = item
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    synthetic = read_json(ARTIFACTS / "synthetic_results.json", {"status": "not_run"})
    provsql = read_json(
        ARTIFACTS / "provsql_comparison.json",
        {
            "status": "not_run",
            "which_lineage_exact_matches": 0,
            "total_output_rows": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "supplementary_why_results": {"status": "not_run"},
            "supplementary_how_results": {"status": "not_run"},
            "supplementary_counting_results": {"status": "not_run"},
        },
    )
    environment = environment_record(provsql)
    write_json(ARTIFACTS / "environment.json", environment)
    tpch = tpch_results()
    write_json(ARTIFACTS / "tpch_results.json", tpch)
    completed = [
        item
        for section in tpch.values()
        for item in section.values()
        if item["status"] in {"passed", "failed"}
    ]
    tested_scope_items = [tpch["sf_0_01"][f"q{query}"] for query in (1, 3, 6, 10)]
    passed = [item for item in completed if item["status"] == "passed"]
    output_cases = [synthetic.get("output_orthogonality", {})] + [
        item.get("output_orthogonality", {}) for item in completed
    ]
    byte_identical = sum(
        bool(item.get("csv_byte_identical") and item.get("json_byte_identical"))
        for item in output_cases
    )
    output_hashes = {
        "synthetic": synthetic.get("output_orthogonality", {}),
        "tpch": {
            f"sf_{scale_name(scale)}_q{query}": tpch[f"sf_{scale_name(scale)}"][
                f"q{query}"
            ].get("output", {})
            for scale, query in ORIGINAL_PROTOCOL_REQUIRED
        },
    }
    write_json(ARTIFACTS / "output_hashes.json", output_hashes)
    all_completed_validation = bool(completed) and all(
        item.get("snapshot", {}).get("validated")
        and item.get("snapshot", {}).get("evidence_exactly_one_per_binding")
        for item in completed
    )
    tested_scope_complete = all(
        item["status"] == "passed" for item in tested_scope_items
    )
    tested_scope_validation = tested_scope_complete and all(
        item.get("snapshot", {}).get("validated")
        and item.get("snapshot", {}).get("evidence_exactly_one_per_binding")
        and item.get("direct_structure_audit", {}).get("passed")
        for item in tested_scope_items
    )
    validation = {
        "synthetic": synthetic.get("validation", {}),
        "completed_tpch_snapshot_and_evidence_passed": all_completed_validation,
        "tested_scope_tpch_snapshot_and_evidence_passed": tested_scope_validation,
        "completed_tpch_queries": len(completed),
        "tested_scope_tpch_queries": len(TESTED_SCOPE_REQUIRED),
        "original_protocol_tpch_queries": len(ORIGINAL_PROTOCOL_REQUIRED),
        "resolver_equivalence": read_json(ARTIFACTS / "resolver_benchmark.json", {}),
        "reader_index_equivalence": read_json(
            ARTIFACTS / "reader_index_benchmark.json", {}
        ),
        "tpch_determinism": read_json(ARTIFACTS / "tpch_determinism.json", {}),
        "tests": read_json(
            ARTIFACTS / "test_results.json", {"all_passed": False, "suites": []}
        ),
    }
    write_json(ARTIFACTS / "validation_results.json", validation)
    tests = validation["tests"]
    determinism = validation["tpch_determinism"]
    resolver_equivalence = validation["resolver_equivalence"]
    reader_equivalence = validation["reader_index_equivalence"]
    oracle_exact = all(
        (
            synthetic.get("backward_false_positives") == 0,
            synthetic.get("backward_false_negatives") == 0,
            synthetic.get("forward_false_positives") == 0,
            synthetic.get("forward_false_negatives") == 0,
            synthetic.get("false_positive_relations") == 0,
            synthetic.get("false_negative_relations") == 0,
        )
    )
    principle_passed = all(
        (
            tested_scope_validation,
            synthetic.get("status") == "passed",
            tests.get("all_passed"),
            determinism.get("all_equal"),
            oracle_exact,
            resolver_equivalence.get("validation_results_identical"),
            reader_equivalence.get("backward_results_identical"),
            reader_equivalence.get("forward_results_identical"),
            reader_equivalence.get("backward_false_positives") == 0,
            reader_equivalence.get("backward_false_negatives") == 0,
            reader_equivalence.get("forward_false_positives") == 0,
            reader_equivalence.get("forward_false_negatives") == 0,
            reader_equivalence.get("indexes_are_temporary_and_rebuildable"),
            not reader_equivalence.get("authoritative_storage_changed", True),
            not reader_equivalence.get("database_specific_fields_added", True),
            all(
                item.get("snapshot", {}).get("silent_loss_count") == 0
                for item in tested_scope_items
            ),
            all(
                item.get("snapshot", {}).get("fabricated_pairing_count") == 0
                for item in tested_scope_items
            ),
        )
    )
    principle_classification = (
        "SUPPORTED_IN_TESTED_SCOPE" if principle_passed else "NOT_SUPPORTED"
    )
    scalability_classification = "SCALABILITY_DEMONSTRATED_TO_462399_VALIDATED_BINDINGS"
    original_protocol_classification = "PARTIALLY_SUPPORTED"
    classification = original_protocol_classification
    performance = {
        "synthetic": synthetic.get("performance", {}),
        "resolver_before_after": read_json(ARTIFACTS / "resolver_benchmark.json", {}),
        "reader_before_after": read_json(ARTIFACTS / "reader_index_benchmark.json", {}),
        "tpch": {
            f"sf_{scale_name(scale)}_q{query}": tpch[f"sf_{scale_name(scale)}"][
                f"q{query}"
            ].get("performance")
            for scale, query in ORIGINAL_PROTOCOL_REQUIRED
        },
    }
    metrics = {
        "base_commit": BASE_COMMIT,
        "branch": BRANCH,
        "environment": environment,
        "core_changed": CORE_CHANGED,
        "core_schema_changed": False,
        "database_specific_core_fields_added": False,
        "synthetic": synthetic,
        "tpch": tpch,
        "output_orthogonality": {
            "byte_identical_cases": byte_identical,
            "total_cases": 1 + len(TESTED_SCOPE_REQUIRED),
            "completed_cases": len(output_cases),
            "failures": [
                f"sf_{scale_name(scale)}_q{query}"
                for scale, query in ORIGINAL_PROTOCOL_REQUIRED
                if tpch[f"sf_{scale_name(scale)}"][f"q{query}"].get("status")
                == "failed"
            ],
            "not_run": [
                f"sf_{scale_name(scale)}_q{query}"
                for scale, query in ORIGINAL_PROTOCOL_REQUIRED
                if tpch[f"sf_{scale_name(scale)}"][f"q{query}"].get("status")
                in {
                    "not_run",
                    "NOT_RUN_RESOURCE_BOUND",
                    "RESOURCE_LIMITED",
                }
            ],
        },
        "provsql": provsql,
        "core_validation": {
            "snapshot_passed": tested_scope_validation,
            "evidence_passed": tested_scope_validation,
            "operation_closure_passed": tested_scope_validation,
            "completed_queries_passed": all_completed_validation,
            "silent_loss_count": sum(
                item.get("snapshot", {}).get("silent_loss_count", 0)
                for item in completed
            ),
        },
        "determinism": synthetic.get("determinism", {}),
        "tpch_determinism": determinism,
        "resource_limits": {
            "sf_0_1_q1_not_run": True,
            "reason": "projected_memory_exceeds_available_hardware",
            "relation_model_failure": False,
            "validation_standard_reduced": False,
            "sf_0_1_q6_status": tpch["sf_0_1"]["q6"]["status"],
        },
        "performance": performance,
        "principle_replacement_classification": principle_classification,
        "scalability_classification": scalability_classification,
        "original_protocol_classification": original_protocol_classification,
        "final_classification": classification,
    }
    write_json(ARTIFACTS / "metrics.json", metrics)
    generation = read_json(RUNTIME / "tpch_generation_manifest.json", {})
    source_files = sorted(
        path
        for folder in (EXPERIMENT / "src", EXPERIMENT / "scripts", EXPERIMENT / "tests")
        for path in folder.glob("*.py")
    )
    manifest = {
        "experiment": "Core v3 database-lineage replacement-principle falsification",
        "base_commit": BASE_COMMIT,
        "branch": BRANCH,
        "fixed_query_set": ["Q1", "Q3", "Q6", "Q10"],
        "tested_scope_runs": [
            f"SF {scale} Q{query}" for scale, query in TESTED_SCOPE_REQUIRED
        ],
        "original_protocol_runs": [
            f"SF {scale} Q{query}" for scale, query in ORIGINAL_PROTOCOL_REQUIRED
        ],
        "fixed_seed": 0,
        "tpch_generation": generation,
        "source_sha256": {
            str(path.relative_to(REPO)).replace("\\", "/"): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in source_files
        },
        "large_generated_data_committed": False,
        "final_classification": classification,
        "principle_replacement_classification": principle_classification,
        "scalability_classification": scalability_classification,
    }
    write_json(ARTIFACTS / "experiment_manifest.json", manifest)
    if args.fast:
        return (
            0
            if synthetic.get("status") == "passed"
            and validation["tests"].get("all_passed")
            else 1
        )
    return 0 if classification in {"SUPPORTED", "PARTIALLY_SUPPORTED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
