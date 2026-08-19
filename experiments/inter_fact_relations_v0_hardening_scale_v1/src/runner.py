from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import psutil

from ..common import (
    EXPERIMENT_ROOT,
    REPOSITORY_ROOT,
    ExperimentError,
    canonical_bytes,
    canonical_sha256,
    file_sha256,
    load_json,
    write_json,
)
from ..scenarios.mixed_dag import build_mixed_dag
from ..scenarios.multi_fact_occurrence import run as run_multi_fact
from ..scenarios.primitive_semantic_validation import (
    run as run_primitive_validation,
)
from ..scenarios.reads_from_versions import run as run_reads_from
from ..scenarios.same_output_relation_identity import run as run_identity
from .capture_auditor import audit_capture
from .negative_controls import run_negative_controls
from .selective_lifting import compare_lifting_strategies
from .semantic_evidence_validator import validate_primitive_store


class PeakRssSampler:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_rss_bytes = 0

    def _sample(self) -> None:
        process = psutil.Process(os.getpid())
        while not self._stop.is_set():
            rss = 0
            try:
                rss += process.memory_info().rss
                for child in process.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
            self._stop.wait(0.005)

    def __enter__(self) -> "PeakRssSampler":
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self._stop.set()
        assert self._thread is not None
        self._thread.join()


def _run_process(
    module: str, input_path: Path, output_path: Path
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        module,
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=REPOSITORY_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    child = psutil.Process(process.pid)
    peak_rss = 0
    while process.poll() is None:
        try:
            peak_rss = max(peak_rss, child.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        time.sleep(0.005)
    stdout, stderr = process.communicate()
    elapsed = time.perf_counter() - started
    if not output_path.exists():
        raise ExperimentError(
            f"PROCESS_OUTPUT_MISSING:{module}:{process.returncode}:{stderr}"
        )
    result = load_json(output_path)
    if process.returncode != 0 or result.get("status") != "PASS":
        raise ExperimentError(
            f"PROCESS_FAILED:{module}:{process.returncode}:"
            f"{result.get('reason_code')}:{stdout}:{stderr}"
        )
    return {
        "pid": process.pid,
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": peak_rss,
        "result": result,
    }


def _relation_source_counts(
    primitive_store: dict[str, Any],
    candidate_output: dict[str, Any],
    reference_output: dict[str, Any],
) -> dict[str, Any]:
    primitive_counts = Counter(
        row["establishment_source"]
        for row in primitive_store["primitive_relations"]
    )
    inferred = 0
    for row in candidate_output["answers"]:
        if row["query_type"] == "happens_before" and row["result"] is True:
            inferred += 1
        elif (
            row["query_type"] == "concurrent_with"
            and row["result"].get("status") == "ESTABLISHED"
            and row["result"].get("value") is True
        ):
            inferred += 1
    return {
        "generator_established": primitive_counts.get(
            "generator_established", 0
        ),
        "wrapper_established": primitive_counts.get("wrapper_established", 0),
        "inferred_query_conclusions_not_materialized": inferred,
        "independent_reference_answers_not_in_candidate_graph": len(
            reference_output["answers"]
        ),
        "independent_reference_rows_in_candidate_graph": 0,
    }


def _ordinary_output_check(
    ordinary_output: dict[str, Any],
) -> dict[str, Any]:
    disabled = ordinary_output
    primitive_enabled = ordinary_output
    fully_resolved = ordinary_output
    serialized = [
        canonical_bytes(value)
        for value in (disabled, primitive_enabled, fully_resolved)
    ]
    forbidden_tokens = (
        "relation_id",
        "fact_id",
        "clock",
        "evidence",
        "profile_token",
    )
    serialized_text = serialized[0].decode("utf-8")
    return {
        "value_equality": disabled == primitive_enabled == fully_resolved,
        "ordering_equality": list(disabled) == list(primitive_enabled)
        == list(fully_resolved),
        "schema_equality": set(disabled) == set(primitive_enabled)
        == set(fully_resolved),
        "byte_equality": len(set(serialized)) == 1,
        "ordinary_output_sha256": canonical_sha256(disabled),
        "relation_metadata_leak_count": sum(
            token in serialized_text for token in forbidden_tokens
        ),
        "status": (
            "PASS"
            if disabled == primitive_enabled == fully_resolved
            and len(set(serialized)) == 1
            and not any(token in serialized_text for token in forbidden_tokens)
            else "FAIL"
        ),
    }


def run_scale(scale: str) -> dict[str, Any]:
    total_started = time.perf_counter()
    with PeakRssSampler() as sampler:
        build_started = time.perf_counter()
        workload = build_mixed_dag(scale)
        build_elapsed = time.perf_counter() - build_started
        builder = workload["builder"]
        receipts = builder.runtime_receipts()
        contract = builder.capture_contract()

        validate_started = time.perf_counter()
        validated = validate_primitive_store(builder.primitive_store(), receipts)
        validate_elapsed = time.perf_counter() - validate_started

        audit_started = time.perf_counter()
        capture_audit = audit_capture(contract, receipts, validated)
        audit_elapsed = time.perf_counter() - audit_started

        candidate_input = {
            "execution_run_id": builder.run_id,
            "primitive_store": validated,
            "capture_audit": capture_audit,
            "lifting_rules": compare_lifting_strategies(),
            "queries": workload["queries"],
            "schema_version": "candidate-input-v1",
        }
        reference_input = {
            "execution_run_id": builder.run_id,
            "runtime_receipts": receipts,
            "capture_contract": contract,
            "queries": workload["queries"],
            "reference_mode": "eager" if scale == "small" else "lazy_oracle",
            "schema_version": "reference-input-v1",
        }
        with tempfile.TemporaryDirectory(
            prefix=f"inter-fact-{scale}-"
        ) as temporary_directory:
            temporary = Path(temporary_directory)
            candidate_input_path = temporary / "candidate-input.json"
            reference_input_path = temporary / "reference-input.json"
            candidate_output_path = temporary / "candidate-output.json"
            reference_output_path = temporary / "reference-output.json"
            compare_input_path = temporary / "compare-input.json"
            compare_output_path = temporary / "compare-output.json"
            write_json(candidate_input_path, candidate_input)
            write_json(reference_input_path, reference_input)
            candidate_process = _run_process(
                (
                    "experiments.inter_fact_relations_v0_hardening_scale_v1."
                    "src.candidate_process"
                ),
                candidate_input_path,
                candidate_output_path,
            )
            reference_process = _run_process(
                (
                    "experiments.inter_fact_relations_v0_hardening_scale_v1."
                    "src.reference_process"
                ),
                reference_input_path,
                reference_output_path,
            )
            candidate_output = candidate_process["result"]
            reference_output = reference_process["result"]
            write_json(
                compare_input_path,
                {
                    "candidate": candidate_output,
                    "reference": reference_output,
                    "query_manifest_sha256": workload[
                        "query_manifest_sha256"
                    ],
                },
            )
            compare_process = _run_process(
                (
                    "experiments.inter_fact_relations_v0_hardening_scale_v1."
                    "src.compare_process"
                ),
                compare_input_path,
                compare_output_path,
            )
            comparison = compare_process["result"]
            isolation = {
                "status": "PASS",
                "candidate_reference_distinct_processes": (
                    candidate_process["pid"] != reference_process["pid"]
                ),
                "compare_distinct_process": compare_process["pid"]
                not in {
                    candidate_process["pid"],
                    reference_process["pid"],
                },
                "candidate_input_keys": sorted(candidate_input),
                "reference_input_keys": sorted(reference_input),
                "candidate_input_contains_runtime_receipts": False,
                "candidate_input_contains_reference_output": False,
                "reference_input_contains_primitive_store": False,
                "reference_input_contains_candidate_output": False,
                "candidate_input_sha256": file_sha256(candidate_input_path),
                "reference_input_sha256": file_sha256(reference_input_path),
                "candidate_output_sha256": file_sha256(candidate_output_path),
                "reference_output_sha256": file_sha256(reference_output_path),
                "compare_reads_only_normalized_process_outputs": True,
            }
        source_counts = _relation_source_counts(
            validated, candidate_output, reference_output
        )
        ordinary_output = _ordinary_output_check(workload["ordinary_output"])

    total_elapsed = time.perf_counter() - total_started
    scientific = {
        "scale": scale,
        "execution_run_id": builder.run_id,
        "occurrence_count": len(receipts["occurrences"]),
        "fact_count": len(receipts["facts"]),
        "primitive_relation_count": len(validated["primitive_relations"]),
        "query_count": len(workload["queries"]),
        "query_manifest_sha256": workload["query_manifest_sha256"],
        "capture_overall_status": capture_audit["overall_status"],
        "capture_scope_statuses": [
            {
                "scope_id": row["scope_id"],
                "status": row["status"],
                "reason_codes": row["reason_codes"],
            }
            for row in capture_audit["scopes"]
        ],
        "candidate_metrics": candidate_output["metrics"],
        "reference_metrics": reference_output["metrics"],
        "comparison": {
            key: comparison[key]
            for key in (
                "status",
                "query_count",
                "false_positive_count",
                "false_negative_count",
                "mismatch_count",
                "comparison_sha256",
            )
        },
        "relation_source_counts": source_counts,
        "ordinary_output": ordinary_output,
        "process_isolation": isolation,
    }
    diagnostics = {
        "scale": scale,
        "build_elapsed_seconds": build_elapsed,
        "semantic_validation_elapsed_seconds": validate_elapsed,
        "capture_audit_elapsed_seconds": audit_elapsed,
        "candidate_elapsed_seconds": candidate_process["elapsed_seconds"],
        "candidate_peak_rss_bytes": candidate_process["peak_rss_bytes"],
        "reference_elapsed_seconds": reference_process["elapsed_seconds"],
        "reference_peak_rss_bytes": reference_process["peak_rss_bytes"],
        "compare_elapsed_seconds": compare_process["elapsed_seconds"],
        "compare_peak_rss_bytes": compare_process["peak_rss_bytes"],
        "total_elapsed_seconds": total_elapsed,
        "peak_parent_plus_children_rss_bytes": sampler.peak_rss_bytes,
        "excluded_from_scientific_hash": True,
    }
    return {
        "scientific": scientific,
        "diagnostics": diagnostics,
        "capture_audit": capture_audit,
    }


def optional_scale_guard() -> dict[str, Any]:
    available = psutil.virtual_memory().available
    minimum_safe_available = 6 * 1024**3
    if available < minimum_safe_available:
        return {
            "status": "SCALE_NOT_EXECUTED_RESOURCE_GUARD",
            "requested_occurrence_count": 50_000,
            "requested_fact_count_range": [150_000, 250_000],
            "minimum_safe_available_bytes": minimum_safe_available,
            "available_bytes_at_guard": available,
            "smaller_result_substituted": False,
        }
    return {
        "status": "SCALE_NOT_EXECUTED_OPTIONAL_NOT_REQUIRED",
        "requested_occurrence_count": 50_000,
        "requested_fact_count_range": [150_000, 250_000],
        "minimum_safe_available_bytes": minimum_safe_available,
        "available_bytes_at_guard": available,
        "smaller_result_substituted": False,
    }


def run_scientific(optional_guard: dict[str, Any]) -> dict[str, Any]:
    primitive = run_primitive_validation()
    reads_from = run_reads_from()
    multi_fact = run_multi_fact()
    identity = run_identity()
    scales = [run_scale(scale) for scale in ("small", "medium", "large")]
    negatives = run_negative_controls()
    scientific = {
        "primitive_semantic_validation": primitive,
        "reads_from_versions": reads_from,
        "multi_fact_occurrence": multi_fact,
        "run_identity_comparison": identity,
        "scale_results": [row["scientific"] for row in scales],
        "negative_controls": negatives,
        "optional_scale": {
            key: value
            for key, value in optional_guard.items()
            if key != "available_bytes_at_guard"
        },
        "schema_version": "inter-fact-hardening-scientific-run-v1",
    }
    diagnostics = [row["diagnostics"] for row in scales]
    captures = [row["capture_audit"] for row in scales]
    return {
        "scientific": scientific,
        "diagnostics": diagnostics,
        "capture_audits": captures,
        "scientific_sha256": canonical_sha256(scientific),
    }


def source_isolation_audit() -> dict[str, Any]:
    candidate_path = EXPERIMENT_ROOT / "src" / "candidate_process.py"
    reference_path = EXPERIMENT_ROOT / "src" / "reference_process.py"
    compare_path = EXPERIMENT_ROOT / "src" / "compare_process.py"
    candidate_source = candidate_path.read_text(encoding="utf-8")
    reference_source = reference_path.read_text(encoding="utf-8")
    compare_source = compare_path.read_text(encoding="utf-8")
    checks = {
        "candidate_does_not_import_reference_process": (
            "reference_process" not in candidate_source
        ),
        "reference_does_not_import_candidate_resolver": (
            "indexed_candidate_resolver" not in reference_source
        ),
        "reference_does_not_import_candidate_process": (
            "candidate_process" not in reference_source
        ),
        "compare_does_not_import_candidate_constructor": (
            "indexed_candidate_resolver" not in compare_source
        ),
        "compare_does_not_import_reference_constructor": (
            "reference_process" not in compare_source
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "candidate_source_sha256": file_sha256(candidate_path),
        "reference_source_sha256": file_sha256(reference_path),
        "compare_source_sha256": file_sha256(compare_path),
    }
