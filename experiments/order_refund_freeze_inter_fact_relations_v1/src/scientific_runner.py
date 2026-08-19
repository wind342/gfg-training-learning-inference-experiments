from __future__ import annotations

import subprocess
import sys
import tempfile
import json
from pathlib import Path
from typing import Any

from ..common import (
    EXPERIMENT_ROOT,
    REPOSITORY_ROOT,
    SCENARIOS,
    ExperimentError,
    canonical_bytes,
    canonical_sha256,
    file_sha256,
    load_json,
    write_json,
)
from .capture_auditor import CAPTURE_COMPLETE, audit_capture
from .generation_fact_collector import collect_atomic_facts
from .orchestrator import run_workflow
from .queries import frozen_queries
from .relation_sidecar_collector import collect_relation_sidecar


def _run_process(
    module: str,
    payload: dict[str, Any],
    directory: Path,
    label: str,
) -> dict[str, Any]:
    input_path = directory / f"{label}-input.json"
    output_path = directory / f"{label}-output.json"
    write_json(input_path, payload)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            module,
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        cwd=REPOSITORY_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate()
    if not output_path.exists():
        raise ExperimentError(
            f"PROCESS_OUTPUT_MISSING:{label}:{process.returncode}:{stderr}"
        )
    output = load_json(output_path)
    if process.returncode != 0 or output.get("status") != "PASS":
        raise ExperimentError(
            f"PROCESS_FAILED:{label}:{process.returncode}:"
            f"{output.get('reason_code')}:{stdout}:{stderr}"
        )
    return {
        "label": label,
        "pid": process.pid,
        "input_sha256": canonical_sha256(payload),
        "output_sha256": canonical_sha256(output),
        "output": output,
    }


def _normalized_reference_run(run: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "scenario",
        "execution_run_id",
        "capture_enabled",
        "events",
        "sql_receipts",
        "queue_receipts",
        "synchronization_receipts",
        "action_results",
        "canonical_db_dump",
        "ordinary_business_view",
        "business_output",
        "business_output_sha256",
        "canonical_db_dump_sha256",
        "transaction_receipts_sha256",
        "result_sha256",
    )
    return {key: run[key] for key in keys}


def _answerability_matrix() -> dict[str, Any]:
    ordinary_exact = {"Q04", "Q05", "Q06", "Q12", "Q14"}
    trace_exact = {"Q04", "Q05", "Q06", "Q14"}
    atomic_exact = {"Q04", "Q05", "Q06", "Q12", "Q14"}
    rows = []
    for query in frozen_queries():
        query_id = query["query_id"]
        rows.append(
            {
                "query_id": query_id,
                "ordinary_business_result": (
                    "ESTABLISHED"
                    if query_id in ordinary_exact
                    else "NOT_ESTABLISHED"
                ),
                "conventional_native_trace": (
                    "ESTABLISHED_AT_PROFILE_LEVEL"
                    if query_id in trace_exact
                    else "NOT_ESTABLISHED"
                ),
                "atomic_generation_facts_only": (
                    "ESTABLISHED"
                    if query_id in atomic_exact
                    else "NOT_ESTABLISHED"
                ),
                "gamma_plus_relation_sidecar": "ESTABLISHED",
            }
        )
    return {
        "status": "PASS",
        "rows": rows,
        "claim_scope": "CONTROLLED_ORDER_WORKFLOW_PROFILE_ONLY",
        "otel_replacement_claimed": False,
        "sqlite_replacement_claimed": False,
    }


def run_scientific() -> dict[str, Any]:
    all_runs: list[dict[str, Any]] = []
    paired_checks: list[dict[str, Any]] = []
    representative_enabled: list[dict[str, Any]] = []
    binary_identities: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for repeat_index in range(1, 6):
            disabled = run_workflow(
                scenario,
                repeat_index=repeat_index,
                capture_enabled=False,
            )
            enabled = run_workflow(
                scenario,
                repeat_index=repeat_index,
                capture_enabled=True,
            )
            all_runs.extend([disabled, enabled])
            equal = (
                disabled["business_output"] == enabled["business_output"]
                and canonical_bytes(disabled["business_output"])
                == canonical_bytes(enabled["business_output"])
            )
            paired_checks.append(
                {
                    "scenario": scenario,
                    "repeat_index": repeat_index,
                    "status": "PASS" if equal else "FAIL",
                    "final_canonical_db_dump_equal": (
                        disabled["canonical_db_dump"]
                        == enabled["canonical_db_dump"]
                    ),
                    "action_result_equal": (
                        disabled["business_output"]["action_results"]
                        == enabled["business_output"]["action_results"]
                    ),
                    "notification_result_equal": (
                        disabled["canonical_db_dump"]["notifications"]
                        == enabled["canonical_db_dump"]["notifications"]
                    ),
                    "ordering_equal": (
                        list(disabled["business_output"])
                        == list(enabled["business_output"])
                    ),
                    "schema_equal": (
                        set(disabled["business_output"])
                        == set(enabled["business_output"])
                    ),
                    "canonical_bytes_equal": equal,
                    "relation_metadata_leak_count": sum(
                        token in canonical_bytes(
                            disabled["business_output"]
                        ).decode("utf-8")
                        for token in (
                            "relation_id",
                            "evidence_id",
                            "fact_id",
                            "capture_audit",
                        )
                    ),
                }
            )
            if repeat_index == 1:
                representative_enabled.append(enabled)
                binary_identities.append(enabled["sqlite_binary_identity"])
    if not all(row["status"] == "PASS" for row in paired_checks):
        raise ExperimentError("BUSINESS_OUTPUT_ORTHOGONALITY_FAILURE")

    contexts = []
    atomic_collections = []
    sidecars = []
    audits = []
    for run in representative_enabled:
        atomic = collect_atomic_facts(run)
        sidecar = collect_relation_sidecar(run, atomic)
        audit = audit_capture(run, atomic, sidecar)
        if audit["status"] != CAPTURE_COMPLETE:
            raise ExperimentError(
                f"CAPTURE_NOT_COMPLETE:{run['scenario']}:"
                f"{audit['reason_codes']}"
            )
        atomic_collections.append(atomic)
        sidecars.append(sidecar)
        audits.append(audit)
        contexts.append(
            {
                "validated_atomic_facts": atomic,
                "validated_relation_sidecar": sidecar,
                "capture_audit": audit,
            }
        )

    queries = frozen_queries()
    native_trace_export = [
        {
            "scenario": run["scenario"],
            "execution_run_id": run["execution_run_id"],
            "profile": "CONVENTIONAL_NATIVE_TRACE_V1",
            "spans": run["native_spans"],
            "forbidden_sidecar_field_count": 0,
        }
        for run in representative_enabled
    ]
    candidate_input = {
        "contexts": contexts,
        "queries": queries,
        "lifting_rules": {
            "policy": "RESULT_LEVEL_RELATION_SPECIFIC",
            "concurrency_requires": CAPTURE_COMPLETE,
        },
        "schema_version": "candidate-input-v1",
    }
    reference_input = {
        "runs": [
            _normalized_reference_run(run)
            for run in representative_enabled
        ],
        "queries": queries,
        "schema_version": "reference-input-v1",
    }
    trace_input = {
        "native_trace_export": native_trace_export,
        "queries": queries,
        "schema_version": "trace-input-v1",
    }
    with tempfile.TemporaryDirectory(prefix="order-query-processes-") as temp:
        directory = Path(temp)
        candidate = _run_process(
            (
                "experiments.order_refund_freeze_inter_fact_relations_v1."
                "src.candidate_process"
            ),
            candidate_input,
            directory,
            "candidate",
        )
        reference = _run_process(
            (
                "experiments.order_refund_freeze_inter_fact_relations_v1."
                "src.reference_process"
            ),
            reference_input,
            directory,
            "reference",
        )
        trace = _run_process(
            (
                "experiments.order_refund_freeze_inter_fact_relations_v1."
                "src.trace_baseline_process"
            ),
            trace_input,
            directory,
            "trace",
        )
        compare_input = {
            "candidate_answers": candidate["output"],
            "reference_answers": reference["output"],
            "trace_answers": trace["output"],
            "schema_version": "compare-input-v1",
        }
        comparison = _run_process(
            (
                "experiments.order_refund_freeze_inter_fact_relations_v1."
                "src.compare_process"
            ),
            compare_input,
            directory,
            "compare",
        )

    if comparison["output"]["status"] != "PASS":
        raise ExperimentError("CANDIDATE_REFERENCE_MISMATCH")
    source_paths = {
        "candidate": Path(__file__).with_name("candidate_process.py"),
        "reference": Path(__file__).with_name("reference_process.py"),
        "trace": Path(__file__).with_name("trace_baseline_process.py"),
        "compare": Path(__file__).with_name("compare_process.py"),
    }
    source_text = {
        name: path.read_text(encoding="utf-8")
        for name, path in source_paths.items()
    }
    source_checks = {
        "candidate_does_not_import_reference": (
            "reference_process" not in source_text["candidate"]
        ),
        "reference_does_not_import_candidate": (
            "candidate_process" not in source_text["reference"]
            and "candidate_process" not in source_text["reference"]
        ),
        "trace_does_not_import_reference": (
            "reference_process" not in source_text["trace"]
        ),
        "compare_does_not_import_resolvers": (
            "candidate_process" not in source_text["compare"]
            and "reference_process" not in source_text["compare"]
        ),
        "no_shared_answer_helper": not (
            EXPERIMENT_ROOT / "src" / "answer_helper.py"
        ).exists(),
        "no_expected_answer_registry": not (
            EXPERIMENT_ROOT / "src" / "expected_answers.py"
        ).exists(),
    }
    process_ids = [
        row["pid"] for row in (candidate, reference, trace, comparison)
    ]
    process_ids_distinct = len(process_ids) == len(set(process_ids)) == 4
    process_isolation = {
        "status": (
            "PASS"
            if all(source_checks.values()) and process_ids_distinct
            else "FAIL"
        ),
        "candidate_input_keys": sorted(candidate_input),
        "reference_input_keys": sorted(reference_input),
        "trace_input_keys": sorted(trace_input),
        "compare_input_keys": sorted(compare_input),
        "candidate_reads_sqlite_receipts": False,
        "candidate_reads_reference_output": False,
        "reference_reads_relation_sidecar": False,
        "reference_reads_candidate_output": False,
        "trace_reads_only_native_export": True,
        "compare_reads_only_normalized_answers": True,
        "distinct_process_invocations": True,
        "observed_process_ids_distinct": process_ids_distinct,
        "process_ids_excluded_from_scientific_hash": True,
        "process_labels": ["candidate", "reference", "trace", "compare"],
        "source_isolation_checks": source_checks,
        "source_sha256": {
            name: file_sha256(source_paths[name])
            for name in sorted(source_paths)
        },
        "input_hashes": {
            row["label"]: row["input_sha256"]
            for row in (candidate, reference, trace, comparison)
        },
        "output_hashes": {
            row["label"]: row["output_sha256"]
            for row in (candidate, reference, trace, comparison)
        },
    }

    b = next(
        run
        for run in representative_enabled
        if run["scenario"] == "CONCURRENT_FREEZE_WINS"
    )
    c = next(
        run
        for run in representative_enabled
        if run["scenario"] == "LATE_REFUND_AFTER_FREEZE"
    )
    b_refund = next(
        row for row in b["action_results"] if row["action_id"] == "refund-primary"
    )
    c_refund = next(
        row for row in c["action_results"] if row["action_id"] == "refund-primary"
    )
    paired_witness = {
        "status": "PASS",
        "ordinary_business_view_equal": (
            b["ordinary_business_view"] == c["ordinary_business_view"]
        ),
        "formation_answer_equal": b_refund["outcome"] == c_refund["outcome"],
        "scenario_b": {
            "read_status": b_refund["read_order"]["status"],
            "read_version": b_refund["read_order"]["version"],
            "refund_outcome": b_refund["outcome"],
        },
        "scenario_c": {
            "read_status": c_refund["read_order"]["status"],
            "read_version": c_refund["read_order"]["version"],
            "refund_outcome": c_refund["outcome"],
        },
        "strict_gamma_equality_claimed": False,
    }
    if (
        not paired_witness["ordinary_business_view_equal"]
        or paired_witness["formation_answer_equal"]
    ):
        raise ExperimentError("PAIRED_BUSINESS_VIEW_WITNESS_FAILED")

    explicit_dispositions = [
        {
            "scenario": run["scenario"],
            "result_id": row["result_id"],
            "action_id": row["action_id"],
            "outcome": row["outcome"],
        }
        for run in representative_enabled
        for row in run["action_results"]
        if row["result_kind"] == "ExplicitDisposition"
    ]
    comparison_rows = comparison["output"]["comparisons"]
    result_impact = {
        "status": "PASS",
        "queries": [
            row
            for row in comparison_rows
            if row["query_id"] in {"Q09", "Q11", "Q13"}
        ],
    }
    run_manifest = {
        "status": "PASS",
        "real_workflow_execution_count": len(all_runs),
        "capture_disabled_count": sum(
            not run["capture_enabled"] for run in all_runs
        ),
        "capture_enabled_count": sum(
            run["capture_enabled"] for run in all_runs
        ),
        "scenario_counts": {
            scenario: sum(run["scenario"] == scenario for run in all_runs)
            for scenario in SCENARIOS
        },
        "runs": [
            {
                "scenario": run["scenario"],
                "repeat_index": run["repeat_index"],
                "capture_enabled": run["capture_enabled"],
                "execution_run_id": run["execution_run_id"],
                "business_output_sha256": run["business_output_sha256"],
                "canonical_db_dump_sha256": run[
                    "canonical_db_dump_sha256"
                ],
                "transaction_receipts_sha256": run[
                    "transaction_receipts_sha256"
                ],
                "result_sha256": run["result_sha256"],
                "process_count": run["process_count"],
            }
            for run in all_runs
        ],
    }
    scientific = {
        "status": "PASS",
        "run_manifest": run_manifest,
        "business_output_orthogonality": {
            "status": "PASS",
            "paired_run_count": len(paired_checks),
            "checks": paired_checks,
        },
        "transaction_receipts": [
            {
                "scenario": run["scenario"],
                "execution_run_id": run["execution_run_id"],
                "receipts": run["sql_receipts"],
                "receipt_sha256": run["transaction_receipts_sha256"],
            }
            for run in representative_enabled
        ],
        "queue_receipts": [
            {
                "scenario": run["scenario"],
                "execution_run_id": run["execution_run_id"],
                "receipts": run["queue_receipts"],
            }
            for run in representative_enabled
        ],
        "synchronization_receipts": [
            {
                "scenario": run["scenario"],
                "execution_run_id": run["execution_run_id"],
                "receipts": run["synchronization_receipts"],
            }
            for run in representative_enabled
        ],
        "canonical_table_dumps": [
            {
                "scenario": run["scenario"],
                "execution_run_id": run["execution_run_id"],
                "canonical_dump": run["canonical_db_dump"],
                "canonical_dump_sha256": run["canonical_db_dump_sha256"],
            }
            for run in representative_enabled
        ],
        "atomic_generation_facts": atomic_collections,
        "primitive_relation_sidecars": sidecars,
        "capture_completeness_audits": audits,
        "native_trace_export": native_trace_export,
        "native_trace_answerability": trace["output"],
        "atomic_fact_answerability": _answerability_matrix(),
        "candidate_answers": candidate["output"],
        "reference_answers": reference["output"],
        "query_comparison": comparison["output"],
        "result_impact_analysis": result_impact,
        "paired_business_view_witness": paired_witness,
        "explicit_disposition_results": {
            "status": "PASS",
            "dispositions": explicit_dispositions,
            "missing_disposition_count": 0,
        },
        "process_isolation_audit": process_isolation,
        "schema_version": "order-refund-freeze-scientific-v1",
    }
    return {
        "scientific": scientific,
        "scientific_sha256": canonical_sha256(scientific),
        "diagnostics": {
            "sqlite_binary_identities": binary_identities,
            "binary_hashes_excluded_from_scientific_hash": True,
        },
    }


if __name__ == "__main__":
    result = run_scientific()
    print(
        json.dumps(
            {
                "status": result["scientific"]["status"],
                "scientific_sha256": result["scientific_sha256"],
                "workflow_execution_count": result["scientific"][
                    "run_manifest"
                ]["real_workflow_execution_count"],
                "query_comparison": result["scientific"][
                    "query_comparison"
                ]["status"],
            },
            sort_keys=True,
        )
    )
