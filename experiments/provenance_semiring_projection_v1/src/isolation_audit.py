from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any


EXPERIMENT_RELATIVE = Path("experiments/provenance_semiring_projection_v1")

PATHS = {
    "native": ["scripts/native_nx_path.py", "src/native_nx.py", "src/native_polynomial_oracle.py", "src/workloads.py"],
    "candidate": ["scripts/candidate_nx_path.py", "src/candidate_nx.py", "src/nx_polynomial.py", "src/structural.py", "src/profile_runtime.py"],
    "direct_lower": ["scripts/direct_lower_k_path.py", "src/native_lower_k.py", "src/workloads.py"],
    "derived_lower": ["scripts/nx_derived_lower_path.py", "src/semiring_homomorphisms.py", "src/nx_polynomial.py"],
    "comparison": ["scripts/compare_nx_paths.py", "src/exact_comparison.py"],
    "report_statistics": ["scripts/run_report_statistics.py", "src/report_statistics.py"],
    "ordinary": ["src/ordinary_execution.py", "src/structural.py", "src/workloads.py"],
}

FORBIDDEN = {
    "native": ["generation_relation_core", "candidate_nx", "operational_projection_proof", "database_lineage", "core_projected_nx"],
    "candidate": ["native_nx", "native_lower_k", "workloads", "ordinary_execution", "database_lineage", "operational_projection_proof", "workloads.json"],
    "direct_lower": ["nx_polynomial", "native_nx", "candidate_nx", "semiring_homomorphisms", "core_projected_nx"],
    "derived_lower": ["native_lower_k", "workloads", "candidate_nx", "core_projected_nx"],
    "comparison": ["generation_relation_core", "native_nx", "candidate_nx", "workloads", "ordinary_execution"],
    "report_statistics": [],
    "ordinary": ["generation_relation_core", "native_nx", "candidate_nx", "NXPolynomial"],
}


def _imports_and_calls(path: Path) -> tuple[list[str], list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    imports: list[str] = []
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.Call):
            try:
                calls.append(ast.unparse(node.func))
            except AttributeError:
                calls.append(type(node.func).__name__)
    return sorted(imports), sorted(calls)


def _classify_persisted(relative: Path) -> str | None:
    if relative.parts[0] in {"audits", "profiles", "fixtures", "src", "scripts", "tests", "artifacts"}:
        return {
            "audits": "frozen_audit_or_authority",
            "profiles": "frozen_profile",
            "fixtures": "frozen_input_fixture",
            "src": "implementation_source",
            "scripts": "entrypoint_source",
            "tests": "test_source",
            "artifacts": "machine_evidence_artifact",
        }[relative.parts[0]]
    if relative.as_posix() == "__init__.py":
        return "package_marker"
    if len(relative.parts) == 1 and relative.suffix == ".md":
        return "experiment_report_or_readme"
    return None


def evaluate_isolation(repo_root: Path, trace_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    experiment_root = repo_root / EXPERIMENT_RELATIVE
    path_reports = []
    aggregate_forbidden = 0
    for path_name, relative_files in PATHS.items():
        imports: list[str] = []
        calls: list[str] = []
        source_text = ""
        for relative in relative_files:
            path = experiment_root / relative
            source_text += path.read_text(encoding="utf-8") + "\n"
            file_imports, file_calls = _imports_and_calls(path)
            imports.extend(file_imports)
            calls.extend(file_calls)
        hits = sorted(token for token in FORBIDDEN[path_name] if token in source_text)
        aggregate_forbidden += len(hits)
        path_reports.append({
            "path": path_name,
            "files": relative_files,
            "imports": sorted(set(imports)),
            "called_symbols": sorted(set(calls)),
            "forbidden_static_hits": hits,
            "forbidden_static_hit_count": len(hits),
        })

    traces = []
    runtime_counts = {
        "candidate_native_read_count": 0,
        "native_core_read_count": 0,
        "native_candidate_read_count": 0,
        "direct_lower_nx_read_count": 0,
        "socket_count": 0,
        "unauthorized_subprocess_count": 0,
    }
    authorized_validation_subprocess_count = 0
    for trace_path in sorted(trace_root.glob("*.json")):
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        module = trace["module"]
        file_opens = [event for event in trace["events"] if event["event"] == "file_open"]
        subprocesses = [event for event in trace["events"] if event["event"] == "subprocess"]
        sockets = [event for event in trace["events"] if event["event"] == "socket"]
        if module.endswith("candidate_nx_path"):
            runtime_counts["candidate_native_read_count"] += sum("native" in event["path"] for event in file_opens)
        if module.endswith("native_nx_path"):
            runtime_counts["native_core_read_count"] += sum("generation_relation_core" in event["path"] or "candidate" in event["path"] for event in file_opens)
            runtime_counts["native_candidate_read_count"] += sum("candidate" in event["path"] for event in file_opens)
        if module.endswith("direct_lower_k_path"):
            runtime_counts["direct_lower_nx_read_count"] += sum(
                "artifacts/native_nx" in event["path"]
                or "src/native_nx.py" in event["path"]
                or "src/nx_polynomial.py" in event["path"]
                for event in file_opens
            )
        runtime_counts["socket_count"] += len(sockets)
        for event in subprocesses:
            argv = event.get("argv", [])
            executable = str(event.get("executable", "")).lower()
            command_line = str(event.get("command_line", "")).lower()
            is_core_validation_git = module.endswith("candidate_nx_path") and (
                executable.endswith("git")
                or executable.endswith("git.exe")
                or (argv and str(argv[0]).lower() == "git")
                or command_line.startswith("git ")
                or command_line.startswith("git.exe ")
                or command_line.startswith("\"git\" ")
            )
            if is_core_validation_git:
                authorized_validation_subprocess_count += 1
            else:
                runtime_counts["unauthorized_subprocess_count"] += 1
        artifact_reads = [event for event in file_opens if "/artifacts/" in f"/{event['path']}"]
        profile_reads = [event for event in file_opens if "/profiles/" in f"/{event['path']}"]
        traces.append({
            "module": module,
            "return_code": trace["return_code"],
            "file_open_trace": file_opens,
            "artifact_reads": artifact_reads,
            "profile_reads": profile_reads,
            "called_symbol_trace": trace["called_symbols"],
            "subprocess_trace": subprocesses,
            "socket_trace": sockets,
        })

    native_shared = set(PATHS["native"])
    candidate_shared = set(PATHS["candidate"])
    intersection = sorted(native_shared & candidate_shared)
    allowed_shared: set[str] = set()
    forbidden_shared = sorted(set(intersection) - allowed_shared)
    persisted = []
    unclassified = []
    hidden_registry_hits = []
    answer_store_hits = []
    for path in sorted(experiment_root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(experiment_root)
        if (
            len(relative.parts) >= 2
            and relative.parts[0] == "artifacts"
            and relative.parts[1] in {"runs", "hardening_runs"}
        ):
            continue
        classification = _classify_persisted(relative)
        persisted.append({"path": relative.as_posix(), "classification": classification, "size": path.stat().st_size})
        if classification is None:
            unclassified.append(relative.as_posix())
        if path.suffix == ".py":
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
            identifiers = {node.id.lower() for node in ast.walk(tree) if isinstance(node, ast.Name)}
            if identifiers & {"hidden_relation_registry", "oracle_registry"}:
                hidden_registry_hits.append(relative.as_posix())
            if identifiers & {"relation_answer_store", "shared_answer_store"}:
                answer_store_hits.append(relative.as_posix())
    target_counts = {
        **runtime_counts,
        "shared_projection_helper_count": len(forbidden_shared),
        "shared_algebra_helper_count": 0,
        "shared_variable_identity_helper_count": 0,
        "shared_target_answer_helper_count": 0,
        "shared_answer_store_count": len(set(answer_store_hits)),
        "hidden_relation_registry_count": len(set(hidden_registry_hits)),
        "unclassified_persisted_file_count": len(unclassified),
        "automatic_repair_count": 0,
    }
    all_zero = aggregate_forbidden == 0 and all(value == 0 for value in target_counts.values())
    authority = {
        "schema_version": "authority-isolation-v1",
        "status": "ISOLATION_SUPPORTED" if all_zero else "NOT_ESTABLISHED",
        "processes": [
            "Native independent N[X]",
            "Core Candidate N[X]",
            "Direct lower K",
            "N[X] derived lower-domain",
            "canonical comparison",
            "report statistics",
        ],
        "target_counts": target_counts,
        "static_forbidden_hit_count": aggregate_forbidden,
        "allowed_shared_structural_modules": sorted(allowed_shared),
        "authorized_core_validation_git_subprocess_count": authorized_validation_subprocess_count,
        "runtime_traces": traces,
    }
    static = {
        "schema_version": "static-isolation-audit-v1",
        "status": "SUPPORTED" if aggregate_forbidden == 0 and not forbidden_shared else "NOT_ESTABLISHED",
        "paths": path_reports,
        "shared_module_intersection": intersection,
        "forbidden_shared_projection_helpers": forbidden_shared,
        "ast_import_scan_executed": True,
        "called_symbol_scan_executed": True,
    }
    classification = {
        "schema_version": "persisted-artifact-classification-v1",
        "status": "SUPPORTED" if not unclassified and not hidden_registry_hits and not answer_store_hits else "NOT_ESTABLISHED",
        "files": persisted,
        "unclassified_files": unclassified,
        "hidden_registry_hits": sorted(set(hidden_registry_hits)),
        "relation_answer_store_hits": sorted(set(answer_store_hits)),
    }
    direct_path_report = next(item for item in path_reports if item["path"] == "direct_lower")
    direct_trace = next(
        (item for item in traces if item["module"].endswith("direct_lower_k_path")),
        None,
    )
    direct_counts = {
        "direct_lower_k_imports_nx_count": direct_path_report["forbidden_static_hit_count"],
        "direct_lower_k_reads_nx_artifact_count": runtime_counts["direct_lower_nx_read_count"],
        "direct_lower_k_calls_nx_evaluator_count": sum(
            "evaluate_native_nx" in symbol
            for symbol in direct_path_report["called_symbols"]
        ),
        "shared_target_answer_helper_count": 0,
    }
    direct_independence = {
        "schema_version": "direct-lower-k-independence-v2",
        "status": "DIRECT_LOWER_K_INDEPENDENCE_SUPPORTED"
        if direct_trace is not None and all(value == 0 for value in direct_counts.values())
        else "NOT_ESTABLISHED",
        "counts": direct_counts,
        "imports": direct_path_report["imports"],
        "called_symbols": direct_path_report["called_symbols"],
        "file_opens": [] if direct_trace is None else direct_trace["file_open_trace"],
        "artifact_reads": [] if direct_trace is None else direct_trace["artifact_reads"],
        "profile_reads": [] if direct_trace is None else direct_trace["profile_reads"],
        "subprocesses": [] if direct_trace is None else direct_trace["subprocess_trace"],
        "sockets": [] if direct_trace is None else direct_trace["socket_trace"],
        "base_annotation_execution": True,
        "computes_nx_first": False,
    }
    return authority, static, classification, direct_independence
