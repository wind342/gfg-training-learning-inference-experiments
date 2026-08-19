from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_policy(experiment_root: Path) -> dict[str, Any]:
    path = experiment_root / "profiles" / "oracle_isolation_policy_v2.json"
    policy = json.loads(path.read_text(encoding="utf-8"))
    if not policy["process_contract"]["separate_processes_required"]:
        raise ValueError("Oracle policy must require separate processes")
    return policy


def _module_name(value: str | None) -> str:
    return "" if not value else value.rsplit(".", 1)[-1]


def _source_map(source_root: Path, overrides: dict[str, str] | None = None) -> dict[str, str]:
    values = {
        path.stem: path.read_text(encoding="utf-8")
        for path in source_root.glob("*.py")
        if path.name != "__init__.py"
    }
    values.update(overrides or {})
    return values


def _ast_facts(module: str, source: str, local_modules: set[str]) -> dict[str, Any]:
    tree = ast.parse(source, filename=module + ".py")
    imports: set[str] = set()
    local_edges: set[str] = set()
    imported_symbols: dict[str, tuple[str, str]] = {}
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                imports.add(item.name)
                target = _module_name(item.name)
                if target in local_modules:
                    local_edges.add(target)
        elif isinstance(node, ast.ImportFrom):
            target = _module_name(node.module)
            if node.module:
                imports.add(node.module)
            if target in local_modules:
                local_edges.add(target)
                for item in node.names:
                    imported_symbols[item.asname or item.name] = (target, item.name)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    called_imports = sorted({
        f"{target}.{symbol}"
        for alias, (target, symbol) in imported_symbols.items()
        if alias in calls
    })
    return {
        "called_imported_symbols": called_imports,
        "imports": sorted(imports),
        "local_edges": sorted(local_edges),
        "module": module,
    }


def _closure(starts: list[str], edges: dict[str, set[str]]) -> set[str]:
    result: set[str] = set()
    pending = list(starts)
    while pending:
        value = pending.pop()
        if value in result:
            continue
        result.add(value)
        pending.extend(edges.get(value, set()) - result)
    return result


def analyze_import_graph(
    source_root: Path,
    policy: dict[str, Any],
    overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    sources = _source_map(source_root, overrides)
    local_modules = set(sources)
    facts = {name: _ast_facts(name, source, local_modules) for name, source in sources.items()}
    edges = {name: set(value["local_edges"]) for name, value in facts.items()}
    candidate_starts = list(policy["candidate_modules"])
    native_starts = list(policy["native_modules"])
    candidate_closure = _closure(candidate_starts, edges)
    native_closure = _closure(native_starts, edges)
    shared_modules = (candidate_closure & native_closure) - set(candidate_starts) - set(native_starts)
    allowed_neutral = set(policy["allowed_neutral_modules"])

    candidate_imports = {item for module in candidate_closure for item in facts[module]["imports"]}
    native_imports = {item for module in native_closure for item in facts[module]["imports"]}
    candidate_calls = {
        item for module in candidate_starts for item in facts[module]["called_imported_symbols"]
    }
    native_calls = {
        item for module in native_starts for item in facts[module]["called_imported_symbols"]
    }
    shared_called_symbols = candidate_calls & native_calls
    allowed_symbols = {
        f"{module}.{symbol}"
        for module, symbols in policy["allowed_neutral_symbols"].items()
        for symbol in symbols
    }
    forbidden_shared_modules = sorted(shared_modules - allowed_neutral)
    forbidden_shared_symbols = sorted(shared_called_symbols - allowed_symbols)
    fragment_hits = sorted({
        symbol
        for symbol in shared_called_symbols
        if any(fragment in symbol.lower() for fragment in policy["forbidden_shared_mapping_symbol_fragments"])
        and symbol not in allowed_symbols
    })
    shared_mapping_details = sorted(set(forbidden_shared_modules + forbidden_shared_symbols + fragment_hits))

    candidate_imports_native = bool(candidate_closure & set(native_starts))
    native_imports_candidate = bool(native_closure & set(candidate_starts))
    native_imports_core = any(name.startswith("generation_relation_core") or ".generation_relation_core" in name for name in native_imports)
    candidate_uses_provo_normalizer = "provo_normalizer" in candidate_closure
    native_uses_provn_parser = "provn" in native_closure
    result = {
        "called_imported_symbols": {
            "candidate": sorted(candidate_calls),
            "native": sorted(native_calls),
            "shared": sorted(shared_called_symbols),
        },
        "candidate_imports_native": candidate_imports_native,
        "candidate_module_closure": sorted(candidate_closure),
        "candidate_uses_provo_normalizer": candidate_uses_provo_normalizer,
        "edges": sorted(
            (
                {"from": source, "to": target}
                for source, targets in edges.items()
                for target in targets
            ),
            key=lambda item: (item["from"], item["to"]),
        ),
        "modules": [facts[name] for name in sorted(facts)],
        "native_imports_candidate": native_imports_candidate,
        "native_imports_core": native_imports_core,
        "native_module_closure": sorted(native_closure),
        "native_uses_provn_parser": native_uses_provn_parser,
        "policy_id": policy["policy_id"],
        "shared_mapping_helper_count": len(shared_mapping_details),
        "shared_mapping_helper_details": shared_mapping_details,
        "shared_neutral_module_count": len(shared_modules & allowed_neutral),
        "shared_neutral_modules": sorted(shared_modules & allowed_neutral),
    }
    result["status"] = "PASS" if all((
        not candidate_imports_native,
        not native_imports_candidate,
        not native_imports_core,
        not candidate_uses_provo_normalizer,
        not native_uses_provn_parser,
        result["shared_mapping_helper_count"] == 0,
    )) else "FAIL"
    return result


def _run_process(experiment_root: Path, mode: str, run_index: int) -> dict[str, Any]:
    script = experiment_root / "src" / "audit_process.py"
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONHASHSEED"] = "0"
    process = subprocess.run(
        [sys.executable, "-B", str(script), "--mode", mode],
        cwd=experiment_root.parents[1],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    value: dict[str, Any] | None = None
    parse_error: str | None = None
    if process.stdout:
        try:
            value = json.loads(process.stdout)
        except json.JSONDecodeError as error:
            parse_error = str(error)
    return {
        "command": ["{python}", "-B", "{experiment}/src/audit_process.py", "--mode", mode],
        "exit_code": process.returncode,
        "mode": mode,
        "parse_error": parse_error,
        "process_boundary": "separate child process",
        "result": None if value is None else value["result"],
        "run": run_index,
        "stderr": process.stderr,
        "trace": None if value is None else value["trace"],
    }


def _paths(run: dict[str, Any]) -> list[str]:
    trace = run.get("trace") or {}
    return [item["path"].lower() for item in trace.get("opened_file_paths", [])]


def _module_names(run: dict[str, Any]) -> list[str]:
    trace = run.get("trace") or {}
    return [item.lower() for item in trace.get("imported_modules", [])]


def _count_matching(values: list[str], fragments: tuple[str, ...]) -> int:
    return sum(any(fragment in value for fragment in fragments) for value in values)


def _runtime_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_runs = [run for run in runs if run["mode"] == "candidate"]
    native_runs = [run for run in runs if run["mode"] == "native"]
    candidate_paths = sorted({path for run in candidate_runs for path in _paths(run)})
    native_paths = sorted({path for run in native_runs for path in _paths(run)})
    candidate_modules = sorted({name for run in candidate_runs for name in _module_names(run)})
    native_modules = sorted({name for run in native_runs for name in _module_names(run)})
    candidate_native_reads = _count_matching(candidate_paths + candidate_modules, (
        "/src/native_reference.", "/src/provo_normalizer.",
        "artifacts/native_reference.ttl", "normalized_prov_dm_reference",
        "w3c_prov_projection_v1.src.native_reference", "w3c_prov_projection_v1.src.provo_normalizer",
    ))
    candidate_expected_reads = _count_matching(candidate_paths, ("expected", "gold_records", "answer_lookup"))
    candidate_old_reads = _count_matching(candidate_paths, ("/experiments/w3c_prov_projection_v1/artifacts/",))
    candidate_hidden_reads = _count_matching(candidate_paths, ("crosswalk", "hidden_lookup", "answer_index"))
    native_candidate_reads = _count_matching(native_paths + native_modules, (
        "/src/candidate_projection.", "/src/provn.", "artifacts/candidate.provn",
        "normalized_prov_dm_candidate", "w3c_prov_projection_v1.src.candidate_projection",
        "w3c_prov_projection_v1.src.provn",
    ))
    native_core_reads = _count_matching(native_paths + native_modules, (
        "/src/generation_relation_core/", "/protocol/core_v3/", "generation_relation_core",
    ))
    native_expected_reads = _count_matching(native_paths, ("expected", "gold_records", "answer_lookup"))
    network_reads = sum(len((run.get("trace") or {}).get("sockets", [])) for run in runs)
    forbidden = sum((
        candidate_native_reads,
        candidate_expected_reads,
        candidate_old_reads,
        candidate_hidden_reads,
        native_candidate_reads,
        native_core_reads,
        native_expected_reads,
        network_reads,
    ))
    return {
        "candidate_expected_answer_read_count": candidate_expected_reads,
        "candidate_hidden_lookup_read_count": candidate_hidden_reads,
        "candidate_native_reference_read_count": candidate_native_reads,
        "candidate_old_artifact_read_count": candidate_old_reads,
        "forbidden_read_count": forbidden,
        "native_candidate_output_read_count": native_candidate_reads,
        "native_core_snapshot_read_count": native_core_reads,
        "native_expected_answer_read_count": native_expected_reads,
        "network_read_count": network_reads,
    }


def run_oracle_process_audit(experiment_root: Path, policy: dict[str, Any] | None = None) -> dict[str, Any]:
    policy = policy or load_policy(experiment_root)
    runs = [
        _run_process(experiment_root, mode, index)
        for index in (1, 2)
        for mode in ("candidate", "native")
    ]
    summary = _runtime_summary(runs)
    candidate = [run for run in runs if run["mode"] == "candidate"]
    native = [run for run in runs if run["mode"] == "native"]
    all_processes_passed = all(run["exit_code"] == 0 and run["result"] is not None for run in runs)
    candidate_consistent = all_processes_passed and candidate[0]["result"] == candidate[1]["result"]
    native_consistent = all_processes_passed and native[0]["result"] == native[1]["result"]
    normalized_equal = all_processes_passed and all(
        candidate[index]["result"]["normalized_record_sha256"] == native[index]["result"]["normalized_record_sha256"]
        and candidate[index]["result"]["normalized_record_count"] == native[index]["result"]["normalized_record_count"]
        for index in range(2)
    )
    result = {
        "all_processes_passed": all_processes_passed,
        "candidate_runs_consistent": candidate_consistent,
        "native_runs_consistent": native_consistent,
        "normalized_results_equal": normalized_equal,
        "policy_id": policy["policy_id"],
        "process_memory_shared": False,
        "run_count_per_path": 2,
        "runs": runs,
        "summary": summary,
    }
    result["status"] = "PASS" if all((
        all_processes_passed,
        candidate_consistent,
        native_consistent,
        normalized_equal,
        summary["forbidden_read_count"] == 0,
    )) else "FAIL"
    return result


def build_oracle_isolation(
    import_graph: dict[str, Any],
    process_trace: dict[str, Any],
) -> dict[str, Any]:
    candidate_runs = [run for run in process_trace["runs"] if run["mode"] == "candidate"]
    native_runs = [run for run in process_trace["runs"] if run["mode"] == "native"]
    summary = process_trace["summary"]
    result = {
        "candidate_expected_answer_read_count": summary["candidate_expected_answer_read_count"],
        "candidate_imports_native": import_graph["candidate_imports_native"],
        "candidate_native_reference_read_count": summary["candidate_native_reference_read_count"],
        "candidate_process_exit_code": max(run["exit_code"] for run in candidate_runs),
        "candidate_runtime_input_roles": ["validated_core_snapshot", "current_candidate_provn_bytes"],
        "candidate_uses_provo_normalizer": import_graph["candidate_uses_provo_normalizer"],
        "native_candidate_output_read_count": summary["native_candidate_output_read_count"],
        "native_core_snapshot_read_count": summary["native_core_snapshot_read_count"],
        "native_imports_candidate": import_graph["native_imports_candidate"],
        "native_imports_core": import_graph["native_imports_core"],
        "native_process_exit_code": max(run["exit_code"] for run in native_runs),
        "native_runtime_input_roles": ["actual_generator_callbacks", "current_native_provo_bytes"],
        "native_uses_provn_parser": import_graph["native_uses_provn_parser"],
        "normalized_record_count": candidate_runs[0]["result"]["normalized_record_count"] if candidate_runs[0]["result"] else None,
        "normalized_results_equal": process_trace["normalized_results_equal"],
        "process_memory_shared": process_trace["process_memory_shared"],
        "shared_mapping_helper_count": import_graph["shared_mapping_helper_count"],
        "shared_neutral_module_count": import_graph["shared_neutral_module_count"],
    }
    result["status"] = "SUPPORTED" if all((
        result["candidate_process_exit_code"] == 0,
        result["native_process_exit_code"] == 0,
        not result["candidate_imports_native"],
        not result["native_imports_candidate"],
        not result["native_imports_core"],
        not result["candidate_uses_provo_normalizer"],
        not result["native_uses_provn_parser"],
        result["shared_mapping_helper_count"] == 0,
        result["candidate_native_reference_read_count"] == 0,
        result["candidate_expected_answer_read_count"] == 0,
        result["native_candidate_output_read_count"] == 0,
        result["native_core_snapshot_read_count"] == 0,
        not result["process_memory_shared"],
        result["normalized_results_equal"],
    )) else "NOT_SUPPORTED"
    return result


def _run_mutation(experiment_root: Path, mutation_id: str, target: Path) -> dict[str, Any]:
    script = experiment_root / "src" / "audit_process.py"
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.run(
        [
            sys.executable,
            "-B",
            str(script),
            "--mode",
            "mutation",
            "--mutation-id",
            mutation_id,
            "--target",
            str(target),
        ],
        cwd=experiment_root.parents[1],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    value = json.loads(process.stdout) if process.returncode == 0 else None
    return {
        "exit_code": process.returncode,
        "result": None if value is None else value["result"],
        "trace": None if value is None else value["trace"],
    }


def _comparison_mutates_before_projection(source: str) -> bool:
    tree = ast.parse(source)
    mutation_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"append", "clear", "extend", "insert", "pop", "remove", "update"}
    ]
    projection_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "project_snapshot"
    ]
    return bool(mutation_lines and projection_lines and min(mutation_lines) < min(projection_lines))


def run_oracle_negative_controls(
    experiment_root: Path,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy = policy or load_policy(experiment_root)
    source_root = experiment_root / "src"
    base_candidate = (source_root / "candidate_projection.py").read_text(encoding="utf-8")
    base_native = (source_root / "native_reference.py").read_text(encoding="utf-8")
    results: list[dict[str, Any]] = []

    def add(number: int, name: str, reason: str, detected: bool, evidence: str) -> None:
        results.append({
            "detected": bool(detected),
            "evidence": evidence,
            "expected_reason_code": reason,
            "mutation": name,
            "number": number,
            "status": "FAIL_CLOSED" if detected else "UNDETECTED",
        })

    graph = analyze_import_graph(source_root, policy, {"candidate_projection": base_candidate + "\nfrom .native_reference import NativeProvCollector\n"})
    add(1, "candidate_imports_native_collector", "CANDIDATE_IMPORTS_NATIVE", graph["candidate_imports_native"], "mutated AST/import graph")
    graph = analyze_import_graph(source_root, policy, {"native_reference": base_native + "\nfrom .candidate_projection import project_snapshot\n"})
    add(2, "native_imports_candidate_projection", "NATIVE_IMPORTS_CANDIDATE", graph["native_imports_candidate"], "mutated AST/import graph")
    graph = analyze_import_graph(source_root, policy, {"native_reference": base_native + "\nfrom generation_relation_core.snapshots import ValidatedSnapshot\n"})
    add(3, "native_imports_core", "NATIVE_IMPORTS_CORE", graph["native_imports_core"], "mutated AST/import graph")

    with tempfile.TemporaryDirectory(prefix="w3c-prov-oracle-negative-") as directory:
        root = Path(directory)
        native_ttl = root / "native_reference.ttl"
        native_ttl.write_text("native", encoding="utf-8")
        mutation = _run_mutation(experiment_root, "candidate_read_native_ttl", native_ttl)
        opened = [item["path"] for item in mutation["trace"]["opened_file_paths"]]
        add(4, "candidate_reads_native_ttl", "CANDIDATE_READS_NATIVE_OUTPUT", any(path.endswith("/native_reference.ttl") for path in opened), "subprocess audit-hook open event")

        candidate_provn = root / "candidate.provn"
        candidate_provn.write_text("candidate", encoding="utf-8")
        mutation = _run_mutation(experiment_root, "native_read_candidate_provn", candidate_provn)
        opened = [item["path"] for item in mutation["trace"]["opened_file_paths"]]
        add(5, "native_reads_candidate_provn", "NATIVE_READS_CANDIDATE_OUTPUT", any(path.endswith("/candidate.provn") for path in opened), "subprocess audit-hook open event")

        shared_helper = "def binding_to_derivation(value):\n    return value\n"
        candidate_mutation = base_candidate + "\nfrom .shared_mapping_helper import binding_to_derivation\nbinding_to_derivation(None)\n"
        native_mutation = base_native + "\nfrom .shared_mapping_helper import binding_to_derivation\nbinding_to_derivation(None)\n"
        graph = analyze_import_graph(source_root, policy, {
            "candidate_projection": candidate_mutation,
            "native_reference": native_mutation,
            "shared_mapping_helper": shared_helper,
        })
        add(6, "shared_binding_derivation_helper", "SHARED_MAPPING_HELPER", graph["shared_mapping_helper_count"] > 0, "mutated call graph and policy")

        expected = root / "expected_normalized_records.json"
        expected.write_text("[]", encoding="utf-8")
        mutation = _run_mutation(experiment_root, "candidate_read_expected", expected)
        opened = [item["path"] for item in mutation["trace"]["opened_file_paths"]]
        add(7, "candidate_reads_expected_records", "CANDIDATE_READS_EXPECTED_RECORDS", any("expected_normalized_records" in path for path in opened), "subprocess audit-hook open event")
        mutation = _run_mutation(experiment_root, "native_read_expected", expected)
        opened = [item["path"] for item in mutation["trace"]["opened_file_paths"]]
        add(8, "native_reads_expected_records", "NATIVE_READS_EXPECTED_RECORDS", any("expected_normalized_records" in path for path in opened), "subprocess audit-hook open event")

        comparison_source = "records = []\nrecords.append({'kind': 'derivation'})\nproject_snapshot(snapshot)\n"
        add(9, "comparison_mutates_before_projection", "COMPARISON_MUTATES_BEFORE_PROJECTION", _comparison_mutates_before_projection(comparison_source), "mutated comparison AST")

        exchange = root / "hidden_relation_exchange.tmp"
        written = _run_mutation(experiment_root, "hidden_exchange_write", exchange)
        read = _run_mutation(experiment_root, "hidden_exchange_read", exchange)
        written_paths = {item["path"] for item in written["trace"]["opened_file_paths"]}
        read_paths = {item["path"] for item in read["trace"]["opened_file_paths"]}
        add(10, "hidden_process_answer_exchange", "HIDDEN_PROCESS_ANSWER_EXCHANGE", bool(written_paths & read_paths), "two subprocess audit-hook traces share an actual temporary answer file")

    detected_count = sum(item["detected"] for item in results)
    return {
        "control_family": "oracle_isolation_mutations",
        "controls": results,
        "detected_count": detected_count,
        "negative_control_count": len(results),
        "status": "SUPPORTED" if detected_count == len(results) == 10 else "NOT_SUPPORTED",
        "undetected_count": len(results) - detected_count,
    }
