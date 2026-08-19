from __future__ import annotations

import ast
import fnmatch
import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROLE_NAMES = {
    "authoritative_candidate_input",
    "native_reference_output",
    "candidate_projection_output",
    "normative_authority_manifest",
    "neutral_shared_schema_or_record_model",
    "diagnostic_report",
    "test_fixture",
    "forbidden_secondary_relation_store",
    "unclassified",
}


def load_policy(experiment_root: Path) -> dict[str, Any]:
    path = experiment_root / "profiles" / "authority_store_audit_policy_v2.json"
    policy = json.loads(path.read_text(encoding="utf-8"))
    if set(policy["roles"]) != ROLE_NAMES:
        raise ValueError("authority-store policy role set is not closed")
    if len(policy["candidate_authorities"]) != 1:
        raise ValueError("authority-store policy must declare exactly one candidate authority")
    return policy


def _excluded(relative: str, policy: dict[str, Any]) -> bool:
    return any(fnmatch.fnmatchcase(relative, pattern) for pattern in policy["scan_exclusions"])


def _classify(relative: str, policy: dict[str, Any]) -> str:
    for rule in policy["classification_rules"]:
        if any(fnmatch.fnmatchcase(relative, pattern) for pattern in rule["patterns"]):
            return str(rule["role"])
    return "unclassified"


def repository_file_list(experiment_root: Path, policy: dict[str, Any]) -> list[str]:
    repo = experiment_root.parents[1]
    experiment_relative = experiment_root.relative_to(repo).as_posix()
    process = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            experiment_relative,
        ],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    prefix = experiment_relative + "/"
    relative = [line[len(prefix):] for line in process.stdout.splitlines() if line.startswith(prefix)]
    return sorted(path for path in relative if not _excluded(path, policy))


def _strings(node: ast.AST) -> list[str]:
    return [child.value for child in ast.walk(node) if isinstance(child, ast.Constant) and isinstance(child.value, str)]


def _candidate_read_findings(text: str, policy: dict[str, Any]) -> list[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return ["CANDIDATE_SOURCE_SYNTAX_ERROR"]
    findings: set[str] = set()
    read_methods = {"open", "read_bytes", "read_text"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
        if name not in read_methods:
            continue
        values = [value.lower().replace("\\", "/") for value in _strings(node)]
        for value in values:
            for fragment, reason in policy["forbidden_candidate_read_targets"].items():
                if fragment.lower() in value:
                    findings.add(str(reason))
    return sorted(findings)


def _walk_json(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _walk_json(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_json(nested)


def _json_findings(value: Any, policy: dict[str, Any]) -> list[str]:
    findings: set[str] = set()
    for obj in _walk_json(value):
        keys = set(obj)
        if "core_to_prov_id" in keys or "core_id_to_prov_id" in keys:
            findings.add("PERSISTENT_CORE_PROV_ID_MAPPING")
        for rule in policy["forbidden_json_key_sets"]:
            all_keys = set(rule.get("all_keys", []))
            any_keys = set(rule.get("any_keys", []))
            if (all_keys and all_keys <= keys) or (any_keys and any_keys & keys):
                findings.add(str(rule["reason_code"]))
    return sorted(findings)


def _content_findings(relative: str, role: str, data: bytes, policy: dict[str, Any]) -> list[str]:
    findings: set[str] = set()
    text = data.decode("utf-8", errors="replace")
    if relative == "src/candidate_projection.py":
        findings.update(_candidate_read_findings(text, policy))
    if relative.endswith(".json") and role not in {"diagnostic_report", "normative_authority_manifest"}:
        try:
            findings.update(_json_findings(json.loads(text), policy))
        except json.JSONDecodeError:
            findings.add("INVALID_JSON_IN_SCANNED_FILE")
    if relative.endswith((".provn", ".ttl")):
        lowered = text.lower().replace("ex:", "")
        for token, reason in policy["forbidden_prov_attribute_tokens"].items():
            if token.lower() in lowered:
                findings.add(str(reason))
    return sorted(findings)


def scan_paths(experiment_root: Path, relative_paths: Iterable[str], policy: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for relative in sorted(set(relative_paths)):
        path = experiment_root / relative
        role = _classify(relative, policy)
        findings = ["UNCLASSIFIED_FILE"] if role == "unclassified" else []
        data = path.read_bytes()
        findings.extend(_content_findings(relative, role, data, policy))
        findings = sorted(set(findings))
        if findings and any(item != "INVALID_JSON_IN_SCANNED_FILE" for item in findings):
            role = "forbidden_secondary_relation_store"
        rows.append({
            "findings": findings,
            "path": relative,
            "role": role,
        })
    roles = Counter(row["role"] for row in rows)
    findings = Counter(item for row in rows for item in row["findings"])
    forbidden_count = roles["forbidden_secondary_relation_store"]
    unclassified_count = sum("UNCLASSIFIED_FILE" in row["findings"] for row in rows)
    result = {
        "authoritative_candidate_input_count": roles["authoritative_candidate_input"],
        "candidate_projection_output_count": roles["candidate_projection_output"],
        "classified_file_count": len(rows) - unclassified_count,
        "expected_answer_artifact_count": findings["CANDIDATE_READS_EXPECTED_ANSWER"],
        "files": rows,
        "forbidden_secondary_relation_store_count": forbidden_count,
        "hidden_binding_crosswalk_count": findings["FORBIDDEN_SECONDARY_RELATION_STORE"],
        "persistent_candidate_lookup_table_count": findings["PERSISTENT_CORE_PROV_ID_MAPPING"],
        "policy_id": policy["policy_id"],
        "receipt_answer_index_count": findings["CANDIDATE_READS_CALLBACK_RECEIPT"],
        "role_counts": dict(sorted(roles.items())),
        "scanned_file_count": len(rows),
        "snapshot_blob_embedded_in_prov_count": findings["SNAPSHOT_BLOB_EMBEDDED_IN_PROV"],
        "classified_file_count_check": len(rows) - unclassified_count,
        "unclassified_file_count": unclassified_count,
        "native_reference_output_count": roles["native_reference_output"],
    }
    result["status"] = "PASS" if unclassified_count == 0 and forbidden_count == 0 else "FAIL"
    return result


def scan_repository(experiment_root: Path, policy: dict[str, Any] | None = None) -> dict[str, Any]:
    policy = policy or load_policy(experiment_root)
    return scan_paths(experiment_root, repository_file_list(experiment_root, policy), policy)


def build_runtime_authority_trace(oracle_process_trace: dict[str, Any]) -> dict[str, Any]:
    by_mode = {
        mode: [run for run in oracle_process_trace["runs"] if run["mode"] == mode]
        for mode in ("candidate", "native")
    }

    def segment(path_id: str, mode: str, input_roles: list[str], output_roles: list[str]) -> dict[str, Any]:
        runs = by_mode[mode]
        opened = sorted({item["path"] for run in runs for item in run["trace"]["opened_file_paths"]})
        imported = sorted({item for run in runs for item in run["trace"]["imported_modules"]})
        subprocesses = sorted({item for run in runs for item in run["trace"]["subprocesses"]})
        sockets = sorted({item for run in runs for item in run["trace"]["sockets"]})
        return {
            "environment_variable_reads": {
                "reliably_observable": False,
                "values": [],
            },
            "imported_modules": imported,
            "input_artifact_roles": input_roles,
            "opened_file_paths": opened,
            "output_artifact_roles": output_roles,
            "path_id": path_id,
            "process_run_count": len(runs),
            "sockets": sockets,
            "subprocesses": subprocesses,
        }

    paths = [
        segment("candidate_projection", "candidate", ["validated_core_snapshot"], ["current_candidate_provn_bytes"]),
        segment("native_reference", "native", ["actual_generator_callbacks"], ["current_native_provo_bytes"]),
        segment("candidate_provn_parser", "candidate", ["current_candidate_provn_bytes"], ["current_candidate_normalized_records"]),
        segment("native_provo_normalizer", "native", ["current_native_provo_bytes"], ["current_native_normalized_records"]),
    ]
    summary = oracle_process_trace["summary"]
    result = {
        "candidate_output_read_by_native_count": summary["native_candidate_output_read_count"],
        "expected_answer_read_count": summary["candidate_expected_answer_read_count"] + summary["native_expected_answer_read_count"],
        "forbidden_read_count": summary["forbidden_read_count"],
        "hidden_lookup_read_count": summary["candidate_hidden_lookup_read_count"],
        "native_reference_read_by_candidate_count": summary["candidate_native_reference_read_count"],
        "network_read_count": sum(len(path["sockets"]) for path in paths),
        "old_artifact_read_count": summary["candidate_old_artifact_read_count"],
        "paths": paths,
        "policy_id": oracle_process_trace["policy_id"],
    }
    result["status"] = "PASS" if all(result[key] == 0 for key in (
        "candidate_output_read_by_native_count",
        "expected_answer_read_count",
        "forbidden_read_count",
        "hidden_lookup_read_count",
        "native_reference_read_by_candidate_count",
        "network_read_count",
        "old_artifact_read_count",
    )) else "FAIL"
    return result


def compute_second_authority_audit(
    scan: dict[str, Any],
    runtime_trace: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    candidate_authorities = policy["candidate_authorities"]
    native_oracles = policy["native_test_oracles"]
    result = {
        "candidate_authority_count": len(candidate_authorities),
        "candidate_authority_identity": candidate_authorities[0]["identity"] if len(candidate_authorities) == 1 else None,
        "candidate_hidden_lookup_read_count": runtime_trace["hidden_lookup_read_count"],
        "candidate_native_reference_read_count": runtime_trace["native_reference_read_by_candidate_count"],
        "candidate_old_artifact_read_count": runtime_trace["old_artifact_read_count"],
        "classified_file_count": scan["classified_file_count"],
        "expected_prov_document_input_count": runtime_trace["expected_answer_read_count"],
        "hidden_binding_crosswalk_count": scan["hidden_binding_crosswalk_count"],
        "native_test_oracle_count": len(native_oracles),
        "persisted_secondary_relation_store_count": scan["forbidden_secondary_relation_store_count"],
        "persistent_candidate_lookup_table_count": scan["persistent_candidate_lookup_table_count"],
        "receipt_answer_index_count": scan["receipt_answer_index_count"],
        "scanned_file_count": scan["scanned_file_count"],
        "snapshot_blob_embedded_in_prov_count": scan["snapshot_blob_embedded_in_prov_count"],
        "unclassified_file_count": scan["unclassified_file_count"],
    }
    conditions = (
        result["candidate_authority_count"] == 1,
        result["candidate_authority_identity"] == "validated Core Snapshot Γ",
        result["unclassified_file_count"] == 0,
        result["persistent_candidate_lookup_table_count"] == 0,
        result["hidden_binding_crosswalk_count"] == 0,
        result["expected_prov_document_input_count"] == 0,
        result["receipt_answer_index_count"] == 0,
        result["candidate_native_reference_read_count"] == 0,
        result["candidate_old_artifact_read_count"] == 0,
        result["candidate_hidden_lookup_read_count"] == 0,
        result["snapshot_blob_embedded_in_prov_count"] == 0,
        result["persisted_secondary_relation_store_count"] == 0,
    )
    result["second_authority_count"] = sum(not condition for condition in conditions)
    result["status"] = "SUPPORTED" if all(conditions) else "NOT_SUPPORTED"
    return result


def run_authority_negative_controls(policy: dict[str, Any]) -> dict[str, Any]:
    cases = [
        ("forged_core_prov_lookup", "forged_core_prov_lookup.json", '{"core_to_prov_id":{"si3_fake":"ex:e_fake"}}', "PERSISTENT_CORE_PROV_ID_MAPPING"),
        ("candidate_reads_native_ttl", "src/candidate_projection.py", "from pathlib import Path\nPath('artifacts/native_reference.ttl').read_text()\n", "CANDIDATE_READS_NATIVE_REFERENCE"),
        ("candidate_reads_normalized_reference", "src/candidate_projection.py", "from pathlib import Path\nPath('artifacts/normalized_prov_dm_reference.json').read_bytes()\n", "CANDIDATE_READS_NATIVE_REFERENCE"),
        ("candidate_reads_old_provn", "src/candidate_projection.py", "from pathlib import Path\nPath('artifacts/candidate.provn').read_bytes()\n", "CANDIDATE_READS_OLD_CANDIDATE_PROVN"),
        ("candidate_reads_expected_answer", "src/candidate_projection.py", "from pathlib import Path\nPath('expected_answer.json').read_text()\n", "CANDIDATE_READS_EXPECTED_ANSWER"),
        ("candidate_reads_callback_receipt", "src/candidate_projection.py", "from pathlib import Path\nPath('runtime/callback_receipt.json').read_text()\n", "CANDIDATE_READS_CALLBACK_RECEIPT"),
        ("hidden_binding_crosswalk", "hidden_binding_crosswalk.json", '{"binding_crosswalk":{"gb3_fake":"ex:d_fake"}}', "FORBIDDEN_SECONDARY_RELATION_STORE"),
        ("snapshot_blob_in_prov", "artifacts/leak.provn", "entity(ex:e, [ex:snapshot_json='{\"snapshot_id\":\"snap3_fake\"}'])", "SNAPSHOT_BLOB_EMBEDDED_IN_PROV"),
        ("evidence_blob_in_prov", "artifacts/leak.ttl", "ex:e ex:evidence_blob \"opaque evidence\" .", "EVIDENCE_BLOB_EMBEDDED_IN_PROV"),
        ("unclassified_relation_file", "mystery_relations.bin", "origin=source activity=render outcome=final", "UNCLASSIFIED_FILE"),
    ]
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="w3c-prov-authority-negative-") as directory:
        root = Path(directory)
        for number, (name, relative, content, expected) in enumerate(cases, start=1):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            scan = scan_paths(root, [relative], policy)
            reasons = sorted({reason for row in scan["files"] for reason in row["findings"]})
            detected = scan["status"] == "FAIL" and expected in reasons
            results.append({
                "detected": detected,
                "expected_reason_code": expected,
                "mutation": name,
                "number": number,
                "observed_reason_codes": reasons,
                "status": "FAIL_CLOSED" if detected else "UNDETECTED",
            })
            path.unlink()
    detected_count = sum(result["detected"] for result in results)
    return {
        "control_family": "secondary_authority_mutations",
        "controls": results,
        "detected_count": detected_count,
        "negative_control_count": len(results),
        "status": "SUPPORTED" if detected_count == len(results) == 10 else "NOT_SUPPORTED",
        "undetected_count": len(results) - detected_count,
    }
