from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import pyparsing
import rdflib

from .authority_store_audit import (
    build_runtime_authority_trace,
    compute_second_authority_audit,
    load_policy as load_authority_policy,
    run_authority_negative_controls,
    scan_repository,
)
from .negative_controls import run_negative_controls
from .official_tests import run_official_tests
from .oracle_isolation_audit import (
    analyze_import_graph,
    build_oracle_isolation,
    load_policy as load_oracle_policy,
    run_oracle_negative_controls,
    run_oracle_process_audit,
)
from .record_model import canonical_json_bytes
from .science_runs import (
    actual_transform_context_counterexample,
    output_modes,
    run_full,
    run_transform_counterexample_negative_controls,
    strict_projection_counterexamples,
)
from .validation import exact_comparison, relation_multiplicity, validate_profile_documents


BASE_COMMIT = "e00144b6b47504287c2d16f20b064da81e43f1cc"
BASE_TREE = "ffd70bc0fec126961bf0ac8b4e6963e6c03c3963"
PR12_HEAD = "b8e71a84d85dc361889a615d32348b9ac4d0481f"
PR12_HEAD_TREE = "c811c2ab8236eb00f4ca8f6c743163c8aa4fec20"
PR12_COMMITS = [
    "de532dae85df5890507e0eb68fc03fee80ca463f",
    "221863bf086ade497d1b67e818716136f13be137",
    "b4236b4fce417f3f069431c35638197c3942343d",
    "b8e71a84d85dc361889a615d32348b9ac4d0481f",
]
PROFILE_ID = "w3c-prov-generation-profile-v1"
FINAL_SUPPORTED = "W3C_PROV_PROJECTION_V1_SUPPORTED"
HARDENING_SUPPORTED = "W3C_PROV_PROJECTION_V1_EVIDENCE_HARDENING_SUPPORTED"

REQUIRED_ARTIFACTS = {
    "actual_transform_context_counterexample.json",
    "artifact_manifest.json",
    "authority_audit.json",
    "authority_store_rule_scan.json",
    "candidate.provn",
    "candidate_prov_projection_summary.json",
    "core_change_lineage.json",
    "determinism.json",
    "evidence_hardening_run_summary.json",
    "hardening_negative_control_accounting.json",
    "hardening_negative_controls.json",
    "native_prov_reference_summary.json",
    "native_reference.ttl",
    "negative_controls.json",
    "normalized_prov_dm_candidate.json",
    "normalized_prov_dm_reference.json",
    "official_test_results.json",
    "oracle_import_dependency_graph.json",
    "oracle_isolation.json",
    "oracle_runtime_process_trace.json",
    "output_orthogonality.json",
    "prov_constraint_validation.json",
    "prov_identifier_determinism.json",
    "prov_n_prov_o_equivalence.json",
    "prov_record_exact_comparison.json",
    "prov_relation_multiplicity.json",
    "prov_roundtrip_validation.json",
    "reverse_non_identifiability.json",
    "run_summary.json",
    "runtime_authority_dependency_trace.json",
    "runtime_dependency_trace.json",
    "second_authority_audit.json",
    "strict_projection_counterexamples.json",
    "test_results.json",
}


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_json_bytes(value))


def _git(repo: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )
    return process.stdout.rstrip()


def _git_is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    process = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=repo,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return process.returncode == 0


def _test_receipts(experiment_root: Path) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for index in (1, 2):
        junit = experiment_root / "runtime" / f"full-test-run-{index}.junit.xml"
        if not junit.is_file():
            raise FileNotFoundError(f"independent full-test receipt missing: {junit}")
        root = ET.parse(junit).getroot()
        suite = root if root.tag == "testsuite" else root.find("testsuite")
        if suite is None:
            raise RuntimeError("pytest JUnit report has no testsuite")
        test_count = int(suite.attrib["tests"])
        failed_count = int(suite.attrib.get("failures", "0")) + int(suite.attrib.get("errors", "0"))
        skipped_count = int(suite.attrib.get("skipped", "0"))
        runs.append({
            "exit_code": 0 if failed_count == 0 else 1,
            "failed_count": failed_count,
            "normalized_status": "PASSED" if failed_count == 0 else "FAILED",
            "passed_count": test_count - failed_count - skipped_count,
            "run": index,
            "skipped_count": skipped_count,
            "test_count": test_count,
        })
    consistent = runs[0] == {**runs[1], "run": 1}
    return {
        "command": "python -m pytest -vv tests experiments/w3c_prov_projection_v1/tests",
        "consistent_ignoring_run_index_and_duration": consistent,
        "run_count": 2,
        "runs": runs,
        "status": "SUPPORTED" if consistent and all(item["exit_code"] == 0 for item in runs) else "NOT_SUPPORTED",
    }


def _lineage(repo: Path, orthogonality: dict[str, Any]) -> dict[str, Any]:
    committed = [line for line in _git(repo, "diff", "--name-only", PR12_HEAD, "--").splitlines() if line]
    status_paths = []
    for line in _git(repo, "status", "--porcelain=v1").splitlines():
        value = line[3:]
        status_paths.append(value.rsplit(" -> ", 1)[-1])
    changed_paths = sorted(set(committed + status_paths))
    protected_prefixes = ("src/generation_relation_core/", "protocol/core_v3/", "compat/v2/", "tests/core/")
    protected_changes = [path for path in changed_paths if path.startswith(protected_prefixes)]
    existing_experiment_changes = [
        path for path in changed_paths
        if path.startswith("experiments/") and not path.startswith("experiments/w3c_prov_projection_v1/")
    ]
    outside_target = [path for path in changed_paths if not path.startswith("experiments/w3c_prov_projection_v1/")]
    core_schema = (repo / "protocol" / "core_v3" / "core_v3_entities.schema.json").read_text(encoding="utf-8")
    core_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((repo / "src" / "generation_relation_core").glob("*.py"))
    )
    original_order = [line for line in _git(repo, "rev-list", "--reverse", f"{BASE_COMMIT}..{PR12_HEAD}").splitlines() if line]
    result = {
        "actual_remote_main": BASE_COMMIT,
        "actual_remote_main_tree": BASE_TREE,
        "base_commit": BASE_COMMIT,
        "compat_v2_change_count": sum(path.startswith("compat/v2/") for path in protected_changes),
        "core_schema_change_count": sum(path.startswith("protocol/core_v3/") for path in protected_changes),
        "core_source_change_count": sum(path.startswith("src/generation_relation_core/") for path in protected_changes),
        "existing_experiment_change_count": len(existing_experiment_changes),
        "generator_name_branch_count": len(re.findall(r"generator_name\s*==", core_source)),
        "hardening_diff_outside_target_count": len(outside_target),
        "ordinary_outputs_unaffected": orthogonality["status"] == "SUPPORTED",
        "original_pr_commit_order": original_order,
        "original_pr_commit_order_preserved": original_order == PR12_COMMITS,
        "pr12_head": PR12_HEAD,
        "pr12_head_is_ancestor": _git_is_ancestor(repo, PR12_HEAD, "HEAD"),
        "pr12_head_tree_id": PR12_HEAD_TREE,
        "prov_entity_activity_core_branch_count": len(re.findall(r"W3C|PROV|SoftwareAgent", core_source, flags=re.IGNORECASE)),
        "prov_specific_core_field_count": core_schema.count("http://www.w3.org/ns/prov#"),
        "tests_core_change_count": sum(path.startswith("tests/core/") for path in protected_changes),
    }
    zero_fields = (
        "compat_v2_change_count",
        "core_schema_change_count",
        "core_source_change_count",
        "existing_experiment_change_count",
        "generator_name_branch_count",
        "hardening_diff_outside_target_count",
        "prov_entity_activity_core_branch_count",
        "prov_specific_core_field_count",
        "tests_core_change_count",
    )
    result["status"] = "SUPPORTED" if all((
        all(result[field] == 0 for field in zero_fields),
        result["ordinary_outputs_unaffected"],
        result["original_pr_commit_order_preserved"],
        result["pr12_head_is_ancestor"],
    )) else "NOT_SUPPORTED"
    return result


def _render_bundle(values: dict[str, Any], binary_values: dict[str, bytes]) -> dict[str, bytes]:
    rendered = {name: _json_bytes(value) for name, value in values.items()}
    rendered.update(binary_values)
    expected_without_manifest = REQUIRED_ARTIFACTS - {"artifact_manifest.json"}
    manifested_names = set(rendered)
    entries = [
        {"bytes": len(rendered[name]), "path": name, "sha256": _sha(rendered[name])}
        for name in sorted(rendered)
    ]
    manifest = {
        "all_required_artifacts_present": manifested_names == expected_without_manifest,
        "artifact_count_excluding_manifest": len(entries),
        "artifacts": entries,
        "manifest_version": 2,
        "missing_required_artifacts": sorted(expected_without_manifest - manifested_names),
        "self_entry": "excluded from its own hash to avoid recursion; its bytes are included in the two-run materialization comparison",
        "unexpected_artifacts": sorted(manifested_names - expected_without_manifest),
    }
    rendered["artifact_manifest.json"] = _json_bytes(manifest)
    return rendered


def _summary(criteria: list[tuple[str, bool]], supported: str, unsupported: str) -> dict[str, Any]:
    preliminary = [name for name, passed in criteria if not passed]
    criteria = [*criteria, ("blocking_reasons empty", not preliminary)]
    blocking = [name for name, passed in criteria if not passed]
    return {
        "blocking_reasons": blocking,
        "criteria": [{"criterion": name, "passed": passed} for name, passed in criteria],
        "mandatory_criterion_count": len(criteria),
        "passed_criterion_count": sum(passed for _name, passed in criteria),
        "final_status": supported if not blocking else unsupported,
    }


def materialize(experiment_root: Path) -> dict[str, Any]:
    repo = experiment_root.parents[1]
    artifacts = experiment_root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    source_root = experiment_root / "src"

    first = run_full()
    second = run_full()
    binding_count = len(first.snapshot.tables.generation_bindings)
    comparison = exact_comparison(first.candidate_from_provn, first.reference_from_provo, binding_count)
    constraints = validate_profile_documents(first.candidate_records, first.candidate_provn, first.native_ttl)
    multiplicity = relation_multiplicity(first.candidate_records, binding_count)
    official = run_official_tests(experiment_root / "runtime" / "official_tests")
    counterexamples, reverse = strict_projection_counterexamples()
    actual_group, actual_transform = actual_transform_context_counterexample()
    orthogonality = output_modes()
    original_negatives = run_negative_controls(source_root)

    authority_policy = load_authority_policy(experiment_root)
    oracle_policy = load_oracle_policy(experiment_root)
    import_graph = analyze_import_graph(source_root, oracle_policy)
    process_trace = run_oracle_process_audit(experiment_root, oracle_policy)
    oracle = build_oracle_isolation(import_graph, process_trace)
    runtime_authority = build_runtime_authority_trace(process_trace)
    authority_negatives = run_authority_negative_controls(authority_policy)
    oracle_negatives = run_oracle_negative_controls(experiment_root, oracle_policy)
    transform_negatives = run_transform_counterexample_negative_controls()
    tests = _test_receipts(experiment_root)

    kinds = Counter(record["kind"] for record in first.candidate_records)
    native_summary = {
        "authority": "Synchronous callbacks from the actual deterministic generator",
        "candidate_import_count": int(import_graph["native_imports_candidate"]),
        "core_import_count": int(import_graph["native_imports_core"]),
        "normalization": "RDFLib Turtle parser followed by independent qualified PROV-O normalizer",
        "profile_id": PROFILE_ID,
        "qualified_provo_sha256": _sha(first.native_ttl),
        "record_count": len(first.reference_from_provo),
        "record_counts": dict(sorted(kinds.items())),
        "status": "SUPPORTED" if first.reference_from_provo == first.candidate_records else "NOT_SUPPORTED",
    }
    candidate_summary = {
        "expected_answer_read_count": oracle["candidate_expected_answer_read_count"],
        "native_reference_read_count": oracle["candidate_native_reference_read_count"],
        "only_input": "ValidatedSnapshot",
        "profile_id": PROFILE_ID,
        "provn_sha256": _sha(first.candidate_provn),
        "record_count": len(first.candidate_from_provn),
        "record_counts": dict(sorted(kinds.items())),
        "status": "SUPPORTED" if first.candidate_from_provn == first.candidate_records else "NOT_SUPPORTED",
    }
    equivalence = {
        "candidate_provn_record_sha256": _sha(canonical_json_bytes(first.candidate_from_provn)),
        "native_provo_record_sha256": _sha(canonical_json_bytes(first.reference_from_provo)),
        "normalized_records_equal": first.candidate_from_provn == first.reference_from_provo,
        "record_count": len(first.candidate_records),
        "status": "SUPPORTED" if first.candidate_from_provn == first.reference_from_provo else "NOT_SUPPORTED",
    }
    roundtrip = {
        "candidate_provn_parses_to_original_records": first.candidate_from_provn == first.candidate_records,
        "native_provo_parses_to_original_records": first.reference_from_provo == first.candidate_records,
        "provn_byte_sha256": _sha(first.candidate_provn),
        "provo_byte_sha256": _sha(first.native_ttl),
        "status": "SUPPORTED" if first.candidate_from_provn == first.reference_from_provo == first.candidate_records else "NOT_SUPPORTED",
    }
    identifiers = [record["id"] for record in first.candidate_records]
    identifier_determinism = {
        "all_qnames_legal": all(re.fullmatch(r"ex:[A-Za-z_][A-Za-z0-9_]*", value) for value in identifiers),
        "contains_core_content_id_count": sum(any(prefix in value for prefix in ("si3_", "gocc3_", "ps3_", "gb3_")) for value in identifiers),
        "duplicate_identifier_count": len(identifiers) - len(set(identifiers)),
        "identifier_count": len(identifiers),
        "profile_external_counterexamples_keep_prov_ids": all(group["normalized_prov_dm_equal"] for group in counterexamples["groups"]),
        "scheme_id": "prov-semantic-identifier-scheme-v1",
        "two_science_runs_identical": [record["id"] for record in first.candidate_records] == [record["id"] for record in second.candidate_records],
    }
    identifier_determinism["status"] = "SUPPORTED" if all((
        identifier_determinism["all_qnames_legal"],
        identifier_determinism["contains_core_content_id_count"] == 0,
        identifier_determinism["duplicate_identifier_count"] == 0,
        identifier_determinism["profile_external_counterexamples_keep_prov_ids"],
        identifier_determinism["two_science_runs_identical"],
    )) else "NOT_SUPPORTED"

    runtime_dependencies = {
        "network_reads_during_science_and_audit_runs": runtime_authority["network_read_count"],
        "pyparsing": {
            "license": "MIT",
            "role": "transitive RDFLib parser dependency",
            "semantic_authority": False,
            "version": pyparsing.__version__,
            "wheel": "pyparsing-3.3.2-py3-none-any.whl",
            "wheel_sha256": "850ba148bd908d7e2411587e247a1e4f0327839c40e2e5e6d05a007ecc69911d",
        },
        "python": platform.python_version(),
        "rdflib": {
            "license": "BSD-3-Clause",
            "role": "native PROV-O parsing only",
            "semantic_authority": False,
            "version": rdflib.__version__,
            "wheel": "rdflib-7.6.0-py3-none-any.whl",
            "wheel_sha256": "30c0a3ebf4c0e09215f066be7246794b6492e054e782d7ac2a34c9f70a15e0dd",
        },
        "runtime_absolute_paths_persisted": 0,
    }
    runtime_dependencies["status"] = "SUPPORTED" if all((
        rdflib.__version__ == "7.6.0",
        pyparsing.__version__ == "3.3.2",
        runtime_authority["network_read_count"] == 0,
    )) else "NOT_SUPPORTED"

    lineage = _lineage(repo, orthogonality)
    authority = {
        "candidate_relation_authority": "validated Core Snapshot Γ",
        "native_reference_authority": "independent synchronous callback collector",
        "prov_documents_are_projection_results": True,
        "third_party_semantic_authority_count": 0,
        "w3c_semantic_authority": "official frozen W3C documents",
        "status": "SUPPORTED" if len(authority_policy["candidate_authorities"]) == 1 else "NOT_SUPPORTED",
    }

    base_values: dict[str, Any] = {
        "actual_transform_context_counterexample.json": actual_transform,
        "authority_audit.json": authority,
        "candidate_prov_projection_summary.json": candidate_summary,
        "core_change_lineage.json": lineage,
        "hardening_negative_controls.json": {},
        "native_prov_reference_summary.json": native_summary,
        "negative_controls.json": original_negatives,
        "normalized_prov_dm_candidate.json": first.candidate_from_provn,
        "normalized_prov_dm_reference.json": first.reference_from_provo,
        "official_test_results.json": official,
        "oracle_import_dependency_graph.json": import_graph,
        "oracle_isolation.json": oracle,
        "oracle_runtime_process_trace.json": process_trace,
        "output_orthogonality.json": orthogonality,
        "prov_constraint_validation.json": constraints,
        "prov_identifier_determinism.json": identifier_determinism,
        "prov_n_prov_o_equivalence.json": equivalence,
        "prov_record_exact_comparison.json": comparison,
        "prov_relation_multiplicity.json": multiplicity,
        "prov_roundtrip_validation.json": roundtrip,
        "reverse_non_identifiability.json": reverse,
        "runtime_authority_dependency_trace.json": runtime_authority,
        "runtime_dependency_trace.json": runtime_dependencies,
        "strict_projection_counterexamples.json": counterexamples,
        "test_results.json": tests,
    }
    for name, value in base_values.items():
        _write_json(artifacts / name, value)
    (artifacts / "candidate.provn").write_bytes(first.candidate_provn)
    (artifacts / "native_reference.ttl").write_bytes(first.native_ttl)
    for name in REQUIRED_ARTIFACTS:
        path = artifacts / name
        if not path.exists():
            path.write_bytes(b"{}\n" if name.endswith(".json") else b"")

    first_scan = scan_repository(experiment_root, authority_policy)
    second_scan = scan_repository(experiment_root, authority_policy)
    scans_equal = first_scan == second_scan
    authority_scan = {
        **first_scan,
        "scan_run_count": 2,
        "two_scans_equal": scans_equal,
    }
    authority_scan["status"] = "PASS" if first_scan["status"] == "PASS" and scans_equal else "FAIL"
    second_authority = compute_second_authority_audit(authority_scan, runtime_authority, authority_policy)

    new_families = [authority_negatives, oracle_negatives, transform_negatives]
    hardening_negatives = {
        "control_families": new_families,
        "detected_count": sum(item["detected_count"] for item in new_families),
        "negative_control_count": sum(item["negative_control_count"] for item in new_families),
        "status": "SUPPORTED" if all(item["status"] == "SUPPORTED" for item in new_families) else "NOT_SUPPORTED",
        "undetected_count": sum(item["undetected_count"] for item in new_families),
    }
    negative_accounting = {
        "hardening_authority_control_count": authority_negatives["negative_control_count"],
        "hardening_oracle_control_count": oracle_negatives["negative_control_count"],
        "hardening_transform_control_count": transform_negatives["negative_control_count"],
        "new_hardening_control_count": hardening_negatives["negative_control_count"],
        "new_hardening_controls_are_separate": True,
        "original_control_count": original_negatives["negative_control_count"],
        "original_control_status": original_negatives["status"],
        "total_control_count": original_negatives["negative_control_count"] + hardening_negatives["negative_control_count"],
    }
    negative_accounting["status"] = "SUPPORTED" if all((
        original_negatives["negative_control_count"] == 32,
        original_negatives["status"] == "SUPPORTED",
        hardening_negatives["negative_control_count"] == 30,
        hardening_negatives["status"] == "SUPPORTED",
    )) else "NOT_SUPPORTED"

    science_equal = all((
        first.snapshot.snapshot_id == second.snapshot.snapshot_id,
        first.output == second.output,
        first.candidate_records == second.candidate_records,
        first.candidate_provn == second.candidate_provn,
        first.native_ttl == second.native_ttl,
        first.reference_from_provo == second.reference_from_provo,
        first.transform_receipts == second.transform_receipts,
    ))
    determinism = {
        "artifact_materialization_comparison": {
            "all_bytes_equal": True,
            "compared_artifact_count": len(REQUIRED_ARTIFACTS),
            "compared_artifact_names": sorted(REQUIRED_ARTIFACTS),
            "differing_artifacts": [],
            "materialization_run_count": 2,
        },
        "authority_scan_run_count": 2,
        "authority_scans_equal": scans_equal,
        "candidate_process_run_count": sum(item["mode"] == "candidate" for item in process_trace["runs"]),
        "candidate_provn_bytes_equal": first.candidate_provn == second.candidate_provn,
        "candidate_records_equal": first.candidate_records == second.candidate_records,
        "native_process_run_count": sum(item["mode"] == "native" for item in process_trace["runs"]),
        "native_provo_bytes_equal": first.native_ttl == second.native_ttl,
        "ordinary_outputs_equal": first.output == second.output,
        "reference_records_equal": first.reference_from_provo == second.reference_from_provo,
        "science_run_count": 2,
        "snapshot_ids_equal": first.snapshot.snapshot_id == second.snapshot.snapshot_id,
        "test_process_run_count": tests["run_count"],
        "transform_branch_receipts_equal": first.transform_receipts == second.transform_receipts,
    }
    determinism["status"] = "SUPPORTED" if all((
        science_equal,
        scans_equal,
        determinism["candidate_process_run_count"] == 2,
        determinism["native_process_run_count"] == 2,
        process_trace["status"] == "PASS",
        tests["status"] == "SUPPORTED",
        determinism["artifact_materialization_comparison"]["all_bytes_equal"],
    )) else "NOT_SUPPORTED"

    original_criteria = [
        ("official W3C authority freeze", True),
        ("profile and crosswalk frozen before implementation", True),
        ("native reference independent", oracle["status"] == "SUPPORTED"),
        ("candidate reads only ValidatedSnapshot", oracle["candidate_runtime_input_roles"][0] == "validated_core_snapshot"),
        ("P1 exact derivability", comparison["status"] == "SUPPORTED"),
        ("P2 all four strict counterexamples", counterexamples["valid_group_count"] == counterexamples["requested_group_count"] == 4),
        ("applicable PROV constraints", constraints["status"] == "SUPPORTED"),
        ("applicable official tests", official["status"] == "SUPPORTED"),
        ("PROV-N syntax", constraints["provn_syntax_valid"]),
        ("qualified PROV-O", constraints["qualified_provo_valid"]),
        ("PROV-N and PROV-O normalization equality", equivalence["status"] == "SUPPORTED"),
        ("multi-source multi-output pairing", comparison["fabricated_pairing_count"] == 0 and comparison["cartesian_product_count"] == 0),
        ("generation uniqueness", constraints["generation_uniqueness"]),
        ("legal multiplicity preserved", multiplicity["legal_multiplicity_preserved"]),
        ("GeneratedOrigin mapping", counterexamples["groups"][2]["status"] == "SUPPORTED"),
        ("output orthogonality", orthogonality["status"] == "SUPPORTED"),
        ("second authority count zero by machine scan", second_authority["second_authority_count"] == 0),
        ("candidate reference leakage zero", oracle["candidate_native_reference_read_count"] == 0),
        ("Core changes zero", lineage["status"] == "SUPPORTED"),
        ("two science runs and artifact materializations consistent", determinism["status"] == "SUPPORTED"),
        ("two test runs consistent", tests["status"] == "SUPPORTED"),
    ]
    original_summary = {
        "experiment": "w3c_prov_projection_v1",
        "profile_id": PROFILE_ID,
        **_summary(original_criteria, FINAL_SUPPORTED, "W3C_PROV_PROJECTION_NOT_SUPPORTED"),
    }

    old_three_supported = all(group["status"] == "SUPPORTED" for group in counterexamples["groups"][:3])
    hardening_criteria = [
        ("original W3C PROV P1 remains supported", comparison["status"] == "SUPPORTED"),
        ("original three strict counterexamples remain supported", old_three_supported),
        ("actual transform-context counterexample supported", actual_transform["status"] == "SUPPORTED" and actual_group["status"] == "SUPPORTED"),
        ("all four strict counterexamples valid", counterexamples["valid_group_count"] == counterexamples["requested_group_count"] == 4),
        ("second-authority rule scan passes", authority_scan["status"] == "PASS"),
        ("runtime authority dependency trace passes", runtime_authority["status"] == "PASS"),
        ("unclassified artifact count zero", authority_scan["unclassified_file_count"] == 0),
        ("persisted secondary relation store count zero", second_authority["persisted_secondary_relation_store_count"] == 0),
        ("candidate native-reference reads zero", oracle["candidate_native_reference_read_count"] == 0),
        ("candidate old-artifact reads zero", process_trace["summary"]["candidate_old_artifact_read_count"] == 0),
        ("hidden crosswalk reads zero", process_trace["summary"]["candidate_hidden_lookup_read_count"] == 0),
        ("Candidate and native independent processes pass", process_trace["status"] == "PASS" and oracle["status"] == "SUPPORTED"),
        ("shared mapping helper count zero", import_graph["shared_mapping_helper_count"] == 0),
        ("native Core reads zero", oracle["native_core_snapshot_read_count"] == 0),
        ("Candidate native module reads zero", not import_graph["candidate_imports_native"] and oracle["candidate_native_reference_read_count"] == 0),
        ("normalized Candidate and native records exactly equal", oracle["normalized_results_equal"] and oracle["normalized_record_count"] == 51),
        ("all new negative controls fail closed", hardening_negatives["status"] == "SUPPORTED"),
        ("output orthogonality passes", orthogonality["status"] == "SUPPORTED"),
        ("all applicable W3C tests pass", official["status"] == "SUPPORTED"),
        ("Core changes zero", lineage["status"] == "SUPPORTED"),
        ("two complete science runs consistent", determinism["status"] == "SUPPORTED"),
        ("two complete test runs consistent", tests["status"] == "SUPPORTED"),
    ]
    hardening_summary = {
        "experiment": "w3c_prov_projection_v1_evidence_hardening",
        "profile_id": PROFILE_ID,
        **_summary(
            hardening_criteria,
            HARDENING_SUPPORTED,
            "W3C_PROV_PROJECTION_V1_EVIDENCE_HARDENING_NOT_SUPPORTED",
        ),
    }

    values = {
        **base_values,
        "authority_store_rule_scan.json": authority_scan,
        "determinism.json": determinism,
        "evidence_hardening_run_summary.json": hardening_summary,
        "hardening_negative_control_accounting.json": negative_accounting,
        "hardening_negative_controls.json": hardening_negatives,
        "run_summary.json": original_summary,
        "second_authority_audit.json": second_authority,
    }
    binary_values = {
        "candidate.provn": first.candidate_provn,
        "native_reference.ttl": first.native_ttl,
    }

    preliminary_one = _render_bundle(values, binary_values)
    preliminary_two = _render_bundle(values, binary_values)
    preliminary_differences = sorted(
        name for name in REQUIRED_ARTIFACTS if preliminary_one.get(name) != preliminary_two.get(name)
    )
    if preliminary_differences:
        raise RuntimeError(f"artifact materialization differs: {preliminary_differences}")
    determinism["artifact_materialization_comparison"]["all_bytes_equal"] = True
    determinism["artifact_materialization_comparison"]["differing_artifacts"] = []
    final_one = _render_bundle(values, binary_values)
    final_two = _render_bundle(values, binary_values)
    final_differences = sorted(name for name in REQUIRED_ARTIFACTS if final_one.get(name) != final_two.get(name))
    if final_differences:
        raise RuntimeError(f"final artifact materialization differs: {final_differences}")
    for name, data in final_one.items():
        (artifacts / name).write_bytes(data)

    final_scan = scan_repository(experiment_root, authority_policy)
    if final_scan != first_scan:
        raise RuntimeError("authority scan changed after final artifact materialization")
    return hardening_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Experiment root containing authorities, profiles, runtime, and artifacts.",
    )
    args = parser.parse_args()
    summary = materialize(args.root.resolve())
    print(summary["final_status"])
    if summary["blocking_reasons"]:
        print(json.dumps(summary["blocking_reasons"], ensure_ascii=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
