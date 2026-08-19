from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from .artifact_io import (
    build_artifact_manifest,
    canonical_json_bytes,
    verify_artifact_manifest,
    write_json,
)
from .hardening_science import EXPERIMENT_ROOT, run_complete_hardening_science


ARTIFACT_ROOT = EXPERIMENT_ROOT / "artifacts"
FROZEN_HEAD = "19eac2a1c5435b378a19c6b37d17a2d275cf794c"
ALLOWED_PREFIX = "experiments/pytorch_autograd_training_lineage_v1/"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run_tests(arguments: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", *arguments, "-q"],
        cwd=EXPERIMENT_ROOT.parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    combined = completed.stdout + completed.stderr
    passed = re.findall(r"(\d+) passed", combined)
    failed = re.findall(r"(\d+) failed", combined)
    result: dict[str, Any] = {
        "command": ["python", "-m", "pytest", *arguments, "-q"],
        "exit_code": completed.returncode,
        "failed_count": int(failed[-1]) if failed else 0,
        "passed_count": int(passed[-1]) if passed else 0,
        "status": "passed" if completed.returncode == 0 else "failed",
    }
    if completed.returncode != 0:
        result["failure_output"] = combined
    return result


def _test_execution() -> dict[str, Any]:
    experiment = _run_tests([
        "experiments/pytorch_autograd_training_lineage_v1/tests"
    ])
    core = _run_tests(["tests/core"])
    return {
        "all_passed": experiment["exit_code"] == 0 and core["exit_code"] == 0,
        "core": core,
        "experiment": experiment,
    }


def _materialize_science_artifacts(artifacts: dict[str, Any]) -> dict[str, str]:
    hashes = {}
    for name, payload in sorted(artifacts.items()):
        data = canonical_json_bytes(payload)
        path = ARTIFACT_ROOT / f"{name}.json"
        path.write_bytes(data)
        hashes[path.relative_to(EXPERIMENT_ROOT).as_posix()] = _sha256(data)
    return hashes


def _artifact_hashes(artifacts: dict[str, Any]) -> dict[str, str]:
    return {
        f"artifacts/{name}.json": _sha256(canonical_json_bytes(payload))
        for name, payload in sorted(artifacts.items())
    }


def _scope_audit() -> dict[str, Any]:
    repository = EXPERIMENT_ROOT.parents[1]
    changed_tracked = subprocess.run(
        ["git", "diff", "--name-only", FROZEN_HEAD, "--"],
        cwd=repository,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    changed_untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=repository,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    changed = sorted(set(changed_tracked) | set(changed_untracked))
    protected_paths = {
        "compat_v2": "compat/v2",
        "core_runtime": "src/generation_relation_core",
        "protocol_core_v3": "protocol/core_v3",
        "tests_core": "tests/core",
    }

    def tree(revision: str, path: str) -> str:
        return subprocess.run(
            ["git", "rev-parse", f"{revision}:{path}"],
            cwd=repository,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()

    trees = {
        name: {
            "current": tree("HEAD", path),
            "frozen": tree(FROZEN_HEAD, path),
        }
        for name, path in protected_paths.items()
    }
    outside_allowed = [path for path in changed if not path.startswith(ALLOWED_PREFIX)]
    return {
        "allowed_prefix": ALLOWED_PREFIX,
        "changed_path_count": len(changed),
        "outside_allowed_paths": outside_allowed,
        "protected_trees": trees,
        "status": (
            "HARDENING_SCOPE_PROTECTED"
            if not outside_allowed
            and all(row["current"] == row["frozen"] for row in trees.values())
            else "HARDENING_SCOPE_VIOLATION"
        ),
    }


def _run_summary(
    hardening: dict[str, Any],
    artifacts: dict[str, Any],
    determinism: dict[str, Any],
    tests: dict[str, Any],
    scope: dict[str, Any],
    *,
    manifest_verified: bool,
) -> dict[str, Any]:
    component = hardening["component_status"]
    comparison = artifacts["gradient_dependency_native_oracle_exact_comparison"]
    v2 = artifacts["query_exact_comparison_v2"]
    isolation = artifacts["gradient_oracle_process_isolation"]
    second_authority = artifacts["gradient_oracle_second_authority_audit"]
    new_controls = artifacts["gradient_oracle_negative_control_accounting"]
    preservation = artifacts["v1_scientific_result_preservation"]
    old_summary = json.loads(
        (ARTIFACT_ROOT / "run_summary.json").read_text(encoding="utf-8")
    )
    mismatches_zero = all(value == 0 for value in hardening["mismatch_counts"].values())
    gates = {
        "01_pr17_original_22_gates_preserved": (
            old_summary["passed_gate_count"] == old_summary["total_gate_count"] == 22
            and preservation["v1_artifact_mismatch_count"] == 0
        ),
        "02_native_backward_node_observation": component["native_backward_observation"]
        == "NATIVE_BACKWARD_NODE_EXECUTION_OBSERVATION_SUPPORTED",
        "03_saved_tensor_baseline_orthogonality": artifacts[
            "saved_tensor_node_assignment"
        ]["status"] == "NATIVE_SAVED_TENSOR_CONSUMPTION_OBSERVATION_SUPPORTED",
        "04_saved_tensor_intervention_oracle": component["saved_tensor_intervention"]
        == "NATIVE_SAVED_TENSOR_GRADIENT_DEPENDENCY_ORACLE_SUPPORTED",
        "05_registered_source_replay_oracle": component["source_replay"]
        == "NATIVE_SOURCE_REPLAY_GRADIENT_DEPENDENCY_ORACLE_SUPPORTED",
        "06_unified_native_gradient_dependency_oracle": component[
            "native_gradient_oracle"
        ] == "NATIVE_GRADIENT_DEPENDENCY_ORACLE_SUPPORTED",
        "07_core_native_dependency_set_exact": component["core_native_exact"]
        == "GRADIENT_DEPENDENCY_NATIVE_ORACLE_EXACT_SUPPORTED",
        "08_all_dependency_mismatch_counts_zero": mismatches_zero,
        "09_v2_reference_has_no_local_gradient_rules": v2[
            "reference_operation_specific_gradient_rule_count"
        ] == 0,
        "10_v2_forward_and_reverse_queries_exact": v2["all_exact"],
        "11_checkpoint_scale_two_independently_validated": component[
            "checkpoint_native_validation"
        ] == "CHECKPOINT_DIVERGENCE_NATIVE_ORACLE_VALIDATED_SUPPORTED",
        "12_candidate_does_not_read_oracle": isolation["candidate_audit"][
            "candidate_oracle_read_count"
        ] == 0,
        "13_native_oracle_does_not_read_core": isolation["native_audit"][
            "native_core_read_count"
        ] == 0,
        "14_shared_gradient_rule_helper_count_zero": second_authority[
            "shared_gradient_rule_helper_count"
        ] == 0,
        "15_original_32_negative_controls_preserved": hardening[
            "original_negative_control_count"
        ] == 32 and preservation["preservation_gates"][
            "original_32_negative_controls_pass"
        ],
        "16_new_20_negative_controls_fail_closed": (
            new_controls["control_count"] == 20
            and new_controls["all_detected"]
            and new_controls["execution_count_total"] == 20
            and new_controls["fail_closed_count"] == 20
            and new_controls["unique_mutation_fingerprint_count"] == 20
            and new_controls["automatic_repair_count"] == 0
        ),
        "17_two_complete_hardening_runs_identical": (
            determinism["all_science_artifacts_exact"]
            and determinism["scientific_reports_exact"]
            and determinism["test_summaries_exact"]
            and not determinism["excluded_fields"]
        ),
        "18_core_and_other_experiments_unchanged": scope["status"]
        == "HARDENING_SCOPE_PROTECTED",
        "19_all_artifacts_independently_rehashed": manifest_verified,
        "20_all_experiment_and_core_tests_pass": (
            tests["all_passed"]
            and tests["run_1"]["experiment"]["passed_count"] >= 56
            and tests["run_2"]["experiment"]["passed_count"]
            == tests["run_1"]["experiment"]["passed_count"]
            and tests["run_1"]["core"]["passed_count"] == 24
            and tests["run_2"]["core"]["passed_count"] == 24
        ),
    }
    supported = all(gates.values())
    return {
        "claim_scope": (
            "the frozen PyTorch 2.13.0+cpu deterministic workloads and the "
            "declared gradient-value dependency profile"
        ),
        "gates": gates,
        "hardening_component_status": component,
        "passed_gate_count": sum(gates.values()),
        "status": (
            "PYTORCH_AUTOGRAD_GENERATION_FACTS_V1_EVIDENCE_HARDENING_SUPPORTED"
            if supported
            else "PYTORCH_AUTOGRAD_GENERATION_FACTS_V1_SUPPORTED_GRADIENT_DEPENDENCY_INDEPENDENCE_NOT_ESTABLISHED"
        ),
        "total_gate_count": len(gates),
    }


def _report_section(
    summary: dict[str, Any],
    artifacts: dict[str, Any],
    determinism: dict[str, Any],
    tests: dict[str, Any],
) -> str:
    comparison = artifacts["gradient_dependency_native_oracle_exact_comparison"]
    saved = artifacts["saved_tensor_gradient_interventions"]
    source = artifacts["source_replay_gradient_interventions"]
    v2 = artifacts["query_exact_comparison_v2"]
    controls = artifacts["gradient_oracle_negative_control_accounting"]
    return f"""## Gradient-dependency oracle hardening

Final hardening status: `{summary['status']}` ({summary['passed_gate_count']}/{summary['total_gate_count']} gates).

The v1 Core capture and receipt reference separately encoded semantically overlapping local reverse rules. That does not overturn the exact Autograd projection, checkpoint graph equality, finite gradient divergence, parameter-update divergence, or v1 lineage results; it limited how independently the gradient-value dependency paths had been validated.

The v2 evidence removes that shared semantic assumption from the reference path. A native-only PyTorch runner uses public `Node.name()`, ordered `Node.next_functions`, node pre/post hooks, leaf tensor hooks, and `saved_tensors_hooks` to observe real backward execution and saved-tensor retrieval. It then performs {saved['attempt_count']} predetermined single-token interventions and {source['attempt_count']} registered-source replay interventions. No Core snapshot, Candidate relation, receipt dependency answer, operation-specific derivative table, or persistent Python object identity is available to that oracle process.

The independently observed native relation has {comparison['native_relation_count']} relations; Core has {comparison['core_relation_count']}. False positives, false negatives, source-identity mismatches, target-gradient mismatches, topology mismatches, duplicate-identity collapses, multiplicity mismatches, missing witnesses, unsupported Core dependencies, and unrepresented native dependencies are all zero. Every declared Core `gradient_value_dependency` therefore has an actual saved-tensor intervention witness, a registered-source replay witness, or both.

For the checkpoint flagship case, the registered recomputation source `source:external:scale:recomputation` with value 2 is independently replayed and changes `step_0:gradient:parameter:p`; real backward node execution and recomputation saved-tensor activity are present. The unchanged stable/divergent native graph is not used as the dependency answer. The v2 forward and reverse comparisons remain exact ({v2['forward_exact_count']}/{v2['forward_query_count']} and {v2['reverse_exact_count']}/{v2['reverse_query_count']}).

Candidate/Core and native oracle run in separate processes and do not read each other's evidence. The v2 receipt reference supplies only forward, backward-boundary, and optimizer edges, accepting gradient-value edges from the native oracle; its operation-specific gradient-rule count is zero. The original 32 negative controls remain preserved, and {controls['control_count']} new controls execute once each, fail closed, have unique fingerprints, and perform no automatic repair.

Two complete hardening runs produced byte-identical scientific artifacts and reports with no excluded scientific fields. Each test execution passed {tests['run_1']['experiment']['passed_count']} experiment tests and {tests['run_1']['core']['passed_count']} unchanged Core tests.

This profile does not evaluate semantic importance, causal contribution magnitude, sample importance, arbitrary PyTorch operators, CUDA or other accelerators, mixed precision, distributed or compiled execution, higher-order/forward-mode AD, in-place and complex alias/view behavior, custom C++ operators, non-SGD optimizers, or reentrant checkpoint. Interventions establish value dependency only for the frozen deterministic workloads and perturbations.

“The original experiment derived gradient-value dependencies in both the Core capture and receipt reference through separately implemented but semantically shared local reverse rules. The hardened experiment removes that shared semantic assumption from the reference path. A native PyTorch oracle observes actual backward-node execution and saved-tensor retrieval, then independently intervenes on saved tensors and registered sources. Within the frozen profile, the resulting native dependency relation is exactly equal to the gradient-value dependency relation delivered by complete generation facts.”

“原实验的Core捕获与receipt参考路径分别实现了语义相同的局部反向规则，因此仍可能共享同类错误。加固实验从参考路径中移除了这一共同假设：原生PyTorch oracle直接观测实际backward节点执行与保存张量取回，并分别对保存张量和注册来源实施独立干预。在冻结profile范围内，由此得到的原生梯度依赖关系与完整生成事实交付的gradient-value dependency关系精确一致。”

**梯度来源关系不再由两套相同规则相互证明，而由真实PyTorch backward发生独立证明。**
"""


def _update_report(section: str) -> None:
    path = EXPERIMENT_ROOT / "EXPERIMENT_REPORT.md"
    original = path.read_text(encoding="utf-8")
    marker = "## Gradient-dependency oracle hardening"
    if marker in original:
        original = original.split(marker, 1)[0].rstrip()
    path.write_text(original + "\n\n" + section, encoding="utf-8", newline="\n")


def materialize() -> dict[str, Any]:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    run_1 = run_complete_hardening_science()
    hashes_1 = _materialize_science_artifacts(run_1["artifacts"])
    artifacts_1 = run_1["artifacts"]
    summary_1 = run_1["hardening_summary"]

    run_2 = run_complete_hardening_science()
    hashes_2 = _artifact_hashes(run_2["artifacts"])
    summary_2 = run_2["hardening_summary"]
    artifact_comparison = {
        path: hashes_1[path] == hashes_2.get(path) for path in sorted(hashes_1)
    }
    del run_2
    gc.collect()

    test_run_1 = _test_execution()
    test_run_2 = _test_execution()
    tests = {
        "all_passed": test_run_1["all_passed"] and test_run_2["all_passed"],
        "run_1": test_run_1,
        "run_2": test_run_2,
    }
    test_summary_1 = {
        "core_passed": test_run_1["core"]["passed_count"],
        "experiment_passed": test_run_1["experiment"]["passed_count"],
        "status": "passed" if test_run_1["all_passed"] else "failed",
    }
    test_summary_2 = {
        "core_passed": test_run_2["core"]["passed_count"],
        "experiment_passed": test_run_2["experiment"]["passed_count"],
        "status": "passed" if test_run_2["all_passed"] else "failed",
    }
    report_1 = {
        "artifact_sha256s": hashes_1,
        "hardening_summary": summary_1,
        "test_summary": test_summary_1,
    }
    report_2 = {
        "artifact_sha256s": hashes_2,
        "hardening_summary": summary_2,
        "test_summary": test_summary_2,
    }
    report_bytes_1 = canonical_json_bytes(report_1)
    report_bytes_2 = canonical_json_bytes(report_2)
    determinism = {
        "all_science_artifacts_exact": all(artifact_comparison.values()),
        "artifact_comparison": artifact_comparison,
        "excluded_fields": [],
        "run_1_scientific_report_sha256": _sha256(report_bytes_1),
        "run_2_scientific_report_sha256": _sha256(report_bytes_2),
        "scientific_reports_exact": report_bytes_1 == report_bytes_2,
        "test_summaries_exact": test_summary_1 == test_summary_2,
    }

    run_1_root = ARTIFACT_ROOT / "hardening_runs" / "run_1"
    run_2_root = ARTIFACT_ROOT / "hardening_runs" / "run_2"
    run_1_root.mkdir(parents=True, exist_ok=True)
    run_2_root.mkdir(parents=True, exist_ok=True)
    (run_1_root / "scientific_reports.json").write_bytes(report_bytes_1)
    (run_2_root / "scientific_reports.json").write_bytes(report_bytes_2)
    write_json(run_1_root / "test_results.json", test_run_1)
    write_json(run_2_root / "test_results.json", test_run_2)
    write_json(ARTIFACT_ROOT / "gradient_dependency_oracle_determinism.json", determinism)
    write_json(ARTIFACT_ROOT / "gradient_dependency_oracle_test_results.json", tests)

    scope = _scope_audit()
    write_json(ARTIFACT_ROOT / "gradient_dependency_oracle_scope_audit.json", scope)
    preliminary = _run_summary(
        summary_1,
        artifacts_1,
        determinism,
        tests,
        scope,
        manifest_verified=False,
    )
    write_json(ARTIFACT_ROOT / "gradient_dependency_oracle_run_summary.json", preliminary)
    provisional_manifest = build_artifact_manifest(EXPERIMENT_ROOT)
    provisional_verification = verify_artifact_manifest(
        EXPERIMENT_ROOT, provisional_manifest
    )
    final_summary = _run_summary(
        summary_1,
        artifacts_1,
        determinism,
        tests,
        scope,
        manifest_verified=provisional_verification["verified"],
    )
    write_json(ARTIFACT_ROOT / "gradient_dependency_oracle_run_summary.json", final_summary)
    _update_report(_report_section(final_summary, artifacts_1, determinism, tests))

    final_manifest = build_artifact_manifest(EXPERIMENT_ROOT)
    final_manifest["independent_verification"] = verify_artifact_manifest(
        EXPERIMENT_ROOT, final_manifest
    )
    write_json(ARTIFACT_ROOT / "artifact_manifest.json", final_manifest)
    final_verification = verify_artifact_manifest(EXPERIMENT_ROOT, final_manifest)
    if not all([
        final_summary["status"]
        == "PYTORCH_AUTOGRAD_GENERATION_FACTS_V1_EVIDENCE_HARDENING_SUPPORTED",
        final_summary["passed_gate_count"] == final_summary["total_gate_count"] == 20,
        final_verification["verified"],
        determinism["all_science_artifacts_exact"],
        determinism["scientific_reports_exact"],
        tests["all_passed"],
    ]):
        raise RuntimeError("GRADIENT_ORACLE_FINAL_EVIDENCE_GATES_FAILED")
    return {
        "artifact_manifest": final_verification,
        "determinism": determinism,
        "run_summary": final_summary,
        "tests": tests,
    }


if __name__ == "__main__":
    print(json.dumps(materialize(), ensure_ascii=False, sort_keys=True))
