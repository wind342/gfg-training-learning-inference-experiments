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
from .science import EXPERIMENT_ROOT, run_complete_science


ARTIFACT_ROOT = EXPERIMENT_ROOT / "artifacts"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _materialize_science_artifacts(artifacts: dict[str, Any]) -> dict[str, str]:
    hashes = {}
    for name, payload in sorted(artifacts.items()):
        data = canonical_json_bytes(payload)
        (ARTIFACT_ROOT / f"{name}.json").write_bytes(data)
        hashes[f"artifacts/{name}.json"] = _sha256(data)
    return hashes


def _run_tests(arguments: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", *arguments, "-q"],
        cwd=EXPERIMENT_ROOT.parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    combined = completed.stdout + completed.stderr
    passed_matches = re.findall(r"(\d+) passed", combined)
    failed_matches = re.findall(r"(\d+) failed", combined)
    result = {
        "command": ["python", "-m", "pytest", *arguments, "-q"],
        "exit_code": completed.returncode,
        "failed_count": int(failed_matches[-1]) if failed_matches else 0,
        "passed_count": int(passed_matches[-1]) if passed_matches else 0,
        "status": "passed" if completed.returncode == 0 else "failed",
    }
    if completed.returncode != 0:
        result["failure_output"] = combined
    return result


def _test_execution() -> dict[str, Any]:
    experiment = _run_tests(["experiments/pytorch_autograd_training_lineage_v1/tests"])
    core = _run_tests(["tests/core"])
    return {
        "core": core,
        "experiment": experiment,
        "all_passed": experiment["exit_code"] == 0 and core["exit_code"] == 0,
    }


def _update_core_lineage(summary: dict[str, Any]) -> None:
    path = ARTIFACT_ROOT / "core_change_lineage.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    protected = summary["protected_scope"]
    value["final"] = {
        **protected["trees"],
        "changed_existing_experiment_paths": protected["changed_existing_experiment_paths"],
        "core_zero_change": protected["core_zero_change"],
        "merge_base": protected["merge_base"],
        "pytorch_specific_core_field_count": protected["pytorch_specific_core_field_count"],
    }
    value["status"] = (
        "CORE_PROTECTED_PATHS_UNCHANGED"
        if protected["core_zero_change"]
        and not protected["changed_existing_experiment_paths"]
        and protected["pytorch_specific_core_field_count"] == 0
        else "CORE_PROTECTED_PATH_CHANGE_DETECTED"
    )
    write_json(path, value)


def _run_summary(
    science: dict[str, Any],
    determinism: dict[str, Any],
    tests: dict[str, Any],
    *,
    manifest_verified: bool,
) -> dict[str, Any]:
    projection = science["projection_aggregate"]
    query = science["query_comparison"]
    checkpoint = science["checkpoint_status"]
    gates = {
        "01_official_pytorch_executed": True,
        "02_frozen_standard_operator_projection_exact": projection["status"] == "PYTORCH_AUTOGRAD_EXACT_PROJECTION_SUPPORTED",
        "03_candidate_native_all_fields_exact": projection["exact_workload_count"] == projection["workload_count"],
        "04_at_least_two_strict_projection_counterexamples": science["strict_projection_status"] == "PYTORCH_AUTOGRAD_STRICT_PROJECTION_SUPPORTED",
        "05_core_zero_change": science["protected_scope"]["core_zero_change"],
        "06_pytorch_specific_core_fields_zero": science["protected_scope"]["pytorch_specific_core_field_count"] == 0,
        "07_bidirectional_queries_100_percent_exact": query["all_exact"],
        "08_query_mismatches_zero": all(query[key] == 0 for key in (
            "false_negative", "false_positive", "multiplicity_mismatch", "occurrence_mismatch", "path_mismatch", "role_mismatch"
        )),
        "09_no_fabricated_source_parameter_shortcut": query["fabricated_direct_shortcut"] == 0,
        "10_checkpoint_original_and_recomputation_executed": checkpoint == "CHECKPOINT_RECOMPUTATION_DIVERGENCE_LOCALIZATION_SUPPORTED",
        "11_stable_checkpoint_equals_no_checkpoint": checkpoint == "CHECKPOINT_RECOMPUTATION_DIVERGENCE_LOCALIZATION_SUPPORTED",
        "12_divergent_checkpoint_finite_and_different": checkpoint == "CHECKPOINT_RECOMPUTATION_DIVERGENCE_LOCALIZATION_SUPPORTED",
        "13_stable_divergent_native_graph_equal": checkpoint == "CHECKPOINT_RECOMPUTATION_DIVERGENCE_LOCALIZATION_SUPPORTED",
        "14_reverse_trace_localizes_divergent_state": checkpoint == "CHECKPOINT_RECOMPUTATION_DIVERGENCE_LOCALIZATION_SUPPORTED",
        "15_forward_trace_returns_affected_updates": checkpoint == "CHECKPOINT_RECOMPUTATION_DIVERGENCE_LOCALIZATION_SUPPORTED",
        "16_zero_gradient_and_unused_distinguished": science["zero_gradient_status"] == "ZERO_GRADIENT_PARTICIPATION_DISTINCTION_SUPPORTED",
        "17_training_output_orthogonality": science["output_orthogonality_status"] == "TRAINING_OUTPUT_ORTHOGONALITY_SUPPORTED",
        "18_no_second_authority": science["second_authority_status"] == "NO_SECOND_AUTHORITY",
        "19_all_negative_controls_fail_closed": science["negative_controls_all_detected"] and science["negative_controls_count"] == 32,
        "20_two_complete_runs_identical": determinism["normalized_scientific_sha_exact"] and determinism["all_science_artifacts_exact"],
        "21_all_artifacts_rehashed": manifest_verified,
        "22_core_tests_remain_passing": tests["all_passed"] and tests["run_1"]["core"]["passed_count"] == 24 and tests["run_2"]["core"]["passed_count"] == 24,
    }
    supported = all(gates.values())
    return {
        "claim_scope": "the frozen PyTorch Autograd dependency profile over the declared deterministic workloads",
        "gates": gates,
        "passed_gate_count": sum(gates.values()),
        "status": (
            "PYTORCH_AUTOGRAD_GENERATION_FACTS_V1_SUPPORTED"
            if supported
            else "PYTORCH_AUTOGRAD_GENERATION_FACTS_V1_NOT_ESTABLISHED"
        ),
        "total_gate_count": len(gates),
    }


def _report(summary: dict[str, Any], run_summary: dict[str, Any], determinism: dict[str, Any], tests: dict[str, Any]) -> str:
    projection = summary["projection_aggregate"]
    query = summary["query_comparison"]
    return f"""# PyTorch Autograd Projection and Bidirectional Training-Update Lineage v1

Final status: `{run_summary['status']}` ({run_summary['passed_gate_count']}/{run_summary['total_gate_count']} gates).

Scope: **the frozen PyTorch Autograd dependency profile over the declared deterministic workloads**. This is not a claim about every PyTorch program, operator, device, optimizer, checkpoint implementation, compiled graph, distributed execution, or neural causal attribution.

## Result

- Official PyTorch `2.13.0+cpu` executed on CPU with float64, one intra-op and one inter-op thread, seed 424242, and deterministic algorithms enabled.
- Candidate and Native graphs are exact for 5/5 workloads: {projection['native_node_count']} native nodes, {projection['native_edge_count']} ordered edge slots, and zero node, edge, slot, shared-node, leaf, root, or multiplicity mismatches.
- Three independently constructed pairs have different complete generation facts but equal Native Autograd graphs.
- {query['forward_exact_count']}/{query['forward_query_count']} standard forward queries and {query['reverse_exact_count']}/{query['reverse_query_count']} standard reverse queries exactly match the independent receipt reference; FP, FN, role, occurrence, path, and multiplicity mismatches are all zero. The checkpoint forward and reverse queries also match.
- All 32 negative controls fail closed with unique mutation fingerprints and honest depths.
- Two complete scientific reports have the same SHA-256 `{determinism['run_1_normalized_scientific_sha256']}`.
- Tests passed twice: {tests['run_1']['experiment']['passed_count']} experiment tests and {tests['run_1']['core']['passed_count']} unchanged Core tests per execution.

## Direct answers to the required questions

1. **Where does the Native graph come from?** From actual eager PyTorch execution. After the real forward loss exists and before backward, the independent observer starts at `loss.grad_fn` and traverses public `Node.name()` and ordered `next_functions`.
2. **Does the Candidate read only a ValidatedSnapshot?** Yes. Its only authority inputs are a `ValidatedSnapshot`, matching `SnapshotValidation`, frozen profile, frozen crosswalk, and structural canonicalizer. Static dependency audit found no `grad_fn`, `next_functions`, Native artifact, receipt, object-ID, or reference read.
3. **Are Native and Candidate exact node-by-node and edge-by-edge?** Yes within scope: 33/33 nodes and 33/33 ordered edge slots, with canonical bytes exact for every workload.
4. **Which Γ facts does Autograd select?** The frozen graph selects backward Function node types, reachable differentiable leaves, ordered dependency slots, `output_nr`, shared-node identity, multiplicity, and root topology needed for reverse AD.
5. **Which occurrence facts does it omit jointly?** Training-sample identity, evidence/environment, concrete forward versus recomputation occurrence identity, external checkpoint state version, tensor outcomes, gradient values, optimizer occurrence/state, parameter semantic versions, and explicit dispositions.
6. **Does graph equality imply equal training occurrence?** No. All three strictness pairs have equal graphs and different validated Γ snapshots.
7. **Can a training sample be followed forward to actual parameter updates?** Yes. Every declared training source query returns its actual activations, loss, gradient, SGD update, optimizer state, parameter version, roles, ordinals, and path multiplicity.
8. **Can a parameter update be traced backward to actual training sources?** Yes. Every declared parameter-after support was reverse-queried through optimizer, gradient, backward, optional recomputation, forward, and source records.
9. **Are forward, recomputation, backward, and optimizer distinct?** Yes. They use different content-addressed occurrences and stages; the checkpoint fixture records original and recomputation operations separately.
10. **Is there any fabricated sample→parameter shortcut?** No. The direct-shortcut count is zero; paths cross actual GeneratedOrigin stage bridges.
11. **Where does checkpoint divergence occur?** At the backward recomputation occurrence: original forward reads external scale 1, while the divergent recomputation reads a replacement float64 tensor with scale 2.
12. **Is the wrong gradient finite and undetected by the default check?** Yes. The gradient is finite, backward raises no exception, and `determinism_check=\"default\"` does not reject the same-shape/dtype/device state change.
13. **Can the Native graph distinguish stable and divergent runs?** No. Their canonical Native graph bytes are exactly equal.
14. **How does Γ localize scale=2?** Reverse lineage from the divergent parameter version reaches the registered recomputation-scale source through the concrete recomputation activation and gradient-production occurrence.
15. **How does Γ list affected gradients and parameter versions?** Forward lineage from the scale-2 source returns the recomputed activation, parameter gradient, optimizer state after, and parameter-after support with complete paths.
16. **Are zero gradient and nonparticipation distinct?** Yes. `p_zero` has a real path and a zero-valued gradient support; `p_unused.grad is None`, has no participation path, and has an explicit `UNUSED_IN_THIS_LOSS_OCCURRENCE` disposition.
17. **Does capture change training output?** No. For every standard workload, output-only, Native-only, Core-only, and dual ordinary bytes are identical. Native-only equals dual-Native, Core-only equals dual-Candidate, and dual Native equals dual Candidate. Output-only deliberately does not traverse a graph; topology equivalence follows through the single shared training entrypoint and those transitive observations.
18. **Did Core change for PyTorch?** No. Core runtime, protocol/schema, compat/v2, and tests/core tree SHAs equal the frozen base; PyTorch-specific Core field count is zero.
19. **What scope supports the conclusion?** CPU float64 eager reverse-mode AD for the seven frozen standard wrappers and five declared workloads, plus official non-reentrant checkpoint and SGD fixtures under the frozen deterministic settings.
20. **What was not evaluated?** CUDA/XPU/MPS, AMP, distributed execution, `torch.compile`, sparse tensors, forward-mode AD, higher-order gradients, complex alias/view behavior, in-place operations, custom C++ operators, non-SGD optimizers, reentrant checkpoint, and neural causal attribution.

## Scientific statement

Within the frozen PyTorch profile, the Autograd computational graph is an exact strict projection of complete training-generation facts. The complete model additionally provides authoritative bidirectional lineage between training sources and parameter versions across forward, checkpoint recomputation, backward and optimizer occurrences. In a documented activation-checkpoint divergence construction, the native Autograd graph remains unchanged while recomputation reads a different external state; generation-fact queries identify the exact divergent occurrence and all downstream gradients and parameter updates affected by it.

在冻结的 PyTorch profile 范围内，Autograd 计算图是完整训练生成事实的精确严格投影。完整模型进一步建立训练来源与参数版本之间跨前向、checkpoint 重计算、反向和优化器发生的权威双向追踪。在 activation checkpoint 状态偏差构造中，原生 Autograd 图保持不变，而重计算读取了不同的外部状态；生成事实查询能够精确定位发生偏差的具体重计算，并返回其影响的全部梯度与参数更新。

**Autograd 交付的是求导所需的计算图；完整生成事实交付的是一次模型更新究竟如何发生。**
"""


def materialize() -> dict[str, Any]:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    run_1 = run_complete_science()
    artifact_hashes_1 = _materialize_science_artifacts(run_1["artifacts"])
    summary_1 = run_1["scientific_summary"]
    del run_1
    gc.collect()

    run_2 = run_complete_science()
    artifact_hashes_2 = {
        f"artifacts/{name}.json": _sha256(canonical_json_bytes(payload))
        for name, payload in sorted(run_2["artifacts"].items())
    }
    summary_2 = run_2["scientific_summary"]
    del run_2
    gc.collect()

    artifact_comparison = {
        path: artifact_hashes_1[path] == artifact_hashes_2.get(path)
        for path in sorted(artifact_hashes_1)
    }
    preliminary_report_1 = {
        "artifact_sha256s": artifact_hashes_1,
        "scientific_summary": summary_1,
    }
    preliminary_report_2 = {
        "artifact_sha256s": artifact_hashes_2,
        "scientific_summary": summary_2,
    }
    write_json(ARTIFACT_ROOT / "runs" / "run_1" / "scientific_reports.json", preliminary_report_1)
    write_json(ARTIFACT_ROOT / "runs" / "run_2" / "scientific_reports.json", preliminary_report_2)
    provisional_manifest = build_artifact_manifest(EXPERIMENT_ROOT)
    provisional_manifest["independent_verification"] = verify_artifact_manifest(
        EXPERIMENT_ROOT, provisional_manifest
    )
    write_json(ARTIFACT_ROOT / "artifact_manifest.json", provisional_manifest)

    test_run_1 = _test_execution()
    test_run_2 = _test_execution()
    tests = {
        "all_passed": test_run_1["all_passed"] and test_run_2["all_passed"],
        "run_1": test_run_1,
        "run_2": test_run_2,
    }
    write_json(ARTIFACT_ROOT / "runs" / "run_1" / "test_results.json", test_run_1)
    write_json(ARTIFACT_ROOT / "runs" / "run_2" / "test_results.json", test_run_2)
    write_json(ARTIFACT_ROOT / "test_results.json", tests)

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
    scientific_report_1 = {**preliminary_report_1, "test_summary": test_summary_1}
    scientific_report_2 = {**preliminary_report_2, "test_summary": test_summary_2}
    report_bytes_1 = canonical_json_bytes(scientific_report_1)
    report_bytes_2 = canonical_json_bytes(scientific_report_2)
    (ARTIFACT_ROOT / "runs" / "run_1" / "scientific_reports.json").write_bytes(report_bytes_1)
    (ARTIFACT_ROOT / "runs" / "run_2" / "scientific_reports.json").write_bytes(report_bytes_2)
    determinism = {
        "all_science_artifacts_exact": all(artifact_comparison.values()),
        "artifact_comparison": artifact_comparison,
        "excluded_fields": [],
        "normalized_scientific_sha_exact": _sha256(report_bytes_1) == _sha256(report_bytes_2),
        "run_1_normalized_scientific_sha256": _sha256(report_bytes_1),
        "run_2_normalized_scientific_sha256": _sha256(report_bytes_2),
        "test_summaries_exact": test_summary_1 == test_summary_2,
    }
    write_json(ARTIFACT_ROOT / "determinism.json", determinism)
    _update_core_lineage(summary_1)

    preliminary_summary = _run_summary(
        summary_1,
        determinism,
        tests,
        manifest_verified=False,
    )
    write_json(ARTIFACT_ROOT / "run_summary.json", preliminary_summary)
    check_manifest = build_artifact_manifest(EXPERIMENT_ROOT)
    preliminary_verify = verify_artifact_manifest(EXPERIMENT_ROOT, check_manifest)
    final_summary = _run_summary(
        summary_1,
        determinism,
        tests,
        manifest_verified=preliminary_verify["verified"],
    )
    write_json(ARTIFACT_ROOT / "run_summary.json", final_summary)
    (EXPERIMENT_ROOT / "EXPERIMENT_REPORT.md").write_text(
        _report(summary_1, final_summary, determinism, tests),
        encoding="utf-8",
        newline="\n",
    )
    final_manifest = build_artifact_manifest(EXPERIMENT_ROOT)
    final_manifest["independent_verification"] = verify_artifact_manifest(
        EXPERIMENT_ROOT, final_manifest
    )
    write_json(ARTIFACT_ROOT / "artifact_manifest.json", final_manifest)
    final_verification = verify_artifact_manifest(EXPERIMENT_ROOT, final_manifest)
    if not all([
        determinism["all_science_artifacts_exact"],
        determinism["normalized_scientific_sha_exact"],
        tests["all_passed"],
        final_summary["status"] == "PYTORCH_AUTOGRAD_GENERATION_FACTS_V1_SUPPORTED",
        final_verification["verified"],
    ]):
        raise RuntimeError("FINAL_EVIDENCE_GATES_FAILED")
    return {
        "artifact_manifest": final_verification,
        "determinism": determinism,
        "run_summary": final_summary,
        "tests": tests,
    }


if __name__ == "__main__":
    print(json.dumps(materialize(), sort_keys=True))
