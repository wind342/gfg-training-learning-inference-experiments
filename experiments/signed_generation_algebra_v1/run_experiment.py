"""Execute, compare, gate and report Signed Generation Algebra v1."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from generation_relation_core.canonical import canonical_json_file_bytes

from .algebra import NaturalPolynomial, SignedPair
from .collector import SignedEffectCollector
from .generator import execute_native
from .query import RegisteredSignedEffectQuery


EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
CONTRACT_ROOT = EXPERIMENT_ROOT / "contracts"
DEFAULT_REPORT_ROOT = (
    EXPERIMENT_ROOT / "reports" / "core_v3_native_v1"
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_file_bytes(value))


def _run_reference(operation_contract_path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            (
                "experiments.signed_generation_algebra_v1."
                "independent_reference"
            ),
            "--contract",
            str(operation_contract_path),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise ValueError("independent reference returned a non-object")
    return value


def _semantic_contributions(
    candidate: dict[str, Any]
) -> list[dict[str, Any]]:
    return sorted(
        [
            {
                key: row[key]
                for key in (
                    "effect_identity",
                    "multiplicity",
                    "occurrence_identity",
                    "relation_role",
                    "sign",
                )
            }
            for row in candidate["algebraic_contributions"]
        ],
        key=lambda row: row["occurrence_identity"],
    )


def _variables(polynomial: dict[str, Any]) -> list[str]:
    return sorted(
        {
            factor["variable"]
            for term in polynomial["terms"]
            for factor in term["monomial"]
        }
    )


def _coefficient_rows(
    polynomial: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        {
            "coefficient": term["coefficient"],
            "monomial": term["monomial"],
        }
        for term in polynomial["terms"]
    ]


def _multiplicity_rows(
    contributions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return sorted(
        [
            {
                "effect_identity": row["effect_identity"],
                "multiplicity": row["multiplicity"],
                "occurrence_identity": row["occurrence_identity"],
            }
            for row in contributions
        ],
        key=lambda row: row["occurrence_identity"],
    )


def _algebra_laws() -> dict[str, Any]:
    x = NaturalPolynomial.variable("x_law_x")
    y = NaturalPolynomial.variable("x_law_y", 2)
    pairs = [
        SignedPair.zero(),
        SignedPair.one(),
        SignedPair(x, NaturalPolynomial.zero()),
        SignedPair(NaturalPolynomial.zero(), x),
        SignedPair(x, y),
    ]
    addition_associative = all(
        left.plus(middle).plus(right)
        == left.plus(middle.plus(right))
        for left in pairs
        for middle in pairs
        for right in pairs
    )
    addition_commutative = all(
        left.plus(right) == right.plus(left)
        for left in pairs
        for right in pairs
    )
    multiplication_associative = all(
        left.times(middle).times(right)
        == left.times(middle.times(right))
        for left in pairs
        for middle in pairs
        for right in pairs
    )
    multiplication_commutative = all(
        left.times(right) == right.times(left)
        for left in pairs
        for right in pairs
    )
    distributive = all(
        left.times(middle.plus(right))
        == left.times(middle).plus(left.times(right))
        for left in pairs
        for middle in pairs
        for right in pairs
    )
    zero_identity = all(
        value.plus(SignedPair.zero()) == value
        and value.times(SignedPair.zero()) == SignedPair.zero()
        for value in pairs
    )
    one_identity = all(
        value.times(SignedPair.one()) == value for value in pairs
    )
    net_additive = all(
        left.plus(right).net_projection()
        == left.net_projection().plus(right.net_projection())
        for left in pairs
        for right in pairs
    )
    net_multiplicative = all(
        left.times(right).net_projection()
        == left.net_projection().times(right.net_projection())
        for left in pairs
        for right in pairs
    )
    unreduced = SignedPair(x, x)
    no_internal_cancellation = (
        unreduced != SignedPair.zero()
        and bool(unreduced.positive.coefficients)
        and bool(unreduced.negative.coefficients)
        and not unreduced.net_projection().coefficients
    )
    checks = {
        "addition_associative": addition_associative,
        "addition_commutative": addition_commutative,
        "distributive": distributive,
        "multiplication_associative": multiplication_associative,
        "multiplication_commutative": multiplication_commutative,
        "net_projection_additive": net_additive,
        "net_projection_multiplicative": net_multiplicative,
        "no_internal_cancellation": no_internal_cancellation,
        "one_identity": one_identity,
        "zero_identity": zero_identity,
    }
    return {
        "checks": checks,
        "failed_checks": sorted(
            key for key, passed in checks.items() if not passed
        ),
        "sample_pair_count": len(pairs),
        "schema_version": "signed-algebra-laws-v1",
    }


def _protected_scope_audit(
    baseline: dict[str, Any]
) -> dict[str, Any]:
    paths = baseline["protected_paths"]
    diff = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            baseline["commit"],
            "--",
            *paths,
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            *paths,
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    actual_trees: dict[str, str] = {}
    for path in paths:
        completed = subprocess.run(
            ["git", "rev-parse", f"HEAD:{path}"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        actual_trees[path] = completed.stdout.strip()
    expected_trees = baseline["protected_tree_sha1"]
    return {
        "baseline_commit": baseline["commit"],
        "diff_exit_code": diff.returncode,
        "expected_tree_sha1": expected_trees,
        "head_tree_sha1": actual_trees,
        "status_entries": [
            line for line in status.stdout.splitlines() if line
        ],
        "unchanged": (
            diff.returncode == 0
            and not status.stdout.strip()
            and actual_trees == expected_trees
        ),
    }


def _import_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level
            names.add(prefix + (node.module or ""))
    return sorted(names)


def _authority_audit() -> dict[str, Any]:
    reference_path = EXPERIMENT_ROOT / "independent_reference.py"
    query_path = EXPERIMENT_ROOT / "query.py"
    reference_imports = _import_names(reference_path)
    query_imports = _import_names(query_path)
    forbidden_reference_prefixes = (
        ".algebra",
        ".collector",
        ".generator",
        ".query",
        "generation_relation_core",
    )
    reference_forbidden = [
        name
        for name in reference_imports
        if name.startswith(forbidden_reference_prefixes)
    ]
    forbidden_query_prefixes = (
        ".collector",
        ".generator",
        ".independent_reference",
    )
    query_forbidden = [
        name
        for name in query_imports
        if name.startswith(forbidden_query_prefixes)
    ]
    query_source = query_path.read_text(encoding="utf-8")
    query_write_tokens = [
        token
        for token in (
            "write_bytes(",
            "write_text(",
            "open(",
            "operation_contract",
            "final_output",
        )
        if token in query_source
    ]
    return {
        "candidate_forbidden_imports": query_forbidden,
        "candidate_write_or_hidden_answer_tokens": query_write_tokens,
        "candidate_snapshot_only": (
            not query_forbidden and not query_write_tokens
        ),
        "independent_reference_forbidden_imports": (
            reference_forbidden
        ),
        "independent_reference_isolated": not reference_forbidden,
        "no_second_authority_store": (
            not query_forbidden and not query_write_tokens
        ),
        "query_imports": query_imports,
        "reference_imports": reference_imports,
    }


def _result_markdown(result: dict[str, Any]) -> str:
    gate_lines = "\n".join(
        f"- `{name}`: {'PASS' if passed else 'FAIL'}"
        for name, passed in sorted(result["release_gates"].items())
    )
    case_lines = "\n".join(
        (
            f"- `{row['case_id']}` / `{row['execution_id']}`: "
            f"bindings={row['binding_count']}, "
            f"positive={json.dumps(row['positive'], sort_keys=True)}, "
            f"negative={json.dumps(row['negative'], sort_keys=True)}, "
            f"net={json.dumps(row['net'], sort_keys=True)}"
        )
        for row in result["execution_summaries"]
    )
    return f"""# Signed Generation Algebra v1 — Result

Final status: **{result["final_status"]}**

## Scope

This exploratory v1 interprets validated Core v3 generation facts under one
frozen signed-effect contract. Sign is not a sixth Core coordinate.
`ExplicitDisposition` is not automatically negative.

The tested hierarchy is:

```text
complete generation state Γ
  -> unreduced signed generation algebra A±(Γ)
  -> net ring projection ν(A±(Γ)) in Z[X]
```

## Executions

{case_lines}

## Strict projection witness

`(0, 0) != (x_record_a, x_record_a)`, while both net projections are zero.
The native final database state is empty for both executions.

## ExplicitDisposition boundary

Misclassified-as-negative count:
`{result["explicit_disposition_misclassified_as_negative_count"]}`.
The filter exclusion remains queryable in the complete state and is interpreted
as `neutral_or_not_applicable` only because the frozen contract says so.

## Release gates

{gate_lines}

## Answers to the research questions

1. Positive and negative actual occurrences remain in separate unreduced
   components: **{result["answers"]["positive_and_negative_retained"]}**.
2. Never-happened and happened-then-cancelled are distinguishable:
   **{result["answers"]["never_and_cancelled_distinguished"]}**.
3. Net projection exactly matches the independent reference:
   **{result["answers"]["independent_net_exact"]}**.
4. ν preserved tested addition and multiplication:
   **{result["answers"]["net_homomorphism_tests_passed"]}**.
5. Z[X] was a strict projection in the executed witness:
   **{result["answers"]["strict_projection_witnessed"]}**.
6. Multiplicity was retained before cancellation:
   **{result["answers"]["multiplicity_retained"]}**.
7. Concrete occurrences remained distinct under equal net values:
   **{result["answers"]["occurrences_distinct"]}**.
8. ExplicitDisposition was not automatically negative:
   **{result["answers"]["explicit_disposition_boundary_preserved"]}**.
9. Frozen Core and manuscript paths changed: **{not result["answers"]["frozen_core_unchanged"]}**.
10. Existing test suite passed: **{result["answers"]["existing_test_suite_passed"]}**.

## Real limitations

- The evidence covers only the frozen deterministic operation family.
- Sign semantics remain a domain-contract interpretation; Core does not infer
  whether a relation is positive or negative.
- The SignedPair aggregates algebraic multiplicities. Concrete occurrence and
  binding identities remain authoritative in Γ and in the relation report,
  not in the aggregate polynomial alone.
- The pure-state reference shares the frozen workload specification with the
  native path, but imports neither Core nor the candidate algebra.
- No literature novelty or general signed-provenance theory is claimed.
"""


def execute_experiment(
    *,
    report_root: Path,
    test_results_path: Path,
    write_reports: bool = True,
) -> dict[str, Any]:
    operation_contract_path = CONTRACT_ROOT / "operation_contract.json"
    operation_contract = _load_json(operation_contract_path)
    signed_contract = _load_json(
        CONTRACT_ROOT / "signed_effect_contract.json"
    )
    query_contract = _load_json(
        CONTRACT_ROOT / "query_contract.json"
    )
    test_results = _load_json(test_results_path)
    execution_ids = [
        execution["execution_id"]
        for execution in operation_contract["executions"]
    ]
    candidates: dict[str, dict[str, Any]] = {}
    native_outputs: dict[str, object] = {}
    snapshot_ids: dict[str, str] = {}
    for execution in operation_contract["executions"]:
        collector = SignedEffectCollector(
            execution["execution_id"], execution_ids
        )
        native = execute_native(execution, collector.capture)
        collected = collector.finalize(native)
        query = RegisteredSignedEffectQuery(
            collected.snapshot,
            collected.validation,
            collected.predicate_registry,
            signed_contract,
            query_contract,
        )
        candidate = query.interpret()
        candidates[execution["execution_id"]] = candidate
        native_outputs[execution["execution_id"]] = (
            native.final_output
        )
        snapshot_ids[execution["execution_id"]] = (
            collected.snapshot.snapshot_id
        )

    reference = _run_reference(operation_contract_path)
    references = {
        row["execution_id"]: row for row in reference["executions"]
    }
    comparisons: list[dict[str, Any]] = []
    for execution_id in execution_ids:
        candidate = candidates[execution_id]
        expected = references[execution_id]
        candidate_contributions = _semantic_contributions(candidate)
        expected_contributions = expected["contributions"]
        positive_exact = (
            candidate["signed_pair"]["positive"]
            == expected["positive"]
        )
        negative_exact = (
            candidate["signed_pair"]["negative"]
            == expected["negative"]
        )
        net_exact = (
            candidate["net_projection"] == expected["net"]
        )
        comparisons.append(
            {
                "coefficient_exact": (
                    _coefficient_rows(
                        candidate["signed_pair"]["positive"]
                    )
                    == _coefficient_rows(expected["positive"])
                    and _coefficient_rows(
                        candidate["signed_pair"]["negative"]
                    )
                    == _coefficient_rows(expected["negative"])
                ),
                "effect_variable_identity_exact": (
                    _variables(
                        candidate["signed_pair"]["positive"]
                    )
                    == _variables(expected["positive"])
                    and _variables(
                        candidate["signed_pair"]["negative"]
                    )
                    == _variables(expected["negative"])
                ),
                "execution_id": execution_id,
                "final_native_output_exact": (
                    native_outputs[execution_id]
                    == expected["final_output"]
                ),
                "multiplicity_exact": (
                    _multiplicity_rows(candidate_contributions)
                    == _multiplicity_rows(expected_contributions)
                ),
                "negative_polynomial_exact": negative_exact,
                "net_polynomial_exact": net_exact,
                "occurrence_identity_exact": (
                    [
                        row["occurrence_identity"]
                        for row in candidate_contributions
                    ]
                    == [
                        row["occurrence_identity"]
                        for row in expected_contributions
                    ]
                ),
                "positive_polynomial_exact": positive_exact,
                "relation_role_exact": (
                    [
                        row["relation_role"]
                        for row in candidate_contributions
                    ]
                    == [
                        row["relation_role"]
                        for row in expected_contributions
                    ]
                ),
                "semantic_contributions_exact": (
                    candidate_contributions
                    == expected_contributions
                ),
                "unmatched_fact_count": len(
                    candidate["unmatched_fact_ids"]
                ),
            }
        )

    pairs = [
        ("case1_never_insert", "case1_insert_then_delete"),
        ("case2_no_update", "case2_update_then_compensate"),
        ("case3_no_increment", "case3_plus5_minus5"),
    ]
    pair_checks = [
        {
            "cancelled_execution": right,
            "complete_generation_states_different": (
                snapshot_ids[left] != snapshot_ids[right]
            ),
            "equal_final_output": (
                native_outputs[left] == native_outputs[right]
            ),
            "equal_net_projection": (
                candidates[left]["net_projection"]
                == candidates[right]["net_projection"]
            ),
            "never_execution": left,
            "signed_states_different": (
                candidates[left]["signed_pair"]
                != candidates[right]["signed_pair"]
            ),
        }
        for left, right in pairs
    ]
    laws = _algebra_laws()
    core_audit = _protected_scope_audit(
        operation_contract["baseline"]
    )
    authority_audit = _authority_audit()
    case5 = candidates["case5_filter_exclusion"]
    fact_kind_by_binding = {
        fact["binding_identity"]: fact["z"]["outcome_kind"]
        for fact in case5["complete_facts"]
    }
    explicit_disposition_misclassified = sum(
        1
        for row in case5["algebraic_contributions"]
        if fact_kind_by_binding[row["binding_identity"]]
        == "disposition"
        and row["sign"] == "negative"
    )
    strict_witness = {
        "a1": candidates["case1_never_insert"]["signed_pair"],
        "a1_not_equal_a2": (
            candidates["case1_never_insert"]["signed_pair"]
            != candidates["case1_insert_then_delete"][
                "signed_pair"
            ]
        ),
        "a2": candidates["case1_insert_then_delete"][
            "signed_pair"
        ],
        "net_a1": candidates["case1_never_insert"][
            "net_projection"
        ],
        "net_a1_equals_net_a2": (
            candidates["case1_never_insert"]["net_projection"]
            == candidates["case1_insert_then_delete"][
                "net_projection"
            ]
        ),
        "net_a2": candidates["case1_insert_then_delete"][
            "net_projection"
        ],
        "witness": "(0,0) != (x_record_a,x_record_a), nu both zero",
    }
    all_exact = lambda key: all(row[key] for row in comparisons)
    gates = {
        "addition_laws_passed": all(
            laws["checks"][key]
            for key in (
                "addition_associative",
                "addition_commutative",
                "zero_identity",
            )
        ),
        "candidate_snapshot_only": authority_audit[
            "candidate_snapshot_only"
        ],
        "equal_final_output_pair_validated": all(
            row["equal_final_output"] for row in pair_checks
        ),
        "equal_net_projection_pair_validated": all(
            row["equal_net_projection"] for row in pair_checks
        ),
        "existing_test_suite_passed": (
            test_results.get("status") == "passed"
            and test_results.get("exit_code") == 0
        ),
        "explicit_disposition_not_auto_negative": (
            explicit_disposition_misclassified == 0
            and case5["explicit_disposition_count"] == 1
            and len(case5["neutral_fact_ids"]) == 1
            and not case5["signed_pair"]["negative"]["terms"]
        ),
        "frozen_core_unchanged": core_audit["unchanged"],
        "independent_negative_exact": all_exact(
            "negative_polynomial_exact"
        ),
        "independent_net_exact": all_exact(
            "net_polynomial_exact"
        ),
        "independent_positive_exact": all_exact(
            "positive_polynomial_exact"
        ),
        "independent_reference_isolated": authority_audit[
            "independent_reference_isolated"
        ],
        "multiplication_laws_passed": all(
            laws["checks"][key]
            for key in (
                "distributive",
                "multiplication_associative",
                "multiplication_commutative",
                "one_identity",
            )
        ),
        "multiplicity_exact": all_exact("multiplicity_exact"),
        "net_projection_additive": laws["checks"][
            "net_projection_additive"
        ],
        "net_projection_multiplicative": laws["checks"][
            "net_projection_multiplicative"
        ],
        "never_vs_cancelled_distinguished": (
            strict_witness["a1_not_equal_a2"]
            and strict_witness["net_a1_equals_net_a2"]
            and pair_checks[0][
                "complete_generation_states_different"
            ]
        ),
        "no_second_authority_store": authority_audit[
            "no_second_authority_store"
        ],
        "occurrence_identity_preserved": all_exact(
            "occurrence_identity_exact"
        )
        and all(
            len(
                {
                    row["occurrence_identity"]
                    for row in _semantic_contributions(
                        candidates[execution_id]
                    )
                }
            )
            == len(
                _semantic_contributions(candidates[execution_id])
            )
            for execution_id in execution_ids
        ),
        "relation_roles_exact": all_exact(
            "relation_role_exact"
        ),
        "signed_pair_no_internal_cancellation": laws[
            "checks"
        ]["no_internal_cancellation"],
        "signed_states_different": all(
            row["signed_states_different"] for row in pair_checks
        ),
    }
    failed_gates = sorted(
        name for name, passed in gates.items() if not passed
    )
    final_status = (
        "SIGNED_GENERATION_ALGEBRA_V1_SUPPORTED"
        if not failed_gates
        else "NOT_SUPPORTED"
    )
    execution_summaries = [
        {
            "binding_count": len(
                candidates[execution["execution_id"]][
                    "complete_facts"
                ]
            ),
            "case_id": execution["case_id"],
            "execution_id": execution["execution_id"],
            "final_native_output": native_outputs[
                execution["execution_id"]
            ],
            "negative": candidates[execution["execution_id"]][
                "signed_pair"
            ]["negative"],
            "net": candidates[execution["execution_id"]][
                "net_projection"
            ],
            "positive": candidates[execution["execution_id"]][
                "signed_pair"
            ]["positive"],
            "snapshot_id": snapshot_ids[execution["execution_id"]],
        }
        for execution in operation_contract["executions"]
    ]
    answers = {
        "existing_test_suite_passed": gates[
            "existing_test_suite_passed"
        ],
        "explicit_disposition_boundary_preserved": gates[
            "explicit_disposition_not_auto_negative"
        ],
        "frozen_core_unchanged": gates["frozen_core_unchanged"],
        "independent_net_exact": gates["independent_net_exact"],
        "multiplicity_retained": gates["multiplicity_exact"],
        "net_homomorphism_tests_passed": (
            gates["net_projection_additive"]
            and gates["net_projection_multiplicative"]
        ),
        "never_and_cancelled_distinguished": gates[
            "never_vs_cancelled_distinguished"
        ],
        "occurrences_distinct": gates[
            "occurrence_identity_preserved"
        ],
        "paired_complete_states_distinct": all(
            row["complete_generation_states_different"]
            for row in pair_checks
        ),
        "positive_and_negative_retained": gates[
            "signed_pair_no_internal_cancellation"
        ],
        "strict_projection_witnessed": (
            strict_witness["a1_not_equal_a2"]
            and strict_witness["net_a1_equals_net_a2"]
        ),
    }
    result = {
        "answers": answers,
        "authority_audit": authority_audit,
        "core_audit": core_audit,
        "execution_summaries": execution_summaries,
        "explicit_disposition_misclassified_as_negative_count": (
            explicit_disposition_misclassified
        ),
        "failed_gates": failed_gates,
        "final_status": final_status,
        "release_gates": gates,
        "schema_version": "signed-generation-algebra-result-v1",
        "test_results": test_results,
    }
    relation_exactness = {
        "comparisons": comparisons,
        "schema_version": "signed-generation-relation-exactness-v1",
    }
    strict_projection = {
        "pair_checks": pair_checks,
        "schema_version": "signed-generation-strict-projection-v1",
        "strict_witness": strict_witness,
    }
    non_collapse = {
        "case_1_witness": strict_witness,
        "case_2_pair": pair_checks[1],
        "case_3_pair": pair_checks[2],
        "partial_cancellation": candidates[
            "case4_add_x_add_y_remove_x"
        ],
        "schema_version": "signed-generation-non-collapse-v1",
    }
    if write_reports:
        report_root.mkdir(parents=True, exist_ok=True)
        _write_json(report_root / "algebra_laws.json", laws)
        _write_json(
            report_root / "authority_audit.json",
            authority_audit,
        )
        _write_json(
            report_root / "candidate_results.json", candidates
        )
        _write_json(
            report_root / "independent_reference.json", reference
        )
        _write_json(
            report_root / "non_collapse_witnesses.json",
            non_collapse,
        )
        _write_json(
            report_root / "relation_exactness.json",
            relation_exactness,
        )
        _write_json(
            report_root / "release_gates.json",
            {
                "failed_gates": failed_gates,
                "final_status": final_status,
                "gates": gates,
            },
        )
        _write_json(report_root / "result.json", result)
        _write_json(
            report_root / "strict_projection.json",
            strict_projection,
        )
        (report_root / "RESULT.md").write_text(
            _result_markdown(result),
            encoding="utf-8",
            newline="\n",
        )
        artifact_files = sorted(
            path
            for path in report_root.iterdir()
            if path.is_file()
            and path.name != "artifact_manifest.json"
        )
        manifest = {
            "artifacts": [
                {
                    "path": path.name,
                    "sha256": hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest(),
                    "size": path.stat().st_size,
                }
                for path in artifact_files
            ],
            "schema_version": (
                "signed-generation-artifact-manifest-v1"
            ),
        }
        _write_json(
            report_root / "artifact_manifest.json", manifest
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-root", type=Path, default=DEFAULT_REPORT_ROOT
    )
    parser.add_argument(
        "--test-results",
        type=Path,
        default=DEFAULT_REPORT_ROOT / "test_results.json",
    )
    args = parser.parse_args(argv)
    result = execute_experiment(
        report_root=args.report_root,
        test_results_path=args.test_results,
    )
    print(
        json.dumps(
            {
                "failed_gates": result["failed_gates"],
                "final_status": result["final_status"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return (
        0
        if result["final_status"]
        == "SIGNED_GENERATION_ALGEBRA_V1_SUPPORTED"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
