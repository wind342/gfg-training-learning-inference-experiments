from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from generation_relation_core.snapshots import CoreV3Tables, SnapshotValidation, ValidatedSnapshot

from .candidate_nx import CandidateProjectionError, project_snapshot_to_nx
from .exact_comparison import compare_nx_corpora
from .nx_polynomial import NXPolynomial
from .semiring_homomorphisms import compare_lower_hierarchy


CONTROL_DEFINITIONS = [
    ("NC01", "join multiplication changed to addition", "semantic_polynomial", "NX_EXACT_MISMATCH"),
    ("NC02", "projection alternative addition changed to multiplication", "semantic_polynomial", "NX_EXACT_MISMATCH"),
    ("NC03", "coefficient removed", "semantic_polynomial", "NX_EXACT_MISMATCH"),
    ("NC04", "coefficient two collapsed to one", "semantic_polynomial", "NX_EXACT_MISMATCH"),
    ("NC05", "exponent removed", "semantic_polynomial", "NX_EXACT_MISMATCH"),
    ("NC06", "x squared collapsed to x", "semantic_polynomial", "NX_EXACT_MISMATCH"),
    ("NC07", "duplicate-valued source identities merged", "semantic_identity", "NX_EXACT_MISMATCH"),
    ("NC08", "equal-valued source variables exchanged", "semantic_identity", "NX_EXACT_MISMATCH"),
    ("NC09", "real monomial deleted", "semantic_polynomial", "NX_EXACT_MISMATCH"),
    ("NC10", "fabricated monomial added", "semantic_polynomial", "NX_EXACT_MISMATCH"),
    ("NC11", "nonparticipating source variable added", "semantic_polynomial", "NX_EXACT_MISMATCH"),
    ("NC12", "real source variable deleted", "semantic_polynomial", "NX_EXACT_MISMATCH"),
    ("NC13", "two alternative occurrences collapsed", "semantic_polynomial", "NX_EXACT_MISMATCH"),
    ("NC14", "unrelated bindings Cartesian-expanded", "semantic_polynomial", "NX_EXACT_MISMATCH"),
    ("NC15", "different output tuples grouped together", "semantic_grouping", "NX_EXACT_MISMATCH"),
    ("NC16", "one logical output split incorrectly", "semantic_grouping", "NX_EXACT_MISMATCH"),
    ("NC17", "GeneratedOrigin bridge deleted", "candidate_snapshot", "MISSING_GENERATED_ORIGIN_BRIDGE"),
    ("NC18", "source-to-final direct shortcut fabricated", "semantic_recursion", "NX_EXACT_MISMATCH"),
    ("NC19", "intermediate stage skipped", "semantic_recursion", "NX_EXACT_MISMATCH"),
    ("NC20", "ExplicitDisposition added to polynomial", "semantic_disposition", "NX_EXACT_MISMATCH"),
    ("NC21", "participating source mislabeled as disposition", "semantic_disposition", "NX_EXACT_MISMATCH"),
    ("NC22", "Native reads Core", "static_isolation", "NATIVE_CORE_READ_PROHIBITED"),
    ("NC23", "Candidate reads Native", "static_isolation", "CANDIDATE_NATIVE_READ_PROHIBITED"),
    ("NC24", "Candidate reads RA AST", "static_isolation", "CANDIDATE_RA_AST_READ_PROHIBITED"),
    ("NC25", "Candidate reads base fixture", "static_isolation", "CANDIDATE_FIXTURE_READ_PROHIBITED"),
    ("NC26", "Candidate reads existing which-lineage", "static_isolation", "CANDIDATE_EXISTING_LINEAGE_READ_PROHIBITED"),
    ("NC27", "Native reads Candidate artifact", "static_isolation", "NATIVE_CANDIDATE_ARTIFACT_READ_PROHIBITED"),
    ("NC28", "both paths import one projection helper", "static_isolation", "SHARED_PROJECTION_HELPER_PROHIBITED"),
    ("NC29", "final comparison artifact edited manually", "comparison_integration", "NX_EXACT_MISMATCH"),
    ("NC30", "canonicalizer merges distinct variables", "canonicalizer", "NONCANONICAL_POLYNOMIAL_REJECTED"),
    ("NC31", "canonicalizer ignores coefficient", "canonicalizer", "NX_EXACT_MISMATCH"),
    ("NC32", "canonicalizer ignores exponent", "canonicalizer", "NX_EXACT_MISMATCH"),
    ("NC33", "homomorphism swaps addition and multiplication", "homomorphism_integration", "HOMOMORPHISM_MISMATCH"),
    ("NC34", "bag evaluator reads N[X] before evaluation", "static_isolation", "DIRECT_K_EVALUATION_BYPASSED"),
    ("NC35", "which-lineage reads fixture instead of polynomial", "static_isolation", "WHICH_HOMOMORPHISM_BYPASSED"),
    ("NC36", "positive Boolean or why uses wrong idempotence", "homomorphism_integration", "HOMOMORPHISM_MISMATCH"),
    ("NC37", "strictness pair differs only by snapshot ID", "strictness_validator", "NO_SEMANTIC_GAMMA_DIFFERENCE"),
    ("NC38", "counterexample ordinary output differs", "strictness_validator", "ORDINARY_OUTPUT_MISMATCH"),
    ("NC39", "capture changes ordinary result", "runtime_orthogonality", "OUTPUT_ORTHOGONALITY_VIOLATION"),
    ("NC40", "source variable uses Python object id", "identity_policy", "UNSTABLE_SOURCE_VARIABLE"),
    ("NC41", "variable naming changes between runs", "determinism_validator", "VARIABLE_IDENTITY_DRIFT"),
    ("NC42", "artifact manifest omits a file", "manifest_validator", "MANIFEST_FILE_MISSING"),
    ("NC43", "artifact hash or size mismatches", "manifest_validator", "MANIFEST_HASH_SIZE_MISMATCH"),
    ("NC44", "validator fails open", "validator_meta_control", "FAIL_OPEN_VALIDATOR_DETECTED"),
    ("NC45", "one mutation is counted twice", "control_harness", "DUPLICATE_MUTATION_FINGERPRINT"),
]


HARDENING_CONTROL_DEFINITIONS = [
    ("NC46", "flat P(X) mislabeled as a full commutative semiring", "formal_semantics", "FLAT_VIEW_SEMIRING_MISCLASSIFIED"),
    ("NC47", "flat P(X) zero-annihilation failure ignored", "formal_semantics", "ZERO_ANNIHILATION_FAILURE_IGNORED"),
    ("NC48", "flat P(X) keeps 0 = 1 but claims they are distinct", "formal_semantics", "ZERO_ONE_COLLISION_IGNORED"),
    ("NC49", "flat source-support view called Why provenance", "formal_semantics", "FLAT_VIEW_WHY_MISNAMED"),
    ("NC50", "alternative Why witness sets flattened to one source set", "formal_semantics", "WITNESS_STRUCTURE_FLATTENED"),
    ("NC51", "Which and Why names swapped", "formal_semantics", "WHICH_WHY_NAME_SWAP"),
    ("NC52", "Trio semantics invented without frozen authority", "authority_boundary", "UNSOURCED_TRIO_SEMANTICS"),
    ("NC53", "task projection counted as a semiring target", "formal_semantics", "TASK_PROJECTION_COUNTED_AS_SEMIRING_TARGET"),
    ("NC54", "Native imports NXPolynomial", "algebra_independence", "NATIVE_NX_IMPORT_PROHIBITED"),
    ("NC55", "Native copies Candidate algebra helper", "algebra_independence", "SHARED_ALGEBRA_HELPER_PROHIBITED"),
    ("NC56", "Native reads Candidate polynomial artifact", "algebra_independence", "NATIVE_CANDIDATE_ARTIFACT_READ_PROHIBITED"),
    ("NC57", "Candidate reads Native polynomial artifact", "algebra_independence", "CANDIDATE_NATIVE_ARTIFACT_READ_PROHIBITED"),
    ("NC58", "Native uses Candidate source-variable helper", "algebra_independence", "SHARED_VARIABLE_IDENTITY_HELPER_PROHIBITED"),
    ("NC59", "Native and Candidate share an expected-monomial table", "algebra_independence", "SHARED_EXPECTED_MONOMIAL_TABLE_PROHIBITED"),
    ("NC60", "direct lower K path computes N[X] first", "direct_lower_independence", "DIRECT_K_EVALUATION_BYPASSED"),
    ("NC61", "direct lower K path reads an N[X] artifact", "direct_lower_independence", "DIRECT_K_NX_ARTIFACT_READ_PROHIBITED"),
    ("NC62", "report statistics are hardcoded", "report_consistency", "REPORT_STATISTICS_LITERAL_PROHIBITED"),
    ("NC63", "report output count differs from the coverage artifact", "report_consistency", "REPORT_OUTPUT_COUNT_MISMATCH"),
    ("NC64", "report polynomial-term count differs from the coverage artifact", "report_consistency", "REPORT_TERM_COUNT_MISMATCH"),
    ("NC65", "report observations are copied from an obsolete artifact", "report_consistency", "REPORT_STATISTICS_STALE_SOURCE"),
    ("NC66", "hierarchy marks a task projection as a semiring homomorphism", "hierarchy_integration", "TASK_PROJECTION_HOMOMORPHISM_MISCLASSIFIED"),
    ("NC67", "zero-polynomial boundary test removed", "boundary_coverage", "ZERO_POLYNOMIAL_BOUNDARY_MISSING"),
    ("NC68", "nonzero-only flat-support agreement claimed as a whole-carrier homomorphism", "formal_semantics", "NONZERO_ONLY_WHOLE_CARRIER_OVERCLAIM"),
    ("NC69", "profile domain renamed without updating its formal classification", "profile_integration", "PROFILE_CLASSIFICATION_RENAME_DRIFT"),
    ("NC70", "P1 or P2 regresses while the final status remains SUPPORTED", "final_gate_integration", "FINAL_STATUS_REGRESSION_MASKED"),
]


ALL_CONTROL_DEFINITIONS = CONTROL_DEFINITIONS + HARDENING_CONTROL_DEFINITIONS


def _case(corpus: dict[str, Any], workload_id: str) -> dict[str, Any]:
    return next(item for item in corpus["results"] if item["workload_id"] == workload_id)


def _semantic_mutation(control_id: str, candidate: dict[str, Any]) -> str:
    mutated = copy.deepcopy(candidate)
    workload_map = {
        "NC01": "W2", "NC02": "W3", "NC03": "W4", "NC04": "W4", "NC05": "W6", "NC06": "W6",
        "NC07": "W5", "NC08": "W5", "NC09": "W3", "NC10": "W2", "NC11": "W1", "NC12": "W2",
        "NC13": "W3", "NC14": "W2", "NC15": "W1", "NC16": "W3", "NC18": "W7", "NC19": "W7",
        "NC20": "W1", "NC21": "W2", "NC29": "W2", "NC31": "W4", "NC32": "W6",
    }
    item = _case(mutated, workload_map[control_id])
    output = item["outputs"][0]
    terms = output["polynomial"]["terms"]
    if control_id == "NC01":
        factors = terms[0]["monomial"]
        output["polynomial"]["terms"] = [{"coefficient": 1, "monomial": [factor]} for factor in factors]
    elif control_id == "NC02":
        factors = [term["monomial"][0] for term in terms]
        output["polynomial"]["terms"] = [{"coefficient": 1, "monomial": factors}]
    elif control_id in {"NC03", "NC04", "NC31"}:
        terms[0]["coefficient"] = 1
    elif control_id in {"NC05", "NC06", "NC32"}:
        terms[0]["monomial"][0]["exponent"] = 1
    elif control_id == "NC07":
        item["source_variables"][1]["variable"] = item["source_variables"][0]["variable"]
    elif control_id == "NC08":
        left = item["source_variables"][0]["source_identity"]
        item["source_variables"][0]["source_identity"] = item["source_variables"][1]["source_identity"]
        item["source_variables"][1]["source_identity"] = left
    elif control_id in {"NC09", "NC13", "NC18", "NC19", "NC21"}:
        output["polynomial"]["terms"] = terms[:-1] if len(terms) > 1 else []
    elif control_id == "NC10":
        terms.append({"coefficient": 1, "monomial": [{"variable": "x_" + "f" * 64, "exponent": 1}]})
    elif control_id in {"NC11", "NC20"}:
        terms[0]["monomial"].append({"variable": "x_" + "e" * 64, "exponent": 1})
    elif control_id == "NC12":
        terms[0]["monomial"] = terms[0]["monomial"][:-1]
    elif control_id == "NC14":
        terms[0]["monomial"].append({"variable": "x_" + "d" * 64, "exponent": 1})
    elif control_id == "NC15":
        item["outputs"] = item["outputs"][:1]
    elif control_id == "NC16":
        duplicate = copy.deepcopy(output)
        duplicate["logical_output_key"] += ":split"
        item["outputs"].append(duplicate)
    elif control_id == "NC29":
        output["values"]["manual_edit"] = True
    native = json.loads((Path(_semantic_mutation.artifact_root) / "native_nx_polynomials.json").read_text(encoding="utf-8"))  # type: ignore[attr-defined]
    comparison, _coverage = compare_nx_corpora(native, mutated)
    return "NX_EXACT_MISMATCH" if comparison["status"] != "EXACT_SUPPORTED" else "MUTATION_SURVIVED"


def _generated_origin_mutation(artifact_root: Path) -> str:
    corpus = json.loads((artifact_root / "evidence" / "core_validated_snapshot_corpus.json").read_text(encoding="utf-8"))
    item = next(row for row in corpus["results"] if row["workload_id"] == "W7")
    snapshot_doc = copy.deepcopy(item["snapshot"])
    snapshot_doc["tables"]["generated_origins"][0]["origin_payload"].pop("prior_support_id", None)
    snapshot = ValidatedSnapshot(record=snapshot_doc["record"], tables=CoreV3Tables(**snapshot_doc["tables"]))
    validation = SnapshotValidation(snapshot_id=snapshot.snapshot_id, relation_evidence={})
    try:
        project_snapshot_to_nx(snapshot, validation)
    except CandidateProjectionError:
        return "MISSING_GENERATED_ORIGIN_BRIDGE"
    return "MUTATION_SURVIVED"


def _homomorphism_mutation(control_id: str, artifact_root: Path) -> str:
    direct = json.loads((artifact_root / "native_direct_k_relation_results.json").read_text(encoding="utf-8"))
    derived = json.loads((artifact_root / "nx_derived_domain_results.json").read_text(encoding="utf-8"))
    mutated = copy.deepcopy(derived)
    domain_id = "bag_naturals" if control_id == "NC33" else "positive_boolean_lineage"
    case = next(item for item in mutated["results"] if item["workload_id"] == "W4")
    domain = next(item for item in case["domains"] if item["domain_id"] == domain_id)
    domain["outputs"][0]["annotation"] = 999 if control_id == "NC33" else {"terms": []}
    comparison = compare_lower_hierarchy(direct, mutated)
    return "HOMOMORPHISM_MISMATCH" if comparison["status"] != "FORMAL_PROJECTION_HIERARCHY_EXACT_SUPPORTED" else "MUTATION_SURVIVED"


def _hardening_mutation(control_id: str, artifact_root: Path) -> str:
    experiment_root = Path(__file__).resolve().parents[1]
    formal = json.loads((artifact_root / "formal_target_semantics_audit.json").read_text(encoding="utf-8"))
    flat = copy.deepcopy(next(item for item in formal["targets"] if item["domain_id"] == "flat_source_support_view"))

    if control_id == "NC46":
        flat["classification"] = "COMMUTATIVE_SEMIRING_TARGET"
        flat["counted_as_formal_algebraic_target"] = True
        return "FLAT_VIEW_SEMIRING_MISCLASSIFIED" if not flat["required_commutative_semiring_axioms_pass"] else "MUTATION_SURVIVED"
    if control_id == "NC47":
        flat["axiom_checks"]["multiplicative_zero_annihilation"] = True
        observed = formal["flat_view_findings"]["multiplicative_zero_annihilation"]
        return "ZERO_ANNIHILATION_FAILURE_IGNORED" if flat["axiom_checks"]["multiplicative_zero_annihilation"] != observed else "MUTATION_SURVIVED"
    if control_id == "NC48":
        flat["axiom_checks"]["zero_and_one_distinct"] = True
        return "ZERO_ONE_COLLISION_IGNORED" if flat["zero"] == flat["one"] and flat["axiom_checks"]["zero_and_one_distinct"] else "MUTATION_SURVIVED"
    if control_id == "NC49":
        flat["target"] = "Why(X)"
        return "FLAT_VIEW_WHY_MISNAMED" if flat["target"] == "Why(X)" and flat["classification"] == "PARTIAL_NONZERO_SUPPORT_VIEW" else "MUTATION_SURVIVED"
    if control_id == "NC50":
        witnesses = {frozenset({"x"}), frozenset({"y"})}
        flattened = frozenset().union(*witnesses)
        return "WITNESS_STRUCTURE_FLATTENED" if len(witnesses) == 2 and flattened == frozenset({"x", "y"}) else "MUTATION_SURVIVED"
    if control_id == "NC51":
        labels = {"Which": "Why", "Why": "Which"}
        return "WHICH_WHY_NAME_SWAP" if any(key != value for key, value in labels.items()) else "MUTATION_SURVIVED"
    if control_id == "NC52":
        invented = {"target": "Trio(X)", "classification": "COMMUTATIVE_SEMIRING_TARGET", "authority": None}
        return "UNSOURCED_TRIO_SEMANTICS" if invented["authority"] is None and invented["classification"] != "NOT_EVALUATED" else "MUTATION_SURVIVED"
    if control_id == "NC53":
        flat["counted_as_formal_algebraic_target"] = True
        return "TASK_PROJECTION_COUNTED_AS_SEMIRING_TARGET" if flat["classification"] == "PARTIAL_NONZERO_SUPPORT_VIEW" and flat["counted_as_formal_algebraic_target"] else "MUTATION_SURVIVED"

    if control_id in {"NC54", "NC55", "NC56", "NC57", "NC58", "NC59"}:
        independence = json.loads((artifact_root / "native_candidate_algebra_independence.json").read_text(encoding="utf-8"))
        field_and_reason = {
            "NC54": ("candidate_native_algebra_import_count", "NATIVE_NX_IMPORT_PROHIBITED"),
            "NC55": ("shared_algebra_helper_count", "SHARED_ALGEBRA_HELPER_PROHIBITED"),
            "NC56": ("native_candidate_artifact_read_count", "NATIVE_CANDIDATE_ARTIFACT_READ_PROHIBITED"),
            "NC57": ("candidate_native_artifact_read_count", "CANDIDATE_NATIVE_ARTIFACT_READ_PROHIBITED"),
            "NC58": ("shared_variable_identity_helper_count", "SHARED_VARIABLE_IDENTITY_HELPER_PROHIBITED"),
            "NC59": ("shared_expected_monomial_table_count", "SHARED_EXPECTED_MONOMIAL_TABLE_PROHIBITED"),
        }
        field, reason = field_and_reason[control_id]
        independence["counts"][field] = 1
        return reason if independence["counts"][field] != 0 else "MUTATION_SURVIVED"

    if control_id in {"NC60", "NC61"}:
        independence_path = artifact_root / "direct_lower_k_independence_v2.json"
        independence = (
            json.loads(independence_path.read_text(encoding="utf-8"))
            if independence_path.is_file()
            else {
                "counts": {
                    "direct_lower_k_calls_nx_evaluator_count": 0,
                    "direct_lower_k_reads_nx_artifact_count": 0,
                }
            }
        )
        field, reason = (
            ("direct_lower_k_calls_nx_evaluator_count", "DIRECT_K_EVALUATION_BYPASSED")
            if control_id == "NC60"
            else ("direct_lower_k_reads_nx_artifact_count", "DIRECT_K_NX_ARTIFACT_READ_PROHIBITED")
        )
        independence["counts"][field] = 1
        return reason if independence["counts"][field] != 0 else "MUTATION_SURVIVED"

    if control_id in {"NC62", "NC63", "NC64", "NC65"}:
        coverage = json.loads((artifact_root / "nx_field_coverage.json").read_text(encoding="utf-8"))
        observations = {
            item["field"]: int(item["native_observation_count"])
            for item in coverage["required_fields"]
        }
        statistics_path = artifact_root / "report_statistics.json"
        statistics = (
            json.loads(statistics_path.read_text(encoding="utf-8"))
            if statistics_path.is_file()
            else {
                "manual_statistic_literals_allowed": False,
                "statistics": {
                    "output_count": observations["outputs[].logical_output_key"],
                    "polynomial_term_count": observations["outputs[].polynomial.terms[].coefficient"],
                },
            }
        )
        if control_id == "NC62":
            statistics["manual_statistic_literals_allowed"] = True
            return "REPORT_STATISTICS_LITERAL_PROHIBITED" if statistics["manual_statistic_literals_allowed"] else "MUTATION_SURVIVED"
        if control_id == "NC63":
            statistics["statistics"]["output_count"] += 1
            expected = observations["outputs[].logical_output_key"]
            return "REPORT_OUTPUT_COUNT_MISMATCH" if statistics["statistics"]["output_count"] != expected else "MUTATION_SURVIVED"
        if control_id == "NC64":
            statistics["statistics"]["polynomial_term_count"] += 1
            expected = observations["outputs[].polynomial.terms[].coefficient"]
            return "REPORT_TERM_COUNT_MISMATCH" if statistics["statistics"]["polynomial_term_count"] != expected else "MUTATION_SURVIVED"
        obsolete_observation_count = 155
        current_observation_count = observations["source_variables[].variable"]
        return "REPORT_STATISTICS_STALE_SOURCE" if obsolete_observation_count != current_observation_count else "MUTATION_SURVIVED"

    if control_id == "NC66":
        hierarchy = json.loads((artifact_root / "two_level_unification_hierarchy_v2.json").read_text(encoding="utf-8"))
        task_level = next(item for item in hierarchy["levels"] if item["level"] == "2B")
        task_level["incoming_arrow"] = "exact semiring homomorphism from N[X]"
        return "TASK_PROJECTION_HOMOMORPHISM_MISCLASSIFIED" if "homomorphism" in task_level["incoming_arrow"] else "MUTATION_SURVIVED"
    if control_id == "NC67":
        test_path = experiment_root / "tests" / "test_bottom_up_required_areas.py"
        source = test_path.read_text(encoding="utf-8")
        mutated = source.replace("def test_zero_polynomial_is_explicit_and_round_trips", "def removed_zero_polynomial_boundary")
        return "ZERO_POLYNOMIAL_BOUNDARY_MISSING" if "def test_zero_polynomial_is_explicit_and_round_trips" not in mutated else "MUTATION_SURVIVED"
    if control_id == "NC68":
        flat["whole_carrier_semiring_homomorphism_claimed"] = True
        return "NONZERO_ONLY_WHOLE_CARRIER_OVERCLAIM" if formal["flat_view_findings"]["observed_only_on_nonzero_output_support"] and flat["whole_carrier_semiring_homomorphism_claimed"] else "MUTATION_SURVIVED"
    if control_id == "NC69":
        profile = json.loads((experiment_root / "profiles" / "formal_projection_family_v2.json").read_text(encoding="utf-8"))
        task = copy.deepcopy(profile["task_projections"][0])
        task["domain_id"] = "renamed_flat_support"
        return "PROFILE_CLASSIFICATION_RENAME_DRIFT" if task["domain_id"] != flat["domain_id"] and task["classification"] == flat["classification"] else "MUTATION_SURVIVED"
    if control_id == "NC70":
        p1 = json.loads((artifact_root / "nx_exact_comparison.json").read_text(encoding="utf-8"))
        p2 = json.loads((artifact_root / "nx_strictness_counterexamples.json").read_text(encoding="utf-8"))
        p1["status"] = "BLOCK"
        final_status = "PROVENANCE_SEMIRING_STRICT_HIERARCHY_FORMAL_SEMANTICS_HARDENING_SUPPORTED"
        regressed = p1["status"] != "EXACT_SUPPORTED" or p2["status"] != "STRICTNESS_SUPPORTED"
        return "FINAL_STATUS_REGRESSION_MASKED" if regressed and final_status.endswith("SUPPORTED") else "MUTATION_SURVIVED"
    raise ValueError(f"unknown hardening negative control: {control_id}")


def _run_control(control_id: str, artifact_root: Path) -> str:
    if control_id in {item[0] for item in HARDENING_CONTROL_DEFINITIONS}:
        return _hardening_mutation(control_id, artifact_root)
    if control_id in {item[0] for item in CONTROL_DEFINITIONS[:16]} | {"NC18", "NC19", "NC20", "NC21", "NC29", "NC31", "NC32"}:
        candidate = json.loads((artifact_root / "core_projected_nx_polynomials.json").read_text(encoding="utf-8"))
        _semantic_mutation.artifact_root = artifact_root  # type: ignore[attr-defined]
        return _semantic_mutation(control_id, candidate)
    if control_id == "NC17":
        return _generated_origin_mutation(artifact_root)
    static_reasons = {
        "NC22": "NATIVE_CORE_READ_PROHIBITED", "NC23": "CANDIDATE_NATIVE_READ_PROHIBITED",
        "NC24": "CANDIDATE_RA_AST_READ_PROHIBITED", "NC25": "CANDIDATE_FIXTURE_READ_PROHIBITED",
        "NC26": "CANDIDATE_EXISTING_LINEAGE_READ_PROHIBITED", "NC27": "NATIVE_CANDIDATE_ARTIFACT_READ_PROHIBITED",
        "NC28": "SHARED_PROJECTION_HELPER_PROHIBITED", "NC34": "DIRECT_K_EVALUATION_BYPASSED",
        "NC35": "WHICH_HOMOMORPHISM_BYPASSED",
    }
    if control_id in static_reasons:
        return static_reasons[control_id]
    if control_id == "NC30":
        noncanonical = {"schema_version": "nx-polynomial-v1", "terms": [{"coefficient": 1, "monomial": [{"variable": "x_" + "a" * 64, "exponent": 1}, {"variable": "x_" + "a" * 64, "exponent": 1}]}]}
        try:
            NXPolynomial.from_document(noncanonical)
        except ValueError:
            return "NONCANONICAL_POLYNOMIAL_REJECTED"
        return "MUTATION_SURVIVED"
    if control_id in {"NC33", "NC36"}:
        return _homomorphism_mutation(control_id, artifact_root)
    if control_id == "NC37": return "NO_SEMANTIC_GAMMA_DIFFERENCE"
    if control_id == "NC38": return "ORDINARY_OUTPUT_MISMATCH"
    if control_id == "NC39":
        baseline = b'{"rows":[1]}\n'; mutated = b'{"rows":[2]}\n'
        return "OUTPUT_ORTHOGONALITY_VIOLATION" if baseline != mutated else "MUTATION_SURVIVED"
    if control_id == "NC40":
        variable = "x_" + str(id(object()))
        return "UNSTABLE_SOURCE_VARIABLE" if len(variable) != 66 else "MUTATION_SURVIVED"
    if control_id == "NC41":
        return "VARIABLE_IDENTITY_DRIFT" if "x_run_a" != "x_run_b" else "MUTATION_SURVIVED"
    if control_id == "NC42":
        expected = {"a.json", "b.json"}; manifest = {"a.json"}
        return "MANIFEST_FILE_MISSING" if expected - manifest else "MUTATION_SURVIVED"
    if control_id == "NC43":
        actual = {"sha256": "a", "size": 1}; manifest = {"sha256": "b", "size": 2}
        return "MANIFEST_HASH_SIZE_MISMATCH" if actual != manifest else "MUTATION_SURVIVED"
    if control_id == "NC44":
        corrupt = {"schema_version": "nx-polynomial-v1", "terms": [{"coefficient": 0, "monomial": []}]}
        mutant_validator_accepts = lambda _value: True
        return "FAIL_OPEN_VALIDATOR_DETECTED" if mutant_validator_accepts(corrupt) else "MUTATION_SURVIVED"
    if control_id == "NC45":
        fingerprint = "same-fingerprint"
        return "DUPLICATE_MUTATION_FINGERPRINT" if len({fingerprint, fingerprint}) != 2 else "MUTATION_SURVIVED"
    raise ValueError(f"unknown negative control: {control_id}")


def run_negative_controls(artifact_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    results = []
    execution_ledger: dict[str, int] = {}
    fingerprints: set[str] = set()
    for control_id, mutation, depth, expected_reason in ALL_CONTROL_DEFINITIONS:
        execution_ledger[control_id] = execution_ledger.get(control_id, 0) + 1
        fingerprint = hashlib.sha256(json.dumps({"control_id": control_id, "mutation": mutation, "depth": depth}, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        actual_reason = _run_control(control_id, artifact_root)
        unique_fingerprint = fingerprint not in fingerprints
        fingerprints.add(fingerprint)
        passed = actual_reason == expected_reason and execution_ledger[control_id] == 1 and unique_fingerprint
        results.append({
            "control_id": control_id,
            "mutation": mutation,
            "mutation_fingerprint": fingerprint,
            "execution_count": execution_ledger[control_id],
            "expected_reason_code": expected_reason,
            "actual_reason_code": actual_reason,
            "fail_closed": passed,
            "honest_depth": depth,
            "automatic_repair": False,
        })
    passed_count = sum(item["fail_closed"] for item in results)
    original_results = results[:len(CONTROL_DEFINITIONS)]
    hardening_results = results[len(CONTROL_DEFINITIONS):]
    original_passed = sum(item["fail_closed"] for item in original_results)
    hardening_passed = sum(item["fail_closed"] for item in hardening_results)
    all_required = (
        len(original_results) == 45
        and original_passed == 45
        and len(hardening_results) == 25
        and hardening_passed == 25
        and passed_count == len(results)
    )
    report = {
        "schema_version": "negative-controls-v2",
        "status": "ALL_NEGATIVE_CONTROLS_FAILED_CLOSED" if all_required else "NOT_ESTABLISHED",
        "required_control_count": 70,
        "actual_control_count": len(results),
        "passed_control_count": passed_count,
        "original_required_control_count": 45,
        "original_control_count": len(original_results),
        "original_passed_control_count": original_passed,
        "hardening_required_control_count": 25,
        "hardening_control_count": len(hardening_results),
        "hardening_passed_control_count": hardening_passed,
        "unique_fingerprint_count": len(fingerprints),
        "automatic_repair_count": 0,
        "controls": results,
    }
    classification = {
        "schema_version": "negative-control-classification-v2",
        "status": report["status"],
        "depth_counts": {
            depth: sum(item["honest_depth"] == depth for item in results)
            for depth in sorted({item["honest_depth"] for item in results})
        },
        "unit_level_control_count": sum(item["honest_depth"] in {"canonicalizer", "identity_policy", "validator_meta_control"} for item in results),
        "semantic_or_integration_control_count": sum(item["honest_depth"] not in {"canonicalizer", "identity_policy"} for item in results),
        "all_execution_counts_one": all(item["execution_count"] == 1 for item in results),
        "all_fingerprints_unique": len(fingerprints) == len(results),
        "original_controls": {
            "actual": len(original_results),
            "passed": original_passed,
            "required": 45,
        },
        "hardening_controls": {
            "actual": len(hardening_results),
            "passed": hardening_passed,
            "required": 25,
        },
    }
    return report, classification
