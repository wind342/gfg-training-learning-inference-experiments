from __future__ import annotations

from itertools import product
from typing import Any

from .native_lower_k import (
    BagCarrier,
    BooleanCarrier,
    FlatSourceSupportCarrier,
    PositiveBooleanCarrier,
)


def _law_checks(carrier: object, samples: list[object]) -> dict[str, bool]:
    zero = carrier.zero()  # type: ignore[attr-defined]
    one = carrier.one()  # type: ignore[attr-defined]
    plus = carrier.plus  # type: ignore[attr-defined]
    times = carrier.times  # type: ignore[attr-defined]
    pairs = list(product(samples, repeat=2))
    triples = list(product(samples, repeat=3))
    return {
        "zero_and_one_distinct": zero != one,
        "addition_identity": all(plus(value, zero) == value and plus(zero, value) == value for value in samples),
        "multiplication_identity": all(times(value, one) == value and times(one, value) == value for value in samples),
        "addition_commutative": all(plus(left, right) == plus(right, left) for left, right in pairs),
        "multiplication_commutative": all(times(left, right) == times(right, left) for left, right in pairs),
        "addition_associative": all(plus(plus(a, b), c) == plus(a, plus(b, c)) for a, b, c in triples),
        "multiplication_associative": all(times(times(a, b), c) == times(a, times(b, c)) for a, b, c in triples),
        "left_distributive": all(times(a, plus(b, c)) == plus(times(a, b), times(a, c)) for a, b, c in triples),
        "right_distributive": all(times(plus(a, b), c) == plus(times(a, c), times(b, c)) for a, b, c in triples),
        "multiplicative_zero_annihilation": all(times(value, zero) == zero and times(zero, value) == zero for value in samples),
        "addition_idempotent": all(plus(value, value) == value for value in samples),
        "multiplication_idempotent": all(times(value, value) == value for value in samples),
        "lattice_absorption": all(
            plus(a, times(a, b)) == a and times(a, plus(a, b)) == a
            for a, b in pairs
        ),
    }


def build_formal_target_semantics_audit() -> dict[str, Any]:
    bag = BagCarrier()
    boolean = BooleanCarrier()
    positive = PositiveBooleanCarrier()
    flat = FlatSourceSupportCarrier()
    carriers = [
        (
            "bag_naturals",
            bag,
            [bag.zero(), bag.one(), 2, 3],
            "COMMUTATIVE_SEMIRING_TARGET",
            "N",
            "natural numbers",
            "natural addition",
            "natural multiplication",
        ),
        (
            "boolean",
            boolean,
            [boolean.zero(), boolean.one()],
            "COMMUTATIVE_SEMIRING_TARGET",
            "B",
            "false and true",
            "logical OR",
            "logical AND",
        ),
        (
            "positive_boolean_lineage",
            positive,
            [
                positive.zero(),
                positive.one(),
                positive.variable("x"),
                positive.variable("y"),
                positive.plus(positive.variable("x"), positive.variable("y")),
                positive.times(positive.variable("x"), positive.variable("y")),
            ],
            "SEMIRING_QUOTIENT_OR_HOMOMORPHIC_IMAGE",
            "PosBool(X)",
            "positive Boolean expressions modulo Boolean-function equivalence",
            "OR with canonical absorption",
            "AND with canonical absorption",
        ),
        (
            "flat_source_support_view",
            flat,
            [
                flat.zero(),
                flat.variable("x"),
                flat.variable("y"),
                flat.plus(flat.variable("x"), flat.variable("y")),
            ],
            "PARTIAL_NONZERO_SUPPORT_VIEW",
            "Vars(N[X])",
            "finite source-variable sets",
            "set union",
            "set union",
        ),
    ]
    targets = []
    for domain_id, carrier, samples, classification, target, carrier_description, addition, multiplication in carriers:
        laws = _law_checks(carrier, samples)
        required_laws = [
            "zero_and_one_distinct",
            "addition_identity",
            "multiplication_identity",
            "addition_commutative",
            "multiplication_commutative",
            "addition_associative",
            "multiplication_associative",
            "left_distributive",
            "right_distributive",
            "multiplicative_zero_annihilation",
        ]
        formal_target = classification in {
            "COMMUTATIVE_SEMIRING_TARGET",
            "SEMIRING_QUOTIENT_OR_HOMOMORPHIC_IMAGE",
        }
        targets.append(
            {
                "domain_id": domain_id,
                "target": target,
                "classification": classification,
                "carrier": carrier_description,
                "zero": carrier.document(carrier.zero()),
                "one": carrier.document(carrier.one()),
                "addition": addition,
                "multiplication": multiplication,
                "axiom_checks": laws,
                "required_commutative_semiring_axioms_pass": all(laws[name] for name in required_laws),
                "counted_as_formal_algebraic_target": formal_target,
                "whole_carrier_semiring_homomorphism_claimed": formal_target,
                "authority": "Green-Karvounarakis-Tannen PODS 2007, proceedings pages 33-34",
                "axiom_check_scope": "executable generating sample set plus frozen authority definition",
            }
        )
    flat = next(item for item in targets if item["domain_id"] == "flat_source_support_view")
    return {
        "schema_version": "formal-target-semantics-audit-v1",
        "status": "FORMAL_TARGET_SEMANTICS_CLASSIFIED",
        "authority": {
            "pods_2007": {
                "sha256": "74e092702db58518afeaf909e1d3380848165b2cb9ae75dc6822b04f66aa5be0",
                "commutative_semiring_definition": "Definition 3.2 and Proposition 3.4, proceedings page 33",
                "homomorphism_condition": "Proposition 3.5, proceedings page 34",
                "nx_universal_property": "Definition 4.1, Proposition 4.2, and Theorem 4.3, proceedings page 34",
                "flat_powerset_calculation": "Section 4 and Figure 5, proceedings page 34",
            },
            "icdt_2001": {
                "sha256": "6c244258ab44a229957a1a16605787f1b9ade11bee0ef6eac8086eb6905e1087",
                "witness_basis": "Section 5 and Definition 6, paper pages 8-9",
                "minimal_witness_basis": "Definition 7, paper page 10",
            },
        },
        "targets": targets,
        "formal_algebraic_target_count": sum(item["counted_as_formal_algebraic_target"] for item in targets),
        "formal_algebraic_target_ids": sorted(
            item["domain_id"] for item in targets if item["counted_as_formal_algebraic_target"]
        ),
        "flat_view_findings": {
            "zero_equals_one": flat["zero"] == flat["one"],
            "multiplicative_zero_annihilation": flat["axiom_checks"]["multiplicative_zero_annihilation"],
            "observed_only_on_nonzero_output_support": True,
            "complete_semiring_homomorphism_supported": False,
            "former_name_why_powerset_accurate": False,
            "replacement_name": "flat_source_support_view",
        },
        "not_evaluated": [
            {
                "target": "Which(X)",
                "classification": "NOT_EVALUATED",
                "reason": "not formally defined in the frozen PODS 2007 authority",
            },
            {
                "target": "Trio(X)",
                "classification": "NOT_EVALUATED",
                "reason": "not formally defined in the frozen PODS 2007 authority",
            },
            {
                "target": "witness-basis Why provenance",
                "classification": "NOT_EVALUATED",
                "reason": "the current flat view does not preserve the set of alternative witnesses",
            },
            {
                "target": "existing Database which-lineage",
                "classification": "NON_SEMIRING_TASK_PROJECTION",
                "reason": "tuple-level source-set result connected separately through Vars(N[X])",
            },
        ],
    }


def render_formal_target_semantics_markdown(audit: dict[str, Any]) -> str:
    rows = []
    for item in audit["targets"]:
        laws = item["axiom_checks"]
        rows.append(
            f"| `{item['domain_id']}` | `{item['classification']}` | "
            f"`{str(item['zero']).lower()}` | `{str(item['one']).lower()}` | "
            f"{item['addition']} | {item['multiplication']} | "
            f"{laws['multiplicative_zero_annihilation']} | "
            f"{item['counted_as_formal_algebraic_target']} |"
        )
    return "\n".join(
        [
            "# Formal target semantics audit",
            "",
            f"Machine status: `{audit['status']}`.",
            "",
            "| Domain | Classification | 0 | 1 | Addition | Multiplication | Zero annihilates | Algebraic target |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
            *rows,
            "",
            "The former `why_powerset` is now `flat_source_support_view`. Its empty set is both additive and multiplicative identity, and it fails multiplicative-zero annihilation for every nonempty value. It is therefore an exact task projection on frozen nonzero output support, not a complete commutative-semiring homomorphism target.",
            "",
            "`Which(X)`, `Trio(X)`, and witness-basis Why provenance remain `NOT_EVALUATED`; no algebraic structure is invented from names alone. Existing tuple-level which-lineage is classified separately as `NON_SEMIRING_TASK_PROJECTION`.",
            "",
            "Authority locations: Green-Karvounarakis-Tannen, PODS 2007, proceedings pages 33-34; Buneman-Khanna-Tan, ICDT 2001, paper pages 8-10.",
            "",
        ]
    )
