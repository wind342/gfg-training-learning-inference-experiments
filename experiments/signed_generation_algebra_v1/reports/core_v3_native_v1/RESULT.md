# Signed Generation Algebra v1 — Result

Final status: **SIGNED_GENERATION_ALGEBRA_V1_SUPPORTED**

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

- `case_1_never_vs_insert_delete` / `case1_never_insert`: bindings=0, positive={"schema_version": "nx-polynomial-v1", "terms": []}, negative={"schema_version": "nx-polynomial-v1", "terms": []}, net={"schema_version": "zx-polynomial-v1", "terms": []}
- `case_1_never_vs_insert_delete` / `case1_insert_then_delete`: bindings=2, positive={"schema_version": "nx-polynomial-v1", "terms": [{"coefficient": 1, "monomial": [{"exponent": 1, "variable": "x_record_a"}]}]}, negative={"schema_version": "nx-polynomial-v1", "terms": [{"coefficient": 1, "monomial": [{"exponent": 1, "variable": "x_record_a"}]}]}, net={"schema_version": "zx-polynomial-v1", "terms": []}
- `case_2_no_update_vs_compensated_update` / `case2_no_update`: bindings=0, positive={"schema_version": "nx-polynomial-v1", "terms": []}, negative={"schema_version": "nx-polynomial-v1", "terms": []}, net={"schema_version": "zx-polynomial-v1", "terms": []}
- `case_2_no_update_vs_compensated_update` / `case2_update_then_compensate`: bindings=2, positive={"schema_version": "nx-polynomial-v1", "terms": [{"coefficient": 7, "monomial": [{"exponent": 1, "variable": "x_scalar_value"}]}]}, negative={"schema_version": "nx-polynomial-v1", "terms": [{"coefficient": 7, "monomial": [{"exponent": 1, "variable": "x_scalar_value"}]}]}, net={"schema_version": "zx-polynomial-v1", "terms": []}
- `case_3_no_delta_vs_opposite_deltas` / `case3_no_increment`: bindings=0, positive={"schema_version": "nx-polynomial-v1", "terms": []}, negative={"schema_version": "nx-polynomial-v1", "terms": []}, net={"schema_version": "zx-polynomial-v1", "terms": []}
- `case_3_no_delta_vs_opposite_deltas` / `case3_plus5_minus5`: bindings=2, positive={"schema_version": "nx-polynomial-v1", "terms": [{"coefficient": 5, "monomial": [{"exponent": 1, "variable": "x_counter"}]}]}, negative={"schema_version": "nx-polynomial-v1", "terms": [{"coefficient": 5, "monomial": [{"exponent": 1, "variable": "x_counter"}]}]}, net={"schema_version": "zx-polynomial-v1", "terms": []}
- `case_4_partial_cancellation` / `case4_add_x_add_y_remove_x`: bindings=3, positive={"schema_version": "nx-polynomial-v1", "terms": [{"coefficient": 1, "monomial": [{"exponent": 1, "variable": "x_item_x"}]}, {"coefficient": 1, "monomial": [{"exponent": 1, "variable": "x_item_y"}]}]}, negative={"schema_version": "nx-polynomial-v1", "terms": [{"coefficient": 1, "monomial": [{"exponent": 1, "variable": "x_item_x"}]}]}, net={"schema_version": "zx-polynomial-v1", "terms": [{"coefficient": 1, "monomial": [{"exponent": 1, "variable": "x_item_y"}]}]}
- `case_5_explicit_disposition_control` / `case5_filter_exclusion`: bindings=1, positive={"schema_version": "nx-polynomial-v1", "terms": []}, negative={"schema_version": "nx-polynomial-v1", "terms": []}, net={"schema_version": "zx-polynomial-v1", "terms": []}

## Strict projection witness

`(0, 0) != (x_record_a, x_record_a)`, while both net projections are zero.
The native final database state is empty for both executions.

## ExplicitDisposition boundary

Misclassified-as-negative count:
`0`.
The filter exclusion remains queryable in the complete state and is interpreted
as `neutral_or_not_applicable` only because the frozen contract says so.

## Release gates

- `addition_laws_passed`: PASS
- `candidate_snapshot_only`: PASS
- `equal_final_output_pair_validated`: PASS
- `equal_net_projection_pair_validated`: PASS
- `existing_test_suite_passed`: PASS
- `explicit_disposition_not_auto_negative`: PASS
- `frozen_core_unchanged`: PASS
- `independent_negative_exact`: PASS
- `independent_net_exact`: PASS
- `independent_positive_exact`: PASS
- `independent_reference_isolated`: PASS
- `multiplication_laws_passed`: PASS
- `multiplicity_exact`: PASS
- `net_projection_additive`: PASS
- `net_projection_multiplicative`: PASS
- `never_vs_cancelled_distinguished`: PASS
- `no_second_authority_store`: PASS
- `occurrence_identity_preserved`: PASS
- `relation_roles_exact`: PASS
- `signed_pair_no_internal_cancellation`: PASS
- `signed_states_different`: PASS

## Answers to the research questions

1. Positive and negative actual occurrences remain in separate unreduced
   components: **True**.
2. Never-happened and happened-then-cancelled are distinguishable:
   **True**.
3. Net projection exactly matches the independent reference:
   **True**.
4. ν preserved tested addition and multiplication:
   **True**.
5. Z[X] was a strict projection in the executed witness:
   **True**.
6. Multiplicity was retained before cancellation:
   **True**.
7. Concrete occurrences remained distinct under equal net values:
   **True**.
8. ExplicitDisposition was not automatically negative:
   **True**.
9. Frozen Core and manuscript paths changed: **False**.
10. Existing test suite passed: **True**.

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
