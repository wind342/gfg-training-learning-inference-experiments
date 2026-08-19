# Formal-semantics hardening preimplementation audit

## Frozen starting point

This additive maintenance branch starts exactly at PR #19 head `f20ff57501b754b111be893565092c0e107c8b73`, tree `906e6e3be5e0085f4c993ef9b6fb5eaf18761e27`. Its machine status is `PROVENANCE_SEMIRING_AS_STRICT_HIERARCHICAL_PROJECTION_SUPPORTED`, with 16/16 gates passing. This audit does not assume that status must survive hardening.

The only writable scope remains `experiments/provenance_semiring_projection_v1/`. Core, protocol, compatibility, Core tests, the Database experiment, and every other experiment remain protected.

## Finding A: the current flat P(X) carrier is not a complete target semiring

At the frozen head, `src/native_lower_k.py::WhyCarrier` uses finite variable sets, `zero() = empty`, singleton base annotations, and set union for both addition and multiplication. `src/semiring_homomorphisms.py::_why` returns the union of every variable occurring anywhere in an N[X] polynomial. The profile names this target `why_powerset` and describes it as `P(X)`.

The frozen PODS 2007 PDF makes the conflict machine-auditable:

- Definition 3.1/3.2 and Proposition 3.4, PDF page 3 / proceedings page 33, require the positive-RA carrier to have distinct `0` and `1` and require a commutative semiring, explicitly including `0*a = a*0 = 0`.
- Section 4, PDF page 4 / proceedings page 34, writes the flat why calculation as `(P(X), union, union, empty, empty)`.
- For any nonempty `A`, `empty union A = A`, not `empty`; multiplication therefore does not have the required annihilating zero. The two designated constants are also equal.
- Proposition 3.5 applies only when both structures are commutative semirings and the map is a semiring homomorphism. It cannot repair the failed premises.

The current carrier is consequently classified as `PARTIAL_NONZERO_SUPPORT_VIEW`, to be renamed `flat_source_support_view`. It can still be tested exactly on the frozen nonzero output support as a task projection `Vars(N[X])`; it must not be counted as a complete semiring homomorphic image.

The phrase “Why provenance” also needs care. Buneman, Khanna, and Tan, ICDT 2001, Section 5 and Definitions 6/7 (PDF pages 9-11, paper pages 8-10) define witnesses, witness bases, and minimal witness bases. A set of witnesses preserves alternative witness structure; the current single flat variable union does not. The hardening will therefore distinguish:

- `flat_source_support_view`: the implemented flat union of source-variable identities;
- `existing Database which-lineage`: a separate tuple-level task projection connected through `Vars(N[X])`;
- witness-basis Why provenance: `NOT_EVALUATED` in this experiment.

The frozen PODS 2007 PDF contains no literal formal targets named `Which(X)`, `Why(X)`, or `Trio(X)`. It does define `PosBool` and the flat `P(X)` calculation. No Which/Trio carrier will be invented; both remain `NOT_EVALUATED`.

## Finding B: report statistics have drifted from formal artifacts

Question 10 of `EXPERIMENT_REPORT.md` hard-codes `155` source-variable observations, `27` outputs, `592` polynomial terms, and `1,112` factor/exponent observations. The frozen formal artifacts instead contain:

| Metric | Report | Formal artifact | Delta |
| --- | ---: | ---: | ---: |
| source-variable observations | 155 | 135 | +20 |
| outputs | 27 | 42 | -15 |
| polynomial terms / coefficient observations | 592 | 197 | +395 |
| monomial factors / exponent observations | 1,112 | 332 | +780 |

The formal sources are `artifacts/nx_field_coverage.json` SHA-256 `02f73faa...` and `artifacts/nx_exact_comparison.json` SHA-256 `e82354f4...`. The report values are stale, manually hard-coded, or mixed from a different execution stage. Artifacts must not be mutated to accommodate the prose. A statistics generator and fail-closed `REPORT_STATISTICS_EXACT_AGAINST_ARTIFACTS` gate are required.

## Finding C: Native and Candidate share algebra and variable helpers

The frozen dependency graph is:

```text
native_nx.py ----> nx_polynomial.py::NXPolynomial <---- candidate_nx.py
      |                                                   |
      +--------> structural.py::variable_for_source <-----+
```

Thus the paths are input-authority-independent but not algebra-implementation-independent. The Native path must receive its own polynomial carrier, coefficient/exponent aggregation, canonical serialization, and SHA-256 variable derivation. Candidate remains on the frozen `NXPolynomial`. Required hardening results are shared algebra helper count `0` and shared variable helper count `0`.

## Preliminary formal classifications

| Target | Classification | Key reason |
| --- | --- | --- |
| bag `N` | `COMMUTATIVE_SEMIRING_TARGET` | Standard natural-number semiring; PODS 2007 p.33 |
| Boolean `B` | `COMMUTATIVE_SEMIRING_TARGET` | OR/AND, false/true, with annihilation; PODS 2007 p.33 |
| `PosBool(X)` | `SEMIRING_QUOTIENT_OR_HOMOMORPHIC_IMAGE` | Positive Boolean expressions modulo Boolean-function equivalence; PODS 2007 p.33 |
| current flat `P(X)` | `PARTIAL_NONZERO_SUPPORT_VIEW` | `0=1=empty`; union multiplication does not annihilate |
| witness-basis Why | `NOT_EVALUATED` | Current representation flattens witness alternatives |
| `Which(X)` | `NOT_EVALUATED` | No formal definition in the frozen main authority |
| `Trio(X)` | `NOT_EVALUATED` | No formal definition in the frozen main authority |
| existing which-lineage / `Vars(N[X])` | `NON_SEMIRING_TASK_PROJECTION` | Tuple-level source-set task result, not an algebraic target |

## Permitted conclusion changes

The flat `P(X)` semiring-homomorphism subclaim must be downgraded. P1, P2, bag, Boolean, PosBool, the Database bridge, lower strictness, joint strictness, ordinary-output equality, and protected-tree equality may be preserved only after fresh execution. Independent Native N[X], zero shared helpers, machine-derived report statistics, and the new highest status may be upgraded only after new machine evidence.

The following conditions block the new highest status: any P1/P2 regression; any Native/Candidate mismatch; any remaining shared algebra or variable helper; fewer than three valid algebraic lower targets; any flat-support misclassification; report/artifact drift; a negative-control failure; nondeterministic hardening runs; protected-tree change; or any need for a semiring-specific Core field.

## Frozen authorities

- Green, Karvounarakis, Tannen, “Provenance Semirings,” PODS 2007, DOI `10.1145/1265530.1265535`, local SHA-256 `74e092702db58518afeaf909e1d3380848165b2cb9ae75dc6822b04f66aa5be0`.
- Buneman, Khanna, Tan, “Why and Where: A Characterization of Data Provenance,” ICDT 2001, DOI `10.1007/3-540-44503-X_20`, local SHA-256 `6c244258ab44a229957a1a16605787f1b9ade11bee0ef6eac8086eb6905e1087`.

Both relevant page ranges were rendered and visually checked in addition to text extraction. Implementation has not started at this audit boundary.
