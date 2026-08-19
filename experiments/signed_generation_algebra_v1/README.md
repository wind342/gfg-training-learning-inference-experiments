# Signed Generation Algebra v1

This independent exploratory experiment tests one bounded construction:

```text
complete generation state Γ
  -> unreduced signed generation algebra A±(Γ) = (P+, P-)
  -> net ring projection ν(P+, P-) = P+ - P- in Z[X]
```

`P+` and `P-` are canonical finite `N[X]` polynomials. Equal terms in the
two components are never cancelled inside `SignedPair`. Cancellation occurs
only in `net_projection`.

The sign is not a sixth generation-fact coordinate. The frozen domain
contract interprets existing `transform_reference`, concrete occurrence,
relation role and declared effect identity. `ExplicitDisposition` is not
negative by default.

## Real executions

The five cases run:

1. an empty SQLite table versus an actual insert followed by an actual delete;
2. an unchanged SQLite scalar versus an actual update followed by a
   compensating update;
3. an unchanged counter versus actual `+5` and `-5` operations;
4. actual multiset operations `+x +y -x`;
5. an actual filter rejection captured as `ExplicitDisposition`.

Each completed operation calls the collector synchronously. The collector
creates Core v3 occurrences, outcomes, atomic bindings and evidence before
the execution is finalized as a validated snapshot.

The candidate query reads only `ValidatedSnapshot` through the registered
Core `QueryEngine` paths. The independent reference is a pure-state evaluator
of the frozen operation sequence; it imports neither Core nor the candidate
algebra.

## Reproduction

From the repository root:

```console
python -m pytest tests/experiments/signed_generation_algebra_v1 -q
python -m experiments.signed_generation_algebra_v1.run_tests
python -m experiments.signed_generation_algebra_v1.run_experiment
```

The compact test result and all machine/human reports are written under
`reports/core_v3_native_v1/`.

## Claim boundary

This v1 is limited to its frozen deterministic operation family. It does not
modify Core, infer signs without a domain contract, equate
`ExplicitDisposition` with a negative effect, or claim literature novelty or
a general signed-provenance theory.
