# Signed Generation Algebra v1 — Frozen Design

## Immutable boundary

The experiment leaves the frozen atomic fact unchanged:

\[
f=(u,\tau,\bar\omega,z;\rho).
\]

The candidate algebra is an interpretation of complete facts, not another
authoritative fact store. The complete fact identity retains source,
transformation, occurrence, outcome, role, scope, multiplicity and binding
identity. Algebraic effect identity separately names the quantity acted on by
positive and negative occurrences.

## Algebra

For `S = N[X]`:

\[
(p^+,p^-)\oplus(q^+,q^-)
  =(p^++q^+,p^-+q^-)
\]

\[
(p^+,p^-)\otimes(q^+,q^-)
  =(p^+q^+ + p^-q^-,p^+q^- + p^-q^+).
\]

Zero is `(0,0)`, one is `(1,0)`, and optional sign reversal swaps the
components. No reduction relation is applied between components.

The net projection is:

\[
\nu(p^+,p^-)=p^+-p^-\in\mathbb Z[X].
\]

The required strictness witness is `(0,0) != (x,x)` with equal zero net
projections.

## Authority paths

```text
frozen operation sequence
  -> native SQLite / counter / multiset / filter execution
  -> synchronous completed-operation receipt
  -> Core v3 occurrence + support/disposition + binding + evidence
  -> ValidatedSnapshot
  -> registered QueryEngine paths
  -> frozen signed-effect interpretation
  -> SignedPair and net projection
```

The isolated reference takes a separate path:

```text
frozen operation sequence
  -> independent pure-state evaluator
  -> expected final output and canonical positive/negative/net polynomials
```

The comparator, not either computation path, checks exact equality.

## Repository audit decision

At the task baseline, the hardened canonical `N[X]` experiment existed on the
diverged repository branch
`maintenance/provenance-semiring-formal-semantics-hardening-v1`, not in the
starting branch. This experiment therefore preserves the same canonical
`nx-polynomial-v1` serialization and normalization rules locally without
merging unrelated branch history or changing an existing experiment.

The current Core API supports the experiment without modification:

- `GenerationOccurrence.transform_reference` carries the realized operation;
- `GenerationBinding` preserves source, occurrence, outcome and role;
- outcomes are typed support or `ExplicitDisposition`;
- evidence and successful operation closure validate every binding;
- `QueryEngine` returns exact binding relations from `ValidatedSnapshot`.

## ExplicitDisposition boundary

The filter exclusion is a real complete fact and remains queryable. Its
contract rule is explicitly `neutral_or_not_applicable`. The implementation
contains no fallback that maps outcome kind `disposition` to the negative
polynomial.

## Limitations

- Contract correctness is assumed for this frozen domain profile and tested
  against an independent execution model; Core does not discover sign.
- Aggregate polynomials retain coefficients but not binding identifiers.
  Occurrence and binding identity remain in the complete state and relation
  reports.
- The operation family does not cover arbitrary transactions, concurrency,
  recursion, SQL difference or every possible compensation protocol.
