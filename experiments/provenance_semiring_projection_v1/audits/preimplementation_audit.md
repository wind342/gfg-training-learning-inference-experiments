# Provenance Semiring Projection v1: preimplementation audit

Audit date: 2026-07-22

Status: `PREIMPLEMENTATION_AUTHORITY_AND_SCOPE_FROZEN`

No implementation existed when this audit was recorded. The only allowed authored path is `experiments/provenance_semiring_projection_v1/`.

## Frozen repository authority

- Base branch: `integration/unified-operational-projection-proof-v2`
- Base commit: `7320fe8a2d690fc87da77d0739b432ea1812d63b`
- Core runtime tree: `03fbdce13249f84abe9d8fb605da31cdc36eda27`
- Core protocol tree: `0b4a2608864e771ebca7cdbfad95aabaed2d0723`
- `compat/v2` tree: `7bbb49d18daf7ea99d7633b40c6df5bc002824ca`
- `tests/core` tree: `280cb44d592ae48d986719638980c11e57aab1f9`
- Existing Database source head: `03caa31b8a6abfe6e112a0544071618c689bb11f`
- Existing Database tree, both source and integrated: `64b5365d9a828a645c99b536254a07a2519f0cc0`
- Frozen which-lineage profile SHA-256: `d32b809931644617d763bad597b62904bba3273e7fa62f9afa963fc74387ac40`
- Frozen Database equivalence artifact SHA-256: `337b92bd88f11a51fe88d6dc2c47896c2bed731470016cd503b7fc2b44b4b8f9`
- Frozen Database canonical candidate/reference SHA-256: `a54037abf452d7308f13a27a287b19a3797b5e9ab77bd62efbb48c1a81672360`

The existing Database, Source Map, OpenTelemetry, Core runtime, protocol, compatibility and Core-test trees are read-only. No semiring-specific Core field is permitted.

## Primary theory authority

Todd J. Green, Grigoris Karvounarakis and Val Tannen, “Provenance Semirings,” PODS 2007, pp. 31-40, DOI `10.1145/1265530.1265535`.

- DOI metadata: https://doi.org/10.1145/1265530.1265535
- Author public PDF: https://www.cs.ucdavis.edu/~green/papers/pods07.pdf
- Frozen local file: `audits/authorities/green-karvounarakis-tannen-pods2007-author-version.pdf`
- Download date: 2026-07-22
- Bytes: 262241
- SHA-256: `74e092702db58518afeaf909e1d3380848165b2cb9ae75dc6822b04f66aa5be0`

The author-hosted PDF has ten pages corresponding to formal proceedings pages 31-40. Definition 3.1 and Definition 3.2 are on PDF page 3 / proceedings page 33. The semiring examples for set, bag and positive-Boolean semantics are on the same page. Proposition 3.5, Definition 4.1, Figure 5, Proposition 4.2 and Theorem 4.3 are on PDF page 4 / proceedings page 34. The ACM-hosted binary was not treated as a second theory source, so byte-level differences from the author-hosted copy are not asserted.

The frozen reading is:

- a K-relation has finite support;
- union combines annotations with semiring addition;
- projection sums annotations of tuples that collapse to the same output tuple;
- selection multiplies by a zero/one predicate annotation;
- natural join multiplies annotations of joinable tuples;
- renaming preserves annotations;
- `N[X]` is the positive-algebra provenance semiring over tuple-identity variables;
- coefficients and exponents are material semantic facts, as illustrated by `2s^2 + rs`;
- semiring homomorphisms commute with positive relational evaluation, and every valuation from `X` into a commutative semiring extends uniquely from `N[X]`.

## Frozen scope

The evaluated language is finite positive relational algebra over base relation, selection, projection, renaming, union/union-all behavior, natural or explicit-key equi-join, self-join, identity-distinct equal-valued tuples and multistage compositions.

Difference, negation, antijoin, `NOT EXISTS`, aggregation, recursion, Datalog, windows, NULL three-valued logic, outer join, arbitrary SQL, probabilistic event evaluation, arbitrary semirings, universal query equivalence, the whole provenance literature and all DBMS implementations are excluded.

Every positive conclusion must begin from the boundary: “Within the frozen positive relational-algebra profile...”

## Authority isolation

The ordinary executor may read only base relations and the frozen RA AST and emits only ordinary tuples/CSV/JSON. The Native K-relation process may read the annotated base relations, RA AST and semiring profile, but no Core or Candidate data. The Core Candidate process may read only a `ValidatedSnapshot`, its matching validation, the frozen N[X] profile, the frozen Core crosswalk and the structural canonicalizer; it may not read fixtures, SQL/RA AST, ordinary results, Native output, existing lineage answers or expected polynomials. The comparison process may read only canonical outputs and may not repair either side.

Structural polynomial serialization may be shared. Operator semantics, occurrence-to-sum/product derivation, expected terms, Native answers and source-output maps may not be shared.

## Planned independent paths

1. Ordinary output process: base relations + RA AST → ordinary tuples and exact CSV/JSON.
2. Native K-relation process: independently annotated relations + RA AST → direct N[X] and lower-K results.
3. Core Candidate process: validated Core-only facts → recursively projected N[X].
4. Comparison process: two canonical outputs → exact mismatch accounting without mutation or repair.

## Hypotheses and falsifiers

The working hypothesis is that stable `SourceInformation` identities supply variables, actual bindings grouped by occurrence and outcome supply multiplication, alternative producers supply addition, and `GeneratedOrigin.prior_support_id` supplies adjacent-stage recursion. Evidence, Environment, ExplicitDisposition and operation identity should remain outside N[X].

The experiment must stop rather than widen scope if Core facts cannot express a real derivation, the Candidate needs the RA AST or Native result, Native and Candidate disagree, a legal workload refutes coefficient/exponent handling, five real strictness pairs cannot be built, or any required change leaves the experiment directory.
