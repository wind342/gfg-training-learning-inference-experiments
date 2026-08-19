# Reproducibility notes

The experiment files are preserved, but several frozen checks intentionally
depend on objects that are not ordinary Python source files.

## Source-history checks

Three formal-semantics hardening checks in GF-P02 and three provenance/compatibility
checks in GF-P01 read frozen objects from the history of
`wind342/source-information-continuity` by exact commit identity. They require
access to that repository's Git objects in addition to this companion checkout.
They are preservation checks, not scientific computation paths.

In a normal companion-only clone, the three GF-P01 checks are reported as
explicit `SKIPPED` history-authority checks. The frozen GF-P02 experiment and
test tree is preserved without modification; its three source-history authority
checks are therefore excluded explicitly when only the companion clone is
available. All six checks pass when run in a checkout whose Git object database
contains the source commits identified in `SOURCE_MANIFEST.md`.

The 47 history-independent GF-P02 tests can also be selected directly with:

```console
python -m pytest experiments/provenance_semiring_projection_v1/tests -q \
  -k "not protected_core_protocol_compat_and_core_tests_match_frozen_base and not existing_database_experiment_tree_matches_frozen_base and not pr19_p1_p2_and_protected_trees_are_preserved_from_frozen_head"
```

The local Core and GF-P01 unit set produces 50 passes; its remaining three
checks require the source commits named in `SOURCE_MANIFEST.md` to be present
in the local Git object database.

## External and generated runtime prerequisites

The complete GF-P01 fail-closed runner also expects the prerequisites frozen by
the five native mechanisms:

- the generated TPCH DuckDB runtime database for database which-lineage;
- the pinned Node dependencies and ignored official test payload for Source
  Maps;
- the frozen official W3C PROV test directory; and
- the PyTorch runtime specified by the Autograd experiment.

Their preparation commands and exact versions are documented in
`experiments/five_profile_unified_projection_proof/ENVIRONMENT.md` and in the
five underlying experiment READMEs. Missing prerequisites remain failures or
skips; this companion repository does not replace them with fabricated data.

## Third-party literature authorities

Two author-version papers used as frozen literature authorities during the
GF-P02 preimplementation audits are not redistributed in this public snapshot.
The audit records retain their original local paths and hashes as historical
evidence. Bibliographic identities and official persistent links are listed in
`THIRD_PARTY_AUTHORITIES.md`. These papers are audit references, not inputs to
the executable projection or scientific result computation.

## Generated evaluation bundles

The direct-prediction and reinforcement-learning independent checkers consume
machine bundles created by their formal runners. The repository retains the
checkers, protocol freezes, implementations and tracked result summaries. The
complete frozen TL-P01 result bundle is also distributed in the publication
archive, so its all-run and confirmation metrics can be recomputed without a
fresh model execution. For an experiment without an archived generated bundle,
a fresh checker execution must first run the corresponding reproduction command
to create the required `MANIFEST.json`/aggregate and evidence-manifest inputs;
the checker must fail if those inputs are absent.

RL-E02 follows the same rule. Its runner creates per-seed base-GFG ledgers,
candidate/credit ledgers and sealed policy checkpoints outside the repository.
Its independent checker reconstructs the trace comparator, regenerates and
validates every formal base GFG, replays every sealed policy, and recomputes
aggregate metrics. The compact terminal replay used for counterfactual volume
is also checked for exact equality with full native episode execution.

RL-E03 and RL-E04 retain compact formal artifacts in their experiment
directories. Their formal runners regenerate the base execution GFGs,
credit-discovery GFGs, exact reference adjudications, optimized adjudications
and held-out policy evaluations. Each experiment provides a separate
independent checker and runtime test entry point; the commands and frozen
scientific boundaries are documented in its `README.md` and
`PROTOCOL_FREEZE.md`.

INF-E01's publication bundle similarly permits independent integrity checking
and recomputation of the strict logit-level derived-GFG result. Fresh
regeneration of the underlying gated forwards and rollback interventions still
requires the original historical parameter checkpoints.

## Large base-GFG payloads

The four omitted base-GFG JSON objects are listed in
`EXTERNAL_ARTIFACTS.md`. Any restored copy must match the recorded SHA-256
before use.
