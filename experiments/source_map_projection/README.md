# ECMA-426 Source Map projection experiment

This is a constructive, falsification-first experiment for ordinary ECMA-426
Source Maps. A deterministic source-to-source transformer emits JavaScript and
synchronous generation receipts. Those same receipts feed two isolated paths:

1. an independent `source-map@0.8.0` native generator/consumer; and
2. existing Core v3 facts followed by a `ValidatedSnapshot`-only projection.

The tested claim is deliberately narrow: original/generated position semantics
are an exact domain projection of selected Core generation facts. No native map
is parsed into Core, no Source Map-specific field is added to Core, and a Source
Map is never treated as the complete generation contract.

## Frozen profile

`contracts/profile_v1.json` fixes ordinary version-3 external sidecar maps,
zero-based lines, JavaScript UTF-16 columns, LF/CRLF, `sourceRoot`,
`sourcesContent`, names, mapped/unmapped segments, and both query directions.
Indexed maps and the other explicit exclusions are not silently generalized.

Official evidence is kept under ignored
`data_private/source_map_projection/official`: the ECMA publication page and
published PDF, a fixed TC39 living-spec checkout/HTML, and
`tc39/source-map-tests` at the exact commits and SHA-256 values in the profile
and report. The published 1.0 HTML endpoint was unavailable through the
configured proxy; that limitation is recorded rather than substituted.

## Install and run

Install the exact dependencies from the experiment directory:

```console
python -m pip install -r experiments/source_map_projection/requirements.lock
pnpm --dir experiments/source_map_projection install --frozen-lockfile
```

Formal execution from the repository root is:

```console
python -m experiments.source_map_projection.scripts.run_all --full
```

The command fails when a mandatory fixture, applicable official test,
projection comparison, query comparison, composition comparison, relation
closure, output-orthogonality check, Oracle-isolation check, or negative
control fails.

It executes output-only, native-only, Core-only, and dual modes; performs two
complete deterministic runs; exercises a 660-mapping medium workload; proves
three same-map/different-Core counterexamples; demonstrates two
same-output/different-source ambiguities; and composes two stages exclusively
through `GeneratedOrigin` bridges.

## Outputs

Stable maps, metrics, official-test results, negative controls, manifests, and
hashes are committed in `artifacts/`. Complete Core snapshots and transient
native files remain under ignored `data_private/`. The authoritative outcome is
`EXPERIMENT_REPORT.md`.

Regression tests are:

```console
python -m pytest tests/experiments/source_map_projection -q
python -m pytest tests/core -q
```
