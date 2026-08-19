# ECMA-426 Source Map projection experiment report

## Outcome

Within the frozen ordinary, non-indexed JavaScript Source Map profile, the experiment **SUPPORTED** the claim that ECMA-426 mappings are an exact cross-space projection of the selected Core v3 generation facts, and **SUPPORTED** the stronger falsification claim that the projection is strict and lossy. It does not claim that a Source Map is a complete generation contract.

> Source Map equality does not imply generation-fact equality.

## Hypotheses

| Hypothesis | Status | Exact evidence |
|---|---|---|
| Native ordinary map equals Core-only projection | SUPPORTED | 685 mapping segments across adversarial, Unicode/CRLF, minified, and 660-segment medium cases; byte-equivalent map documents and exact normalized records |
| Native and projected bidirectional query behavior agrees | SUPPORTED | 1385/1385 queries exact; 0 false positives, 0 false negatives, 0 name/source/position mismatches |
| Projection is strict/lossy | SUPPORTED | 3 independent same-map/different-Core counterexamples |
| Result bytes alone identify the source | NOT_SUPPORTED | 2 same-output/different-source-map ambiguity cases |
| Two-stage relations compose through GeneratedOrigin | SUPPORTED | 5 mappings exact; 0 broken bridges, ambiguity, cycles, invented mappings, or direct shortcuts |
| Output is orthogonal to metadata mode | SUPPORTED | Four modes are byte-identical; output-only recorded 0 receipts; 0 metadata token hits |
| Full ECMA-426 surface is established | PARTIALLY | Official non-indexed tests 80/80 passed; 19 indexed-map cases and the declared non-JavaScript surfaces are excluded; published 1.0 HTML bytes were unavailable |

## Frozen scope and exclusions

The profile covers ordinary version-3 external sidecar maps, zero-based lines, JavaScript UTF-16 columns, LF/CRLF, names, `sourceRoot`, `sourcesContent`, mapped and unmapped segments, and generated-to-original plus original-to-generated queries. Indexed maps/sections, WebAssembly, CSS, DevTools UI, stack traces, remote source retrieval, `sourceMappingURL` parsing, proposal fields, and arbitrary compilers are excluded.

The official test identity is commit `2965987bf4c96afa400c9356c8e620cb340aaee2`. The living spec identity is commit `62f8e694b62f5e6708523dc97563580bbf17591c`. The published PDF SHA-256 is `3a9092125d8ae2a5a9809ff8de38d366aa4f41aaa71ae2a2d076fa90058c67c3`.

Unavailable evidence: `https://426.ecma-international.org/1.0/index.html` could not be retrieved through the configured proxy because repeated requests ended in TLS handshake/HTTP 502 errors. It was not silently substituted. The official publication page, published PDF, living HTML/repository, and fixed official tests were separately hashed and verified.

## Exact experiment metrics

- Official tests: 80/80 applicable passed; 19 indexed cases excluded from 99 total.
- Negative controls: 30/30 produced the frozen reason code; no partial output, repair, or frozen-input mutation.
- Medium workload: 660 mappings from 3 sources (minimum required: 600).
- Multi-stage: 5 M1, 5 M2, 5 composed mappings, 5 GeneratedOrigin bridges.
- Determinism: 2/2 normalized complete scientific runs byte-identical; digest `f721d3802c713251ae7fa3e0ea36772b4f77f917714332c34cb31261358d68f6`.
- Core schema/source/compat/core-test changes: 0 files.
- Secondary authority mapping stores: 0.

## Core usage

No Core v3 schema or protected Core implementation change was required. Synchronous transformer receipts create ordinary `SourceInformation`, `GenerationOccurrence`, `PerceptualSupport`, `ExplicitDisposition`, `GenerationBinding`, evidence, operation results, and `GeneratedOrigin` records. `Π_SM` accepts only a validated Core snapshot. It selects exactly one `source_map_anchor:` relation for mapped support, retains unmapped generated anchors, and fails closed on duplicate or conflicting anchors. Dispositions and secondary `participation:` bindings intentionally do not project.

Complete snapshots and transient native files are retained only under ignored `data_private/source_map_projection/formal_runs`; committed artifacts contain stable results and hashes, not original/private data.

## Reproduction

```console
python -m experiments.source_map_projection.scripts.run_all --full
python -m pytest tests/experiments/source_map_projection -q
python -m pytest tests/core -q
```
