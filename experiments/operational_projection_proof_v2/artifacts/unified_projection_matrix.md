# Unified Projection Matrix

| Domain mechanism | P1 | P2 | P3 | P3 subtype | Declared scope |
|---|---|---|---|---|---|
| Database which-lineage | SUPPORTED | SUPPORTED | NOT_APPLICABLE | domain root/wider projection | fixed deterministic tuple-level profile |
| OpenTelemetry trace | SUPPORTED | SUPPORTED | SUPPORTED | cross-domain hierarchical projection | deterministic in-process occurrence/execution/causal shadow |
| ECMA-426 Source Map | SUPPORTED | SUPPORTED | SUPPORTED | multistage generation composition | ordinary non-indexed JavaScript profile |

The ordinary non-indexed Source Map profile is `SUPPORTED`; the full ECMA-426 standard surface remains `PARTIAL` because indexed maps and declared non-JavaScript surfaces are excluded.

Conjunctive matrix status: `SUPPORTED`.
