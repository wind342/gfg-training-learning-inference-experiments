# Executable Generation-Fact Graph v2

This experiment preserves the frozen v1 falsification and tests the layered
heterogeneous graph:

```text
G_e = (V_F, V_O, E_I, E_R; Sigma)
```

- `V_F` contains one complete five-coordinate generation fact per
  `GenerationBinding`.
- `V_O` contains concrete execution occurrences, including legitimate
  occurrences with no `GenerationBinding`.
- `E_I` is the exact `realizes_fact` incidence derived from each fact's
  occurrence.
- `E_R` preserves primitive relation endpoint kinds and identities without
  forced lifting.
- `Sigma` is the frozen schema, relation registry, evidence, capture and
  canonicalization contract.

An occurrence node is an execution-skeleton graph node, not a sixth atomic
fact coordinate.

## Reproduction

From the repository root:

```text
python -m experiments.executable_generation_fact_graph_v2.scripts.run_all
python -m experiments.executable_generation_fact_graph_v2.scripts.run_tests
python -m experiments.executable_generation_fact_graph_v2.scripts.finalize
```

`run_all` intentionally reports the test gates as pending. Only `finalize`
may emit `EXECUTABLE_GENERATION_FACT_GRAPH_V2_SUPPORTED`, and only after the
focused, frozen Core and full-repository suites all pass.

The formal scientific claim is exact construction, validation, direct graph
query and projection for the frozen Signal, Order, Scale and Signed Algebra
profiles. In the Order profile, all 56 frozen answers are produced directly
from `ValidatedGenerationFactGraphV2` and compared with an independent
receipt-backed reference; the old order candidate is retained only as
projection-compatibility evidence. Compensation targets are resolved by a
frozen Order policy rather than equated with generic graph reachability.

Recompilation validation detects structural mutation but shares the compiler
implementation and is therefore not treated as an independent semantic
oracle. Weighted shortest path, critical-path optimization and general
performance optimality are not established.
