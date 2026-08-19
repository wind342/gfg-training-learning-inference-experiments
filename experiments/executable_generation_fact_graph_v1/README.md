# Executable Generation-Fact Graph v1

Final status:
`EXECUTABLE_GENERATION_FACT_GRAPH_V1_NOT_SUPPORTED`.

The original fact-only vertex hypothesis was falsified. The frozen order
relation sidecar contains mandatory occurrence-level primitive relations whose
endpoints have no `GenerationBinding`, and the scale profile contains
occurrences with multiple fact mappings. The v1 compiler therefore fails
closed instead of dropping, fabricating, reattaching, or Cartesian-expanding
relations. See `artifacts/V1_RESULT.md` for the exact machine census.

This experiment compiles validated Core v3 atomic generation facts and
validated inter-fact relations into one typed, attributed, run-scoped
multigraph:

```text
G_e = (V_e, E_e)
```

Every `GenerationBinding` instance becomes exactly one graph node. Edges are
either adjacent-stage `GeneratedOrigin` dependencies, validated primitive
relations, or explicitly traceable derived relations. The graph is queryable
only after fail-closed validation.

The implementation does not modify Core v3, add a sixth fact coordinate,
materialize a global transitive closure, or treat a report as an authority.
The Signal-only graph projects exactly back to its admitted atomic state.
The complete order primitive relation store cannot project into the v1
fact-only vertex model; this failed gate is the central v1 result. Signed
Generation Algebra remains an exact downstream projection of the admitted
fact state.

Run the complete experiment:

```console
python -m experiments.executable_generation_fact_graph_v1.scripts.run_all
```

Run tests:

```console
python -m pytest experiments/executable_generation_fact_graph_v1/tests
```

The frozen scope is the real multi-stage Signal pipeline, the real
order/refund/freeze SQLite and multiprocessing workflow, the frozen signed
effect contract, and the 10,000-occurrence/30,000-fact controlled scale.
