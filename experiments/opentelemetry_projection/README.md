# OpenTelemetry Trace Projection Experiment

This experiment asks a falsifiable question: within the controlled,
deterministic relational executor, can the occurrence/execution/causal part of
an OpenTelemetry trace be projected exactly from an already validated Core v3
Snapshot, without adding an OpenTelemetry authority to Core?

It is not a general OpenTelemetry integration and it does not claim that a
trace is the complete generation-fact contract.

## Frozen mapping

The mapping is fixed before comparison:

| Core/runtime fact | Canonical OTel projection |
| --- | --- |
| One query run | One root span |
| One `GenerationOccurrence` | One child span |
| `occurrence_index` | Deterministic logical order |
| `occurrence_stage` | Selected stage attribute |
| `transform_reference.operator_type` | Span name and operation-type attributes |
| Support outcome or explicit disposition | Selected outcome attribute and event |
| GeneratedOrigin → prior support → producing occurrence | Span Link |
| SourceInformation → occurrence → outcome binding | Deliberately not projected |

Every occurrence span has the query-run span as its parent. Cross-stage causal
predecessors are Links because one occurrence may consume multiple prior
outputs and Core does not select one of them as a unique parent. Binding edges
never become spans or parent edges.

Random `trace_id`, `span_id`, and wall-clock timestamps are normalized away.
Their deterministic semantic replacements are derived from the run ID and
stable occurrence identity. Names, parent identities, every Link (including
multiplicity), status, selected attributes, selected events, and logical order
remain in the comparison. Duplicate semantic span candidates fail before any
dictionary lookup.

The independent projection schema is
[`src/canonical_otel.schema.json`](src/canonical_otel.schema.json). It uses
`additionalProperties: false`, so source, support, binding, lineage, and Core
IDs cannot leak into the narrow trace projection.

## Independent paths

The experiment compares three paths:

1. Official Python SDK spans created synchronously inside actual executor
   capture callbacks.
2. `ValidatedSnapshot` + `SnapshotValidation` → canonical OTel trace.
3. `ValidatedSnapshot` + `SnapshotValidation` → immutable database-domain
   projection → canonical OTel trace.

The direct projector does not import native capture, the Oracle, expected
fixtures, tests, or artifacts. The database-to-OTel projector does not read a
Snapshot or native span. Static checks, runtime traps, and native-record
deletion tests enforce those boundaries.

## Reproduction

Python 3.12.10 is fixed for the recorded run. The official OTel wheels and all
of their transitive dependencies used here are hash-locked:

```powershell
python -m pip install -e .
python -m pip install -r experiments/database_lineage/requirements.lock
python -m pip install --require-hashes -r experiments/opentelemetry_projection/requirements.lock
python -m experiments.opentelemetry_projection.scripts.run_experiment
python -m experiments.opentelemetry_projection.scripts.run_tests
```

The formal command expects the database experiment's reproducible SF0.01
DuckDB fixture at
`experiments/database_lineage/runtime/tpch_sf_0_01.duckdb`. Pass another path
with `--database-path` if needed.

For a fast development-only run that omits the formal workload:

```powershell
python -m experiments.opentelemetry_projection.scripts.run_experiment --skip-formal
```

This option is not used for the recorded final result.

## Official definitions

Definitions were taken only from official sources, accessed 2026-07-21:

- [OpenTelemetry Specification 1.59.0](https://opentelemetry.io/docs/specs/otel/)
- [Tracing API: parent Context, Links, events, and status](https://opentelemetry.io/docs/specs/otel/trace/api/)
- [Tracing SDK and SpanExporter](https://opentelemetry.io/docs/specs/otel/trace/sdk/)
- [Official Python SDK export API](https://opentelemetry-python.readthedocs.io/en/stable/sdk/trace.export.html)
- [Official OpenTelemetry Python repository](https://github.com/open-telemetry/opentelemetry-python)

Pinned packages are `opentelemetry-api==1.44.0`,
`opentelemetry-sdk==1.44.0`, and the SDK-required
`opentelemetry-semantic-conventions==0.65b0`.

## Scope

Included: deterministic in-process relational execution, one root span, one
span per occurrence, parent cardinality, causal Links, status, selected
attributes/events, and output orthogonality.

Excluded: metrics, logs, baggage, sampling, remote propagation, collectors,
vendor exporters, full semantic conventions, arbitrary asynchronous or
distributed causality, and recovery of complete generation facts from a trace.

OpenTelemetry trace equality does not imply generation-fact equality. The two
strict-projection counterexamples in the artifacts have different sources,
bindings, direct relations, and lineage answers but identical normalized
traces.

See [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md) and the machine-readable
[`artifacts/`](artifacts/) directory for exact results.
