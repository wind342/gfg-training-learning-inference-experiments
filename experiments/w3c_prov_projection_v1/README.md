# W3C PROV projection v1 experiment

This independent, falsification-first experiment evaluates whether the frozen `w3c-prov-generation-profile-v1` is an exact strict projection of validated complete generation facts.

Machine statuses:

- `W3C_PROV_PROJECTION_V1_SUPPORTED`
- `W3C_PROV_PROJECTION_V1_EVIDENCE_HARDENING_SUPPORTED`

Both have empty blocking-reason lists. These statuses are scoped to the declared profile, crosswalk, fixture family, frozen authorities, and validators. They do not claim that W3C PROV is incomplete, incorrect, non-general, non-extensible, or unable to represent other facts under another profile.

## Hardened design

The deterministic generator reads four row-level sources, performs selection and a multi-source/multi-output join, records one `ExplicitDisposition`, carries two intermediate supports through `GeneratedOrigin`, and produces byte-stable CSV, JSON, and text outputs.

The two scientific paths are independently executed and traced:

- Candidate process: actual generator -> Core collector -> `ValidatedSnapshot` -> candidate projection -> deterministic PROV-N -> experiment-owned PROV-N parser.
- Native process: actual generator -> `NativeProvCollector` -> qualified PROV-O Turtle -> RDFLib parser -> independent PROV-O normalizer.

The processes are started separately, share no memory objects, and compare only canonical normalized-record hashes after both paths finish. `sys.addaudithook` records imports, opened files, subprocesses, and socket events. The Candidate reads no native output, expected record set, old artifact, callback-answer receipt, or hidden crosswalk. The native path reads no Core Snapshot or Candidate output.

`authority_store_audit_policy_v2.json` classifies every repository file in the experiment and applies content/schema rules. The resulting 78-file scan has zero unclassified files, zero forbidden secondary relation stores, and zero Snapshot/Evidence blobs in PROV. `second_authority_audit.json` is computed from that scan and the runtime trace; it is not a fixed declaration.

## Actual transform-context counterexample

The fourth strict-projection group executes two real integer code paths:

- `generator._execute_left_associative`: `(x + y) + z`, intermediate values `[11, 11]`;
- `generator._execute_right_associative`: `x + (y + z)`, intermediate values `[6, 11]`.

With `x=5`, `y=6`, and `z=0`, both branches produce final value `11`. That value directly enters the ordinary JSON/text output. Branch ID, code path, evaluation order, intermediate-state digest, transform reference, occurrence payload, and operation-result hash are captured from execution. Candidate never reads the branch receipt, and native PROV does not project the profile-external context.

Both snapshots validate and have different IDs. Ordinary output, source/result entities, binding semantics, 51 Candidate records, canonical PROV-N bytes, and normalized native PROV-O remain identical. The declared W3C PROV generation profile does not select these occurrence-specific transformation facts.

## Frozen dependencies and references

Full W3C publications and 53 applicable official test inputs stay in ignored `runtime/`. URLs and hashes are committed under `authorities/`. Parser dependencies are frozen as:

- `rdflib==7.6.0`, wheel SHA-256 `30c0a3ebf4c0e09215f066be7246794b6492e054e782d7ac2a34c9f70a15e0dd`;
- `pyparsing==3.3.2`, wheel SHA-256 `850ba148bd908d7e2411587e247a1e4f0327839c40e2e5e6d05a007ecc69911d`.

They are implementation dependencies, not semantic authorities.

## Reproduce

From the repository root, with proxy variables set if required:

```powershell
python -m venv --system-site-packages experiments/w3c_prov_projection_v1/runtime/venv
experiments/w3c_prov_projection_v1/runtime/venv/Scripts/python.exe -m pip download --only-binary=:all: --dest experiments/w3c_prov_projection_v1/runtime/wheels rdflib==7.6.0 pyparsing==3.3.2
experiments/w3c_prov_projection_v1/runtime/venv/Scripts/python.exe -m pip install --no-index --find-links experiments/w3c_prov_projection_v1/runtime/wheels rdflib==7.6.0 pyparsing==3.3.2
$env:PYTHONPATH = "src;."
experiments/w3c_prov_projection_v1/runtime/venv/Scripts/python.exe -m experiments.w3c_prov_projection_v1.src.bootstrap_references
experiments/w3c_prov_projection_v1/runtime/venv/Scripts/python.exe -m pytest -vv tests experiments/w3c_prov_projection_v1/tests --junitxml=experiments/w3c_prov_projection_v1/runtime/full-test-run-1.junit.xml
experiments/w3c_prov_projection_v1/runtime/venv/Scripts/python.exe -m pytest -vv tests experiments/w3c_prov_projection_v1/tests --junitxml=experiments/w3c_prov_projection_v1/runtime/full-test-run-2.junit.xml
experiments/w3c_prov_projection_v1/runtime/venv/Scripts/python.exe -m experiments.w3c_prov_projection_v1.src.experiment
```

The materializer performs two complete science runs, two authority scans, two Candidate processes, two native processes, and two complete in-memory artifact materializations. It consumes two independent JUnit receipts. Downloaded references, wheels, venv files, and JUnit reports remain ignored.

## Headline machine evidence

- P1: 51 Candidate records equal 51 native records; all FP/FN, field, multiplicity, dangling-reference, duplicate-ID, fabricated-pairing, missing-binding, Cartesian-product, and constraint metrics are zero.
- Relations: 14 Usage, 6 Generation, 14 Derivation, and 3 Association.
- P2: all 4/4 requested strict-projection groups are valid.
- Oracle isolation: 2/2 Candidate and 2/2 native child processes pass; shared mapping helper count is zero.
- Authority store: 78/78 files classified; unclassified and forbidden-store counts are zero.
- Official tests: 53/53 applicable frozen W3C cases pass.
- Controls: the original 32/32 controls remain separate and pass; 30/30 new authority, Oracle, and transform controls fail closed.
- Complete tests: 47/47 and 47/47 in independent processes.
- Protection: Core source/schema, `compat/v2`, `tests/core`, other experiments, and PROV-specific Core fields all have zero changes.
- Artifacts: all 34 required files are present; two materializations are byte-identical.

See `EXPERIMENT_REPORT.md` and `artifacts/evidence_hardening_run_summary.json` for the complete gate.
