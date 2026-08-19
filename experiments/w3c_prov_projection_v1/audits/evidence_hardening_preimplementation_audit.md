# W3C PROV evidence-hardening preimplementation audit

This is a read-only audit of PR #12 at head `b8e71a84d85dc361889a615d32348b9ac4d0481f` (tree `c811c2ab8236eb00f4ca8f6c743163c8aa4fec20`), based on `main` at `e00144b6b47504287c2d16f20b064da81e43f1cc`. The isolated maintenance worktree was clean before these two audit files were created. The primary worktree already contained five unrelated changes and was not modified.

The four original PR commits are retained in order. This audit distinguishes an evidence-generation weakness from a scientific counterexample; it does not presume that the existing `SUPPORTED` status will survive hardening.

## 1. Current second-authority evidence

The present result is a hand-constructed dictionary. In `src/experiment.py:189-196`, `persistent_candidate_lookup_table_count`, `hidden_binding_crosswalk_count`, `expected_prov_document_count`, `receipt_answer_index_count`, and `second_authority_count` are assigned zero, and `status` is assigned `SUPPORTED`. The committed `artifacts/second_authority_audit.json` contains those assignments.

Those values are not currently derived from repository enumeration, artifact-role classification, content/schema scanning, runtime reads, candidate input dependencies, or native-reference input dependencies. The materializer runs the science paths and legacy controls at `src/experiment.py:86-101`, then constructs the second-authority object without passing any measured scan inputs. This is an audit-implementation insufficiency, not a scientific falsification of P1 or P2.

## 2. Current Oracle-isolation evidence

`src/experiment.py:38-46` implements real AST import extraction. The booleans `native_imports_core`, `native_imports_candidate`, and `candidate_imports_native` are computed from that extraction at `src/experiment.py:169-174`. In contrast, `shared_mapping_helper_count` and `expected_answer_read_count` are directly assigned zero at `src/experiment.py:177-178`.

The candidate contract is real: `project_snapshot` accepts and type-checks one `ValidatedSnapshot` at `src/candidate_projection.py:35-40`, then independently constructs its entities, activities, Usage, Generation, Derivation, and Association records at `src/candidate_projection.py:46-214`. The native collector imports neither Core nor candidate, receives callbacks at `src/native_reference.py:36-73`, builds its own normalized relations at `src/native_reference.py:75-188`, and serializes qualified PROV-O at `src/native_reference.py:190-241`. Neither path currently has measured runtime-read evidence.

The current `run_full` is not a process-level isolation proof. It creates both collectors in one process and supplies both to one generator invocation at `src/science_runs.py:28-36`; `_emit` forwards the same callback event object to each sink at `src/generator.py:24-26`. This remains useful output-orthogonality evidence, but separate subprocess executions and read traces are required for Oracle isolation.

## 3. Shared foundations

Legal neutral sharing:

- Event dataclasses and the callback protocol (`src/events.py:7-69`).
- The frozen deterministic generator fixture (`src/generator.py:37-139`). Independent runs may use the same fixture version but must not share process memory or each other's outputs.
- Normalized-record ordering and canonical JSON (`src/record_model.py:8-24`), which contain no PROV mapping knowledge.

The paths do not currently share semantic-ID or relation-construction helpers: candidate implementations are at `src/candidate_projection.py:16-32,46-214`, while native implementations are at `src/native_reference.py:16-33,75-188`.

Forbidden sharing includes any Core-to-PROV or callback-to-PROV crosswalk implementation, semantic mapping table, binding-pairing implementation, relation-ID answer, Usage/Generation/Derivation constructor, or expected-record builder. The hardened audit must compute such sharing from AST/import/call evidence rather than assert zero.

## 4. Existing strict-projection counterexamples

The existing groups at `src/science_runs.py:47-121` change these complete facts:

1. `evidence_profile_external_difference` changes evidence extraction details and the resulting EvidenceRecord, EvidenceLink, and GenerationBinding identities (`src/science_runs.py:50-55`).
2. `environment_and_operation_result_difference` changes EnvironmentRecord, manifest dependency hashes, and GeneratorOperationResult operation name (`src/science_runs.py:56-61`).
3. `generated_origin_bridge_difference` changes GeneratedOrigin identity and profile-external payload plus producer operation result (`src/science_runs.py:62-67`).

All three are legitimate for their declared omitted complete facts. None changes an actual transform algorithm, branch, parameter, or executed code path. `GeneratorVariant` currently contains only evidence, environment, operation-name, and bridge-detail variants (`src/events.py:72-77`); the generator computes the same intermediate rows and outputs without a variant-dependent transform branch (`src/generator.py:84-139`). The current validity gate checks output, PROV, snapshot identity, and path validity but no branch or intermediate-state difference (`src/science_runs.py:71-99`).

## 5. Required actual-transform counterexample

The hardening implementation should execute two real deterministic integer branches: `left_associative` computes `(x + y) + z`, and `right_associative` computes `x + (y + z)`. The branch ID, executed code path, intermediate values, transformation-plan digest, and execution summary must be captured from the executed branch—not added later as a report label.

The following declared-profile facts must remain equal: occurrence stage/type/stable key/index, operation type, Entity and Agent identities, Usage, Generation, Derivation, Association, role, ordinal, canonical PROV-N bytes, and normalized PROV-O records. Source and result entities, ordinary output bytes, and GenerationBinding semantics must also remain equal.

Complete Γ must differ in actual execution facts, including `GenerationOccurrence.transform_reference`, occurrence-payload evaluation context, executed branch, intermediate-state digest, transformation-plan digest, operation-result execution summary, and therefore snapshot identity. Branch receipts may prove execution but must not be candidate input, native projection input, or a relation-answer source.

## 6. Claim separation and implementation gate

- Scientific conclusion: existing P1/P2 evidence is not invalidated merely by an audit gap, but all conclusions must be rerun without assuming support.
- Candidate implementation: currently maps one ValidatedSnapshot locally; runtime dependency isolation remains unmeasured.
- Native reference: currently maps callbacks locally; runtime dependency isolation remains unmeasured.
- Audit implementation: direct zero assignments are inadequate and must become fail-closed measurements.
- Runtime dependency: current dual callback is in-process and must be supplemented with independent subprocess traces.
- Strict-projection counterexample: the fourth group must arise from different actual transform execution, not metadata-only mutation.

No implementation code was modified before this audit commit.
