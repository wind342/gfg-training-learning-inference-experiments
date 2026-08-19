# W3C PROV-DM generation profile evidence-hardening report

## Result

Final machine status: `W3C_PROV_PROJECTION_V1_EVIDENCE_HARDENING_SUPPORTED` with 23/23 mandatory criteria and no blocking reasons. The original `W3C_PROV_PROJECTION_V1_SUPPORTED` result also remains supported.

The PR #12 head `b8e71a84d85dc361889a615d32348b9ac4d0481f` is an ancestor of this maintenance history, its original four commits remain in their original order, and its tree was not rewritten. All changes are confined to `experiments/w3c_prov_projection_v1/`.

P1 remains 51 Candidate records = 51 independently normalized native records: 10 Entity, 3 Activity, 1 SoftwareAgent, 14 Usage, 6 Generation, 14 Derivation, and 3 Association. FP, FN, field mismatch, multiplicity mismatch, fabricated pairing, Cartesian product, dangling reference, duplicate identifier, missing binding, and constraint violation counts are all zero.

All four strict-projection groups are valid: the original Evidence, Environment/OperationResult, and GeneratedOrigin groups plus `actual_transform_context_difference`. The reverse mapping remains non-identifiable within the declared profile.

## Evidence hardening

The second-authority result is now derived from a policy-controlled 78-file repository scan, content/schema inspection, and runtime dependency trace. All files are classified, and forbidden secondary stores, persistent lookup tables, hidden binding crosswalks, expected-answer inputs, receipt-answer indexes, embedded Snapshot blobs, old-artifact reads, and Candidate-to-native reads all count zero.

Candidate and native results are built in separate child processes. Candidate reads the frozen generator and Core collector/projection/parser modules, Core validation modules, and the three frozen Core protocol/schema JSON files. Its runtime projection input is exactly one `ValidatedSnapshot`, followed by the just-created PROV-N bytes for parsing. It reads no committed experiment artifact. Native reads the generator, event/record/native/PROV-O normalizer modules, RDFLib modules and metadata, actual callbacks, and the just-created Turtle bytes. It reads no Core module, Snapshot, Candidate module, Candidate PROV-N, or normalized Candidate output.

AST/import/call analysis finds one shared neutral module, `record_model`, used only for deterministic record ordering. The shared mapping-helper count is computed as the intersection of actually called imported symbols across both mapping closures minus policy-allowed neutral symbols, plus any shared non-neutral module. The resulting count is zero. Semantic-ID, pairing, Usage, Generation, and Derivation construction remain separately implemented.

## Actual transform-context evidence

The generator executes two distinct functions on fixed integers `x=5`, `y=6`, `z=0`:

1. `generator._execute_left_associative` computes `(x + y) + z`, retaining intermediate values `[11, 11]`.
2. `generator._execute_right_associative` computes `x + (y + z)`, retaining intermediate values `[6, 11]`.

Both actual branches return `11`; that value is used directly in ordinary generated output. Execution receipts independently record different branch IDs, code paths, evaluation orders, intermediate-state hashes, transform-reference hashes, occurrence-payload hashes, and operation-result hashes. The receipt is not Candidate input and is not mapped into native PROV.

The two Core snapshots differ because their render occurrence contains different executed transform references and occurrence-specific transform contexts, which content-address the occurrence and downstream evidence/operation facts differently. Both snapshots validate. Their source/result entities, binding semantics, ordinary output, profile-selected Activity semantics, 51 Candidate records, PROV-N bytes, and normalized native PROV-O are identical. This is not a report-only, Evidence-only, or metadata-only mutation.

## Required questions

1. **Is second-authority count zero a machine scan or a handwritten declaration?** It is now automatically computed from policy, two repository scans, content/schema rules, and two-run runtime traces. The result is zero and `SUPPORTED`.

2. **Which files and runtime inputs did Candidate actually read?** Generator/Core/candidate/PROV-N/neutral module code (or their compiled modules), Core validation code, `canonical_serialization_v3.json`, `core_v3_entities.schema.json`, and `core_v3_protocol.json`; runtime inputs are one current `ValidatedSnapshot` and the current in-memory PROV-N bytes.

3. **Did Candidate read native PROV, expected records, or an old artifact?** No. All three measured counts are zero.

4. **Did native reference read Core or Candidate output?** No. Native Core/Snapshot and Candidate-output read counts are zero.

5. **Do Candidate and native share a PROV mapping helper?** No. They share only the neutral `record_model` ordering representation; mapping-helper count is zero.

6. **How is `shared_mapping_helper_count` computed?** The audit builds AST import closures and called-import symbol sets for Candidate and native. It intersects actually called shared symbols, subtracts policy-approved neutral symbols, adds shared non-neutral modules, and fails closed on any remainder.

7. **Are Candidate and native results built in independent processes?** Yes. Two Candidate and two native child processes ran separately, exchanged no relation-answer file, and compared only final canonical hashes.

8. **Do PROV-N and PROV-O still normalize to the same 51 records?** Yes, in both independent process pairs and the complete science runs.

9. **Do the original three strict counterexamples still hold?** Yes, all three remain `SUPPORTED`.

10. **Does the new transform-context counterexample come from different code paths?** Yes: `_execute_left_associative` and `_execute_right_associative` execute distinct branches and expressions.

11. **Are their actual intermediate states different?** Yes: `[11, 11]` versus `[6, 11]`, with different intermediate-state SHA-256 values.

12. **Are ordinary output and PROV still identical?** Yes. Output bytes/metadata, Candidate records, PROV-N bytes, and normalized native PROV-O are identical.

13. **Why are the complete Snapshots different?** Executed transform reference, plan digest, evaluation order, branch/code path, intermediate state, occurrence payload, occurrence identity, and execution-derived operation/evidence facts differ.

14. **Is the new counterexample only a metadata mutation?** No. The value used in output is produced by two actually executed functions with different intermediate computations. `report_only_mutation=false` and `metadata_only_mutation=false`.

15. **Were any PROV-specific Core changes introduced?** No. Core schema/source, Core tests, `compat/v2`, generator-name Core branching, and PROV-specific Core fields all remain zero.

16. **Do the original W3C constraints and official tests still pass?** Yes. PROV-N/qualified PROV-O validation passes and all 53/53 applicable frozen W3C cases pass.

17. **Does the original scientific conclusion survive hardening?** Yes. P1 is unchanged, all four P2 groups are valid, output orthogonality passes, the original 32 controls pass, all 30 new controls fail closed, and two full test runs each pass 47/47.

## Scoped conclusion

“In addition to evidence-, environment-, operation-result-, and reintroduction-specific differences, two validated executions followed different actual transformation paths and retained different occurrence-specific transformation contexts while producing identical ordinary outputs and identical W3C PROV profile projections. The strictness of the tested projection therefore extends beyond omitted audit metadata to concrete execution-specific generation facts.”

The declared W3C PROV generation profile does not select these occurrence-specific transformation facts. This report does not claim that W3C PROV cannot express transformation context; a different declared profile or extension could select additional facts.
