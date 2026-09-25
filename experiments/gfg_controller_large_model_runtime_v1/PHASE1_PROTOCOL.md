# Runtime architecture first; no scaled-model training in this phase

Scientific baseline d17f4510adf46ebae4fc391382ca6be1134eec03. Work only in this
new experiment directory and a new external evidence directory. Frozen release
commit remains 36dab5ce347dbbdac157ef23205f556606d18294. No old file changes.

The user's latest priority is architecture BEFORE 70M/160M validation. This phase
does not create/train those models and does not claim transport to either scale.
No cadence/probe reduction, no competitor pruning, no change of alpha candidates
or quadratic_complete coordinates. Native fused AdamW recurrence unchanged.

## Certified runtime path

Use two full-size buffers (candidate, direction), not five. On controlled steps,
temporarily point the SAME Parameter objects at copied candidate storage while
native fused AdamW executes, always restore original data pointers in finally.
Original model parameters remain the receiving state for adjudication. Form the
actual rounded delta as candidate minus original, never substitute an analytic
unrounded optimizer direction. After selection, alpha1 copies native candidate
bytes; alpha<1 reuses the now-consumed delta as scaled endpoint. Keep optimizer
moments unscaled. A failed proposal invalidates the step; no continued training
with partly advanced moments after failure.

Boundary core tiles targets and competitors; retains reference weight differences,
float64 arithmetic, explicit float32 coordinate rounding and endpoint association.
No GEMM distributive rewrite (mathematical equivalence is not byte identity).
All competitors remain, streaming four minima. Runtime does not retain full
coordinate arrays, three-dimensional all-competitor differences, or CPU parameter
hashes. Full pre-update logits/head weights are still materialized: limitation
explicitly retained, not a claim of vocab-sharded or fully GPU-native execution.

Keep proven global nested-JVP/shared-derivative path for certification. Separately
test blockwise second-order jet propagation including propagated input curvature
and mixed parameter/input effects, against reference h/h'/h''. Never substitute
blockwise into the certified path if any byte mismatch is observed. This is a
predeclared component-level diagnostic, not a new scientific strategy.

## Gates and checks

Seal source/protocol/checker, all baseline Python hashes, assignment and old
dependency seal before outcomes. Refuse existing raw-output directory.
1. Synthetic boundary tests at shapes (1,2,3), (7,5,17), (53,11,32),
   (212,24,64), (7,4097,65), plus all-zero/tied case. Compare every coordinate,
   competitor order, all four minimum margins, scores and alpha exactly.
2. Native candidate/storage/optimizer checks at four alphas and three consecutive
   steps, with explicit independent displacement arithmetic and byte equality.
3. All 12 existing histories x three existing LRs (.001,.003,.01), unchanged
   batches/seeds/fresh optimizer/cosine200/stop first99. Compare old controller,
   new AUDIT mode, and new RUNTIME mode. No new training histories.
4. AUDIT paired replay checks native proposal, h/h'/h'', coordinates, minima,
   scores, alpha, post-parameters, logits, crossings and final optimizer exactly.
   Bridge to old persisted d17f451 dynamic artifacts too. RUNTIME replay retains
   external test-harness hashes (NOT inside runtime) to establish same trajectory.
5. Independent auditor reopens raw logits/coordinates, redoes strict gap>0,
   decisions, hash chains, crossings and all source/artifact identities.

Runtime equivalence PASS only if all three gates and all 72 new trajectories
(36 in each mode, including initial-success zero-step cases) match the old
scientific oracle. FAIL on any certified-path mismatch; no tolerance substitution.
Blockwise component separately PASS if ALL three arrays are byte-identical at
every tested decision, otherwise PARTIAL/NOT_CERTIFIED with maximum errors and
all failure locations retained. It cannot downgrade equality of the separate
certified global-JVP path, and cannot be advertised as certified if it differs.

Store timings only as diagnostic bridge times: old and audit modes include extra
comparison work and blockwise diagnostics; NOT a controlled speed benchmark.
Record exact two-buffer allocation formula and measured allocation counts.
Do not infer total peak-memory reduction from persistent-buffer reduction alone.
Formal 70M/160M memory/time/scientific comparisons follow only after this gate.

If software fails, preserve failed seal/output, repair note and original source;
re-seal before retry. Results are not retrospectively relabeled or overwritten.
