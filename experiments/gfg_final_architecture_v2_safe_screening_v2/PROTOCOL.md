# Phase A engineering repair: fixed-shape FP32 evaluation

User authorised repair after v1 stopped at step12. The v1 source/protocol/failure
authority is retained byte-for-byte. New output ROOT, no overwrite or rebranding
of that run. Parent remains 41ca46bef060815d2f106f8235764fb8b03e67b7.

Change only the numerical affine operator: every Linear head forward uses a
256-row FP32 tile, zero padding the last tile, TF32/autocast disabled. Both
full checker and sparse frontier evaluator use exactly this primitive. The
original full-batch arithmetic is not a reliable batch-invariant authority.
Do not promise historical head-byte identity; report it separately.

The production Hildreth function is called with a LOCAL dependency-injected
affine primitive (identical function bytecode, copied globals, no imported
module mutation). Selection, EPS, reserve formula, dual coordinates, sweeps,
fixed relation floors, AdamW and data are untouched. The v1 conservative
certificate and anchor refresh rule are unchanged. Selected constraint rows
remain fully evaluated. No tolerance widening, unsafe commit or old loss.

Preflight on the actual task9 head: verify fixed full/gathered logits identical
for several deterministic row subsets, including changed row positions and a
partial last tile. Check task10 entry counts against historical K. If entry
counts differ, stop and report rather than silently changing K. Verify a
one-step full-reference vs sparse-controller differential before formal run.
The known offending step12 native candidate will be reached in the new run;
all candidate checks continue to record variable-shape historical differences.

Exactly one task10 run of 210 scheduled updates from the same task9 head and
moments. Every receiving/native/sweep/commit candidate has an isolated full
fixed-tile observer and a variable-shape historical numerical diagnostic.
Only fixed-tile sparse outputs feed the controller. Count/sign mismatch to
fixed full evaluator aborts. Every scheduled update has the unchanged production
algorithm with fixed affine as a paired reference; all decisions and head bytes
must match this reference. Original historical trajectory/head/prediction
comparisons are reported, not used to steer training.

All v1 certificate, memory, test isolation, 100-sweep and timing conditions still
apply. Count useful exact rows AND padded rows (256*ceil(batch/256)), full fixed
reference tile rows, and FP64 anchor refresh rows. No assertion that row savings
are FLOPs or speedup. Include all screening/refresh costs. This is a numerical
engineering revision, not a change in capability-floor meaning.

Final test evaluated once after successful completion. No Banking77 until this
phase is reported. Store compact traces/hashes and one final head, not BERT or
per-step large model banks. Preserve any failure without automatic tuning.
