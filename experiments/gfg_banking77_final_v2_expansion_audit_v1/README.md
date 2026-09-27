# Banking77 nonlinear Final V2: zero-update class-expansion audit

Training completed for both arms through task11. Verification **PASS**: all18
boundaries independently recomputed and all7,560 continuation updates replayed
without the production observer. See `results/AUDIT.json` and `results/FINAL.json`.

Base: `534130d8995fbf9f6cec176afb915f1a56845702`.
The original task2 controller is imported unchanged. Each arm continues from
its own verified task2 head and AdamW checkpoint. No older incorrect full-run
checkpoint is used. No main, frozen tag, previous source, result or artifact is
changed. This is an observer-only study, not expansion protection.

## Three answers

1. **Yes.** Class expansion damages old algorithmic controls without a training
   update on both corrected nonlinear Final V2 trajectories.
2. **The effect can be catastrophic on these controls.** The 210 arm has a
   158 -> 19 boundary (139 losses, 22 newly-zero classes). The 630 arm has
   153 -> 0 (32 newly-zero classes) and 221 -> 9 (212 losses, 25 newly-zero
   classes). These are not merely a few isolated flips.
3. **Every observed flip is strict new-class competition.** Old-class logits
   are unchanged; a newly admitted class has a larger logit than the true old
   class. Parameter/moment/counter/gradient/RNG state is unchanged during each
   audit. No optimizer step, backward or cutter intervention occurs inside it.
   Historical training created the boundary state, but no update at that boundary
   causes these flips. The audit does not prove which earlier parameter changes
   created the competing logits, or attribute their history solely to W1.

## Exact readout isolation

Frozen BERT cache; MLP 768 -> 256 -> GELU -> 77. The head already has all 77
outputs. Expansion changes only the allowed class slice, not model dimensions.
For each boundary, independent scratch buffers compute all77 logits using the
existing fixed256 FP32 arithmetic. BEFORE and AFTER are slices of that same
output. A second full forward must reproduce the bytes exactly.

For old true label y,
`margin_after = min(margin_before, true_logit - max_new_class_logit)`.
Therefore pure expansion cannot turn a strict-margin wrong prediction correct.
The observer asserts monotone margins and zero wrong->correct. Ties are separately
distinguished from strict new-class steals and from argmax accuracy. No ties
occurred in these runs.

Identity checks cover the live head, shadow head, both gradient buffers, AdamW
moments, counters, hyperparameters, module mode and Python/NumPy/Torch/CUDA RNG.
The observer uses no-grad forward/readout only. K is independently formed by
the original production Bank **after expansion**; it is not taken from the
observer or replaced by pre-expansion counts. Equality to the observed after
counts is checked, not used to choose K.

## 210 updates/task

Ten controls per arrived old class. Correctness means strict margin > 0.
Relations-to-zero counts only classes with positive BEFORE count and zero AFTER.

| Boundary | Old controls | Correct before | Correct after | Expansion loss | C->W | W->C | Relations losing | Relations to zero |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T2->T3 |140|91|85|6|6|0|4|0|
| T3->T4 |210|125|49|76|76|0|18|5|
| T4->T5 |280|86|86|0|0|0|0|0|
| T5->T6 |350|129|115|14|14|0|10|4|
| T6->T7 |420|158|19|139|139|0|30|22|
| T7->T8 |490|60|60|0|0|0|0|0|
| T8->T9 |560|112|112|0|0|0|0|0|
| T9->T10 |630|159|159|0|0|0|0|0|
| T10->T11 |700|212|212|0|0|0|0|0|

## 630 updates/task

| Boundary | Old controls | Correct before | Correct after | Expansion loss | C->W | W->C | Relations losing | Relations to zero |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T2->T3 |140|108|108|0|0|0|0|0|
| T3->T4 |210|155|151|4|4|0|3|0|
| T4->T5 |280|200|99|101|101|0|23|2|
| T5->T6 |350|153|0|153|153|0|32|32|
| T6->T7 |420|63|63|0|0|0|0|0|
| T7->T8 |490|116|116|0|0|0|0|0|
| T8->T9 |560|170|167|3|3|0|2|0|
| T9->T10 |630|221|9|212|212|0|28|25|
| T10->T11 |700|67|63|4|4|0|3|1|

## Cumulative events

| Metric |210|630|
|---|---:|---:|
| Expansion-only correct->wrong events |235|477|
| Distinct control IDs flipped |228|466|
| Expansion-only wrong->correct events |0|0|
| Largest single-boundary loss |139|212|
| Largest single-class loss (out of10 controls) |9|10|
| Class-positive-to-zero events |31|60|
| Boundaries with zero expansion loss |5 /9|3 /9|
| New-class tie events |0|0|

Cumulative events can include the same control/class at multiple boundaries;
they are not asserted to be unique examples/classes. All reported effects refer
to algorithmic controls, not held-out test accuracy or all population capability.

After the 630 T5->T6 expansion, all35 old classes have K=0 (32 newly zero,
three already zero). This is the unchanged algorithm's post-expansion K rule.
Subsequent per-update floor safety does not restore or retrospectively protect
the pre-expansion correct controls. All later training proceeds normally.

## Training and audit boundaries

Continued tasks3-11: 1,890 updates in the210 arm, 5,670 in the630 arm; 7,560
total. Original task order, local batch generator, seed2021, batch16, LR5e-5,
AdamW history, predictor, cutter, rho, EPS, solver caps and tolerances preserved.
Every scheduled update forms the actual AdamW proposal and runs the historical
actual-update predictor. Exact endpoint checks and corrected INNER_LIMIT
acceptance remain the final authority. No old CE backward or test data reads.
All executed training updates satisfy their post-expansion K. Predictor
false-safe count is0 in both arms. The210 arm has0 rollbacks; the630 arm has200
(129 in task8 and71 in task11), all preserved under the unchanged controller.
These within-task checks do not erase the separately recorded boundary losses.

The observer never reacts to a catastrophic result. No calibration, replay,
new-class gate, logit scaling, bias adjustment, predictor/cutter tuning or K
change was introduced. No test evaluator was run in this task.

`diagnostics/verify.py` independently implements boundary endpoints, strict
comparison counts and per-flip attribution from predecessor checkpoint states.
It also reexecutes each continuation **without the production observer**,
checking every batch, native proposal, predictor component, solver decision,
executed head, optimizer moment/counter and endpoint against recorded training.
That replay reuses the unchanged production controller, not a second independent
solver implementation. Independent boundary recomputation is separate.

The first-step observer-on/off differential for each arm also passes. Source,
cache, checkpoint and BERT hashes are checked; the Python-level read allowlist
rejects test evidence and external writes. Its scope excludes native-library/OS
internals. Some read-only verification runs overlapped other work on the same
device; wall figures in task records are diagnostics, not speed benchmarks.

## Files and reproduction

New source only in this directory:

- `common.py`: immutable source imports, input allowlist, identity/hash guards.
- `observer.py`: full77 readout slices, distributions and strict/tie attribution.
- `run.py`: preflight and tasks3-11 continuation; controller imported unchanged.
- `diagnostics/verify.py`: independent boundary checks and observer-free replay.
- `diagnostics/report.py`: cumulative summary and compact artifact packaging.
- `PROTOCOL.md`: frozen before formal execution.
- `results/`: machine-readable summaries, per-step ledgers, complete boundary
  logits/IDs/labels, flip records and checks. No raw text or BERT feature copies.
- `ARTIFACT_MANIFEST.json`: SHA-256/size of public artifacts and the18 external
  small head/moment checkpoints.

Local authority:
`E:/gfg-evidence/banking77-final-v2-expansion-audit-v1`.
Inputs remain in the original task2/nonlinear authorities named in FREEZE.
Runtime: `E:/gfg-python/governor-v4-bench/Scripts/python.exe`, PyTorch/CUDA
versions and source hashes in FREEZE. These are create-only runs, not overwrite
or resume commands:

```text
python -B -m experiments.gfg_banking77_final_v2_expansion_audit_v1.run
python -B -m experiments.gfg_banking77_final_v2_expansion_audit_v1.diagnostics.verify 210
python -B -m experiments.gfg_banking77_final_v2_expansion_audit_v1.diagnostics.verify 630
python -B -m experiments.gfg_banking77_final_v2_expansion_audit_v1.diagnostics.verify
python -B -m experiments.gfg_banking77_final_v2_expansion_audit_v1.diagnostics.report
```

Use a fresh authority location on a separate reproduction machine/workspace
with the declared input mappings. Never delete old results to rerun. Frozen
scientific results and prior failed/corrected authorities remain untouched.
