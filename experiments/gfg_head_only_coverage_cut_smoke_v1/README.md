# Head-only protection coverage / boundary-tight cut smoke

Independent extension of `df98982d476b57976872d9f6fb21b6a62404a4a3`.
The original `10/current` experiment is historical evidence, not rerun.
Three new arms only; see [PROTOCOL.md](PROTOCOL.md) for the frozen contract.

## Results

All three arms ran once to210 scheduled updates before any new-arm test evaluation.
The table uses the SAME primary test IDs and relation-macro accuracy as the
historical baseline (1120 old /1119 new examples; one pre-existing exclusion).

| Method | Old test | New test | All test |
|---|---:|---:|---:|
| Historical10 + current cut |76.6071%|89.1958%|82.9015%|
| A:140 + current cut |94.8214%|73.6588%|84.2401%|
| B:10 + boundary-tight cut |76.5179%|89.1958%|82.8568%|
| C:140 + boundary-tight cut |95.0893%|69.0140%|82.0516%|

| Method | Final old protected correct/total | Projection steps | Native violating sample-update occurrences | Executed old correct-to-wrong | Skips | Training wall(s) |
|---|---:|---:|---:|---:|---:|---:|
| Historical10/current |78/80|174|1246|0|0|25.1732|
| A140/current |1092/1120|207|2298|0|112|79.0285|
| B10/tight |78/80|174|1691|0|0|24.8806|
| C140/tight |1090/1120|207|4712|0|137|73.9932|

Mean geometry on projection-triggered steps:

| Method | Attempted norm ratio | Attempted cosine | Attempted relative displacement | Executed norm ratio | Executed cosine(nonzero only) | Executed relative displacement |
|---|---:|---:|---:|---:|---:|---:|
| Historical10/current |0.921469|0.886963|0.448264|0.921469|0.886963|0.448264|
| A140/current |0.888124|0.821553|0.540411|0.419265|0.856348|0.752811|
| B10/tight |0.905293|0.905128|0.413857|0.905293|0.905128|0.413857|
| C140/tight |0.844796|0.850330|0.499498|0.309968|0.916453|0.779456|

### Interpretation of the three fixed comparisons

1. Coverage: A raises old test by18.2143 percentage points, but reduces new test
   by15.5370 points. All rises1.3386 points. A skips112/210 updates: this establishes
   a substantial effect of the larger protection-bank policy, but DOES NOT isolate
   a coverage-only explanation of the old-test gain independently of suppressed
   new learning. It does not achieve the desired high retention/high plasticity
   combination under this two-pass rule.
2. Cut: B leaves new macro accuracy exactly unchanged and old accuracy falls by
   one test example(0.0893 percentage points). The proposed tight-cut modification
   did not improve accuracy in this fixed smoke. The earlier observation of
   redundant corrections is real, but is not evidence of an accuracy benefit from
   removing them. Different trajectories make mean geometry an outcome, not a
   same-update paired comparison.
3. Combination: C has95.0893% old retention, but new accuracy falls to69.0140% and
  137/210 parameter updates are skipped. The requested combination did not deliver
   both benefits. The recorded skips expose the limitation of the present sequential
   two-pass execution rule; they do not prove that the feasible set is empty or
   that a frozen linear head fundamentally cannot preserve and learn both tasks.

No epsilon/pass limit changes, parameter sweeps, retraining, or result-driven
rule changes were made. Failed candidate projections and all skips remain in the
records. There were no formal training-process failures or restarts.

Integrity: **PASS**. All32 preflight tests passed. Independent reconstruction
verified all630 new-arm steps,2,150,400 endpoint margin values and10,764 pair
certificates, including failed candidate projections. No executed update lost an
initially-correct-for-that-step old protected example. All three training-result
hashes were fixed before test evaluation. Original experiment files and the paper
freeze tag remain unchanged. These are integrity checks, not an accuracy success
gate.

## Scope

FewRel task2, seed2021,210 updates, LR5e-5, frozen/eval BERT, existing linear
head and inherited task1 AdamW moments. Identical current batch order. No replay,
Router, BERT updates, scalar shrink, sweep, extra seeds, or result-driven rerun.
The larger bank is1120 unique old validation examples, disjoint from training and
the existing primary test by exact encoded input. These examples actively constrain
updates, so their correctness is not independent held-out performance.

A retains the parent's current-cut implementation without editing it. B/C use
`+1e-6` boundary correction and recheck each pair immediately before correction.
All arms have at most2 passes and skip unsafe candidates. AdamW moments advance
once on every scheduled update, including parameter skips. Eligibility is current
receiving correctness, not forced recovery of initially incorrect examples.

The B comparison changes both correction magnitude and within-pass rechecking;
it does not separately identify their individual effects. Sequential local
corrections are not a joint optimal projection, and two passes need not find a
safe candidate. FP32 full-bank endpoint checks authorize the actual update.

## Execution and evidence

Run from the repository root using the existing local FewRel/BERT environment:

```powershell
python -B -m experiments.gfg_head_only_coverage_cut_smoke_v1.preflight
python -B -m experiments.gfg_head_only_coverage_cut_smoke_v1.run
python -B -m experiments.gfg_head_only_coverage_cut_smoke_v1.check
python -B -m experiments.gfg_head_only_coverage_cut_smoke_v1.package
```

Formal outputs use exclusive-create files. Do not rerun over an existing result.
The runner completes all3 trainings and writes a hashed completion barrier before
evaluating any new-arm test result. Shared test features are evaluated only after
that barrier; historical baseline results are referenced from their original files.

`results/RESULTS.json` contains the requested comparison and diagnostics.
`results/CHECKS.json` contains independent reconstruction checks.
`records/*/STEPS.jsonl.gz` contains receiving/native/cut/executed margins, pair
certificates and attempted/actual geometry. `PUBLIC_RECORDS_MANIFEST.json` hashes
the public records. Model tensors, original text/token inputs and cached features
remain local at `E:/gfg-evidence/head-only-coverage-cut-smoke-v1` and are NOT published.
Full local head snapshots allow the checker to reconstruct every cut without
calling the production cut or retraining any model.

Geometry is averaged over projection-triggered steps. On a skipped update the
attempted cut is NOT executed: actual norm ratio is0, cosine is undefined, and
relative displacement from the native proposal is1. Attempted and executed
geometry are therefore reported separately; undefined cosines are excluded from
their mean. Norm retention must not be confused with directional change.

Training wall includes the formal training loop and step logging, excludes cache
construction, final snapshot serialization, test evaluation and independent audit.
Historical wall measurements are descriptive, not a controlled timing benchmark.
