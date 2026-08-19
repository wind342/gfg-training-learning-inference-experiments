# GFG capability-mechanism discovery report

## Result

The selected executable mechanism is **cyclic rule completion with adaptive-optimizer relaxation**. The capability transition is at optimizer step **1200** under the frozen definition: step 1200 is the first point of the first three-point window (1200, 1300, 1400) with train and validation accuracy at least 0.99 and 0.90 respectively, after earlier validation accuracy at or below 0.30. The formed capability is not dynamically stable in the strict sense. The supported classification is `TRANSIENT_DEGRADATION_RECOVERY`.

All temporal claims below follow exact optimizer-step identities, `reads_from`/fact incidence, `GeneratedOrigin`, `realizes_fact`, and declared order. Each cited fact block was expanded only according to its own ordered source-role and outcome records; sources or outcomes from different facts were never Cartesian-recombined. Numeric similarity was used only to calculate diagnostics after identity alignment, never to join records.

## What rule was formed

The materialized training batch contains 317 distinct triples `(left, operator, right)` over 23 operand tokens and a separate operator token. The target is only at the last position. Reindexing the 23 operand tokens as unknown elements of the prime cyclic group gives one linear constraint per observed generation, `coordinate(output) = coordinate(left) + coordinate(right) mod 23`. Exact modular elimination has rank 22, one expected scale degree of freedom, 23 distinct recovered coordinates, and satisfies **317/317** training relations. Thus the training facts uniquely determine all 529 pair results up to a harmless choice of generator; no validation label was used.

Applying that derived rule to the validation input objects reproduces every recorded evaluation accuracy exactly from the corresponding validation logits. At step 500 the exact evaluation fact (`occ_63c60c...`, metric object `obj_e6f2eb...`) has train accuracy 1.0 but validation accuracy 0.3537736. The unseen-pair true-class margin distribution still has negative mean (-1.713) and median (-1.44). By step 1100, validation accuracy is 0.8113208 and mean margin is positive (3.236), but the lower tail remains negative. At the step-1200 evaluation occurrence `occ_dcba7fe3...` (metric `obj_23f39e...`), validation accuracy is 1.0 and the minimum unseen-pair margin is positive (1.65). The following exact evaluation occurrences at steps 1300 and 1400 also form 1.0 validation accuracy, sealing the transition.

The update producing parameter version 1200 is occurrence `occ_d37f3b3b...`. For the position embedding alone, fact block `factblock_7cc73ede...` binds the exact version-1199 parameter, clipped gradient `obj_e68e87fb...`, optimizer configuration, and three version-1199 Adam state objects to four distinct version-1200 outcomes. The whole checkpoint displacement from 1100 to 1200 is distributed: global parameter L2 displacement is 5.616; the largest components are block-1 MLP input (2.763), block-1 MLP output (2.561), and block-1 attention input (2.413). It is not a head-only rescaling.

## Finite-state mechanism

The executable uses these states:

1. `SAMPLE_MEMORIZATION`: train accuracy is below 0.99.
2. `RULE_FORMING`: train accuracy is at least 0.99 but fewer than 90% of rule-derived unseen margins are positive.
3. `RULE_CANDIDATE`: a single grid point passes, but the three-point persistence guard is not yet sealed.
4. `RULE_FORMED`: the frozen three-point window passes.
5. `TRANSIENT_DEGRADED`: an Adam relaxation excursion makes a formed parameter version lose rule margins.
6. `RULE_RECOVERED`: clipped full-batch gradients have restored the formed rule.

The runtime state deliberately retains only evaluation-derived scalars and prefix observations. It does not retain token coordinates, validation answers, graph object identifiers, graph paths, or native tensors. Forecast time is a deterministic function of the prefix cut, its validation coverage deficit, and the observed evaluation grid. The future curve is a formula (quadratic approach to the predicted formation step, followed by a high plateau with relaxation excursions), not a stored trajectory.

## Mechanisms proposed and falsified

### H1 — selected: distributed rule completion

Prediction: the training relations must be consistent with a low-dimensional composition law; training memorization must precede unseen generalization; and the transition must coincide with uniform positive unseen margins rather than merely low training loss.

Falsification attempt and result: exact modular elimination could have been inconsistent, rank-deficient beyond the generator symmetry, or unable to reproduce validation metrics. Instead it fits 317/317 relations at rank 22 and reconstructs all recorded validation accuracies from logits. Train accuracy was already 1.0 at step 100 while validation remained 0.0660; it was still only 0.0991 at steps 300 and 400. Uniform positive validation margins appear at the sustained transition. H1 survives.

### H2 — rejected: incremental lookup-table acquisition

Prediction: validation improvement should be explained by new examples, changed batch membership, or continued acquisition of training correctness.

Falsification attempt and result: all 10,000 `before_batch` occurrences form input and target outcomes with the same two content hashes (`0cd1af...` and `bfe2fe...`) and the same 317 sample identities. Train accuracy saturates long before validation. No new samples are generated at the transition. The step-1199 batch fact `factblock_f04a9cdb...` leads through forward/loss occurrence `occ_037c63dc...`, backward occurrence `occ_1891f77f...`, clip occurrence `occ_a380ea06...`, and optimizer occurrence `occ_d37f3b3b...`; the sources are the same complete batch, not added evidence. H2 is falsified.

### H3 — rejected: head norm or global parameter norm is the mechanism

Prediction: a scalar head/global norm threshold should align with formation and remain aligned with capability.

Falsification attempt and result: the tied token/head weight L2 norm is 4.039 at step 400 (validation 0.099), 3.296 at step 1100 (validation 0.811), and 3.389 at transition; it is not a monotone formation coordinate. Global parameter norm continues from 41.27 at a capable step 4000 to 41.76 at the collapsed step 4100, and later to 42.61 at recovered step 4200. The transition checkpoint displacement is largest in the second transformer block, not the head. H3 is falsified.

### H4 — stability hypothesis, rejected as strict stability: a permanently robust formed basin

Prediction: after formation, all later evaluation parameter versions should retain high train and validation accuracy despite small optimizer oscillations.

Falsification attempt and result: parameter version 4100 is evaluated by occurrence `occ_130732b0...` and forms train accuracy 0.1956, validation accuracy 0.1840, and loss 5.7124 in metric object `obj_0126e801...`. Parameter version 8600 is independently evaluated by `occ_a62f9a5b...` and forms train accuracy 0.8864, validation accuracy 0.6792, and loss 0.4553 in `obj_351e658d...`. Both failures recover by the next grid point (1.0/1.0 at 4200; 1.0/0.9906 at 8700), so persistent degradation is also falsified.

H4's alternative transient mechanism is supported. Before the first large episode, global Adam `exp_avg_sq` L2 falls from 0.00991 at step 3700 through 0.00131 and 0.000174 to 0.0000231 at step 4000 while the model remains correct. At the collapsed version, the next exact full-batch forward occurrence `occ_e48c79c...` forms training loss 4.6623; backward occurrence `occ_617b2630...` forms global gradient norm 30.98; clipping occurrence `occ_edf65a97...` forms norm 1.0. The analogous second episode produces loss 0.4039, gradient norm 8.23, and clipped norm 1.0 through occurrences `occ_b47df891...`, `occ_4cc4d954...`, and `occ_4d23ea62...`. These are observable preconditions and responses, not inferred concurrency.

The causal parameter chain is explicit. At the first episode, step-4099 batch outcomes feed forward fact `factblock_37d21308...`; loss `obj_c4cbcc82...` and version-4099 position parameter generate gradient `obj_e6704ca9...`, which generates clipped gradient `obj_347aa25c...`. Optimizer fact `factblock_8652778e...` reads that clipped gradient plus exact parameter and Adam-state versions and forms position-parameter outcome `obj_285bd809...` at version 4100. The evaluation fact reads the complete exact version-4100 parameter set. During recovery, the same batch hashes continue to generate loss, gradients, clipped gradients, and successive parameter versions; optimizer occurrence `occ_089fb210...` forms version 4200 and evaluation occurrence `occ_7c10002e...` records full recovery. The second episode has the same primitive pattern through optimizer fact `factblock_a9cc482f...` and recovery optimizer fact `factblock_68f18b8...`.

## Stability forecast

The state machine forecasts two inclusive post-formation risk intervals centered on the calibrated relaxation episodes: one at formation plus 2900 optimizer steps and the next 4500 steps later. Each interval includes one grid step on either side to express observation resolution. The curve forecast explicitly degrades at the centers and returns to the formed plateau at the following grid point. This makes `TRANSIENT_DEGRADATION_RECOVERY`, rather than an assumption of stability, part of the sealed forecast.

## Intervention and causal prediction

The submitted intervention uses only `before_optimizer_step`. For exactly 800 calls it sets every current parameter gradient to `None`. In AdamW, a parameter with no gradient is skipped: the parameter, its step counter, moments, and decoupled weight decay do not advance. This changes one mechanism variable, `effective_optimizer_updates`, while leaving task, data, evaluator, capture, and validation contracts untouched.

Prediction: `DELAY`, with a transition-step shift interval of +700 to +900 and central expectation +800. The graph's exact equality of all full-batch source contents makes an 800-update time translation executable rather than merely correlational. The predicted stability effect is `NO_CHANGE`: formation and relaxation risk are shifted in declared optimizer time, but the formed-state oscillator and clipped-gradient recovery mechanism are unchanged.
