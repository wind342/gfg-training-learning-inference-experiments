# GFG executable mechanism discovery report

## Sealed claim

The submitted scientific claim is `CapabilityFormationMechanism`, not this report. Its finite operative state couples two processes:

1. formation of a token-map-independent cyclic rule circuit, measured by probability assigned to the group-consistent answer; and
2. an endogenous AdamW reservoir cycle in which small gradients and depleted second moments increase normalized-update stress until a clipped-gradient burst resets the reservoir.

The capability transition occurs when the rule probability is already rising and the next state-predicted slingshot/recovery cycle moves the lower rule-margin tail through the success boundary. The same reservoir cycle persists after formation, so the sealed stability result is `TRANSIENT_DEGRADATION_RECOVERY`, with uncertainty in the exact future burst phase.

Only the released, validated participant GFG was used. Its manifest reports 111,313 hash-chained blocks, 4,642,344 atomic facts, 1,062,946 objects, 111,313 occurrences, and 711,267 explicit edges. `gfg_validation.json` passes exact fact expansion, exact object references, parameter-version chaining, evaluation binding, program order, and the prohibition on approximate temporal joins.

## Primitive query semantics and evidence chain

For every citation below I expanded one fact block over its ordered `(source object, relation_role)` entries and its exact outcomes. Each resulting atomic fact remains attached to that block's one concrete occurrence through `realizes_fact`; I did not join independent source and outcome sets by step or value. `reads_from` is the source-incidence dependency supplied by that expansion. `GeneratedOrigin` and `program_order` were followed only where explicitly recorded.

The forward query used this traversal:

`training_task -> batch_materialization occurrence -> exact batch input/target outcomes -> training forward/loss -> backward gradient -> gradient_clip -> optimizer_parameter_update -> exact new parameter version -> evaluation outcome`.

The reverse query began at an evaluation metric or logit outcome, followed its evaluation fact to the exact evaluated parameter objects, then followed those parameter versions through `GeneratedOrigin` and the optimizer facts that formed them. Numeric equality and timestamp proximity were never used as identity.

Representative primitive records are:

- Step-0 batch occurrence `occ_10bace233ceff9d389539539b8f34521532eac7311b793176f7a386784e52382`, fact block `factblock_62abea22d5803a5f33cdb12eb5135f799b4085e9f5922ac1c24691abba6ec9de`, reads the exact task object and forms the exact batch inputs and targets.
- The transition-forming burst is recorded by gradient-clip occurrence `occ_5e57bead45e7b08291448747afe51fd4eeb23aeb162f55e237fc77ed3066d86a` at training step 1135, with `total_norm=52.422943115234375`. Representative atomic fact block `factblock_dcd5737c143835f529cf7572ff9f88485c1ad6a20891fe8b0976325def998c47` maps unclipped gradient `obj_f9023d73cb8ea944671d5b5ac2a7c3148782a9314a3583e368058d8026291327` to clipped gradient `obj_97bbadf79d9b9bd8cc7a57200f6721884998df8e5a01c3899845cfc88d46138e`.
- Optimizer occurrence `occ_df4cc4327a812c8f80c1068043b9246853063742de63d9564f35c1e670d11814` at parameter version 1136 follows that clip in explicit `program_order`. Fact block `factblock_6565b5a7165c4db20c89f17ab4ec48d064e1c53e74433911692f018c231a04de` reads the exact prior parameter, the cited clipped gradient, optimizer configuration, and optimizer moments, then forms the new parameter and moment outcomes. Its block contains 60 primitive `GeneratedOrigin` edges for the parameter/moment version chain.
- Evaluation occurrence `occ_9c9f9d31bb591247f9d4ccb1ced1b1ee657bbe0801240fa99760e520687fcc81` reads parameter version 1200. Fact block `factblock_f081922c343a456d265cb6978c381e780be5937f48eef62e2791ed6061d621fd` forms its train predictions, logits, validation predictions, validation logits, and metric result.

## Capability formation

The training triples define a cyclic group operation under an opaque token map. From training outcomes alone, the executable solves

`coordinate(target) - coordinate(left) - coordinate(right) = 0 (mod p)`.

On this graph the modular equation matrix has a one-dimensional nonzero nullspace and yields a bijection over all 23 operand tokens. The arbitrary global nonzero coordinate scale cancels from the operation. This lets the executable derive the correct answer for each observed validation input without storing this execution's token map or consulting future labels.

Training-sample memorization is not the transition. At evaluation step 100, occurrence `occ_0c1b875c44c35453540a0fae81dc1a585bdeb5ca25f12a089c63dc12366f97af` records train accuracy 1.0 but validation accuracy only 0.0283018872. Its exact evaluation fact is `factblock_2bb913feb9e97b5580f8c67d836792e11b8e190e68115835742683a79a5ec226`.

The rule state then changes prospectively:

| Evaluation step | Validation accuracy | Mean rule margin | 10th-percentile margin | Mean rule probability |
|---:|---:|---:|---:|---:|
| 700 | 0.1604 | -4.1902 | -8.7758 | 0.1527 |
| 800 | 0.5802 | 0.9515 | -3.5745 | 0.5690 |
| 900 | 0.7123 | 2.0353 | -3.0265 | 0.6665 |
| 1000 | 0.8255 | 3.5853 | -2.2379 | 0.7770 |
| 1100 | 0.8679 | 5.0726 | -0.8483 | 0.8467 |
| 1200 | 0.9906 | 5.5234 | 2.6854 | 0.9608 |

The exact evaluation facts at steps 800, 900, 1100, and 1200 are respectively `factblock_6537b598d6126e168b67b9897620ba24b39530011632046f0240d16e7050c445`, `factblock_34f077d4a2985531921934ddc10b7d640ff7f057b1c7668863eb3f0860cef6c5`, `factblock_01e0f05ea6bc4f7da0792de8c9488213aee1c2217bfb06119ca9b285da0e2fb1`, and `factblock_f081922c343a456d265cb6978c381e780be5937f48eef62e2791ed6061d621fd`.

The frozen transition definition is first satisfied at step 1200: an earlier evaluation is below 0.3, and steps 1200, 1300, and 1400 all have train accuracy at least 0.99 and validation accuracy at least 0.9. The latter two evaluations are occurrences `occ_f532f4d0abb52b4e4e9bcfbb62ebb4138121f7b9165b92d4c9f411beeee98d5d` and `occ_1e070367153ca70282fe44740dbfaa4e92eba9d1b59ba0f462963d64f20c8060`.

## Optimizer reservoir and stability

The gradient-clip occurrence payload supplies a value at every training step, not a timestamp join. Major bursts occur after a long norm decay, a local minimum, and a short accelerating precursor. Examples are:

- steps 1125–1135: `0.00024103 -> 52.42294312`;
- steps 2980–2988: `0.00016475 -> 49.68954086`;
- steps 8892–8898: `0.00038309 -> 76.15070343`;
- steps 9590–9598: `0.00047274 -> 85.82303619`;
- steps 9981–9989: `0.00046041 -> 193.22958374`.

The representative step-2988 clip is occurrence `occ_cb0531025c8fd8c7f2b7970d9c04714a3abb422892969af746b915a9f0fba419`, fact `factblock_445813a68a5c8f5675326afc826d1de3e56f06f693acdcc392a270a7b3e72774`. Step 8898 is occurrence `occ_adff8c62bbfb3eaec1bc133072feb3c1a965cff0b1dfa5dd7de588634c23f7ca`, representative fact `factblock_3ca7aa09b83d80039061deb12a44affb4f963186cd1cb71af6c9d4d6c8e0d024`. Step 9598 is occurrence `occ_ffe707e7a97f1c6218fe889726ac1ba656bbffdb9a73c5b03ea454b23dc63b34`, representative fact `factblock_6ad42b3fa7c62fd247efe51424d3f89d76942263a81e2f381f433dac203432eb`. Step 9989 is occurrence `occ_90f8a887849de30764624ba6fbc4a049da4c0e9a0ead945e5e5b73c94ca9dbfc`, representative fact `factblock_796fb8cb1da1f3e32ac3132a9e1d2b84c6e85047a5e9144eb444f46651b7cc04`.

Derived only from the exact parameter, `exp_avg`, and `exp_avg_sq` tensor outcomes, the relative normalized Adam step rises from 0.000404 at evaluation 2900 to 0.009434 at 3000; from 0.000167 at 8800 to 0.010756 at 8900; and from 0.000152 at 9500 to 0.009537 at 9600. These are reservoir-state changes, not equal-value identity claims.

The capability is dynamically unstable but repeatedly recovers:

| Burst / evaluation facts | Degraded evaluation | Next evaluation | Result |
|---|---:|---:|---|
| step-2988 clip -> `factblock_9860abc73adf5849befbe3b73425ad03bf80e6fd95d1456414b77c74dc5c02e8` | step 3000: train 0.9685, validation 0.8019 | step 3100: 1.0 / 1.0, `factblock_5d975592991f31d33e8bd9514a772881e65cff95115e2caf16ae429062d5c518` | recovery |
| step-8898 clip -> `factblock_65afceb815f640405e1055015699ec5b69e59418c7cd5538edd506eb5e0c7eae` | step 8900: 0.2776 / 0.1934 | step 9000: 1.0 / 1.0, `factblock_0ff670c1d3426cd7956fe6437d23e286a26c686fcf5a28a843db8e899246354b` | recovery |
| step-9598 clip -> `factblock_05ff7a5f1f65e5d01f3ae1ac844cbcf13a6b027de3e436ef1ce8df45a82fd6ec` | step 9600: 0.2997 / 0.2264 | step 9700: 1.0 / 1.0, `factblock_235cbe45f553d7d9a06d926de860c5fe17d688f1caa8e73b545df73a3759f31a` | recovery |

The final step 10000 is censored during recovery after the step-9989 burst: train accuracy is 0.6782 and validation accuracy 0.5943 in `factblock_234d36e1aaceb89ac1ff6b78af0a11fddf1a0b9b6a47d2f1b41a0b9aa738f588`. No suffix exists to prove that individual recovery. The prediction of recovery follows the repeated same-state mechanism above; it is not asserted from a missing edge. This supports `TRANSIENT_DEGRADATION_RECOVERY` rather than `STABLE`; persistent degradation is not supported because the observed prior degradations recover.

## Hypotheses and falsification

| Hypothesis | Prefix-only or suffix test | Outcome |
|---|---|---|
| Selected: group-rule probability plus Adam reservoir/slingshot state | Initialize separately at steps 800 and 900, infer state from each prefix, and forecast the unread suffix | Not falsified in the allowed cut domain; both forecast transition step 1200 |
| Rejected: training memorization/loss alone causes generalization | Compare evaluation 100 with later evaluations: train accuracy is already 1.0 while validation remains 0.0283; many equally memorized states have different rule probabilities and futures | Falsified state sufficiency |
| Rejected: once formed, capability is monotone and stable | Replay after formation through exact evaluation/parameter facts at 3000/3100, 8900/9000, and 9600/9700 | Falsified; transient degradation and recovery are required state transitions |
| Rejected: slingshots are a fixed-period clock | Major-peak spacings include 128, 243, 321, 406, 448, 463, 479, 566, and mature values from 341 to 393 steps | Falsified; the executable uses a prefix-inferred interval and reservoir-conditioned recurrence, not an absolute timestamp |

## Prefix-only replay and closure

The formal prediction cut requires no prior transition, train accuracy 0.99 on the last three evaluations, and current validation accuracy between 0.2 and 0.75. The discovery run therefore supplies two eligible historical cuts:

| Cut | Prefix state | Sealed transition | 200-step interval | 500-step interval | Unread-suffix validation RMSE |
|---:|---|---:|---|---|---:|
| 800 | rule probability 0.5690; mean margin 0.9515; q10 -3.5801; next burst 1124 | 1200 | [1100, 1300] | [1000, 1400] | 0.13 |
| 900 | rule probability 0.6665; mean margin 2.0353; q10 -3.0370; next burst 1128 | 1200 | [1100, 1300] | [1000, 1400] | 0.13 |

The candidate reads no facts after either cut. A near-transition cut at 1100 also forecasts step 1200. Post-formation state replays predict the next burst within their declared ±100-step risk interval from cuts 1400 (predicted 1629; observed 1583), 2500 (2534; observed 2525), 5000 (5139; observed 5142), and 8800 (8901; observed 8898).

The retained failure is burst amplitude at the late cut 8800: although the next risk interval contains the burst, point-curve RMSE over the short remaining suffix is about 0.258 because later phase drift and collapse amplitude are underdetermined. This cut is outside the sealed initialization domain, which requires no prior transition, but it limits the strength of the stability-amplitude claim. The early eligible-prefix forecasts, which are the submitted use case, remain below the frozen 0.2 normalized-RMSE bound on this discovery execution.

Closure counterexample search produced two required state enlargements:

- train accuracy/loss alone grouped step-100 and later memorized states despite incompatible futures, so rule probability was retained;
- rule probability alone grouped many post-formation evaluations near 1.0 despite incompatible next evaluations, so burst phase and optimizer reservoir were retained.

No smaller tested state closed both capability and stability predictions.

## Executable correspondence and use audit

| Mechanism claim | Primitive GFG evidence | Executable field | Forecast expression |
|---|---|---|---|
| cyclic rule formation | batch fact plus validation-logit evaluation facts | `rule_probability`, `rule_probability_velocity_per_grid` | advances predicted rule state, curve, and `will_transition` |
| next transition needs a slingshot/recovery | all prefix `gradient_clip` occurrences and optimizer fact chains | `next_burst_step`, `cycle_interval_estimate` | 0.65 burst / 0.35 probability transition blend |
| reservoir controls future burst phase | exact Adam `exp_avg_sq` outcomes | `reservoir_depletion` | scales the mature interval recurrence |
| normalized update controls degradation expression | exact parameters plus `exp_avg` and `exp_avg_sq` | `relative_adam_step`, `current_gradient_norm` | sets transient hazard amplitude |
| optimizer drive controls rule progress | optimizer configuration source | `optimizer_learning_rate`, `optimizer_weight_decay` | scales rule-probability velocity |
| recurring bursts imply transient instability | prefix major-burst occurrences | `major_burst_count` | selects `TRANSIENT_DEGRADATION_RECOVERY` versus `UNDETERMINED` |

Perturbing rule velocity to zero changes `will_transition` to false and returns `NO_TRANSITION`; holding rule probability fixed changes the future curve. Removing projected burst state removes instability intervals and hazard corrections. Holding reservoir depletion or normalized-update state fixed changes projected burst phase or degradation amplitude. These are operative uses. `rule_margin_mean`, `rule_margin_q10`, `rule_positive_margin_fraction`, raw `second_moment_mean`, and the rule-recovery success flag are explicitly diagnostic and are not described as independent controls.

## Intervention audit

`TrainingIntervention` runs only at `before_optimizer_step`. It sets the current optimizer groups' learning rates to zero for 1600 optimizer steps, then restores the exact prefix-native values. This is an allowed optimizer-group hyperparameter mutation. It injects no validation sample, label, parameter, answer table, or evaluator change.

The affected optimizer operation reads current parameters, clipped gradients, optimizer configuration, first moments, and second moments, and writes new versions of all four state classes, as shown by the step-1136 optimizer fact above. With learning rate zero:

- parameters and therefore rule probability are held;
- weight-decay parameter motion is also held because AdamW scales it by learning rate;
- forward, loss, backward, clipping, and gradient formation continue;
- Adam first and second moments continue to evolve.

Thus this is not claimed as a pure time translation. The reservoir re-equilibrates while the rule-bearing parameter state is fixed. After restoration, the same state law resumes from that altered reservoir. The predicted direction is `DELAY`, with shift interval [1300, 1900], safely above the 600-step minimum. The predicted stability effect is `NO_CHANGE`: the intervention delays formation but does not eliminate the recurrent post-formation reservoir cycle.

## Cross-run status

Fixed structure: modular coordinate recovery, rule-probability state, burst detection from gradient facts, and the reservoir recurrence form.

Prefix-inferred state: token coordinates, rule margins/probability/velocity, burst peaks and interval, current gradient norm, second-moment depletion, relative Adam step, and optimizer hyperparameters.

Discovery-estimated constants: recovery lag, mature interval attractor, blend weights, expected post-formation baseline, and hazard width.

Uncertainty: one execution does not establish cross-run transport of those constants. Fixed-rate, fixed-period, and absolute-step alternatives were falsified within the discovery run. The submitted mechanism is now sealed; only prospective execution on the distinct unseen continuous run can validate transport.
