# Discovery report: clipped-Adam rule-margin oscillator

## Result

The selected mechanism is a finite-state relaxation oscillator in rule margin. The model first memorizes the complete fixed training set, while its held-out margins remain negative. Under the exact AdamW parameter/optimizer-state chain, long quiet intervals reduce gradient and second-moment scale; recurrent gradient releases are clipped at norm 1 and produce substantial parameter-version motion. Successive releases move the model into basins with increasingly positive margins for the task's latent cyclic rule. Capability forms when the lower held-out margin tail becomes positive. The same oscillator remains active after formation, so the supported stability result is `TRANSIENT_DEGRADATION_RECOVERY`.

The formal transition is optimizer step 1600. The independently frozen diagnostic interval is 1500–1700 (width 200), and the primary interval is 1400–1800 (width 400). These values are not stored as a trajectory in the executable: the candidate regenerates them from a previously unseen prefix using its log-odds equation.

## Graph method and identity discipline

All claims below follow concrete graph records. I expanded no sources and outcomes into a Cartesian product. For each cited path I kept the atomic fact block's source role, occurrence, outcome, and parameter version together. Time is optimizer-step/occurrence identity, not wall clock. Evaluations read the exact `theta_t` formed at step `t`; the optimizer fact blocks read the clipped gradient, the prior parameter and Adam states, and the frozen optimizer configuration; their `GeneratedOrigin` edges bind each result parameter to the next use.

The participant-safe graph passed hash-chain, exact-fact, exact-object, exact-program-order, parameter-chain, and evaluation-parameter-binding validation. It contains 10,000 optimizer steps and 101 evaluation occurrences.

## The learned rule and a rule-level observable

The task object declares 23 operand/result tokens plus an operator token. Every training example is of the form `(a, operator, b) -> y`. Treating the 317 exact training relations as equations

`phi(y) - phi(a) - phi(b) = 0 (mod 23)`

gives rank 22 and a one-dimensional nullspace. Its nonzero solution is a permutation of all 23 residues and satisfies every training relation; the identity is also uniquely determined. Thus the frozen examples identify a cyclic group operation up to the irrelevant choice of generator. This analysis uses token identities only to establish the mechanism on the discovery graph; the executable does not retain this token map and is invariant to the unseen token map.

Using that training-derived closure, I computed for every held-out pair the correct-class logit margin, `correct logit - maximum incorrect logit`. Its positive fraction equals the graph's validation accuracy. More importantly, its lower tail distinguishes confident memorization from a robust rule:

| Step | Train accuracy | Validation accuracy | Mean rule margin | 10th-percentile rule margin |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 1.000 | 0.104 | -4.145 | -7.046 |
| 500 | 1.000 | 0.269 | -2.719 | -7.916 |
| 1500 | 1.000 | 0.816 | 4.422 | -2.095 |
| 1600 | 1.000 | 0.925 | 4.982 | +0.773 |
| 1800 | 1.000 | 0.953 | 6.610 | +2.272 |

The sign change of the lower tail at step 1600 is independently aligned with the first of three formal passing evaluations (1600, 1700, 1800).

## Selected mechanism: formation by recurrent gradient release

The `before_batch` graph facts show the same input content hash and same target content hash at all 10,000 occurrences, with the same 317 sample identities. Data novelty or ordering cannot cause the state changes. The optimizer configuration object instead fixes AdamW at learning rate 0.003, betas `(0.9, 0.98)`, parameter-group weight decays 1.0 and 0.0, and global gradient clipping at 1.0.

Contiguous runs with recorded pre-clip norm above 1 form an oscillator. The formation-relevant runs and the next bound evaluations are:

| Pre-clip episode | Peak norm | Validation before | Validation after |
| --- | ---: | ---: | ---: |
| 452–469 | 38.832 | 0.104 at 400 | 0.269 at 500 |
| 770–784 | 34.993 | 0.302 at 700 | 0.415 at 800 |
| 1128–1147 | 37.688 | 0.557 at 1100 | 0.689 at 1200 |
| 1508–1521 | 59.679 | 0.816 at 1500 | 0.925 at 1600 |

The optimizer-state tensors corroborate the quiet/release cycle: aggregate mean `exp_avg_sq` is `2.858e-9` at step 1500 after the quiet phase and `5.740e-7` at step 1600 after the release, a 201-fold increase. Analogous quiet-to-release changes occur from step 400 to 500 (`6.665e-8` to `1.778e-6`) and from 700 to 800 (`3.131e-8` to `2.297e-6`).

Across the last formation episode, the aggregate L2 change over all 15 parameter tensors from step 1500 to 1600 is 5.082, while the held-out lower-tail margin crosses zero. The initiating clip occurrence is `occ_4a01db982f915de3be28b167c8cb9cea42e28d357a5a32c21809780f1c3c72ff`; its first atomic fact block is `factblock_d209639de3ab8deb5d482e95d36bac2659b239c9a39a4ae2758c880a15bb6248`. The step-1600 optimizer occurrence is `occ_5bd64ffe3021c90b0b6201c5895ab42063bca6d35680194bd9779e1559f2f652`. For the tied token/head parameter, `edge_dd9f8c3e25c34f5ca5de3d565083634a2f24f3f631f83df459d92e4cb541` is the primitive `GeneratedOrigin` from version 1599 to version 1600. The evaluation occurrence `occ_b6e97eeea7a656bf62d9b2eee05970d3568ba19c12092c86002913166e22a0d5` realizes `factblock_ab4b02e2159dd7b8e3bc39bc22ed60d28ed3a8618b8a7ecd0545e45329572daa` and reads the resulting exact parameter objects.

The executable state therefore tracks train/validation state, validation log odds as a portable rule-margin proxy, the positive log-odds gain, clip-episode starts, and inferred mature pulse period. Its transitions are `FITTING_SAMPLES -> SAMPLE_MEMORIZATION -> RULE_EXTRACTION -> RULE_MARGIN_CROSSING -> RULE_GENERALIZED`, with a reversible `SHOCK_DEGRADED_RECOVERING` state after formation.

## Prefix-only forecast self-test

I truncated the graph at the earliest legal prediction cut, step 500. The cut has three prior train evaluations at 1.0, current validation accuracy 0.269, and no prior transition. From only steps 100 and 500, the endpoint validation-log-odds gain is 0.002889 per optimizer step. Solving for validation accuracy 0.9 gives step 1606.8, which rounds to evaluation step 1600.

The three prefix pulse starts are 19, 179, and 452. Extrapolating their two recurrence gaps gives a mature period of 369 steps. The candidate uses that recurrence, rather than stored future steps, to emit inclusive instability-risk intervals and its future finite-state evolution. On the sealed discovery suffix, the smooth logistic curve has RMSE 0.1283; the frozen 0.10 oscillator-risk decrement reduces RMSE to 0.1167. Both are below the required normalized RMSE maximum of 0.2.

## Post-formation stability and recovery

Stable generalization is falsified. After the formal transition, validation falls below 0.9 at inclusive grid intervals `[1900,1900]`, `[4500,4500]`, `[6300,6300]`, `[7100,7100]`, `[7800,7800]`, `[8200,8200]`, and `[8600,8600]`. Every one returns above 0.9 at the immediately following evaluation, so persistent degradation is also falsified. The supported enum is `TRANSIENT_DEGRADATION_RECOVERY`.

The observable precondition is a quiet low-gradient phase followed by a recurrent release. For example, the 100 steps before evaluation 4500 have median pre-clip norm 0.0007; step 4499 jumps to 1.069 and is clipped. The exact optimizer occurrence `occ_a003b476c7730441706151fd5fdc8ec2563897f0ba9c8ad35216325d592b5e0b` forms parameter version 4500. Primitive edge `edge_ccb40f69aa3e9d09a77de256e89408cb429d6541ffeac8bcbdbebb9e605fc725` binds the tied token/head parameter version 4499 to 4500. Evaluation occurrence `occ_c29a656359e7da6a18eaa0c742e23669ca0ac8fe1630f529f94a6823be37a1a2` then records train/validation accuracies 0.274/0.274 and mean held-out correct margin -5.946.

Recovery is not inferred from proximity. The collapsed `theta_4500` is the `GeneratedOrigin` read by subsequent full-batch forwards; their losses realize backward facts, which form gradients; clip facts bound those gradients to clipped outcomes; optimizer facts read those outcomes plus Adam states and form successive parameter versions. At evaluation 4600, occurrence `occ_fe9f38ebdebec4ac193de429e5292cd6217990689845a11ff63f9e51f3cce1d6` reads `theta_4600` and records train/validation 1.000/0.995 with mean rule margin +6.892. The aggregate parameter L2 motion from 4500 to 4600 is 5.713. Equivalent exact degradation/recovery chains occur at 6300/6400, 7800/7900, and 8200/8300.

The mature-period forecast from the prefix places risk centers about every 369 steps after formation. With the sealed ±75-step uncertainty, it contains all seven observed below-threshold instability evaluations without encoding their steps in the executable.

## Falsification record

### H1 — selected: accumulated rule-margin basins plus clipped-gradient relaxation

Prediction: memorization precedes held-out lower-tail margin formation; recurrent clip episodes produce stepwise margin gains; after formation the same episodes can transiently remove margin and corrective exact-gradient paths restore it. I attempted to falsify H1 by searching for a passing three-evaluation window before a sustained pulse, a formation pulse not connected through optimizer facts to the evaluated parameter version, a post-formation collapse without a preceding release, or a collapse without an exact corrective-gradient recovery chain. None occurs. All four positive predictions are present, and the transition episode, parameter `GeneratedOrigin`, evaluation binding, subsequent collapses, and recoveries are cited above. H1 survives.

### H2 — rejected: ordinary sample memorization or training-loss threshold is the capability

Prediction: transition should coincide with training accuracy/loss convergence. It does not. Training accuracy is already 1.0 at step 100, fifteen evaluation intervals before the transition, while validation is 0.104 and the mean correct held-out margin is -4.145. At step 4500 the recorded current training loss is only 0.000949 immediately before a parameter update whose exact evaluation has train accuracy 0.274, showing that a small pre-update training loss is not a stable capability state. H2 is falsified.

### H3 — rejected: batch novelty or changing sample composition creates the rule

Prediction: formation steps should read new or redistributed sample sources. Instead, the training-batch input hash `83a27e7e4bdaa2ae7bc0ee03bd47f6e065056370bf453773b97eeb46edb6df61` and target hash `97264ccc2c4ae2bac49572f91f55e183d7be69d6496c1c022dcde0585bcd066e` each occur at all 10,000 batch materializations. Each contains the same 317 sample identities. H3 is falsified by exact source identity, not a step join.

### H4 — rejected as sufficient: any large clipped-gradient pulse causes degradation

Prediction: larger pre-clip norms should always produce larger accuracy loss. They do not. The episode beginning at 3438 reaches norm 107.143, yet evaluation 3500 is train/validation 1.000/1.000. The episode beginning at 5201 reaches 124.559, yet evaluation 5300 is also 1.000/1.000. Pulse impact depends on the current rule-margin basin and oscillator phase. This falsifies clipping magnitude alone while preserving it as a necessary transition driver in H1.

### Stability hypothesis — rejected: capability is permanently stable after formation

The seven below-threshold intervals and their exact next-grid recoveries falsify permanence. Because no observed failure persists for two evaluation intervals and the endpoint remains 1.0, persistent degradation is also unsupported. The graph supports transient degradation with recovery.

## Intervention

The executable intervention uses only `before_optimizer_step` and the allowed optimizer-group-hyperparameter mutation. It captures native learning rates, sets current group learning rates to zero for exactly 1600 hook calls, then restores the captured values. No samples, validation facts, parameters, gradients, task, evaluator, or capture behavior are altered.

Direction is `DELAY`; predicted transition shift is `[1400,2000]`, exceeding the required 600-step effective shift. During the pause, `rule_log_odds` and oscillator phase cannot advance through parameter motion; after restoration the original mechanism resumes. Consequently the predicted post-formation stability effect is `NO_CHANGE`: the instability/recovery oscillator is translated rather than removed.
