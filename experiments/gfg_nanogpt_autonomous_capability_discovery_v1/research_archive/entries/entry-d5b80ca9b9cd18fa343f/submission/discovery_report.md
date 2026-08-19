# Executable mechanism-discovery report

## Result

The selected mechanism is the executable in `mechanism.py`. It treats capability formation as a change from a memorization state to a positive held-out algebraic rule margin under the exact AdamW parameter/optimizer-state chain. A slow decay-exposure state advances rule formation; a separate optimizer-stress phase explains why a formed rule can degrade and recover. The supporting report is not used at replay time.

On the discovery graph, the contract transition is optimizer step **1900**: steps 1900, 2000 and 2100 are the first consecutive three-point window with training accuracy at least 0.99 and validation accuracy at least 0.9 after an earlier validation result at or below 0.3. The stability result is **TRANSIENT_DEGRADATION_RECOVERY**, not stable permanence: multiple later exact parameter versions lose capability and recover at the next evaluation.

The sealed candidate contains no run/date lookup, raw discovery token map, frozen answer trajectory, filesystem access, or future GFG read. `initialize` reads only the supplied prefix and returns canonical finite state; `forecast` evolves that state to step 10000.

## Query boundary and exact task structure

All scientific input came from the validated participant GFG. Derived quantities below preserve the source-object, occurrence, outcome and role tuples in their fact blocks; no facts were joined merely by equal step or equal value.

The initial full-batch occurrence `occ_10bace233ceff9d389539539b8f34521532eac7311b793176f7a386784e52382` realizes fact block `factblock_7efe76ccf0a227c13051ad9cfd669a388c747de67e7dbfbbf6f9c8e45803cdf0`. Its exact `training_task` source is `obj_018c2e327d7c6fc249f5c4d48c9413f0abcfbeff7619f9f9b1d372f16098f697`, and its two separate outcomes are training inputs `obj_22325fab349ee60fe771275fa2daa4ab84b8111920bc2c9d6f79f5a739c386a2` and targets `obj_c8fb3261a0795290389cb806bf7a0dcdce283295e2634a61ba0378edcab5d655`. The objects contain 317 distinct training examples.

Writing a latent coordinate for each of the 23 non-operator tokens, every training fact supplies one equation

`coordinate(outcome) = coordinate(left) + coordinate(right) (mod 23)`.

Gaussian elimination over the 317 exact equations has rank 22 and a one-dimensional nullspace. Fixing its arbitrary nonzero scale produces 23 distinct coordinates and satisfies all 317 equations; 198 observed reversed-input pairs also have identical outcomes. Thus the training graph identifies a cyclic addition rule under an arbitrary token permutation. The executable repeats this solve from each unseen prefix; the discovery permutation itself is not stored.

The optimizer configuration object is `obj_da2829e61f93dd4bc2897d220a554791ad8db1ac3ff5b66b55341acd55621a2c`: AdamW, learning rate 0.003, betas (0.9, 0.98), clipping threshold 1.0, and a decayed group with weight decay 1.0. This object is an `optimizer_configuration` source, rather than an inferred timestamp association, in each optimizer update fact.

## Formation evidence

Early memorization is not rule formation. At step 100, evaluation occurrence `occ_0c1b875c44c35453540a0fae81dc1a585bdeb5ca25f12a089c63dc12366f97af` forms metrics object `obj_4707da50f48e5f06cbfa3802c46a44175bd621a93c04dcf1a396e37155408f2f`: training accuracy 1.0 but validation accuracy 0.0235849. Its atomic evaluation fact block is `factblock_8eb525b3b7699cb147e57bd9e23489cfc4ee76281f48022c0ab6328b0876f944` and has 15 exact `evaluated_parameter_version` sources.

For each evaluation I decoded the held-out label only from the training equations, then computed the correct logit minus the largest incorrect logit. The 10th percentile of that margin is the continuous rule state: crossing zero means at least 90% of held-out algebraic cases have the correct class above every alternative. Separately, I ordered the token-embedding rows by their inferred cyclic coordinate and measured the largest nonzero Fourier-pair energy divided by total nonzero Fourier energy. These are derived from exact materialized tensors, not extra outcomes.

At initialization the executable also expands the prefix fact blocks and accepts the WTE version for this calculation only when its exact object ID is an `evaluated_parameter_version` source of the fact block whose outcomes include the selected validation-logits object. The optimizer-step filter is therefore a query bound, not a temporal join used to invent dependence.

| Step | Train accuracy | Validation accuracy | Rule-margin q10 | Cyclic spectral concentration | Exact WTE parameter result |
|---:|---:|---:|---:|---:|---|
| 100 | 1.0000 | 0.0236 | -8.2290 | 0.1713 | `obj_291be941e7eaff54b83df024a693d444eb360bf9f07753b4c567185643033131` |
| 1100 | 1.0000 | 0.2123 | -8.4323 | 0.1498 | `obj_817117a02f61430b77d9025c6f195542e15da4461171bb51a3c342cf556ba1b2` |
| 1500 | 0.9968 | 0.4340 | -5.3038 | 0.1807 | `obj_a7250449b171d283f69d3b36e48ceb4ee2a0e8a413ccd089f45c741b896c4aa5` |
| 1800 | 1.0000 | 0.7406 | -2.7719 | 0.1736 | `obj_d4f23a0bfed22f9087f28cbd6271f122afd13493ce584bf38122f5080d1de875` |
| 1900 | 1.0000 | 0.9151 | +0.4251 | 0.2144 | `obj_5ce4e1231f6efcfb79ce25a9caedceef89571332722291052fe03e5a3e34332d` |
| 2300 | 1.0000 | 0.9953 | +2.3529 | 0.2193 | `obj_376961078325cf40291448cd91c4cba903af1280b60243b5a848e9c47b420310` |

The step-1900 WTE object was formed by optimizer occurrence `occ_3a60bf05f648710c67fca9f04cad95e600bda14c83ddeefb78191ddfa9ebe1b5`, fact block `factblock_d7911f48a9f2427f12119e0ead6d6182b8fcc4d4a0f8bbf2ed3e784380ebebdd`. Its six source roles are the prior parameter, clipped gradient, optimizer configuration, prior first moment, prior second moment and prior optimizer step. Primitive edge `edge_dbd832c9b9ac38f3647b59433abcb961e02a732459bbc42d18c28446a63bbed8` records the `GeneratedOrigin` from WTE version 1899 to version 1900.

The transition itself is bound to those exact parameter results:

| Step | Evaluation fact block | Metrics result | Train | Validation |
|---:|---|---|---:|---:|
| 1900 | `factblock_4b681123851d4ad1682c6c78e5d2dde0aca888ab1665a4996bfb9ef6852ffe15` | `obj_8185a4d7efd6be2daaf616ec62754590f5828b98d7491abd63bbd94dbd8b2f64` | 1.0000 | 0.9151 |
| 2000 | `factblock_9a0b7d1119cc6243763ed4023c7f5e7e1094a1a4d47a4ccef3a6a668df96c246` | `obj_517fafcec0bafab375e547eff6fef3f059d91725c0d0b55cad5da5a26ec34bfe` | 1.0000 | 0.9434 |
| 2100 | `factblock_e2e39ea7e3e2d7b8a05e126acec684d9cfe55d99a9b1d1e92ffc28c7ccd4a582` | `obj_1e327c9bef4132570dfbce7c9d223b9dab7b568757b197fd100752ccc64ec74b` | 1.0000 | 0.9575 |

At step 1900 the validation token-input object `obj_a5df4cc6ab3dd63cc7183361511f8ec6e3b1e22665ebd4351ff87e25ba0f7172` is a `layer_input_0` source in fact block `factblock_fc1d4ac56754ef2f9096c2d4d71c3fcc13f91a6013b0cd328d042f0f8ad2e98f`; the evaluation validation-logits result is `obj_a6f363dea2653c4543f88a626740fc01433207d71ed9a662f71608b05dfe4137`. Those exact objects, plus the training equations above, are the boundary of the margin derivation.

## Post-formation stability

Permanent stability is directly falsified. The graph contains repeated low-accuracy parameter results followed by recovery under later parameter versions:

| Degradation step | Train / validation / loss | Global raw gradient norm at same parameter version | Next evaluation and recovery |
|---:|---|---:|---|
| 2200 | 0.9432 / 0.5189 / 0.4638 | 3.6895, clipped to 1.0000 | 2300: 1.0000 / 0.9953 |
| 4100 | 0.9369 / 0.7217 / 0.2260 | 5.3645, clipped to 1.0000 | 4200: 1.0000 / 0.9623 |
| 6900 | 0.4984 / 0.3302 / 2.3991 | 21.8486, clipped to 1.0000 | 7000: 1.0000 / 0.9623 |
| 8600 | 0.4606 / 0.3255 / 2.3345 | 35.8003, clipped to 1.0000 | 8700: 1.0000 / 0.9340 |
| 9900 | 0.2145 / 0.1132 / 4.8609 | 25.0058, clipped to 1.0000 | 10000: 1.0000 / 0.9387 |

There are shallower sub-0.9 dips at 3400, 8200 and 9100, also followed by recovery. The raw-gradient facts at a degradation step occur after that step's evaluation in declared program order, so I use them as the exact response of the same parameter version, not as a retroactive cause of its evaluation. The cause-side state is the parameter plus Adam moment results formed by the preceding optimizer occurrence; both are retained in executable state diagnostics.

Representative exact identities are:

- Step 2200 evaluation occurrence `occ_cad7482981c28ee5a41712fbb8a7039b3faa76882b15813f5ce89ed7f5151db4`, fact block `factblock_a3dc882abbba4686911292ba39bd43b33d74b0f809717363a8f50d7295f64682`, metrics result `obj_532736d6853fa5201eaf78d59e543b45285ef155f01a4d9f955f68f245d02ead`; WTE version 2200 is formed by optimizer occurrence `occ_841de0ccf60cbd39f83b63e3e7f385d793f0aafc745bb1477b753f422031ae13` and update fact `factblock_ad66c2a32613631e75e75e72e3af099bf012e14f662494311a0c789c92bbe6e4`.
- Step 2200 backward occurrence `occ_883aad764064c5c81ad38b2141f8da5a71bba15511a041dcc28b4065c274b6d4` forms WTE gradient `obj_fd58c5e60464911cf4aa0ebf54b7a3727a1a0faa9016349916abf1cd825cbbef` in `factblock_40203f5cb98dc2b0448099424a9824a135bb54aabe222b1fec4e47f63b7e9fc7`; clip occurrence `occ_8ee176f790396409cee2fc0281526fb79d480f0f14fedb191dd324933274cfa2` forms the clipped result in `factblock_cc655586397d47273c34fb03b57a4a99864826cde03acbe457bfe2b072896880`.
- Step 2300 evaluation fact `factblock_77a5ed819d5a18a8dc5c93cd7629c938574ee445ed2875107b0cb886969dac75` forms recovery metrics `obj_e43b0561701f2c6e373e20314d83fdfc9ab327a9256e2582d0d4c35ff5cd0c54` from its 15 exact parameter-version sources.
- Later evaluation/recovery pairs are `factblock_491301b6aa04544b3fd5b14f60c6892539b0e7b819757cce194bcd4e87af792f` / `factblock_dde819ed54a2ce9230a8406fe94a6904c53523b2cf1b15d00ccc744f6a3f248b` (4100/4200), `factblock_ce04da1af5315720d8cf79ffed6d51d319dcd1305cdd8f24583c09a2ebe9898a` / `factblock_2db4f4b87a017d014220e49ecf18af9fe896af37b9db7168d154065e0b4bca6b` (6900/7000), and `factblock_9a347f2b5e43658ac30e3eb1b2facdffc857435119331b55c4ded78d344298dc` / `factblock_54ad22b4d9b1ca5ae5f80a4b6aa2f34297e550e428462b67987788320d0933cc` (8600/8700).

Stress peaks on the evaluation grid recur near a roughly 400-step phase, but observed peak separations range from 300 to 500 steps and their amplitudes vary greatly. The executable therefore predicts inclusive 200-step vulnerability intervals around a 400-step mean phase and only a bounded mean dip in the numeric curve. It does not claim a fixed catastrophic amplitude or exact invariant period.

## Mechanisms proposed and falsification attempts

### H1 — selected: decay-exposure rule margin plus optimizer stress

AdamW update facts propagate parameter, gradient and both moment states. Training equations identify the cyclic rule; the held-out q10 rule margin changes from negative to positive exactly at the qualifying transition, while cyclic spectral concentration increases. After formation, a distinct optimizer/moment stress phase produces clipped high-gradient responses and recoveries. This hypothesis survives the required prefix replays below at normalized RMSE below 0.2.

Falsification attempt: remove stress phase and layer-scale state. This makes post-formation states with essentially equal memorization and high rule score operationally identical, although one advances to a large degradation and another remains near the plateau. The reduction is rejected. Falsification attempt: assert an exact 400-step pulse. The observed 300–500 step peak spacing rejects that stronger invariant; the selected mechanism widens timing and amplitude uncertainty instead.

### H2 — rejected: training memorization drives the transition

This predicts formation soon after training accuracy reaches 0.99. It fails immediately: step 100 has training accuracy 1.0 and validation accuracy 0.0236, while the qualifying rule transition is 1800 steps later. It also cannot distinguish step 2100 (1.0/0.9575) from step 6900 (0.4984/0.3302) using its proposed state update. Training memorization remains a gate in the selected state, but it is insufficient.

### H3 — rejected: unconstrained retrospective sigmoid or recent-slope extrapolation

Eight-point logit-linear fits made using only each prefix predicted the 0.9 crossing at 2479 (cut 1200), 2582 (cut 1300), 2625 (cut 1500), 2237 (cut 1700) and 2154 (cut 1800). These inconsistent predictions miss a 500-step interval at several early admissible cuts. The selected law instead uses the repeatable graph-defined rule-onset macrostate (first validation score at least 0.3), decay exposure, and a frozen 600-step lag.

### H4 — rejected stability hypothesis: formation is permanently stable

The exact evaluation and recovery facts at 2200/2300, 4100/4200, 6900/7000, 8600/8700 and 9900/10000 reject both `STABLE` and `PERSISTENT_DEGRADATION`. Recovery always appears on the next 100-step evaluation in these severe cases. The supported class is `TRANSIENT_DEGRADATION_RECOVERY`.

## State sufficiency and prefix-only replay

The executable state is not just optimizer step. It contains memorization score, rule score, rule-onset anchor, decay exposure, inferred task rank, held-out rule-margin q10, cyclic spectral concentration, layer-norm scale, raw/clipped gradient norms, Adam first/second-moment norms, and stress phase. Tensor diagnostics are computed only when present in the supplied prefix; unavailable evidence is explicit `null`, never imputed from a future.

Counterexample search produced these state revisions:

| Proposed reduction | Counterexample | Revision |
|---|---|---|
| Training accuracy/loss only | Train accuracy is 1.0 at steps 100 and 2100 but validation is 0.0236 versus 0.9575. | Add algebraic rule score/margin and task structure. |
| Rule score only after formation | High-rule states are followed by both plateau behavior and sharp degradation. | Add optimizer moments, gradient stress, layer scale and phase. |
| Exact fixed-period stress | Observed peak gaps are 300–500, not constant. | Use an interval-valued phase and bounded mean amplitude. |
| Minimal curve fit | Prefix crossing estimates vary by hundreds of steps. | Add the rule-onset anchor and decay-exposure law. |

For each replay, only evaluations/tensors at or before the cut initialized the state; `forecast` then ran without a GFG handle. RMSE is on the hidden discovery suffix and is already normalized to the [0,1] accuracy range.

| Prefix cut | Validation at cut | Predicted center | 200-step interval | 500-step interval | Suffix NRMSE |
|---:|---:|---:|---|---|---:|
| 1100 | 0.2123 | 1900 | [1800, 2000] | [1700, 2100] | 0.1394 |
| 1200 | 0.2736 | 1900 | [1800, 2000] | [1700, 2100] | 0.1402 |
| 1300 | 0.3302 | 1900 | [1800, 2000] | [1700, 2100] | 0.1410 |
| 1400 | 0.3632 | 1900 | [1800, 2000] | [1700, 2100] | 0.1418 |
| 1500 | 0.4340 | 1900 | [1800, 2000] | [1700, 2100] | 0.1427 |
| 1600 | 0.5755 | 1900 | [1800, 2000] | [1700, 2100] | 0.1432 |
| 1700 | 0.6887 | 1900 | [1800, 2000] | [1700, 2100] | 0.1440 |
| 1800 | 0.7406 | 1900 | [1800, 2000] | [1700, 2100] | 0.1449 |

Additional cuts at 1900, 2100 and 2300 test at/after formation; suffix NRMSE is 0.1457, 0.1475 and 0.1426 respectively, and the detected transition remains 1900 once the three-point outcome is present. These tests include pre-transition, near-transition and post-transition regions. Failures of H2 and H3 were retained rather than hidden by refitting the report.

Operational closure is asserted at the forecast contract tolerance, not at exact next-logit equality: every admissible pre-transition replay is below NRMSE 0.2 and contains the transition in both intervals. Catastrophic pulse amplitude remains a declared uncertainty because one graph cannot establish it as cross-run invariant.

## Report-to-code correspondence

| Mechanism claim | Primitive GFG evidence | Executable state field | Update-law location |
|---|---|---|---|
| Training facts define a cyclic rule under token relabeling. | Step-0 batch occurrence/fact and exact input/target objects listed above. | `task_structure_rank`, `task_structure_size` | `_latent_addition_map`, `_tensor_features` |
| Memorization precedes rule formation. | Step-100 evaluation fact and metrics result. | `memorization_score`, `rule_score` | `initialize`, `_preformation_curve` |
| AdamW decay exposure advances formation. | Optimizer configuration plus six-source update facts and `GeneratedOrigin` parameter chain. | `optimizer_learning_rate`, `optimizer_weight_decay`, `decay_exposure` | `initialize`, `forecast` state evolution |
| Positive held-out q10 margin is the rule transition state. | Validation token-input/language-model/evaluation fact blocks and logits results. | `rule_margin_q10`, `rule_score`, `predicted_transition_step` | `_tensor_features`, `forecast` q10 and log-odds margin updates |
| Cyclic representation concentrates as rule state forms. | Exact WTE parameter versions formed by optimizer facts. | `algebraic_spectral_concentration` | `_tensor_features`, spectral state update |
| Formed capability has transient optimizer stress. | Exact degradation/recovery evaluation facts; backward and clip facts. | gradient/moment norms, `layernorm_scale`, `stress_phase_steps` | `_stress`, post-formation curve and intervals |
| Pulse amplitude is not invariant. | 300–500-step peak gaps and unequal degradation magnitudes. | interval list and bounded `optimizer_stress` | `_stress`, interval loop |

Every state field is either updated in `forecast`, frozen from the allowed prefix as a diagnostic/configuration, or explicitly nullable when the prefix interface does not expose materialized tensors.

## Intervention and complete state audit

`intervention.py` applies only at `before_optimizer_step`. It records each native optimizer-group learning rate, sets the current rates to zero for exactly 800 optimizer calls, then restores them. This is an allowed optimizer-group hyperparameter mutation. It predicts **DELAY** with shift interval **[600, 1100]** steps, satisfying the minimum effective shift criterion.

This is not modeled as a pure translation:

| State or operation | During pause |
|---|---|
| Parameters and decoupled weight-decay progress | Frozen because group LR is zero. |
| Rule score/margin and cyclic representation | Predicted to remain near intervention-start state. |
| Training forward, loss and raw gradient | Continue. |
| Gradient clipping | Continues under the native threshold. |
| Adam `exp_avg`, `exp_avg_sq` and step counters | Continue to update. |
| Stress phase | Rephased by the parameter-progress / optimizer-age mismatch. |
| Group learning rates | Restored to their exact captured values after call 800. |

The mechanism-state prediction is therefore 800 suppressed parameter-progress steps but 800 advanced optimizer-age steps. Moment adaptation makes the shift interval wider than the nominal pause. The predicted stability effect is **NO_CHANGE**: pulse timing is rephased, but the intervention does not remove moment/gradient feedback or change the eventual learning rate, clipping threshold or parameter groups, so the long-run class remains transient degradation with recovery.

## Invariance and uncertainty audit

Fixed structural parts are the contract transition definition, exact GFG traversal, prefix-only boundary, algebraic-equation solve, AdamW state categories, and executable update form. State inferred afresh from each prefix includes token coordinates (used transiently, not serialized), rule anchor, current score/margin, optimizer configuration, spectral concentration, moment/gradient norms and stress phase.

The 600-step anchor lag, 400-step mean stress period, plateau and pulse depth are discovery-estimated parameters. Only one run is available, so they are not presented as established cross-run laws. Fixed-period and fixed-amplitude alternatives were explicitly falsified by the discovery sequence; timing is therefore interval-valued and amplitude is conservative. The 200-step transition interval is an extra precision diagnostic, while the 400-wide primary interval carries cross-initialization/task uncertainty. The unseen run may change task, split, token map, initialization and data order; the candidate uses none of those as an identity key and recomputes all available state from its prefix.
