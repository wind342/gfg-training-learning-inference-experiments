# Discovery report: executable rule-margin relaxation mechanism

## Result and authority

The sealed scientific claim is `CapabilityFormationMechanism` in `mechanism.py`, not this report. It forecasts a transition by optimizer step 10,000 from a finite prefix state consisting of memorization strength, rule logit, and progress through a weight-decay/Adam relaxation-burst cycle. The discovered run's contract transition is optimizer step **1500**. Its post-formation classification is **TRANSIENT_DEGRADATION_RECOVERY**.

The graph is a closed, participant-safe capture: 111,313 blocks, 1,062,946 objects, 111,313 occurrences, 4,642,344 atomic facts, 711,267 explicit edges, and 101 evaluations. `gfg_validation.json` reports PASS for exact fact expansion, object references, parameter-version chain, evaluation binding, program order, and absence of approximate temporal joins. I treated occurrence and object identity as the time authority throughout.

## Primitive traversal and the observed transition

The forward query was: exact training-task object → `batch_materialization` fact → batch input/target outcomes → `training_forward_and_loss` fact → loss outcome → `autograd_backward` facts → parameter-gradient outcomes → `gradient_clip` facts → clipped-gradient outcomes → `optimizer_parameter_update` facts → parameter and Adam-state outcomes. Each optimizer fact reads the exact `parameter_before_update`, `clipped_gradient`, `optimizer_configuration`, `exp_avg_before_update`, `exp_avg_sq_before_update`, and `step_before_update` sources. Primitive `GeneratedOrigin` edges connect those outcomes to the next parameter/state versions. I never joined records merely by equal value or timestamp.

The reverse query began at each concrete evaluation metric object and its `capability_evaluation` occurrence, then followed its fact block's 15 `evaluated_parameter_version` sources and task source to exact parameter results. From each parameter result I followed `GeneratedOrigin` to its forming optimizer fact and the fact's exact gradient/configuration/moment sources. `program_order` was used only to preserve declared execution order, never as a substitute for `reads_from`.

The frozen transition contract requires an earlier validation point at or below 0.3 followed by the first three-point train/validation window at least 0.99/0.9. The exact evidence is:

| theta / evaluation occurrence | train | validation | evaluation fact block |
|---|---:|---:|---|
| 1400 / `occ_eadb6aa0390d94672366937ded251250f4d59054d3631bd64d164d1abbb0a0ad` | 1.0000 | 0.669811 | `factblock_c98180f1358fad7635589e3d6ea7d72b1d0cc3e11820244a426eddaacf0b0dd2` |
| 1500 / `occ_e7d608b55a11d114110ee32ae21d42c9fd8b9f2fbc0de95876bf40f1432923e9` | 1.0000 | 0.929245 | `factblock_48762d3cf79034b8058ada5011a94ec4d7fc442032f693e3cf872bede1f4a428` |
| 1600 / `occ_4cba2fd2e99f2a9387f87a32ac14b8150820bb02b677e505c8f06ad80577fd7e` | 1.0000 | 0.948113 | `factblock_da8535e0bc9d8e39e8ebd66d528cc06a69c9b7958ffe3623706ca385305617f5` |
| 1700 / `occ_9a44ce86d1126f746fe632df01a6084ffc85cf549e5a04681a8b71a73f566b7c` | 1.0000 | 0.966981 | `factblock_b317f0582a3258b727836fdbf78871be58b358debc4cedd062b557a778e91fd6` |

Thus 1500 is the first qualifying window. Training accuracy had already reached 1.0 at step 100 while validation was 0.018868, so sample memorization is a necessary gate but cannot itself be the formation state.

## Mechanism discovery

The optimizer-configuration object is `obj_28c0cc00de601d18e2e0a6922cc1d98504e5307572e3a80da4819aa734a30520` (content `36e6ed…22c9`): AdamW, learning rate 0.003, betas (0.9, 0.98), clipping max norm 1.0, positive decoupled weight decay 1.0 for the decayed group, and zero decay for the other group. That same configuration content is an exact source in every one of the 10,000 optimizer updates.

After memorization, the graph exhibits a relaxation oscillator rather than monotone convergence. Contiguous `gradient_clip` occurrences whose recorded pre-clip `total_norm` is at least 1 have concrete peak steps:

`181, 423, 730, 1061, 1428, 1798, 2176, 2567, 2974, 3363, 3780, 4213, 4635, 5039, 5401, 5742, 6115, 6478, 6844, 7212, 7570, 7951, 8338, 8713, 9110, 9492, 9892`.

The transition-forming burst is the concrete clipping occurrence `occ_f8367724dc21332011c4bf21c5fb20dbb6cd5b6068e3d583ee15b8a68408ddc1` at step 1428: total norm 37.937111 and fact block `factblock_3e9d8d0f977d4e9e0090a033b1272d104f067670c6581ab558f0b366994b41f6`. Its clipped-gradient result is read by the exact optimizer facts, whose parameter and moment outcomes become later `GeneratedOrigin` sources. This burst falls between theta1400's 0.669811 validation state and theta1500's first qualifying 0.929245 state.

Materialized parameter and layer outcomes give an independent state check. From theta1400 to theta1500, the exact layer-0 attention matrix `transformer.h.0.attn.c_attn.weight` changes from object `obj_fe9d540aef4d6dd500febdd8a2e015f39ae6609578b2a42b8f3e17cf3de0f851` (L2 2.16899) to `obj_c7f3cbb68af3d97840875b777e8668b29cb77ae59dc1ffa408426bfcad71d61e` (L2 3.11898); `transformer.h.0.mlp.c_fc.weight` changes from `obj_7d06d1425a648c8bdba094c5b74eda80c23498417ca5809c9c9d10792e3b8c50` (L2 1.77658) to `obj_040f422b9c416c9702ea47c6fe3e19cd40d1793bf39576e1e9b4070f9ba3077e` (L2 2.86472). The validation block-1 activation changes from `obj_7374f63175d2354531e96fe009f6a77180692ee5d2be996061eb4d506aae4651` (RMS 0.071849) to `obj_a61695e1fc13e3b2200319929e538cbdef4cf3efaef629d5fae57f2c17498c22` (RMS 0.381868). These are exact formed outcomes, not value-based joins.

The executable compression is therefore:

1. `memorization_strength` gates whether a transition can satisfy the train criterion.
2. `rule_logit` is the prefix validation rule margin and directly generates the validation forecast through a sigmoid.
3. `decay_progress`, the realized burst intervals, and positive weight decay locate the next relaxation instability from prefix state.
4. A burst increases rule logit; the first burst that makes three future grid points exceed 0.9 forms the capability.
5. After formation, burst/evaluation resonance can temporarily suppress both train and validation performance; clipped gradients plus continuing Adam moment/parameter updates recover it.

This is a state law, not a claim that a burst at a particular absolute optimizer step is invariant.

## State closure and counterexample search

I explicitly searched for omitted-state counterexamples.

- Equal memorization states are incompatible: train accuracy is 1.0 from step 100 through the pre-transition region, yet validation ranges from 0.018868 to 0.698113 and then crosses 0.9. Training accuracy alone is not closed.
- Similar validation accuracy is incompatible without oscillator phase: theta1200, 1300, and 1400 have validation 0.674528, 0.698113, and 0.669811, but only the latter is immediately adjacent to the 1428 burst. The elapsed time since the concrete 1061 burst distinguishes the remaining future.
- Equal post-formation accuracy is incompatible without recovery state: many evaluation points are 1.0, but a nearby high-gradient episode can produce a later collapse while a point far from an episode remains stable.
- High gradient norm alone is also insufficient. The step-6478 peak is 155.402588 (`occ_3e63a7c0b2a4ee6c5541371498b38a14f7d06e39903932b8197bddd51fe489f5`), yet theta6500 remains train 0.993691 / validation 0.938679 and theta6600 recovers to 0.995283. Burst phase, rule margin, and recovery buffer must accompany norm state.
- A generated next state not represented by the update law was treated as a failure during development. Adding explicit burst gain and recovery resonance removed the observed formation and first-collapse failures; later rare resonance remains forecast uncertainty rather than a stored suffix.
- Replaying from a cut exactly at theta1800 initially exposed a closure failure: carrying only its low instantaneous rule margin predicted a prolonged loss, whereas the exact suffix recovered at theta1900. I retained `recovery_active_at_cut` and the prefix's pre-collapse `stable_rule_logit`; the executable now performs the evidenced one-grid recovery without reading the suffix.

No exact duplicate of the full compressed state occurs, so the discovery execution can only fail to falsify closure; it cannot prove cross-run sufficiency.

## Competing hypotheses and falsification

1. **Selected: weight-decay/Adam relaxation bursts increment a rule margin.** The optimizer configuration, exact gradient→clip→optimizer→parameter/moment paths, burst at 1428, layer/parameter outcome change, transition, repeated later bursts, and prefix replay all support it. It was not falsified within this execution.
2. **Rejected: training-loss or memorization threshold causes generalization.** Train accuracy is already 1.0 at step 100 with validation 0.018868 and remains 1.0 over more than a thousand steps before formation. Holding only this proposed state gives incompatible futures.
3. **Rejected: fixed-period or absolute-step clock.** The 26 inter-burst intervals range from 242 to 433 steps (median 376.5); the early intervals are 242, 307, 331, and 367. A fixed timestamp/period loses the state-dependent relaxation. The executable estimates phase and period anew from the unseen prefix.
4. **Rejected stability hypothesis: capability is permanently stable after formation.** Exact evaluation facts show repeated degradations followed by recovery, including severe failures at 1800, 9500, and 9900. Persistent degradation is also rejected because the next grid point recovers in every severe case.

## Prefix-only replay before sealing

For each cut I exposed only evaluations and clipping occurrences at or before that cut, initialized a new finite state, and forecast the withheld suffix without future graph reads. RMSE is on the withheld validation grid; the range is already normalized to an accuracy scale of one.

| cut | prefix validation | inferred next-period | 200-step interval | 500-step interval | suffix RMSE |
|---:|---:|---:|---:|---:|---:|
| 800 | 0.273585 | 346.0 | [1400,1600] | [1300,1700] | 0.1358 |
| 900 | 0.278302 | 346.0 | [1400,1600] | [1300,1700] | 0.1365 |
| 1000 | 0.316038 | 346.0 | [1400,1600] | [1300,1700] | 0.1378 |
| 1100 | 0.632075 | 375.5 | [1400,1600] | [1300,1700] | 0.1379 |
| 1200 | 0.674528 | 375.5 | [1400,1600] | [1300,1700] | 0.1385 |
| 1300 | 0.698113 | 375.5 | [1400,1600] | [1300,1700] | 0.1394 |
| 1400 | 0.669811 | 375.5 | [1400,1600] | [1400,1800] | 0.1401 |

All seven independently initialized 200- and 500-step intervals contain the actual step 1500; every 200 interval has width 200 and every primary interval has width 400. All suffix RMSE values are below 0.2.

I also replayed stability suffixes at and after the transition rather than stopping at formation:

| cut | region | recovery active at cut | suffix RMSE |
|---:|---|---|---:|
| 1500 | candidate crossing (window not yet complete) | false | 0.1228 |
| 1700 | completed formation window | false | 0.1243 |
| 1800 | inside first degradation | true | 0.1485 |
| 1900 | immediately recovered | false | 0.1494 |
| 5400 | inside later degradation | true | 0.1501 |
| 5500 | immediately recovered | false | 0.1659 |
| 9500 | inside severe degradation | true | 0.0187 |
| 9600 | immediately recovered | false | 0.0207 |

These cuts test before, at, and after formation and both sides of observed state transitions. All remain below the 0.2 curve gate. In particular, the explicit recovery state corrected the failed theta1800 replay without consulting theta1900 during initialization.

## Post-formation stability

The supported state is **TRANSIENT_DEGRADATION_RECOVERY**, not STABLE or PERSISTENT_DEGRADATION.

| burst/evaluation evidence | degradation | next-grid recovery |
|---|---|---|
| peak 1798: `occ_cd948e244e0ff70b8bad1c66f7cd7a9d5f6a2658d35857b23687233a6772c1cd`, norm 77.024544; theta1800 `occ_ab10ce89e2b4d61c86996887ce03dc3b25b466842a518175f41f247a19c6c473` | train 0.261830, validation 0.117925, norm 26.693605 | theta1900 `occ_82fc553c6f90064e63496f14e7e78b841d548eb813189b1a36da836cc857e527`: train 1.0, validation 0.995283, norm 0.002370 |
| theta5400 `occ_f6463cf487e842986a29755e2a07513ebede91ba48dfec5f3e82574540d82b42`, norm 31.411760; peak 5401 `occ_8ba64ce366584d036854c92069886398c82e3852fe1eba67269a26df13b05d7d`, norm 58.931824 | train 0.867508, validation 0.839623 | theta5500 `occ_c283441a26624b14140c7f4c04e4f516996b4c1554b0fcef150c599897474e0c`: train/validation 1.0, norm 0.002465 |
| peak 9492: `occ_8f466659e183e054e939088ad76566adc948588d1c2b12df425971a9e498dea3`, norm 88.207993; theta9500 `occ_f31555a077983264bfdb6a8e9448593fc899f14916a40edec15529b283d4e2b7` | train 0.555205, validation 0.400943, norm 12.655365 | theta9600 `occ_1ffe1a6f6b61db98d1f919a9d6d26f8885de06fb7ce48e93f8cb1748d1a4b201`: train/validation 1.0, norm 0.001546 |
| peak 9892: `occ_288fa5b2b03508b84401c9096c9b4ab78e776090a136efa8de5f45420388f616`, norm 80.693581; theta9900 `occ_1950a054ab198a2f6da74c0e587f6cc9f3c4a4bfb858a21700bde7955bc36d30` | train 0.274448, validation 0.179245, norm 6.500341 | theta10000 `occ_ee32793c55dde0f0e6583a3b684b9d70968da4ffcb5c7339bc4011bb219f0457`: train 1.0, validation 0.995283 |

Observable preconditions are advanced decay progress, a rising/high gradient-norm episode, and evaluation-grid resonance while the rule-margin buffer is finite. Recovery is not inferred from proximity: each clipping fact forms clipped-gradient outcomes; exact optimizer facts read those outcomes and the prior Adam `exp_avg`, `exp_avg_sq`, and step objects, form new moment/parameter outcomes, and `GeneratedOrigin` carries those exact results into later forwards and the recovered evaluation.

## Report-to-code correspondence and executable-use audit

| claim | primitive/derived GFG evidence | executable field | code path |
|---|---|---|---|
| memorization is necessary but insufficient | evaluation facts; train=1 at step100 while validation=0.018868 | `memorization_strength` | sustained-window train gate in `forecast` |
| relaxation phase predicts next burst | concrete clipping occurrence series and total norms | `decay_progress`, `estimated_next_period`, `positive_weight_decay` | next-event recurrence in `forecast` |
| bursts advance rule formation | 1061/1428 burst facts and later bound evaluations | `rule_logit`, `rule_logit_gain_per_burst` | sigmoid rule update and transition scan |
| capability can transiently degrade | exact 1798→1800→1900 and later fact chains | `adam_recovery_active`, `stability_buffer` | operational accuracy cap and instability intervals |
| a prefix cut inside degradation recovers | prior formed evaluations plus current sub-0.9 evaluation and continuing optimizer facts | `recovery_active_at_cut`, `stable_rule_logit` | next-grid recovery base in `forecast` |
| optimizer state mediates update/recovery | each optimizer fact reads clipped gradient and three prior Adam-state roles | update-law structure; no tensor is silently ignored in intervention audit | event recovery law; intervention state audit |

Perturbations at the step-1200 prefix confirmed operational use:

- Base: interval [1400,1600], predicted validation at 1500 = 0.930780.
- Decreasing `rule_logit` by 0.6 shifts the interval to [1800,2000] and lowers step-1500 prediction to 0.880665; increasing it by 0.6 raises the prediction to 0.960787.
- Decreasing `decay_progress` by 0.25 shifts the interval to [1500,1700].
- Setting burst gain to zero shifts formation to [4800,5000].
- Setting `memorization_strength` to 0.5 changes `will_transition` to false.
- `positive_weight_decay` changes the period equilibrium and therefore later curve and instability state. `stability_buffer` changes the executable mature-resonance width; `adam_recovery_active` directly caps the forecast curve.

`learning_rate`, raw burst peak norms, and observed prefix rows are explicitly diagnostic/frozen-prefix records; the mechanism spec does not mislabel them as operative causal state.

## Intervention and full state audit

`TrainingIntervention` declares only `before_optimizer_step` and changes only current optimizer-group `lr`. It captures each group's current rate, sets all rates to zero for 1200 optimizer applications, then restores the captured values. The predicted direction is **DELAY**, with signed (intervention minus baseline) shift **[600,1800]** optimizer steps. This meets the 600-step causal minimum throughout the eligible cut region because a prefix with validation at most 0.75 cannot change parameter formation during the 1200-step hold.

The intervention is not modeled as a pure time translation. State audit:

- Parameters and decoupled weight-decay progress are held because both the Adam update and decoupled decay are multiplied by learning rate.
- Rule logit is held because the parameter version is unchanged.
- Forward, loss, backward, raw gradients, and clipping continue.
- Adam step, `exp_avg`, and `exp_avg_sq` continue updating even while `lr=0`; the code records this explicitly.
- On restoration, formation dynamics resume from changed Adam moments, so the shift interval includes uncertainty rather than asserting an exact 1200-step translation.

The predicted stability effect is **NO_CHANGE**: the graph supports delay of formation but does not support a directional change from the long-run transient-degradation/recovery class after restoration. A restricted-runtime unit replay verified zero rates for exactly 1200 hook calls and restoration on call 1201. No counterfactual training branch was present in the scientific input, so the effect remains sealed for prospective validation rather than claimed observed.

## Cross-run invariance and limitations

Fixed structure: exact training/evaluation binding, the transition contract, gradient clipping, Adam moment roles, the relaxation-state form, and prefix-only update procedure. Prefix-inferred state: current rule logit, memorization, burst peaks, period, decay progress, and optimizer constants. Discovery-estimated parameters: total-norm threshold 1.0, equilibrium period 380, per-burst rule-logit gain 1.75, and drift 0.0004. Uncertainty: mature burst/evaluation resonance and the intervention's changed moment state.

The wide 242–433 interval variation falsifies a fixed-period claim. Passing the seven within-run cuts establishes only that this compression was not falsified by the discovery execution. The candidate contains no run/date key, absolute evidence path, or stored discovery suffix and makes no claim that correct GFG state semantics alone proves transport. Transport is tested only by the sealed unseen continuous execution.
