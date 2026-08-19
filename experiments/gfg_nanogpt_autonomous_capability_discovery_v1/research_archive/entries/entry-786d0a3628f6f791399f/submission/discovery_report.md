# Discovery report: clipped-burst rule-margin mechanism

## Result

The sealed scientific claim is `CapabilityFormationMechanism` in `mechanism.py`. It forecasts a capability transition by step 10,000 and initializes every operative quantity from the supplied prefix. On the discovery execution the formal transition is step **1500**: evaluations at 1500, 1600 and 1700 all have train accuracy 1.0 and validation accuracy at least 0.981, after an earlier validation accuracy below 0.3.

The selected finite state says that memorization and rule formation are separated by the lower tail of the validation correct-class logit margin. Full-batch AdamW evolves in long quiescent intervals punctuated by global-gradient bursts that are clipped at norm 1. Those bursts change the parameter/optimizer state. Formation occurs when the burst-conditioned update raises the lower-tail rule margin enough for the contract's three-point window. The same relaxation bursts continue after formation, so the capability is not dynamically stable in the strict sense: it undergoes transient degradation followed by recovery.

The forecasted stability class is therefore `TRANSIENT_DEGRADATION_RECOVERY`. Instability intervals are generated from prefix burst phase and recurrence uncertainty; they are not a stored list from this execution.

## GFG query semantics and evidence chain

All citations below refer to primitive objects, occurrences, one-outcome fact blocks, or primitive edges. A fact block is expanded only into its own source-to-outcome `reads_from` relations; sources and outcomes were never cross-joined. Derived statements use this traversal:

1. select an exact evaluation metric result and its realizing capability-evaluation occurrence;
2. follow that fact block's exact prediction, target and latest-loss sources;
3. follow the bound validation logits to the exact evaluated parameter objects;
4. for a parameter update, follow `GeneratedOrigin` from the prior parameter, and the update fact's clipped-gradient and Adam-state sources;
5. follow the clipped gradient to its exact unscaled gradient, global norm and clip configuration;
6. follow the gradient to the loss, and the loss to the concrete full-batch targets and forward logits.

`program_order` is used only to check declared execution order. Numeric equality, optimizer-step proximity and missing edges were not used as identity or dependency evidence.

### Concrete formation chain

- At step 1400, metric object `obj_cfb71756bd2401a5572d6f9947171c7cdd126d07b3a742b9c28c1f75472fb082` is formed by occurrence `occ_9fa8f8eb2a6b3179e54e1ff3fe189a126d2b751fd54aef2f72034da3a9283aae` in fact block `factblock_9b12bd490c51ff0aad4614360d56c0925374523d17d9e4ad09b5430362db5132`. It records train accuracy 1.0 and validation accuracy 0.688679. The materialized validation logits and exact targets give correct-class margin q10 = -3.7962, q25 = -0.9158 and median = 3.1978.
- Batch occurrence `occ_c712d4c4dfb734056929fb877ce3740f0d23197bcc0bce83603497f80e9e6d33` at step 1429 realizes input and target facts `factblock_0999bebd8e7d1a1751dd38108b953866e68e51089c91d0f31b899c4ea1b06dbe` and `factblock_976d6304a933a08e9ca60664241072dadec389f0ab1a9621c4036ef2e5edfd6b`. The batch has all 317 training samples; the selection order is an exact source of both outcomes.
- Loss occurrence `occ_ede9c8b5c6ea8cf60df7a3981c92c7779172a3e1b39b453ee10d262c8c27cffb` realizes `factblock_e2ef27a5f4bc586368cd78640003b49c01567d4307529096aa833e3e5783a5bf`, whose exact sources are training logits `obj_2cc9e75bb3669a0b4e72ca4cec7300d128473b461e704c2e3d6744a0decd53da` and batch targets `obj_1d560f13ef5c7fcf751f82b353256635d5b690c65b1bffa8b535b5c55c75bcfe`.
- Backward occurrence `occ_dc2aa4522c63ae81674b69af7cb21604d1f7aad376106836929291e9d925ad57` realizes the token-embedding gradient fact `factblock_947034742f67808d7a8579d69dc104fbbbf685950b38bad80e25a7434530e973`, reading the exact loss and differentiated parameter.
- Global-norm occurrence `occ_ad29d22a86bd5110213821098b92253ebd8aee342bdcdeca8d066942a1e1eee6` realizes `factblock_ab6e15ec11d6a51aa448f3ac8a50cc356bc175b43b56fe2eaf4e46ed6a6540ab`. Its outcome `obj_289e9a74e53405792150e46a699f0a51b12fcb303579e78be742cc3c828bf4e5` is 49.8211 and reads all 15 exact gradient objects. Clip occurrence `occ_099feeadd7449bf42ebccf64a600ffe8984e660f65291b46888cb4d5b3e33426` uses that norm, the unscaled token-embedding gradient and the norm-1 configuration in `factblock_bdb11ca4b50b241e40a02d5645996c7f09cbfbb603e7efb4e36dba6f530d931b`.
- Optimizer occurrence `occ_a2b07d27961d3555f84924ec5008162a57cc541973c77059ae6cd6937f057ddd` reads the exact prior token embedding, clipped gradient, configuration, first moment, second moment and step in `factblock_19712f3808c7a6e8339f7a8c735da5101424a8b1c228489ac9a2a4b3620e40d7`, forming parameter `obj_a0a48bc43e14fc480f57887fd188a831d1db68c3e1eea289ba5a78a88b1613bb`. Primitive edge `edge_f8ffdf9916403130315ee5239c828c7b671eb0389e64cb6847f27ab527428402` is the exact `GeneratedOrigin` relation from version 1429 object `obj_103d5f4e521a01e6baf6fbea9b0f5a4f92770035e085c45680ad99447feb08b4` to version 1430.
- At step 1500, `factblock_8f8525e0adbab46f042c9a47c3a6871c1bbf2d695a0f28d21e5eab31d2dab5c8` and occurrence `occ_0582f7300e49b6144e8cce450ce63d7379da333209851c274a165aee45f7481b` form metric `obj_e16a624002cf5e83febdb02f1534289bb08a34bde927a6f6e096e2fd80755eb1`: train accuracy 1.0 and validation accuracy 0.981132. The exact logit/target margins are now q10 = 4.2481, q25 = 5.3047 and median = 6.6801. Facts `factblock_1358fc6c380908df0edb06a021c70cb3013fc6852b448f83387c2671b9a31330` and `factblock_6f623dedbada02a3defacaa372a6cb85c83efb236eef97effcbbadeee8b775d2` record the sustaining 1600 and 1700 results (both 0.990566 validation accuracy).

This chain supports the update law's use of a clipped-burst phase and rule-margin reserve. It does not imply that the single step 1429 alone caused all formation; it is the observed final high-stress event in a prefix-conditioned sequence of parameter and optimizer states.

## Candidate mechanisms and falsification

### H1 — selected: burst-conditioned lower-tail rule margin

Hypothesis: the executable state must retain (a) current validation margin distribution, (b) clipped-burst phase/recurrence and (c) Adam moment tension. Formation is predicted when the remaining state-conditioned burst cycles raise the lower margin tail through the transition region. Post-formation bursts can erase margin temporarily, while full-batch updates restore it.

Evidence supporting H1:

| Evaluation | Validation accuracy | Margin q10 | Margin median | Latest major burst peak |
|---:|---:|---:|---:|---:|
| 800 | 0.2123 | -8.1988 | -4.2729 | 743 |
| 1000 | 0.2783 | -7.9140 | -3.8737 | 743 |
| 1100 | 0.3443 | -6.5706 | -1.1452 | 1069 |
| 1300 | 0.5708 | -5.0143 | 0.8846 | 1069 |
| 1400 | 0.6887 | -3.7962 | 3.1978 | 1069 |
| 1500 | 0.9811 | 4.2481 | 6.6801 | 1429 |

Falsification attempt: initialize and seal at historical cuts without reading later facts. At every listed cut, predict all later grid points, then compare to the recorded suffix.

| Prefix cut | State cycles remaining | Forecast transition | 200-step interval | Suffix curve RMSE |
|---:|---:|---:|---:|---:|
| 500 | 4 | 1500 | [1400, 1600] | 0.1761 |
| 800 | 2 | 1500 | [1400, 1600] | 0.1685 |
| 1000 | 2 | 1500 | [1400, 1600] | 0.1635 |
| 1100 | 1 | 1500 | [1400, 1600] | 0.1607 |
| 1200 | 1 | 1500 | [1400, 1600] | 0.1606 |
| 1300 | 1 | 1500 | [1400, 1600] | 0.1606 |
| 1400 | 1 | 1500 | [1400, 1600] | 0.1607 |

The cuts cover before and near formation. The suffix errors include every later transient failure; no discovery suffix values are read during each replay. H1 was not falsified within this execution, but this is not proof of cross-run transport.

### H2 — rejected: memorization/loss threshold

Hypothesis: near-zero training loss or perfect training accuracy is sufficient state. It fails closure. At step 100, fact `factblock_c99b1487ad7d3b0f5d59ce511e2efb26c037ee27cf46caa6845e02c3af6fdc29` records train accuracy 1.0, loss 0.00994 and validation accuracy only 0.0613. Training accuracy remains 1.0 and loss becomes much smaller through steps 400–1400 while validation futures remain incompatible. Equal perfect train accuracy is therefore not equal capability formation, and training loss is not a sufficient executable state.

### H3 — rejected: curve-only or absolute-clock transition

Hypothesis: extrapolate the recent validation-accuracy trend or use a fixed absolute transition step. A 300-step linear prefix extrapolation forecasts the 0.9 crossing at approximately 3756 from cut 500, 3373 from cut 800, 2198 from cut 1000, 2362 from cut 1100, 1913 from cut 1200, 1638 from cut 1300 and 1584 from cut 1400. It is retrospectively unstable and misses early prefix intervals. A fixed step 1500 can fit this one execution but has no cross-run state semantics and is deliberately absent from code. The selected executable changes its transition when its margin, cycle count, recurrence period or period drift is perturbed.

### H4 — stability test; stable and persistent alternatives rejected

The formed capability is neither strictly stable nor persistently lost:

- Step 2600 fact `factblock_2563e354a31f83e8bdfb7bc50fa8ba974c152a73e1023308ac6d6e271717516a` records validation accuracy 1.0. A norm-55.3504 burst at step 2694 is formed by `factblock_ae34ac28f50ea7e0d1ac926016cd2175f336e0493243f452c9dd990ca2efdcb0`; step 2700 fact `factblock_e9c7404b869ff7fa3631192f1d6e38623496ea5ba42f0efabd63a7e2256db082` drops to train 0.4227 and validation 0.1698. Step 2800 fact `factblock_b3e2e504df0179f420c225036e8fc8c2a5b03ca1fe6e025b8c95980b734602c1` recovers both to 1.0.
- Step 5400 is 1.0 (`factblock_4e20019b57faa995c1e387f5d5ec6bdab99d667fb52566ce5555f9964ddcf76b`), step 5500 is 0.1321 (`factblock_9ee5405811f5532048ba9017df2824aadf586443f7cb023dafa2684c99b3413b`), and step 5600 is 1.0 (`factblock_a45411a088253443d5214fbb4bc2da4a5e6b1439836cf6a7946ff2c1038d034a`).
- Step 7800 is 1.0; after the norm-78.3574 burst realized by `occ_04e9748b3fdc7b940a51389201509c05ce6accf2021c7fe40897478ae23ea70a`, accuracy is 0.4198 at 7900, 0.8868 at 8000 and 0.9764 at 8100. The exact metric facts are `factblock_0daa8e7b98d272b64a6422b99ff1018698ca9b254949156d24860da3c54a92e5`, `factblock_d14354a4a276cab9150bec4f03f0f6f8f898e57bad870277dc4c673e4a376633`, `factblock_2bf466c9a8a22a9b3f1875af686960b469e0cfd530f27d5fb74530973cc29cb3` and `factblock_198745985b79a2d433ddfd90389177ca7d6d91a1d08ad1289b415621cb216f35`.
- A norm-125.4153 burst outcome `obj_3569f0c83661020696a2e896f36b579c4fe6a5630dd57cd1aef3c30a252d95cd`, realized by `occ_13392dd75189d9e5bdefa4c6b1072f285eb4c113aae8aea4fd815acb8ace990b`, precedes the step-8400 validation drop to 0.3019 (`factblock_93ec97236f2a7ee7e72381d1cf79e6f84489d58ff0fdf236b76acd54ed435d86`) and step-8500 recovery to 0.9764 (`factblock_a1e6dc10eeb96ca0a4e6cedaf5d1d64274b875c5bece21114d6c506674b3c9ad`).

These exact formed outcomes falsify `STABLE`; their recoveries falsify `PERSISTENT_DEGRADATION`. The mechanism retains an oscillator phase and a margin reserve rather than hiding the result in a broader narrative.

## State-sufficiency and closure audit

Two matched-state counterexample searches shaped the state:

1. Validation accuracy 0.1321 occurs at both 500/600 before formation and at 5500 during a formed-state failure. The former next evaluations remain low; 5500 recovers to 1.0 at 5600. Accuracy alone is insufficient. `formation_status`, lower-tail margin, burst phase and optimizer tension separate the futures.
2. Validation accuracy at 1000 (0.2783) is close to that at 8400 (0.3019), but their next values are 0.3443 and 0.9764. The earlier state has never formed and has q10 -7.914; the later state is a formed capability inside a burst-recovery phase. This is a matched-observable/divergent-future counterexample to compressing state to validation accuracy.

Within the discovery execution, the full submitted state had no transition-interval counterexample at the seven replay cuts above. Exact late instability timing is not closed by the pre-formation prefix, so the executable widens burst-risk windows rather than claiming exact pulse times. Passing these replays does not establish a correct cross-run law.

## Report-to-code correspondence

| Mechanism claim | Primitive GFG evidence | Executable field | Code location |
|---|---|---|---|
| Formation requires rule-level margin, not train memorization | exact evaluation logits/targets and capability facts at 100, 800–1700 | `validation_accuracy`, `rule_margin_q10`, `cycles_remaining` | `_validation_margins`, `initialize`, `forecast` |
| Clipped relaxation events alter formation state | norm and clip facts at 1429, exact optimizer fact at 1430 | `major_burst_count`, `last_burst_peak_step`, `last_burst_period`, `burst_period_delta` | `_major_bursts`, `_period_state`, `_future_bursts` |
| Adam state changes burst vulnerability | optimizer update reads first/second moments and step | `optimizer_tension` | `_optimizer_tension`; hazard amplitude and risk-window width in `forecast` |
| Capability can degrade and recover | exact 2600–2800, 5400–5600, 7800–8100 and 8300–8500 facts | `burst_phase`, predicted margin reserve and burst stress | `forecast` generated state evolution and instability intervals |
| Pause delays formation but changes restart state | optimizer occurrence reads gradient, lr/configuration, moments and step | intervention retention fields and `formation_progress_frozen_updates` | `TrainingIntervention.apply` |

## Executable-use audit

Every field claimed operative has a forecast path:

- `validation_accuracy` anchors the generated curve's log odds.
- `rule_margin_q10` selects the remaining formation-cycle regime in `initialize`; at cut 1100 forcing it below -7 changes `cycles_remaining` from one to two and changes the predicted transition from 1500 to 1900.
- `last_burst_peak_step`, `last_burst_period` and `burst_period_delta` generate future burst centers. At cut 800, holding the observed period delta at zero changes the predicted transition from 1500 to 1400.
- `optimizer_tension` changes both the validation hazard decrement and every predicted instability-window width. Setting it to zero versus two changes hazard amplitude from 0.04 to 0.08 and changes the first half-width by 160 steps.
- `burst_phase` changes the first recurrence interval when a prefix ends inside an active burst; later phase is generated from burst centers. `last_gradient_total_norm` is not claimed as an independent forecast control after it has been compiled into phase and recurrence, so the raw norm is a diagnostic in the sealed transition.
- q25, median, recent validation points and train accuracy are retained as diagnostics and consistency checks. They are not described as independent controls.

Thus the executable is not controlled by an absolute step or a stored trajectory. Perturbing the operative prefix state changes required outputs.

## Intervention and complete state audit

The intervention predicts `DELAY` by 800–1200 optimizer steps. At `after_backward` it zeros only current training gradients. At `before_optimizer_step` it sets all current optimizer-group learning rates to zero for exactly 800 optimizer occurrences, then restores the captured native rates.

State changed, frozen or left evolving:

- parameters and learning-rate-coupled weight decay are held fixed by lr = 0;
- current gradients are explicitly zeroed, so no new training signal enters Adam moments;
- Adam first moments continue to decay by beta1 = 0.9, second moments by beta2 = 0.98, and optimizer step counters continue to advance;
- RNG and selection order continue, but every recorded training batch contains all 317 training samples, so the pause does not selectively inject or omit validation examples;
- validation data, labels, task, evaluator, capture and future parameter state are untouched.

The executable intervention state records all of these consequences. This is not claimed to be a pure time translation: moment decay changes the restart. Eight hundred suppressed parameter updates provide the lower shift bound; up to 400 steps of moment and clipping re-equilibration provide the upper bound. The predicted stability effect is `WORSEN` because reduced moment buffers increase normalized restart sensitivity and transient margin-erasure risk.

## Invariance and limitations

Structural parts of the candidate are the exact GFG traversal, full-prefix state extraction, margin definition, clip-relative burst detection and finite update equations. Prefix-inferred quantities are current margins, burst centers/period drift, formation status and Adam tension. Thresholds -7 and 0.18, the 10× major-burst ratio, delta damping 0.5 and the 0.95 transition target are estimated from the discovery execution. Late pulse timing and cross-run coefficient transport are uncertain.

Fixed-period, fixed-rate and fixed-transition alternatives were explicitly rejected. The burst recurrence begins from the unseen prefix's measured period and period change, and its change is damped; the transition moves when state moves. Nevertheless, one execution cannot establish invariance across an unseen task, token map, split, initialization and data order. The sealed prospective run is the required cross-run test. A future matched-state/divergent-future counterexample would falsify this compression and require additional optimizer or representation state.
