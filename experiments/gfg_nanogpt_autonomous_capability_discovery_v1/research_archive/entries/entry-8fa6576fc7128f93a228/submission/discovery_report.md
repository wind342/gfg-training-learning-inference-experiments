# Executable capability-formation mechanism

## Sealed claim

The scientific claim is the code in `mechanism.py`. It initializes only from the supplied GFG prefix and forecasts without another graph read. Its formation state is the held-out correct-class logit-margin distribution coupled to the prefix-observed growth of the final layer-normalization parameter norm. Its separate stability state is the ratio of that gain to the weight-decayed token/output circuit norm, together with gradient-relaxation phase, optimizer pressure, shock load, and recovery. The report documents that executable; it is not a substitute for it.

The discovery execution makes the formal capability transition at optimizer step **800**: an earlier evaluation is at or below 0.30, training accuracy is already 1.0, and validation accuracy at steps 800, 900, and 1000 is respectively 0.981132, 0.990566, and 0.990566. Post-formation behavior is **TRANSIENT_DEGRADATION_RECOVERY**, not stable and not persistently degraded.

## Exact GFG basis and traversal

The task graph contains 317 training and 212 validation pairs over 23 opaque operand tokens plus operator token 23. The split covers 529 distinct operand pairs. I did not infer equality from token values. Forward evidence was traversed by exact identity: dataset/batch sources to the concrete forward and loss occurrences, then loss to backward gradients, clipping, Adam parameter/state facts, the resulting parameter objects, and evaluation facts that read those exact objects. Reverse evidence began at a capability-evaluation result and followed its prediction, logits, exact evaluated-parameter, optimizer, gradient, and batch incidences. `program_order` was used only as ordering; it was never treated as `reads_from`. Fact blocks were kept atomic and never source/outcome Cartesian-recombined.

At cut 500, the validation-logits fact is `factblock_b8500bf0188359387b5c6f5f9aa96f9a5bb8728142df56e336344d29452fc41a` (occurrence `occ_389471c423725107e8fcae0d36726ea4411995c69796cbad88c1f82ca82594fe`). It reads the exact validation inputs and all 15 parameter-version objects. Its result is used by prediction fact `factblock_5a7e5d3bbf5e881920ad5bad10b98ab21809ef50f4687f78bf8813d3a26f2bfe`, and the resulting validation predictions participate in capability fact `factblock_f0c9b1bc4d502ba2467843f77cafa8df9d8b450c3e38bd5d28fccf6d84c9b646`.

At transition step 800, the corresponding exact facts are validation logits `factblock_f029219eee6bde14584847a8247f9bb4641d3eb34608d69f4cd37c0a38df9c95`, validation prediction `factblock_eb0e8c0a5c724d63d0a4cf866489c08222cbc032f9c00571ed481ab4cb17bfca`, and capability evaluation `factblock_bb8223335e63c83517d587fc52c675851cea9df35475b23a06ef06898b7f083a`. The exact capability occurrence is `occ_4f764bd04382f7a02e1631851c2407c080b480b73e251cc1a600c718bb510888`.

The formation coordinate is not an unattached diagnostic. At step 500, optimizer occurrence `occ_c1d54b1c3fe3866b9db4b91472239dab8c66133a26b0e4bad4f58827efe4bf76` forms the current final-layer-normalization object `obj_5f833366f9efea5440c6b266b96e263a87287655811c85e1250c8bb0956c6e70` in `factblock_96e55c3bfd2ae84ff940df10b3701d2c59a95566324709475a7c283da26c4e4d`. That object is one of the exact parameter sources of the cut-500 validation-logits fact. At step 800, `factblock_0898281ab8ac8f4537438892b2067f017344bbceae84f6af7424f86cfec672b5` forms its later version `obj_9760f21f26d9c975d7c062d49d1ffa0e4110e8f416d372cf5059d2fe98505143`, which is read by the step-800 evaluation. The selected state is therefore attached to the actual optimizer-to-evaluation formation chain.

## Formation observations

Correct-class margin means and lower tails were computed from each materialized validation-logits result and the exact validation-target object. Parameter norms came from the matching materialized optimizer result, not from similarly valued tensors.

| Step | Train acc. | Validation acc. | Mean validation margin | 10th-percentile margin | Final-LN norm | Token/output norm |
|---:|---:|---:|---:|---:|---:|---:|
| 300 | 1.0000 | 0.1415 | -3.6445 | -7.8204 | 11.7272 | 4.0069 |
| 400 | 1.0000 | 0.1887 | -3.6630 | -8.0512 | 12.5147 | 4.0281 |
| 500 | 1.0000 | 0.6792 | 1.4297 | -4.6452 | 13.5043 | 4.1489 |
| 600 | 1.0000 | 0.7547 | 2.3435 | -3.0589 | 14.0602 | 3.7007 |
| 700 | 1.0000 | 0.8349 | 3.5263 | -1.1007 | 14.7947 | 3.7025 |
| 800 | 1.0000 | 0.9811 | 5.2622 | 1.7499 | 15.7040 | 3.7992 |
| 900 | 1.0000 | 0.9906 | 5.4963 | 2.2572 | 16.2376 | 3.3161 |
| 1000 | 1.0000 | 0.9906 | 6.5941 | 3.3614 | 17.0118 | 3.3136 |

Training memorization is complete by step 100 while the held-out mean and lower-tail margins remain negative. The selected state therefore does not equate memorization with rule formation. `forecast` advances every prefix margin by twice the prospectively updated final-LN norm increment. Accuracy is regenerated as the fraction of margins above zero. The coefficient 2.0 is a discovery-run estimate; the final-LN rate and all margins are re-estimated from the unseen prefix. No absolute formation step is a model input.

## Post-formation stability

The graph contains repeated gradient-relaxation episodes. A concrete example is gradient-norm fact `factblock_afa28acdf4b15854f96294ab6a932388dfb280ad1207e2b3ebfcf7cadf887c5d` at training step 3987. Occurrence `occ_55dfcd7e39accd9a3648f167eb1fb65510d1ccee5c46898d454e719af0febd25` records total norm 117.900177 before clipping. The exact optimizer result then participates through the parameter-version chain in later evaluations.

The instability is transient and recurrent:

| Region | Exact capability fact(s) | Train / validation accuracy | Result |
|---|---|---|---|
| 4000→4300 | `factblock_740f433e8748ee8e3118535bf6c9363232d3a484e6a65cf666ec2236359ded1e` → `factblock_38e8032f652dd897badeb6bfd33f7e5a5ab0b4510c09d7defc102187b80ce22f` | .703/.500 → 1/1 | degradation then recovery |
| 4400→4500 | `factblock_46707c4500ccb92fa4e212b4b415329a148e369e8f364bfaf4c497935869e423` → `factblock_7837cd0405585cb04b0fa44ef24864ea4380d1364d08581ef2848b8052ea2608` | .735/.453 → 1/1 | degradation then recovery |
| 5100→5200 | `factblock_553a46461100f3478a945f2db8266d48ce6b11ff8fa95b39ac8fac85c6ddfac1` → `factblock_ef6d2532e3c6d0b7337bb305855a4c24fac106ee684f5524a0ca9488e336bf7b` | .801/.698 → 1/1 | degradation then recovery |
| 6500→6600 | `factblock_369dce329be2876ee5d8a717d883cbbd16c81f6d50cfa3326c2727fa8dff6a74` → `factblock_8e6bfc023b07ee03b85cf0eaf2c1b97c18fdc6f0c787e38b93600187588d1700` | .700/.627 → 1/1 | degradation then recovery |

The executable detects burst onsets from primitive recorded total norms, debounces each episode, and retains the estimated relaxation period as state. It also computes the exact prefix Adam normalized-momentum RMS from matching `exp_avg` and `exp_avg_sq` results. Period is allowed to grow only to a discovery-tested cap; it is not a fixed absolute-step lookup. The circuit-fragility ratio is final-LN gain divided by the prospectively decaying token/output norm. Current Adam pressure relaxes, phase generates the next pressure peak, and pressure plus fragility create a temporary shock load; that load returns to zero outside the interval. This same update law emits both the curve depression and `predicted_instability_intervals`.

Prefix-only next-onset tests support phase as useful but uncertain state:

| Prefix cut | Forecast next onset | Recorded next onset | Absolute error |
|---:|---:|---:|---:|
| 500 | 732 | 743 | 11 |
| 1000 | 1097 | 1071 | 26 |
| 1300 | 1448 | 1420 | 28 |
| 1700 | 1800 | 1805 | 5 |
| 2100 | 2185 | 2173 | 12 |
| 3900 | 3984 | 3987 | 3 |
| 4300 | 4367 | 4388 | 21 |

Amplitude is not closed as tightly as phase, so the forecast uses an expected maximum shock load of 0.08 and reports intervals rather than pretending to know future burst amplitude. This uncertainty is why the state is not labeled `STABLE` even though most evaluation points remain near one.

## Candidate mechanisms and falsification

### Rejected 1: absolute step or accuracy-only transition

An absolute step-800 threshold identifies this run and cannot transport. Accuracy alone is also not a closed post-formation state. For example, steps 5000 and 5400 both have train and validation accuracy 1.0, yet their next evaluations are 0.698113 at 5100 and 0.985849 at 5500. The missing relaxation phase and circuit/optimizer state separate those futures. This candidate was rejected rather than patched with a timestamp.

### Rejected 2: independent linear margin extrapolation

I fit each validation example's margin independently from only its last three prefix evaluations. From cut 500 this predicts the first sustained 0.9 window at step 1800; the recorded transition is 800. From cut 700, several negative individual slopes prevent a sustained window altogether. The failure shows that margins are reorganized by a shared parameter-state update; carrying margin values without a coupled gain law is insufficient.

### Selected: gain-coupled margins plus a separate fragility cycle

The selected formation law retained the exact prefix margin distribution and made final-LN gain and its rate operational. The table reports forecasts produced without reading each suffix. RMSE is over the suffix evaluation points through step 1200; the cut-400 miss was retained rather than hidden.

| Prefix cut | Forecast transition | Actual transition | Transition error | Suffix RMSE |
|---:|---:|---:|---:|---:|
| 300 | 800 | 800 | 0 | 0.1076 |
| 400 | 900 | 800 | 100 | 0.2387 |
| 500 | 800 | 800 | 0 | 0.0397 |
| 600 | 800 | 800 | 0 | 0.0417 |
| 700 | 800 | 800 | 0 | 0.0492 |

The required cut rule first qualifies the discovery execution at step 500, where both transition intervals emitted by the sealed model contain 800. The cut-400 curve error is evidence of uncertainty before the margin-reorganization event; it motivated retaining burst phase rather than claiming a globally smooth curve. Passing the other within-run tests is non-falsification, not proof of cross-run invariance.

## Report-to-code and executable-use audit

| Claim | Primitive evidence | State field | Operative code path |
|---|---|---|---|
| Held-out rule formation is a margin-distribution transition | cut-500 and step-800 validation logits/prediction facts listed above | `rule_margins` | `margin_shift`; fraction of shifted margins above zero directly sets every future accuracy |
| Shared parameter gain advances those margins | final-LN optimizer facts `factblock_96e55...` and `factblock_089828...`; exact evaluated-parameter incidence | `final_layer_norm_gain`, `final_layer_norm_rate` | `gain = current_gain + gain_rate * delta`; the rate changes formation timing, while absolute gain also changes the independent fragility ratio |
| Weight-decayed circuit reserve controls vulnerability | WTE optimizer facts `factblock_bb125824811e7149080875343dd5462fa5c2068d65dddba6a8f8d4652d3cccf7` and `factblock_7da8c83fc6c6fbe853ec07a1837e778504c7fe55285c817d4af9d99598df0200` | `circuit_norm`, `circuit_decay_per_step` | generated circuit norm is the denominator of `circuit_fragility`; holding it fixed suppresses/delays instability intervals |
| Gradient relaxation and Adam state produce transient shocks | exact norm fact/clipping occurrence at 3987, optimizer `exp_avg`/`exp_avg_sq` facts, and repeated recovery facts | `adam_pressure`, `burst_onsets`, `cycle_period`; generated `optimizer_pressure` | current pressure relaxes and `_phase_load` generates its next peak; normalized pressure creates `shock_load`, so removing pressure or phase changes the curve depression |
| Recovery is part of the same state law | exact 4000→4300, 4400→4500, 5100→5200, and 6500→6600 facts | `shock_load` in generated evolution | phase load returns to zero and accuracy returns to its margin-controlled baseline |

No listed controlling field is merely serialized. The formation interval changes if margins, gain, rate, or their coupling is perturbed. The stability curve and intervals change if circuit norm/decay, phase, period, or pressure is perturbed. Exact fact and occurrence identifiers are supporting provenance only and are deliberately absent from executable state.

The executable-use audit was run at the qualifying cut 500. Baseline transition center was 800 with 22 generated instability intervals. Reducing gain rate from 0.008886 to 0.003 moved the center to 1300; reducing margin/gain coupling from 2.0 to 0.5 moved it to 1600. Multiplying circuit reserve by ten left formation unchanged but delayed the first instability interval from [1733, 1983] to [6293, 6543] and reduced the count to 10. Changing phase period from 258 to 350 moved the first interval to [1475, 1725]. Thus the report's controlling variables reach required outputs through tested expressions.

## Intervention audit

The intervention uses only `before_optimizer_step` and mutates only current gradients. For 800 optimizer-hook invocations it sets every exposed parameter's `grad` to `None`. In the observed AdamW update facts, a parameter result reads parameter-before-update, clipped gradient, optimizer configuration, `exp_avg`, `exp_avg_sq`, and parameter-local step; for example `factblock_96e55c3bfd2ae84ff940df10b3701d2c59a95566324709475a7c283da26c4e4d` records those sources for final-LN at step 500. With no gradient, native AdamW skips that parameter, so the parameter, moment, variance, and local step state are all held rather than silently evolving.

Forward/backward computation, global optimizer-loop calls, RNG, data order, and measured gradient-relaxation phase continue. Therefore I do not assert an exact time translation. The forecast is **DELAY** with shift interval **[700, 1100]** optimizer steps. Formation margin, gain, circuit reserve, and Adam state are frozen together for 800 updates; the continuing inputs and phase widen the interval. The predicted stability effect is **NO_CHANGE**: reserve and Adam state have no supported directional change, while the continuing phase creates uncertainty rather than evidence for IMPROVE or WORSEN. The executable intervention state explicitly lists both the held mechanism fields and the continuing external fields.

## Invariance boundary

Fixed structural parts are the exact transition definition, the margin/gain update form, circuit-fragility form, and relaxation-state form. Prefix-inferred parts are margins, gain, gain rate, circuit norm, current pressure, burst onsets, and phase. Discovery-estimated parameters are the margin/gain coefficient, decay approximation, burst threshold/debounce, period growth/cap, and expected shock amplitude. Limited observations leave uncertainty in shock amplitude and in continuing RNG/data-order effects. The model contains no run/date lookup, absolute evidence path, future graph read, or stored future trajectory.
