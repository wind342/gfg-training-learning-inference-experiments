# Discovery report: cyclic rule formation with Adam renewal

## Result

The selected mechanism is an executable finite state consisting of (i) a token-map-invariant cyclic rule code recovered from exact training generation facts, (ii) the signs of the held-out correct-rule-class logit margins, and (iii) the phase of recurrent AdamW second-moment renewal/clipping episodes. Training memorization is already complete while the rule margins are mostly negative. Renewal episodes progressively move those margins positive; the frozen three-point capability transition occurs when their positive fraction remains at least 0.90. The same episodes continue after formation and can briefly consume margin reserve, while subsequent full-batch gradients replenish Adam state and restore the rule margins. The supported stability class is `TRANSIENT_DEGRADATION_RECOVERY`.

`mechanism.py` is the scientific authority. This report documents why that executable was selected and how each claim compiles into it.

## Graph validity and query semantics

The sealed input reports 111,313 hash-chained blocks, 1,062,946 objects, 111,313 occurrences, 4,642,344 atomic facts, 711,267 explicit edges, and 101 evaluations. `gfg_validation.json` passes exact fact expansion, object references, parameter-version chains, program order, and evaluation/parameter binding.

I treated each fact block as its specified ordered source-role/outcome expansion. For example, batch occurrence `occ_10bace233ceff9d389539539b8f34521532eac7311b793176f7a386784e52382` realizes `factblock_84db7c974c202d02228fac662dd41cc098dbd1f32b4528cac9d55c4f1d92f873`: exact task source `obj_502c435fd2670a1ab952f9ff6370a6e465f5520ce69674faf79ddf9bd02616e0` forms exact input and target outcomes `obj_a7fe0b1405d812dd59fed00fefe17c350c2dbf8cbf02effc90f0fe7d8254f306` and `obj_6a0bb6a94d0447b5404fcd132e98171592bcf9f0414c89d2993674d0fcc4f880`. I did not split sources and outcomes and recombine them.

Forward traces use actual source roles from batch -> forward/loss -> backward -> clip -> optimizer outputs and `GeneratedOrigin` parameter chains. Reverse traces from an evaluation use its exact `evaluated_parameter_version` sources, not equal values or timestamps. `program_order` was used only as order, never as a substitute for `reads_from`/incidence.

## Rule variable discovered from training facts

The task has 23 operand/result tokens and a distinct operator token. For each exact training input/target outcome, I formed the constraint

`code(left) + code(right) - code(outcome) = 0 mod 23`.

Gaussian elimination over GF(23) gives rank 22 and a one-dimensional nullspace. Choosing any nonzero scale yields a bijection of the 23 tokens; all 317 training facts satisfy the same cyclic operation. The scale ambiguity is harmless because it leaves the decoded outcomes unchanged. This is recomputed from each supplied prefix; no discovery token map is embedded in the candidate.

The exact validation token inputs and evaluation logits then define, for case `i`,

`margin_i = logit_i[decoded_rule_outcome_i] - max(other logits_i)`.

At every materialized evaluation grid point, the fraction of positive margins equals the recorded validation accuracy exactly. This supplies an executable, token-map-invariant rule state rather than a retrospective accuracy trend.

The frozen transition is witnessed by exact evaluation facts. An earlier low point is `factblock_d799c61b30830d6b3e381b2330475b13efc29fc5462dea88893eb39e71038b7f` / occurrence `occ_bbebcada347726f9ca26d54cf5242f5eeaea722a7daca9dc2703c71e1d24f4bb`, whose metric `obj_0930be6e7961b4b79cdd53594c4ea96fb395993d35e9a6cd386f45e1b4abf15e` has train accuracy 1.0 and validation accuracy 0.2264. Immediately before transition, `factblock_f027c7edbe9a2caf56c9cc96bda4b1698297b03a6d807182fa5adff409ee40bf` has validation 0.8491. The exact three qualifying facts are:

- `factblock_905e19e85543b4f02c89c78f87646ca2e60a34802625641f4f1bc819d9696e20`, occurrence `occ_181574a7d03b26b69389912ccd2d988ca6ac5a363394d9c4bac5c6d968830c5e`, metric `obj_67649846043236de4b4aaec6b467fc6a764fe26c59ab62d4b7014e9d559e2507`: train 1.0, validation 0.9151.
- `factblock_9b87493af52d8f2c683eec9a5893c0eea04cad9821d3a280bb17ee531380ffcf`, occurrence `occ_b6e97eeea7a656bf62d9b2eee05970d3568ba19c12092c86002913166e22a0d5`, metric `obj_8e5c0c99e8508303267c27792f66dd6d3a9521552f382eb4eba678416474d6c2`: train 1.0, validation 0.9387.
- `factblock_9f440a22fa1e141cd187f3caca6126d3df6f84e41c53ed790e3d2297f6fb8543`, occurrence `occ_4fca1bdf61d3ecca2f861136577bef366452514f6ce663bceaa3df2a4c467907`, metric `obj_70afb507219872a5d0169e1259259183634c487b4b43677a686dda8c73b00a05`: train 1.0, validation 0.9858.

## Candidate mechanisms and falsification

### H1 — cyclic rule margins plus Adam renewal (selected)

Prediction: positive rule-margin fraction grows modestly between renewal events and by a larger increment at each event. The event phase is inferred from actual `gradient_clip` occurrences; recurrence growth is coupled to the prefix Adam `beta2`, not to a transition timestamp. This state predicts both formation and later shocks.

The formation episode centered at training step 1435 provides an exact primitive chain:

1. Forward/loss occurrence `occ_3319b25ef4295cebd54f47cbfecaefc4c7d541df076dc83934b630cbd0972b0c`, `factblock_330e59692c4ea0448125177c71bf17e10284c9a9617ab807b40b3921a55abc85`, forms loss `obj_7b6185b2af2c455c5777484bdb7809f942447f9c50fccafcd014488b8667cdde` from exact inputs, targets, and 15 parameter sources.
2. Backward occurrence `occ_bcb75b379a578263a9aed6d0b372e266be44a8b0489e933f949743cb721048ff` realizes 15 separate facts. The token-embedding fact is `factblock_3d5d425b8cd8d4f884d70cb7017a4acacc193a336425c31c1fc08caf3d1360dd`.
3. Clip occurrence `occ_c603171b849803703cfbdd911f63410b5a798d4ff89eb87fb6d965c11d40d567` records total norm 47.9784. `factblock_b7e5f6413eee2d078567879a65226437a952daf863dde65ee5004eb6819bfbd1` maps the exact unclipped token-embedding gradient to its clipped result.
4. Optimizer occurrence `occ_e2a4f3ad8ac6cfd72b6bd233ab25911968ea5bfab2436d36aa7769641bec5605` realizes `factblock_9025fff6fda809543e8f310dfc57cf39667ed21b600c636aa6ba7ef2eed671b0` from parameter, clipped gradient, configuration, `exp_avg`, `exp_avg_sq`, and Adam step. Edge `edge_15a588d53c4334e3664ee2218f7904b2fd352e5f9002dc6d36e491a851455850` is the exact `GeneratedOrigin` from token-embedding version 1435 to 1436.

The predicted consequence—positive-margin fraction crossing the frozen threshold—is then independently recorded by the three evaluation facts above.

### H2 — training accuracy or training loss is sufficient (rejected)

This fails closure. Step 700 and step 2000 both have train accuracy 1.0 and nearly identical loss (0.0003424 versus 0.0003395), but validation accuracy is 0.2547 versus 1.0. The exact evaluation facts are `factblock_479813fe9a0ba3e477d1a6aa00cb50386e59d8075dfcaae456c77106e9f8e0c1` and `factblock_b3e53aaeedb0e5ca26ec16b79ee2e52b2c4417362559eaed5c4988f68fa32682`. Thus memorization/loss state admits materially incompatible next capability behavior and cannot be the executable state.

### H3 — token-embedding cyclic geometry alone is sufficient (rejected)

I computed the fraction of off-diagonal embedding Gram variance explained by cyclic token difference. It is 0.888849 at step 1900 and 0.888821 at step 5800, yet validation accuracy is 1.0 versus 0.2264. The corresponding exact token-embedding parameter objects are `obj_dd2971cab021dd0547004c0a08a7e8b77958a3b6e76b39970c60e42577cf9eeb` and `obj_9dbc99b7836196e649c338ebb0c60b297c96223fd562c08a927c47cd11d2a452`. The omitted optimizer state separates the cases: derived all-parameter RMS of `exp_avg/(sqrt(exp_avg_sq)+1e-8)` is 0.00926 versus 0.58690. Geometry is a useful diagnostic but not closed state.

### H4 — a fixed transition timestamp/one-run sigmoid (rejected)

A timestamp can reproduce this one transition but has no primitive source relation to the held-out outcomes and cannot distinguish the equal-loss or equal-geometry counterexamples. Perturbing the rule-fraction initial state or renewal phase changes the sealed forecast, whereas an absolute timestamp would not. Therefore absolute step remains only the declared evaluation coordinate.

### H5 — post-formation state is permanently stable (rejected in favor of transient degradation/recovery)

At training step 5797, forward `factblock_bd4bacd1d53a4b0230cd6a16cfb07a9f464dfd0b6af71ea6e784f9003703c1ae` forms loss `obj_04ca897e5a14404deecf627acf77f88a5995c9d6d57844657023d7b10625653c`; backward token-embedding fact `factblock_8d8b3abe9ad0e365307efada079a4ac3051a3e795575ecf1e9ea419cda4f8e83` forms gradient `obj_aff265a8c008ee27925525c68a9e9221cca29bf8ed7926e4bdfb3fd083694cdf`. Clip occurrence `occ_cd456312402b8b8bda668df4f642a588f6f63a09aa8944fc2a822fab02f80911` records norm 45.7588 and `factblock_00e994adbaa27541dfa877f78d6affe17754da3d200f432243c5f496c9c32ecf` forms clipped gradient `obj_a4c995493cdaee1d5e7dda0ac4e0f2c6335bb0f1a30eca6bfb465854ba1ee99d`.

Optimizer occurrence `occ_89e4047b1e767ab6cf90d95d4dcf9c0907dec9f1835c708fa560d1e275dba3f5` reads that gradient plus exact parameter/Adam states in `factblock_dff2ba2ccd240d1aa4db74e076214f6ebcd446507c1130277693e3ec48a3482d`; `edge_672b62f01c526fea59647559733b0a4d4c0086487613e29fcb69f29a2b96cade` generates the next token-embedding parameter version. Evaluation `factblock_e6b579c2c26ded11ad60f8db9712927f43cbaca07f0d722e63badd7aa2ab482c` then records train 0.3249 and validation 0.2264. At the next grid, exact evaluation `factblock_e7328fd8f50710fc84d365b5283c89627738e252b39711077a5d0c7d2cc47b41` records train 1.0 and validation 0.9953. Similar one-grid recoveries occur around other renewal episodes. Persistent degradation is therefore falsified; stable-with-no-dips is also falsified.

The observable precondition is a long low-gradient interval that decays second-moment reserve, followed by a clipped-gradient renewal and unusually large preconditioned update. Recovery is actual continued full-batch loss -> backward -> clip -> Adam state replenishment -> new parameter versions, not time proximity alone.

## Prefix-only replay and closure tests

Each replay exposed only blocks at or before its cut, reconstructed the cyclic code and current margins, detected only past clip episodes, and generated the remaining curve without suffix reads.

| Cut | Region | Predicted transition center | 200 interval | 500 interval | Future-curve RMSE |
|---:|---|---:|---|---|---:|
| 500 | early rule formation | 1500 | 1400–1600 | 1300–1700 | 0.1310 |
| 600 | between renewals | 1500 | 1400–1600 | 1300–1700 | 0.1317 |
| 700 | immediately before renewal | 1500 | 1400–1600 | 1300–1700 | 0.1326 |
| 800 | immediately after renewal | 1500 | 1400–1600 | 1300–1700 | 0.1186 |
| 900 | between renewals | 1500 | 1400–1600 | 1300–1700 | 0.1206 |
| 1000 | immediately before renewal | 1500 | 1400–1600 | 1300–1700 | 0.1193 |
| 1100 | near transition, after renewal | 1500 | 1400–1600 | 1300–1700 | 0.1118 |

Post-transition cuts at 1700 and 3500 preserve the formed phase; cuts immediately before and at the large instability (5700 and 5800) preserve the prior transition and invoke the one-grid recovery branch. This specifically checks the state transition and stability branches rather than refitting a full run.

State counterexample searches produced H2 and H3. Every submitted state transition is generated by `forecast`; no future object, suffix metric, or stored discovery trajectory is available to it.

## Executable-use audit

| Mechanism claim | Primitive GFG evidence | Executable state | Exact code path |
|---|---|---|---|
| A cyclic rule is identifiable | Batch occurrence/fact and exact input/target outcomes above | `rule_code_by_token` | `_training_triples` -> `_recover_rule` -> validation-label decoding |
| Rule formation is held-out margin formation | Exact validation logits/outcomes and evaluation facts | `current_rule_fraction` | `_margin_history`; initial `q` in `forecast`; threshold scan in `_observed_transition` |
| Adam renewal changes formation | `gradient_clip` occurrences and optimizer facts reading clipped gradient, `exp_avg`, `exp_avg_sq`, step | `renewal_episodes`, `adam_beta2` | `_renewal_episodes`; recurrence interval expressions; renewal gain branch |
| Instability is transient | 5797 backward/clip/update chain plus exact 5800/5900 evaluation facts | formed/recovery phase and future renewal peaks | lag-dependent shock branch followed by below-0.90 recovery branch |
| Intervention delays formation | Optimizer facts enumerate parameter, gradient, configuration, both moments and step | intervention hold counter | `TrainingIntervention.apply`: gradients become `None`, making AdamW skip all per-parameter reads/writes |

Perturbation audit:

- At cut 500, reducing the operative `current_rule_fraction` by 0.10 moves the predicted transition from 1500 to 1700.
- Shifting the last renewal peak by +100 moves it to 1600. Thus renewal phase is executed, not merely serialized.
- Permuting the recovered code changes decoded labels, margin signs, and the transition forecast through `_margin_history`.
- Changing `adam_beta2` changes both interval-growth expressions.
- Rank/count checks, margin history, negative-margin count, and mean/q10 margins are explicitly labeled diagnostics in `mechanism_spec.json`; no independent causal claim is made for stored fields that do not alter the forecast.

## Intervention audit

The intervention predicts `DELAY` by 800–1000 steps and holds exactly 900 optimizer occurrences. At `after_backward` and `before_optimizer_step`, every exposed parameter gradient becomes `None`. For AdamW this skips the parameter write, per-parameter step, `exp_avg`, `exp_avg_sq`, and decoupled weight decay. A zero-learning-rate fallback is restored after the occurrence if parameters are not exposed.

This is not an unaudited freeze of one visible metric. The affected optimizer fact explicitly reads all six roles (`parameter_before_update`, `clipped_gradient`, `optimizer_configuration`, `exp_avg_before_update`, `exp_avg_sq_before_update`, `step_before_update`); the intervention freezes every mutable per-parameter state in that fact together. Full-batch tensors have invariant content hashes and memberships throughout the graph, and layer-forward facts expose no stochastic source, so advancing batch occurrences does not change an operative data/RNG state. After release, formation and renewal resume from the same state. The predicted stability effect is therefore `NO_CHANGE`: instability/recovery intervals translate with the delayed mechanism rather than changing state-relative severity.

## Invariance and uncertainty

The cyclic constraint, exact margin definition, GFG lineage, and Adam state roles are fixed structural laws. The token code, current margins, beta2, and renewal phase are inferred anew from the supplied prefix. Renewal gain, interval cap, and shock amplitude are discovery-estimated scalar parameters—not a stored trajectory. Only one complete execution is available, so fixed-amplitude and perfectly periodic alternatives were actively rejected; the submitted intervals express the remaining timing uncertainty. No run identity, date, wall clock, absolute evidence path, future table, or discovery token map appears in executable state.
