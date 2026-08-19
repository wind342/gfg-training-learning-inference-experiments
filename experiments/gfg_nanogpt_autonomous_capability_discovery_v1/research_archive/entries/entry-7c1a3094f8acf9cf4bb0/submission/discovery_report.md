# Discovery report: optimizer-driven rule-circuit formation

## Result

The selected executable mechanism is a finite-state relaxation oscillator coupled to rule-circuit progress. During memorization, exact training outcomes are already saturated while validation margins remain negative. AdamW's decoupled weight decay and adaptive state generate separated global-gradient bursts. A burst updates the exact parameter-version chain; once the validation rule circuit is partially formed, the next burst moves nearly all held-out examples together and forms the sustained capability. The same oscillator continues after formation. As normalized weight reserve declines, some late bursts cause a one-evaluation degradation followed by recovery, so the supported stability result is `TRANSIENT_DEGRADATION_RECOVERY`.

This report supports the claim; [mechanism.py](mechanism.py) is the sealed scientific authority. Its initialization reads only the supplied prefix. Its forecast performs no GFG reads.

## Exact GFG query boundary

All quantitative reductions start from outcome-specific fact blocks. I expanded each fact as its exact ordered sources, concrete occurrence, one outcome, and relation roles. `reads_from` below means the derived traversal from an outcome through that fact block's source incidences; it is never a join on step or value. Parameter propagation uses primitive `GeneratedOrigin`, and ordering claims use primitive `program_order`. I did not form a source-by-outcome Cartesian product.

One representative forward chain is the training iteration beginning from `theta_712`:

- batch occurrence `occ_36b85d2a946766425eb930a6ddd848ed36947fc0758f1cb2fcf0d368c3aeef35` realizes input and target facts `factblock_78c446988cdb9c7077e8d6c68e922a674ed956caf8d66dd0ec76589de9b35264` and `factblock_ab27b473cdcbbdec658f21a47eef59941cae6d0c252921f9b9aa4c2ae1520add`;
- forward occurrence `occ_b2d95c42875c03b0cb2478aba681cad3d965099f64f90df77fd775d56ca878c6` and loss occurrence `occ_3b50aac541ddc76a6b2f329f7acb63327faa6d8244e47c61956e4304f5f0a123` realize `factblock_bda599bdf6e4d802ba4a239517d1d32bd89b1ae7dfded79d55a8a9ac903b3127` and `factblock_96cf3976ae7bc6856c9365ec1a4ed8955f4a1941f5f50d50268ff0612f4a5c47`;
- backward occurrence `occ_49330c45f029693ef6688d77016d03408831381ce4b0379c221ca4ee2c3ff718` forms the individual parameter gradients;
- norm occurrence `occ_045049e15e84d576c4bb73e25a5b51e5f03d30bf66e61590c519bc0d2d44cfbd` realizes global-norm fact `factblock_5139b5fae2c6bcf934b152592ac5b23b6ec6ea0f8a7c37021997b7e8a889ad05`, whose exact scalar object is `obj_a4f384769585ade0538b37fac307d97bd5be3b3f5fc429b2e016efd40506514c` = 44.185173;
- clip occurrence `occ_ab6a075071c156c12bbe0990ab7a84ad20c99b8e8886787e2884c8253e36658f` forms the clipped gradients;
- optimizer occurrence `occ_cadee8363f06e7381ea582878facf3107db5e5e9f2c003849f09ba3afd21422d` reads those exact clipped gradients and optimizer states and forms `theta_713`. For the tied token/output weight, `factblock_af4400ccad168f5378d86db6c8abba84e0c0ff83fa72396917fc6f2c482dedd5` has sources with roles `parameter_before_update`, `clipped_gradient`, `optimizer_configuration_for_parameter_update`, `exp_avg_before_parameter_update`, `exp_avg_sq_before_parameter_update`, and `step_before_parameter_update`, and outcome `obj_799c93d3969a50094e359567782211b272f8bd9730053a45f5cfc6742f24a2c3`;
- primitive edge `edge_df9e2d41c5a39c9bc3954ec4faff3e7fc05d8b3a205cc0c522223bd5221153c3` orders clip before that optimizer occurrence, while primitive `GeneratedOrigin` edge `edge_36da2169acf2759dcd4120a2737bb6bd19e54c6fbff0368cb56060dbc576ee5e` links the exact version-712 object to the version-713 object.

Evaluation facts were traversed in reverse from their exact capability outcomes through predictions, targets, latest loss, logits, and evaluated parameter-version sources.

## Observations and formation boundary

The contract transition is step 900. An earlier evaluation is at or below 0.30, and steps 900, 1000, and 1100 are the first three-point window with train accuracy at least 0.99 and validation accuracy at least 0.90:

| Step | Train accuracy | Validation accuracy | Exact capability fact |
|---:|---:|---:|---|
| 500 | 1.0000 | 0.2264 | `factblock_b2b1a2065ed0a89ee4c50503d8958ab49760a872c8c5809ebf119441898440b9` |
| 600 | 1.0000 | 0.2642 | `factblock_0f53c95bfd7b33bc864e008886de766d0699126cfe17b16d227281520ef250d4` |
| 700 | 1.0000 | 0.3396 | `factblock_52a4097f9d2667c89c73df2d597b114196d60478f766355e317f582a25a57272` |
| 800 | 1.0000 | 0.8679 | `factblock_02adbb0e7a87a2e086fae6f2413947e0a16b368fae8d12ae6e062b4cdeeaf4cc` |
| 900 | 1.0000 | 0.9104 | `factblock_f43f51b8003a0f60c17dfc896a07ddd6223d02ec924fddc749d08cf89f30879f` |
| 1000 | 1.0000 | 0.9434 | `factblock_6cc383bf502e4906c1defefca80873537aad883fcee7770b3720a1fcf5c61bda` |
| 1100 | 1.0000 | 0.9292 | `factblock_7cfe6bebb7932956650f360704b0ff271f883619d3fa3b3464bf5b003845e4da` |

The logits are separate formations bound to their exact evaluated parameter versions. At step 700, validation occurrence `occ_629c349b60c23ca2ac1003ca803406d4c9e887cf42abb686e58ebafa20fdb5a3` realizes `factblock_7ab5f0a0392d31fbcf2fff07848afed59c101b877b4ca54c2b6b3d0c3a76207a`, producing logits object `obj_83eb0fef41e7b70e7948854cbfee7f352164c897c88e4dd5f390b2018f866aaa` from the validation inputs and 15 exact parameter-version sources, including tied-weight version object `obj_ff1c4f7eeb9812d556d125063d3f85a3018f7130cbbbf3b9357d0cb50c5cb596`.

For each exact validation-logit row I derived `correct-target logit - maximum other logit`, using the target from the same evaluation fact chain. Median margins at steps 500, 600, 700, 800, and 900 were -2.9568, -2.2075, -1.7767, 3.5786, and 4.3709. The 10th-percentile margins were -6.7920, -6.3195, -6.4986, -0.7190, and 0.5143. Thus accuracy did not merely drift through 0.90: a shared parameter formation moved the entire held-out margin distribution after the burst.

## Selected mechanism: state-conditioned relaxation burst

The optimizer configuration is exact inline object `obj_da2829e61f93dd4bc2897d220a554791ad8db1ac3ff5b66b55341acd55621a2c` in block 0, `block_239e46c0b7e8d96edf51496e1be2d6a006e514bc7a7ca7363605d0605b954612`. It records AdamW with betas (0.9, 0.98), learning rate 0.003, and a decoupled-weight-decay group with weight decay 1.0. Optimizer fact `factblock_01ffabb249333493c1844df3a214dbfceec07989c5fd95efc4691d3df561abb5` at step 700 demonstrates operational dependence: exact prior parameter, clipped gradient, configuration, first moment, second moment, and step sources form tied-weight object `obj_ff1c4f7eeb9812d556d125063d3f85a3018f7130cbbbf3b9357d0cb50c5cb596`. Primitive `GeneratedOrigin` edge `edge_32a684bbb1498ff63eb5f7ab0953fa2fab311b96cae01667c9888a1f8759c59c` links its exact prior and result versions.

Significant contiguous global-gradient bursts have centers 29, 163, 396, 712, and 1080. The first four peak norm facts are:

- 3.479858 at step 29: `factblock_222e04de1cba1d355e000153ea1686ad972db2af9eb2e48ebaf836e200e32ef7`;
- 13.824698 at step 163: `factblock_48ce7d512c9868cfa1ef20b6648399a8da8a5386b40ccffb5225764729ba1eca`;
- 40.336540 at step 396: `factblock_af6a4b06cc9b82a29f8effafc7cc6cec541161e07663d547600eb14371f28d4a`;
- 44.185173 at step 712: `factblock_5139b5fae2c6bcf934b152592ac5b23b6ec6ea0f8a7c37021997b7e8a889ad05`.

Their early center intervals are 134, 233, and 316 steps. The submitted early-phase law retains 0.84 of interval acceleration: `233 + 0.84*(233-134) = 316.16`, forecasting the next center at 712 from the prefix ending at any of steps 500, 600, or 700. Crucially, code derives 712 from prefix burst state rather than absolute step. The first post-burst evaluation is forecast as a formation state and the next as the sustained transition candidate. Rule progress and the exact median validation margin both affect that response in code.

The operative finite state is therefore:

1. rule-circuit progress and margin;
2. oscillator phase and the last two burst periods;
3. learning-rate/weight-decay law;
4. normalized tied-weight stability reserve.

This is not asserted to be the smallest state. It is the tested state that survived the available closure checks.

## Competing mechanisms and falsification

### Rejected A: validation-accuracy trend alone

A last-three-point least-squares extrapolation to 0.90 predicts step 1228 from the step-500 prefix, 1139 from the step-600 prefix, and 1701 from the step-700 prefix. The same proposed compressed state law produces incompatible remaining futures across adjacent valid cuts and misses the exact burst-linked step-900 formation. Its suffix shape also cannot generate the collective margin sign change between 700 and 800. This candidate was rejected.

### Rejected B: fixed step-900 clock

A hard-coded step 900 fits this execution but has no executable path from prefix graph state and cannot transport to the independently changed task, token map, initialization, split, or data order. The graph instead supplies changing burst intervals and prefix-conditioned margin state. A clock field perturbation would change the answer without changing any primitive formation state. It therefore fails the operational-use and cross-execution-invariance audits and was rejected even though it retrospectively names the observed boundary.

### Rejected C: stable forever after formation

This stability hypothesis is directly falsified. Exact capability outcomes show 1.0000 validation accuracy at 7400, 0.8160 at 7500, and 0.9858 at 7600. The corresponding facts are `factblock_6a52685f8225073fa657890d9a20905a7e2cfdd75c4c3a80d91fe07b8a875f9d`, `factblock_daa055428d2979522642a5ce8d6c4a724f8afbabaadf74675bea4f3b2010cddb`, and `factblock_4fed3116ef95baf132d5379a33e11852cae6b781340d10c9c3c36c52666ef659`. The degradation is preceded by the step-7475 global norm 58.218964, exact fact `factblock_d8d2cc9836df44fd9ab875134ec1ad74d69763b5de83d7e759e4e1488e3d9f00`.

The pattern repeats more strongly: accuracy 1.0000 at 8200, 0.5330 at 8300, and 0.9953 at 8400 around the step-8290 burst (`factblock_5de42944b1e77f189d6196288af77388f4c5431e87dc3f22391bdaa65d5fb09c`); and 1.0000 at 9100, 0.6179 at 9200, and 0.9953 at 9300 around the step-9190 burst (`factblock_1d3fcb0ae351c7a92335de18306631bfb1f5fa4ac00a15483013549743147244`). Because each degradation recovers on the next evaluation grid, persistent degradation is also rejected. The supported class is `TRANSIENT_DEGRADATION_RECOVERY`.

### State-sufficiency enlargement

Current gradient magnitude alone is insufficient: steps 700 and 1000 have nearly equal exact total norms, 0.0006741 and 0.0006858, but the next significant centers are only 12 and 80 steps away. Burst-period phase separates them. Likewise, high accuracy alone cannot distinguish a quiet generalized state from a low-reserve state approaching a damaging burst. The executable therefore retains oscillator phase and normalized tied-weight reserve rather than hiding these counterexamples in a broad narrative.

## Prefix-only replay

For each cut, initialization was restricted to blocks and evaluations at or before that optimizer step. The suffix was read only after the forecast object had been produced for scoring.

| Cut | Prefix phase | Predicted formation burst | 200-step interval | 500-step interval | Actual transition | Future accuracy RMSE |
|---:|---|---:|---:|---:|---:|---:|
| 500 | RULE_CIRCUIT_ACCUMULATION | 712 | [800, 1000] | [700, 1100] | 900 | 0.08999 |
| 600 | RULE_CIRCUIT_ACCUMULATION | 712 | [800, 1000] | [700, 1100] | 900 | 0.09162 |
| 700 | RULE_CIRCUIT_ACCUMULATION | 712 | [800, 1000] | [700, 1100] | 900 | 0.09150 |

RMSE is on the [0,1] validation-accuracy scale over every suffix evaluation through step 10000. The selected state is not falsified by these within-execution cuts. This does not prove cross-execution transport; that remains a prospective claim for the sealed unseen execution.

## Report-to-code and operational-use audit

| Mechanism claim | Primitive GFG evidence | Executable state | Update location |
|---|---|---|---|
| Partial rule circuit controls post-burst formation | exact validation-logit facts at 500–900 and their exact target/parameter incidences | `validation_accuracy_tail`, `validation_grid_progress`, `validation_margin_median` | `_margin_state`; formation response in `forecast` |
| Adaptive/decay state generates recurrent bursts | exact norm facts at centers 29, 163, 396, 712; optimizer configuration and parameter-update facts | `burst_centers_tail`, `learning_rate`, `weight_decay`, `asymptotic_burst_period` | `_burst_runs`, `_predict_burst`, `_future_pulses` |
| Weight reserve controls instability | tied-weight parameter-version formations; late burst and evaluation facts | `weight_scale_reserve`, `reserve_multiplier_per_cycle` | `_weight_reserve`, low-reserve branch in `forecast` |
| Recovery follows a damaging burst | exact 7400/7500/7600, 8200/8300/8400, and 9100/9200/9300 capability facts | phase generated as `TRANSIENT_DEGRADATION` then `RECOVERY` | state-evolution loop in `forecast` |

Executable perturbations confirmed operational use. Moving the most recent burst center from 396 to 430 changes the forecast burst and transition interval. Setting the median margin to -12 changes the formation-response curve. Lowering weight reserve moves the first predicted instability earlier. Changing weight decay from 1.0 to 0.5 changes both the asymptotic period and stability prediction. Fields that merely identify evidence objects are not serialized into state.

## Intervention and complete state audit

The submitted intervention is an 800-update optimizer pause at `before_optimizer_step`, predicting `DELAY` by 700–900 steps. At every paused hook it detaches each current parameter gradient and sets each optimizer-group learning rate to zero. With gradient absent, AdamW skips that parameter, including its first moment, second moment, and step state; zero learning rate also prevents a parameter or decoupled-weight-decay change. Before the following update, the exact original group rates are restored.

The affected state audit is:

- parameters: held;
- current gradients: removed after they are formed, so they cannot be read by the optimizer;
- Adam first and second moments and per-parameter step: held because the parameter has no optimizer input gradient;
- weight decay: held because no parameter update occurs and learning rate is zero;
- rule progress, validation margins, oscillator update count, and normalized reserve: consequently held;
- RNG and batch-selection order: left evolving. The graph's batch fact uses the complete 317-example training input object at every inspected step, with selection order as a separate exact source, so the report does not claim that RNG state time-translates. This is the source of the ±100-step shift uncertainty.

Pausing 800 operative updates yields a predicted effective shift interval [700,900], exceeding the required 600 steps. It does not selectively improve or worsen reserve; after restoration the same state law resumes, so the predicted stability effect is `NO_CHANGE`, with instability intervals delayed approximately with the formation state.

## Limits

The fixed equations, prefix-inferred state, discovery-estimated constants, and uncertainty are separated in `mechanism_spec.json`. Passing three historical cuts only means the candidate was not falsified here. In particular, the 0.84 interval-acceleration retention, 440-step asymptote, and reserve decay exponent are estimates from one continuous execution. They are not claimed as established invariants until the same sealed code is tested prospectively on the distinct unseen execution.
