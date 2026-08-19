# Executable discovery report

## Result

The selected theory is a clipped-gradient relaxation mechanism with two
separately executable dynamics. `FormationDynamics` treats the recurrent
bounded optimizer impulses as rule-reorganizing events: a low validation
charge rises sharply when an episode occurs, then closes toward a sustained
rule capability. `StabilityDynamics` retains independent oscillator phase,
post-formation impulse burden, fragility, recovery memory, and degradation.
The two outputs are composed only as
`clamp(formation capability - stability degradation, 0, 1)`.

On the discovery execution, the transition is optimizer step 2800. The exact
evaluation fact blocks at 2800, 2900, and 3000 report train accuracy 1.0 and
validation accuracies 0.910377, 0.985849, and 1.0 respectively:
`factblock_fe464b6f27e5659e9f7f3a6489f994e0d9b3387c7640ce79b230e78371057cbf`,
`factblock_2d70f16b124bf35c316b91e5f78134f9f2e89e8827ed8358e88cee2d1be950ee`,
and
`factblock_98389960c2c59325e8343d0b74271209afaacdc28f1bd4645ea28b0607a484a7`.
The corresponding capability occurrences are
`occ_794a5a927c5a8c19fe69b09e3c34c454faef85fece365d4e62b028468901c664`,
`occ_68d92779668073ba73ecdcf174d81429d0a3ee3c8bf50ef4523ec173cf806686`,
and
`occ_eafbe2c77f1ff00daff14ec8c35d868615c3498ea36a2a7475f4d86981c5e735`.

Post-formation stability is `TRANSIENT_DEGRADATION_RECOVERY`. Exact
evaluation facts show 1.0 at step 3500, 0.561321 at 3600, and 0.990566 at
3700; 1.0 at 7800, 0.188679 at 7900, and 1.0 at 8000; and 1.0 at 9400,
0.754717 at 9500, and 1.0 at 9600. The three degradation facts are
`factblock_335fcc0526f1eac72406f983d0ceb95864b8fd4e7754d4b39d034f94be413829`,
`factblock_8abaa30776fa2060d5b54b331e667b73393acabb782ce2519b0aca9e8b82efe2`,
and
`factblock_701070a2bd12756f7102a8bff3e5f561fb62cc428e42243baac4a52530b6aa96`.
Their recoveries are independently formed outcomes, not equal-value joins.

## Graph basis and query boundary

Only the validated participant GFG was used. Its 111,313 hash-chained blocks
contain 1,083,354 objects, 131,717 occurrences, 3,598,293 atomic facts, and
731,671 explicit edges. Queries followed exact outcome-specific fact blocks;
no source/outcome Cartesian reconstruction, time join, run lookup, wall-clock
lookup, or network source was used.

The materialized dataset objects
`obj_332edf293e4dea8f9359eb4d1b04d1f8461c1520efe43af617a6679632c24cc9`
and
`obj_e86dcfa8fc233fa4ea1d8148392fffe1c13f590304c064605a12344577478a83`
cover 317 training and 212 validation inputs. Their exact targets establish a
complete 23-by-23 commutative, associative Latin operation with identity token
13; the union has 529 distinct ordered pairs. This algebra is a diagnostic
that distinguishes rule-level generalization from sample memorization. It is
not an operative forecast field and is not claimed as a transportable token
map.

At step 2100, capability fact
`factblock_3e34e3e14ee9c248093a2fa43f560efa905129709ea402d0d5cda9a88d3b2280`
is realized by
`occ_97ae8c78ca305a43d41d453bb3733dd9f6505d9c1c6a3a7d22577a607f4141e7`
and reports train 1.0 but validation 0.325472. The next evaluation at 2200 is
0.839623 (`factblock_fd88b87330048e5df4a05e432d2a3a38eb7053f37fd4e5b5eef9631680b18b48`).
Between them, step 2125 norm fact
`factblock_cc6f895489f48131b38a151c9cdb99302e83f59c0aa158f4bdf36769e45722fa`
forms the exact global-norm object
`obj_eed5c5e198f895c2cf1be8772a55f9221346d3e7b01b9efc48f6423e26d5c26d`
from 15 named gradients. It is realized by
`occ_ed84b7fc6c544f26c73c6290e88f945b7387adee4ed9fc584eb32f416bcc0dc7`.
The clip occurrence
`occ_4ac322326f84fd6cb5ba491e909ef01d838d0afdfb30430235f05d12baa496a9`
reads that norm, each exact gradient, and configuration object
`obj_a97703be1b60e36d5141597a5fe60bc25547541fb6235dcde2d9460856291b0b`.
Its `max_norm` is 1.0.

The next optimizer occurrence
`occ_95a54ada83fee69461e6e0e994660774e40d4e38b25f4889c6df4f0ff5a03a81`
forms parameter version 2126. Its embedding outcome fact
`factblock_87cdf4392c6c6d4b5ee2e5724f3c6b9437c8a15467c9952e7e73d846e963f5e7`
keeps the prior parameter, clipped gradient, AdamW configuration, step,
`exp_avg`, and `exp_avg_sq` sources together. Primitive edge
`edge_b1bb5b7fe3b0b6e0401e93c3b0774e38b353abb3720e38bac6f7589761634c5a`
is the exact `GeneratedOrigin` from embedding version 2125 to 2126. Thus the
claimed impulse path is an actual gradient -> norm/clip -> optimizer state and
parameter-version path, not mere program order.

The same structure appears before instability. Step 3591 norm occurrence
`occ_78d3106a2906a76e62f2c478b557535026fcd8a860e16e150f3759de38989f46`
and clip occurrence
`occ_18c65287a2e18132d1e96bf7568b1b44fb0bc05c26cf4298ab03e69e6129392c`
record total norm 8.838923 immediately before the 3600 deficit. Step 9479
records norm 168.437866 in
`factblock_85c4c61f9d6155bc71d3058e55479ece20d7f19d2d14a7a113794659ad654e99`
before the 9500 deficit. Recurrent episode spacings are not constant: selected
successive peak spacings include 268, 327, 354, 335, 356, 340, 380, 390,
324, 405, and 437 steps. This falsifies a fixed-period calendar.

## Candidate mechanisms and falsification

1. **Selected: clipped-impulse formation plus burden/recovery stability.**
   State is initialized from the prefix evaluation facts and, when ordinary
   tensor support is present, materialized global-gradient norms. A recharge
   episode changes formation capability through an explicit gain expression.
   After formation, every episode increments an independent burden; formation
   fragility and threshold discharge produce degradation, and the same state
   recursively recovers. This candidate survived the prefix replays below.

2. **Rejected: absolute-step sigmoid or fixed-delay grokking.** At step 2100
   a retrospective linear extrapolation from the last three accuracies cannot
   generate the 0.325472 to 0.839623 change at 2200. Removing executable
   episode impulses from the selected replay moves its transition from 2800
   to 6900. A clock fit also has no closed stability state and cannot explain
   variable episode spacings. This alternative was rejected rather than used
   as a hidden timestamp threshold.

3. **Rejected: capability-only stability or one fixed oscillator phase.**
   Capability near one is a matched compressed state with divergent futures:
   step 3500 (1.0) is followed by 0.561321, step 7800 (1.0) by 0.188679, while
   many other 1.0 points remain at 1.0. Fixed-period replay generates false
   windows because recharge intervals vary. Adding phase alone still fails
   after a discharge: recovery memory changes the subsequent recharge law.
   Retaining phase, burden, fragility, and recovery memory resolves the tested
   divergent futures.

## Prefix-only replay and closure tests

The candidate was sealed and recursively replayed without suffix queries on
the complete 100-step grid. Results below compare later to the already
recorded suffix; RMSE is raw validation-accuracy RMSE.

| Cut | Region | Predicted transition interval (200 / 500) | Future event points | RMSE |
|---:|---|---|---|---:|
| 2100 | before transition | 2700-2900 / 2600-3000 | 3600, 7900, 9500 | 0.018448 |
| 2700 | near transition | 2700-2900 / 2600-3000 | 3600, 7900, 9500 | 0.017373 |
| 3500 | after formation, before first deficit | observed 2700-2900 / 2600-3000 | 3600, 7900, 9500 | 0.017794 |
| 7800 | before the second deficit | observed transition | 7900, 9500 | 0.016749 |
| 9400 | before the third deficit | observed transition | 9500, then recovery | approximately 0.011 |

For the 2100 and 3500 full-horizon replays, degradation-event precision and
recall are both 1.0 (three true grid events and no false event). For the 7800
replay, both are again 1.0 on its two-event suffix. The 9400 replay predicts
validation 0.767 at 9500 versus 0.755 observed and recovery at 9600. These are
prefix tests of this discovery execution, not proof of cross-run invariance.

The executable-use audit produced the following counterfactuals from cut
2100:

- Removing formation's episode-impulse expression delays the sustained
  crossing from 2800 to 6900.
- Holding `impulse_burden` at zero eliminates all three predicted degradation
  events.
- Shifting `next_episode_onset` by 100 steps shifts the predicted event points
  from 3600/7900/9500 to 3700/8000/9600.
- Setting both operative `impulse_scale` fields to 0.25 moves the transition
  to 4700 and changes the stability classification from transient degradation
  with recovery to stable.

These tests establish operational use rather than mere serialization.

## Report-to-code correspondence

| Claim | Primitive GFG evidence | Executable state | Update location |
|---|---|---|---|
| Rule capability accumulates and crosses | exact capability fact blocks at 2100, 2200, 2800-3000 | `capability`, `formation_velocity`, `formed` | `FormationDynamics.step` baseline and gain |
| Clipped episodes advance formation | norm/clip occurrences at 2125 and exact optimizer fact at 2126 | `episode_impulses_used`, `impulse_scale` | `FormationDynamics.step` episode loop |
| Capability remains formed after transition | three-point transition facts and later recoveries | `formed`, `transition_step` | formation floor and exponential closure |
| Episode timing controls exposure | materialized norm episodes and variable spacings | `next_episode_onset`, `recharge_period`, `phase` | `StabilityDynamics.step` recursive episode loop |
| Repeated impulses create distinct stability risk | 3500/3600/3700, 7800/7900/8000, 9400/9500/9600 facts | `impulse_burden`, `fragility_discharge_done` | burden increment and threshold branches |
| Recovery history changes later dynamics | exact recovery facts after each deficit | `recovery_memory`, `shock_onset`, `shock_amplitude` | recharge selection and exponential recovery |
| Final validation is dual composition | component interface contract | both complete substates | `CapabilityDynamicsMechanism.forecast` clamp |

The cyclic-operation diagnostic and embedding spectral summaries used during
discovery are intentionally absent from operative state: perturbation tests
showed that the prefix validation charge plus exact optimizer-episode state
was sufficient for the declared forecasts on the tested cuts. They are not
silently claimed as causal controls.

## Intervention audit

The single intervention runs only at `before_gradient_clip` and changes the
current clip control from 1.0 to 0.25. It does not alter samples, validation
information, parameters directly, future state, task, evaluator, or capture.
The affected operation reads all 15 current gradients, the global norm, and
the clip configuration; AdamW momentum, second moment, parameter versions,
and data order continue evolving. Therefore the prediction is not a pure
time translation.

In the same update law the intervention changes
`formation_state.impulse_scale` and `stability_state.impulse_scale` from 1.0
to 0.25. The sealed counterfactual predicts `DELAY`, with a shift interval of
600-4000 steps (point replay shift 1900), because rule-reorganizing gains are
smaller. It predicts stability `IMPROVE`, because burden increments are four
times smaller and no threshold degradation is generated in the replay.

## Invariance and uncertainty

Fixed structure consists of the two-state decomposition, bounded clip impulse,
burden/recovery closure, and exact output composition. Prefix-derived state
includes capability, velocity, oscillator phase, formation age, burden, and
recovery memory. Numeric gains and recharge responses were estimated from one
discovery execution. Matched-state counterexamples falsified smaller states,
but passing within-run replays does not establish cross-run transport. The
sealed executable is the prospective test on the distinct unseen continuous
execution; no run identity, date, wall clock, or stored answer trajectory is
used.
