# Executable discovery report: gain–recovery capability mechanism

## Result and scientific authority

The selected mechanism is a two-coordinate finite state: an exact final-layer-normalization gain coordinate controls rule formation, while a clipped-gradient/Adam recovery oscillator controls temporary loss of the formed capability. The scientific claim is the executable `CapabilityFormationMechanism` in `mechanism.py`; this report only documents its evidence and tests.

On the discovery execution, the frozen transition definition gives step 1500. Step 1400 has validation accuracy 0.7971698, so no earlier qualifying window can begin there. Steps 1500, 1600 and 1700 have train accuracy 1.0 and validation accuracies 0.9622642, 0.9764151 and 0.9528302. The exact evaluation metric objects are respectively `obj_13ee7d4a006d54db9c5a30e4be2a74f244fac82ed4900696fbdc2349a929d374` (step 1400), `obj_1b0de07fe9bb98b505cc6f8e851241a9dde42be299514ca98e5b95d2bacd9a9c`, `obj_89dc48d9c92d3288a5d5c91e104b3cb11c2e5e449f213ac01f823855129a3083`, and `obj_b73ebbe4a4d8af79e88076c6135b48050a3e2cfee1043123bdd1510cf718dea4`. An earlier low result exists, including the step-600 metric object `obj_c7363a8b74bca3800425786cbc7a7bb249b5df64b326ffd4e75b7dae0d87ef5c` with validation accuracy 0.2075472.

The discovered post-formation classification is `TRANSIENT_DEGRADATION_RECOVERY`. On the evaluation grid, validation-below-0.9 intervals after formation are [2400,2400], [2700,2700], [5100,5100], [5500,5500], [7100,7100], [7500,7500], [8300,8300], and [8800,9000]. Every interval recovers; for example, the last interval is followed by 0.9433962, 0.9858491 and 0.9952830 at steps 9100–9300. Thus neither `STABLE` nor `PERSISTENT_DEGRADATION` describes the complete graph.

## Primitive GFG evidence and derivation

The optimizer is not inferred from timing. Object `obj_287f26acb0e6fc5d8a60ae1f9631329e0c7e7c9aa1e609bef69dcea916500397` is the exact AdamW configuration source: lr 0.003, betas (0.9, 0.98), gradient max norm 1.0 in the clipping occurrences, weight decay 1.0 for one group and 0.0 for the other. The initial batch occurrence `occ_03768472a75d6d68670c667ba04a22a4e04ce384867ee6224dbebda5c349e075` realizes `factblock_e42dc3a8244c3c3d3507cd71f70f8ee7b963f579c46ef9744a0cb9f95c931b0d`. All 10,000 `before_batch` occurrences contain 317 sample identities, and the input and target tensors each have one content hash across the complete run. This is exact full-batch training, not a timestamp join.

Formation uses exact parameter identities. At step 600, final-norm gain object `obj_308221e2f248ffaf5475c3ffd8f54b4c2cbcdfa3eb17d2115b9c5e61b7aeace7` has content hash `cab7dd25...` and derived L2 norm 14.4197585. Primitive `GeneratedOrigin` edge `edge_79565f159e3041c3f6e6036f53be3a1ae14174faa4e1809416efa771d6f4f2e8` links its exact version-599 source to this version-600 result. Evaluation occurrence `occ_4f215cbcd0f9f01c581f7f6ba5a84b3e6648065ae5001cd535c2103357fbc2c3` realizes `factblock_5bcba1eafa53b1796532a9d8a537844874c295d4a5647cc04f25b3c73ef03892` and reads the exact version-600 parameter set.

At transition, final-norm object `obj_954954ba221cbb7a531a90c56022d64c54e55e8b7c3fd85fce5f434cb468eca2` has L2 norm 21.1382. Optimizer occurrence `occ_e9e02e2f80fdc38b2ed172ccf4165e85ff029b2d98c8106e1f4f7024afdc4217` realizes final-norm `factblock_2c2f33dc2af96e0f91d7807863e9348d6e24e4045bb2825694faf4fbf4529d06`; its six ordered sources are the exact parameter, clipped gradient, optimizer configuration, `exp_avg`, `exp_avg_sq`, and optimizer step before update. Evaluation occurrence `occ_63440ac08d746fd742cc16218e6ba886661dfd0cfb0bbd97fd518beb5f173101` realizes `factblock_541c83d82352a4088004c01740e0cdb0e5ae782590eb666dfc4701ff801e8484`, whose sources include that exact final-norm version and the other exact version-1500 parameters. The norm is a derived statistic of the identified tensor; it never equates tensor identities.

The instability coordinate is also occurrence-bound. At optimizer step 7099, clipping occurrence `occ_75b2564fc5055bd0539cb939ebf07442b78110c438959c2e4efe6cb2ec33f9af` records raw total norm 67.7109833 and max norm 1.0. It realizes final-norm clipping `factblock_ae138352fd039f9a451696b29cacf41a244e4ec584604270c9dbe02b88a1a6c9`, mapping raw gradient object `obj_81b8b655c623e0aea45000fd109dfedf6540ec35d604665dd88be3baf343bfa8` to clipped object `obj_79f22e39af3c8a45b35e4f79c42d2b04ffdf6d28871e24af621ba38415c27ee4`. The following optimizer occurrence `occ_8617500cea98732e6887552206031a0deba4b4d6fb7f5415ab62503faaae47d0` realizes `factblock_5e10d94004b8c4c158ea1e24d2996126980b415b800138077045f1468654a6a2`; that fact reads the clipped gradient plus the exact parameter and Adam states and forms version-7100 final-norm object `obj_0f5874d14744f68c63d57440eb709f6a9be27ed4fd6274bc37fd8a5e5d728439`. Evaluation `factblock_6f0259a43e9d1f012a0bd0956afd2e5b1b007255c2f2077bd392ff400623ed9a` reads the exact version-7100 parameter set and forms metric object `obj_30c1f1edc0bf01d729a23dd83255c12114abb4e845ebd0d0c7f630f27e68a335`: train 0.2176656, validation 0.1603774, loss 7.5840874. Recovery evaluation `factblock_e6df50ee3920c4e7c606383d5b6b32e69b1da9c624af139edfe7dfe495e345b2` forms step-7200 metric `obj_92b0868ebec79b08895c83fb4c52ce3952d17227f2dfdbf0351980b9f27bd1d3`, with train 1.0 and validation 0.9952830.

The late prolonged case has the same primitive type but a longer recovery. Clipping occurrence `occ_a77bf143fbbabf9bf55b475b3d3c7b380196b29f7bafa1a7bbbcff8ad05c842f` at step 8716 records total norm 118.5426331 and realizes the 15 exact parameter-gradient clipping facts. Metrics `obj_ddf9bb10ae966c1de1f7ec0e0e339af957c17be62dbedf7c6111d01c18c9a8c6` and `obj_3a7f1aaeb6c7decba3282aa8437af602ea2005b3cfb04b765bfedaba6c91d82a` show degradation at 8800 and 9000; `obj_00a7ee3a5b672ff7ee94cf8b35d925e7d1e157048e21df20d3c222ef983670c3` shows recovery at 9100. These are linked through actual optimizer state and parameter `GeneratedOrigin` chains. Program order is used only to order the already identified occurrences, not to invent dependency.

## Mechanisms proposed and falsification attempts

| Hypothesis | Test | Result |
|---|---|---|
| Absolute-step sigmoid or stored accuracy trajectory | Remove graph state and let only optimizer step control the forecast; then apply the required 800-update state-pause intervention. The clock continues while every parameter and Adam state is held. | Rejected. It cannot execute the predicted delay and is unsupported as a cross-run invariant. A one-run fit does not repair that closure failure. |
| Final-norm gain alone; capability is permanently stable once its threshold is crossed | Search for similar gain with incompatible next/evaluation behavior. At 7000 and 7100 the final-norm L2 values are 48.8181 and 49.2225 (0.83% apart), but validation is 1.0 versus 0.1603774 and train is 1.0 versus 0.2176656. | Rejected. Gain is necessary for formation but insufficient for stability; optimizer recovery state separates the futures. This directly falsifies the proposed stable-after-formation mechanism. |
| Coupled gain plus clipped-Adam recovery oscillator | Initialize only from historical prefixes; forecast the sealed suffix; perturb operative fields; search for unmatched transitions and persistent failures. | Selected, not falsified by this execution. It retains gain, gain velocity, burst phase/period and high-gain recovery status. Cross-run validity remains prospective. |

The selected update is state-conditioned. If `g` is final-norm L2 and `d` is effective updates after the cut, code executes

`g(d) = g_cut + 0.0043 d + (r_cut - 0.0043) 2000 (1 - exp(-d/2000))`.

`r_cut` blends a robust prefix slope with a gain-conditioned structural slowdown. Rule strength is `sigmoid(4.5 (g/sqrt(n) - 2.11))`. Burst recurrence mean-reverts from the observed period toward `270 + 17 gain_rms`; normal recovery decays for 70 effective updates. The first burst at high gain has a longer, 300-update recovery and 600-update recharge. Validation is generated by rule strength minus recovery load, clipped to [0,1]. No future fact updates these variables.

## Prefix-only replay and closure

Each replay constructed a bounded graph at the cut, initialized the candidate from that prefix, and withheld every later occurrence. Accuracy RMSE is normalized by the frozen [0,1] accuracy range.

| Cut | Region and prefix state | Forecast 200 interval | Forecast 500 interval | Hidden-suffix curve NRMSE | Outcome |
|---:|---|---|---|---:|---|
| 600 | pre-formation; gain L2 14.4198; two observed bursts | [1400,1600] | [1300,1700] | 0.1204 | contains transition 1500 |
| 800 | pre/early formation; gain L2 16.0955; three bursts | [1400,1600] | [1300,1700] | 0.1199 | contains transition 1500 |
| 1100 | near formation and right-censored next burst | [1400,1600] | [1300,1700] | 0.1173 | contains transition 1500 |
| 2300 | after formation; transition already observed in prefix | retained exact 1500 | retained exact 1500 | 0.1248 | suffix remains compatible with transient recovery |
| 8700 | immediately before the prolonged instability | retained exact 1500 | retained exact 1500 | 0.0460 | predicts transient interval [8701,9001] and recovery |

The early cuts cover before and near formation; 2300 covers after formation; 8700 challenges the stability state. Passing these tests means only that the discovery execution did not falsify the compression.

Field perturbations establish operational use. Holding both gain rates at zero at cut 600 changes `will_transition` from true to false. Raising the resonance threshold beyond reach removes the predicted instability interval and changes stability from `TRANSIENT_DEGRADATION_RECOVERY` to `UNDETERMINED`. Perturbing the last burst period changes future burst phase. `recent_evaluations` is explicitly diagnostic; it is serialized for audit but is not claimed to control the forecast.

## Report-to-code correspondence

| Claim | Primitive evidence | Executable field | Exact code path |
|---|---|---|---|
| Gain accumulation controls formation | exact final-norm versions, optimizer fact blocks and evaluation bindings at 600 and 1500 | `gain_l2`, `gain_rate_l2_per_step`, `gain_dimension` | `_gain_at` -> `_rule_strength` -> curve and transition scan |
| Adam/clipping recovery controls instability | clipping facts, six-source optimizer update facts, exact degraded and recovered evaluations | `last_burst_step`, `last_burst_period`, `recovery_load` | `_project_pulses` -> `_pulse_load` -> curve and state phase |
| A high-gain pulse can recover slowly | step-8716 clipping occurrence and 8800–9100 exact metrics | `resonance_used`, `resonance_gain_rms` | `_project_pulses` 300-step branch -> instability intervals |
| Formation is sustained, not a one-point crossing | exact frozen evaluation facts at 1400–1700 | `had_low_validation`, `observed_transition_step` | exact prefix scan in `initialize`; three-point future scan in `forecast` |
| State pause delays formation | optimizer update facts read parameter, gradient and all Adam state | intervention counters and saved group rates | set every gradient to `None` and lr to zero at `before_optimizer_step`; restore after 800 successful pauses |

## Intervention audit

The intervention predicts `DELAY` with shift [700,900]. At each of 800 intervention-relative optimizer hooks, every current parameter gradient is set to `None` and all group learning rates are set to zero. PyTorch AdamW therefore skips the parameter: parameter value, `exp_avg`, `exp_avg_sq`, optimizer step and decoupled weight decay are all held. Native learning rates are saved as finite state and restored on the next hook. The mechanism change is an 800-update deficit: gain and oscillator phase stop advancing together.

RNG and data-order state are deliberately not frozen. The graph shows identical full-batch input and target content on all 10,000 batches, so neither can change model or Adam state while the optimizer inputs are removed. They can affect the resumed branch only if the unchanged model code consumes RNG independently; the ±100 shift interval covers that remaining uncertainty. The task, evaluator, validation data, capture, endpoint and all branch identities are untouched. Because all operative state is paused together and then resumes, the predicted stability effect is `NO_CHANGE`, not an unsupported claim that freezing one visible quantity freezes hidden state.

## Cross-run status and limits

The tensor scale, observed gain velocity, burst phase and burst period are inferred anew from the unseen prefix. The functional form and its constants were estimated from this one discovery execution. There is no run/date lookup and no discovery trajectory table in code. Matched-state divergent-future searches forced retention of optimizer recovery state, but one run cannot prove that no additional context is required. The unseen continuous execution is therefore the first test of cross-run transport, not a foregone conclusion.
