# Manuscript claim-to-evidence map

This map connects the principal claims and quantitative results in the final
manuscript structure to their frozen experimental protocols, executable
implementations, machine-readable results and independent checks. The
manuscript itself is not stored in this repository.

This documentation-only revision updates the mapping to the current manuscript
structure. All experimental evidence, validators, archive hashes and executable
authorities remain frozen at tag
`paper-experiments-cross-system-feedback-release`, commit
`36dab5ce347dbbdac157ef23205f556606d18294`, and Zenodo record
[`10.5281/zenodo.22032772`](https://doi.org/10.5281/zenodo.22032772).

## Generation facts and structural projections

| Manuscript location | Claim or result | Evidence authority |
|---|---|---|
| Introduction and Section 2 | The five-coordinate atomic generation fact, binding, scoped validation, multi-stage `GeneratedOrigin` and GFG construction are executable structures rather than retrospective labels. | [Core v3 protocol](protocol/core_v3/), [generation-relation implementation](src/generation_relation_core/), [executable GFG v2](experiments/executable_generation_fact_graph_v2/) |
| Introduction | Database which-lineage, Source Maps, OpenTelemetry, W3C PROV and PyTorch Autograd are exact and strict projections under their frozen profiles; each generation-fact coordinate has a deletion witness. | **GF-P01:** [experiment report](experiments/five_profile_unified_projection_proof/EXPERIMENT_REPORT.md), [machine summary](experiments/five_profile_unified_projection_proof/artifacts/five_profile_summary.json), [unified manifest](experiments/five_profile_unified_projection_proof/artifacts/unified_manifest.json) |
| Introduction | Canonical `N[X]` provenance and its registered lower algebraic and task views are exact projections; distinct generation states can retain the same polynomial. | **GF-P02:** [experiment report](experiments/provenance_semiring_projection_v1/EXPERIMENT_REPORT.md), [report statistics](experiments/provenance_semiring_projection_v1/artifacts/report_statistics.json), [unification result](experiments/provenance_semiring_projection_v1/artifacts/unification_of_unification_result_v2.json) |
| Section 3 | Scientific investigation can accumulate, query and recompile a family of GFGs, while interventions, replays and validations establish further execution-grounded facts. | The cumulative experiment lineage indexed in [EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md), source identities in [SOURCE_MANIFEST.md](SOURCE_MANIFEST.md), and the independent cross-domain signal witness **GF-S01:** [report](experiments/signal_multistage_generated_origin_v1/EXPERIMENT_REPORT.md) |

Section 3 is a methodological synthesis of the executed experiment lineage; it
is not presented as an additional measurement with a separate accuracy score.

## Training–learning dynamics

| Manuscript claim | Experiment | Protocol and implementation | Result and validation authority |
|---|---|---|---|
| Familiar observables such as loss, accuracy, absolute step, a single margin and LayerNorm gain are not sufficient training states; formed capability can subsequently remain, decline or recover. | **TL-E01** | [scientific contracts](experiments/gfg_nanogpt_autonomous_capability_discovery_v1/contracts/), [theory and counterexample ledger](experiments/gfg_nanogpt_autonomous_capability_discovery_v1/research_archive/THEORY_LEDGER.md) | [public run summary](experiments/gfg_nanogpt_autonomous_capability_discovery_v1/reports/public_summary.json), [archived submissions](experiments/gfg_nanogpt_autonomous_capability_discovery_v1/reports/formal_submissions/) |
| Pausing effective parameter–optimizer evolution delays formation, whereas the clipping-threshold negative control does not reproduce the result. | **TL-E02** | [intervention contract](experiments/gfg_nanogpt_autonomous_capability_discovery_v1/contracts/intervention_api.json), [checkpoint-fork implementation](experiments/gfg_nanogpt_autonomous_capability_discovery_v1/checkpoint_fork.py) | [theory ledger](experiments/gfg_nanogpt_autonomous_capability_discovery_v1/research_archive/THEORY_LEDGER.md), [causal-evaluation archive](experiments/gfg_nanogpt_autonomous_capability_discovery_v1/research_archive/entries/) |
| The same realized update can produce different functional responses in different receiving states. | **TL-E03** | [reciprocal matched-pair protocol](experiments/gfg_nanogpt_stepwise_support_transition_v1/RECIPROCAL_MATCHED_PAIR_PROTOCOL_V2.json), [local-response protocol](experiments/gfg_nanogpt_stepwise_support_transition_v1/P2_RECIPROCAL_LOCAL_RESPONSE_PROTOCOL.json) | [reciprocal validator](experiments/gfg_nanogpt_stepwise_support_transition_v1/reciprocal_validator.py), [GFG validator](experiments/gfg_nanogpt_stepwise_support_transition_v1/reciprocal_gfg_validator.py), [independent replay validator](experiments/gfg_nanogpt_stepwise_support_transition_v1/independent_replay_validator.py) |
| Complete realized-update responses include saturation, acceleration, turnback and sign reversal; a fixed local linear or quadratic extrapolation is not a general full-response law. | **TL-E04** | [frozen amplitude-path protocol](experiments/gfg_nanogpt_stepwise_support_transition_v1/B_UPDATE_AMPLITUDE_PATH_PROTOCOL.json) | [path validator](experiments/gfg_nanogpt_stepwise_support_transition_v1/amplitude_path_validator.py), [GFG validator](experiments/gfg_nanogpt_stepwise_support_transition_v1/amplitude_path_gfg_validator.py) |
| The primary conditioning coordinates are target boundary state, target-specific update geometry and parameter–Adam receiving state; native history and prior response curves add no stable independent gain after these coordinates in the executed analysis. | **TL-E05** | [factor-analysis contract](experiments/gfg_nanogpt_response_factor_analysis_v1/FACTOR_ANALYSIS_CONTRACT.md), [analysis](experiments/gfg_nanogpt_response_factor_analysis_v1/analysis.py) | [validator](experiments/gfg_nanogpt_response_factor_analysis_v1/validator.py) |
| Capability is supported distributively, with component necessity, substitution, backup, synergy and failure slack. | **TL-E06** | [CSRG experiment](experiments/gfg_nanogpt_support_redundancy_v1/README.md), [CSRG-4C-v1 capture contract](experiments/gfg_nanogpt_support_redundancy_v1/capture_contract_v2.json) | [archive verifier](experiments/gfg_nanogpt_support_redundancy_v1/verify_archive.py) |
| A realized update persistently reallocates support and can change the primary supporting component. Across 72 sections, the experiment formed 1,656 target-group transitions, 1,636 valid allocation transitions and 228 primary-support hand-offs. | **TL-E07** | [analysis scope](experiments/gfg_nanogpt_support_reallocation_audit_v1/ANALYSIS_SCOPE.md), [analysis](experiments/gfg_nanogpt_support_reallocation_audit_v1/analysis.py) | [independent checker](experiments/gfg_nanogpt_support_reallocation_audit_v1/INDEPENDENT_CHECKER.py), [independent recomputation](experiments/gfg_nanogpt_support_reallocation_audit_v1/independent.py) |
| Identity-aligned target margins connect internal response and support change to the four observable transitions across a target-specific readout boundary. | **TL-E08** | [identity-aligned ledger construction](experiments/gfg_nanogpt_identity_aligned_margin_crossing_v1/experiment.py), [actual-update boundary protocol](experiments/gfg_nanogpt_actual_update_boundary_v1/PROTOCOL_FREEZE.md) | The archived 15,264-row ledger retains target identities, pre/post-update margins and all four truth transitions; its identities, hashes, counts and transitions are recomputed by the [independent checker](experiments/gfg_nanogpt_actual_update_boundary_v1/INDEPENDENT_CHECKER.py). A separate local-quadratic readout candidate was falsified and is not used as authority for the boundary facts. |

Together, TL-E01–TL-E08 support the mechanism chain synthesized in Section 4;
no single experiment is treated as establishing the entire chain alone.

## Direct prediction

| Manuscript result | Experiment | Evidence authority |
|---|---|---|
| The three response coordinates were used after the actual update was formed and before post-update target outputs were read. | **TL-P01** | [frozen protocol](experiments/gfg_nanogpt_actual_update_boundary_v1/PROTOCOL_FREEZE.md) |
| Frozen confirmation split: 4,652/5,088 post-update target-boundary outcomes; 91.43% accuracy, 92.17% balanced accuracy and 91.49% macro-averaged recall across the four transitions. | **TL-P01** | [formal results](experiments/gfg_nanogpt_actual_update_boundary_v1/RESULTS.md), [independent checker](experiments/gfg_nanogpt_actual_update_boundary_v1/INDEPENDENT_CHECKER.py) |
| All 12 eligible fully materialized runs: 14,069/15,264 outcomes, 92.17% accuracy and 91.30% four-transition macro recall. | **TL-P01** | [formal results](experiments/gfg_nanogpt_actual_update_boundary_v1/RESULTS.md), [independent checker](experiments/gfg_nanogpt_actual_update_boundary_v1/INDEPENDENT_CHECKER.py) |

The second-order predictor is an endpoint boundary predictor derived from the
three coordinates. It is not presented as a fixed quadratic law for the whole
finite-amplitude response curve falsified by TL-E04.

## Frozen inference, feedback dynamics and cross-system validation

| Manuscript location | Claim or result | Evidence authority |
|---|---|---|
| Section 6 | Across thirteen training histories and 52 checkpoints, frozen inference used exact training-formed versions, recruited target- and query-conditioned support, combined support non-additively and changed under component-version rollback. | **INF-E01:** [frozen protocol](experiments/gfg_nanogpt_training_learning_inference_projection_v1/PROTOCOL_FREEZE.md), [analysis](experiments/gfg_nanogpt_training_learning_inference_projection_v1/analysis.py), [independent checker](experiments/gfg_nanogpt_training_learning_inference_projection_v1/independent.py), [strict audit](experiments/gfg_nanogpt_training_learning_inference_projection_v1/strict_audit.py) |
| Sections 6.2–6.3 | The scale and attention passages are mechanism-level deductions from the established frozen-projection relation, connected to their cited literature; they are not separate experiment IDs or additional benchmark comparisons. | **INF-E01** plus the manuscript references cited in those passages. |
| Section 6.4 | Selective positive feedback concentrated support around the reinforced behaviour and was accompanied by reduced performance in other skills; rebalanced feedback largely recovered those skills while preserving the reinforced skill. The stronger preregistered temporal-precedence claim in RL-E05 was not supported. | **RL-E05:** [protocol](experiments/gfg_rl_selective_positive_feedback_support_concentration_v1/PROTOCOL_FREEZE.md), [results](experiments/gfg_rl_selective_positive_feedback_support_concentration_v1/RESULTS.md), [scientific assessment](experiments/gfg_rl_selective_positive_feedback_support_concentration_v1/SCIENTIFIC_ASSESSMENT.md). **RL-E06:** [protocol](experiments/gfg_rl_selective_positive_feedback_dose_recovery_v1/PROTOCOL_FREEZE.md), [results](experiments/gfg_rl_selective_positive_feedback_dose_recovery_v1/RESULTS.md), [independent checker](experiments/gfg_rl_selective_positive_feedback_dose_recovery_v1/independent_checker.py). |
| Section 7.1 | Receiving-state-conditioned functional response and persistent support reorganization were independently reproduced in ResNet-18/CIFAR-100 with SGD momentum and in a diffusion U-Net/CIFAR-10 system with AdamW. | **TL-G01:** [protocol](experiments/gfg_resnet_cifar_training_learning_generalization_v1/PROTOCOL_FREEZE.md), [results](experiments/gfg_resnet_cifar_training_learning_generalization_v1/RESULTS.md), [independent checker](experiments/gfg_resnet_cifar_training_learning_generalization_v1/INDEPENDENT_CHECKER.py). **TL-G02:** [protocol](experiments/gfg_ddpm_cifar_training_learning_generalization_v1/PROTOCOL_FREEZE.md), [results](experiments/gfg_ddpm_cifar_training_learning_generalization_v1/RESULTS.md), [independent checker](experiments/gfg_ddpm_cifar_training_learning_generalization_v1/INDEPENDENT_CHECKER.py). |
| Section 7.2 | Frozen inference in all three ResNet seeds and all three diffusion seeds preserved exact learned state, recruited query-conditioned support, combined support non-additively and depended on training-formed component versions. | **INF-G01:** [protocol](experiments/gfg_cross_system_frozen_inference_projection_v1/PROTOCOL_FREEZE.md), [results](experiments/gfg_cross_system_frozen_inference_projection_v1/RESULTS.md), [independent checker](experiments/gfg_cross_system_frozen_inference_projection_v1/INDEPENDENT_CHECKER.py). |
| Section 8 | The conclusion synthesizes the nanoGPT training–learning results, prospective prediction, frozen-inference evidence, bounded feedback result and cross-system validation. | **TL-E01–TL-E08**, **TL-P01**, **INF-E01**, **RL-E05–RL-E06**, **TL-G01–TL-G02** and **INF-G01**. |

## Foundational test of the recursive scientific process

The manuscript Methods section uses RL-E01–RL-E04 as a separate cumulative
test of the GFG-based recursive scientific process: establish a relation,
discover it without being supplied the true relation, compile the discovery
computation itself to reorganize causal adjudication, and retain the method
under occurrence-addressed stochastic variation. The complete mapping is in
[RL_EVIDENCE_CHAIN.md](RL_EVIDENCE_CHAIN.md).

This sequence includes the reported 64-to-9 candidate reduction, exact recovery
of signed credit and pair interactions, a 90.64% reduction in native replay
transitions, a 2.38-fold end-to-end speedup and the held-out stochastic-policy
results. Formation ancestry is never counted as causal credit without matched
replay or coalition intervention.

## Figure and table authority

- **Figure 1** is the methodological synthesis represented by Section 3 and
  the cumulative experiment lineage; it is not an additional measured result.
- **Figure 2** summarizes the response morphologies established by TL-E04.
- **Table 1** is generated from TL-P01's frozen confirmation results.
- **Table 2** is generated from INF-E01's frozen inference comparisons.
- **Table 3** is generated from RL-E06's frozen dose, endpoint trade-off,
  rebalancing-recovery and fresh-reproduction results.

## Artifact availability boundary

Compact protocols, implementations, summaries and independent checkers are
tracked in this repository. Large base-GFG payloads and the RL-E02 formal
runtime bundle, the TL-P01 target ledger and the INF-E01 derived GFGs are not
duplicated in Git. Their public archive locations and verification boundaries
are fixed in [PUBLIC_ARCHIVE.md](PUBLIC_ARCHIVE.md) and
[PUBLIC_EVIDENCE_MATRIX.md](PUBLIC_EVIDENCE_MATRIX.md); top-level byte sizes
and SHA-256 hashes are fixed by the archive's `ARCHIVE_MANIFEST.json`.
