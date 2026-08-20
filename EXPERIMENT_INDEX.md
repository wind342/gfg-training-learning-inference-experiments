# Primary experiment index

The original sixteen-entry evidence set was sealed by `paper-experiments-v2`.
Later publication-evidence releases retain the same experiment identities and
add portable evidence bundles or documentation corrections without rewriting
the frozen results. A manuscript-section-level map is provided in
[MANUSCRIPT_EVIDENCE_MAP.md](MANUSCRIPT_EVIDENCE_MAP.md).
Together, RL-E01–RL-E04 form the cumulative experimental chain summarized in
[RL_EVIDENCE_CHAIN.md](RL_EVIDENCE_CHAIN.md).

| ID | Experiment | Main scientific role | Code, protocol and evidence |
|---|---|---|---|
| GF-P01 | Five-profile unified projection proof | Establishes the five mature domain mechanisms as exact and strict projections under their frozen profiles. | [five-profile experiment](experiments/five_profile_unified_projection_proof/) |
| GF-P02 | Classical `N[X]` provenance-semiring projection | Establishes exact strict projection to canonical `N[X]` and the frozen lower algebraic/task projections. | [semiring experiment](experiments/provenance_semiring_projection_v1/) |
| TL-E01 | Cross-run state sufficiency and post-formation trajectory audit | Falsifies loss, accuracy, absolute step, a single margin and LayerNorm gain as sufficient states; establishes post-formation decline and recovery as phenomena requiring explanation. | [theory and counterexample ledger](experiments/gfg_nanogpt_autonomous_capability_discovery_v1/research_archive/THEORY_LEDGER.md) |
| TL-E02 | Optimizer pauses and clipping negative controls | Tests whether effective parameter–optimizer evolution, rather than a superficial clipping change, moves capability-formation time. | [intervention experiment](experiments/gfg_nanogpt_autonomous_capability_discovery_v1/) |
| TL-E03 | Full/skip, receiving-state exchange and reciprocal response | Establishes that the same realized update produces different functional responses in different receiving states. | [stepwise causal experiment](experiments/gfg_nanogpt_stepwise_support_transition_v1/) |
| TL-E04 | Multi-amplitude realized-update paths | Falsifies a globally fixed linear or quadratic local extrapolation and establishes finite-amplitude nonlinear functional response. | [frozen amplitude-path protocol](experiments/gfg_nanogpt_stepwise_support_transition_v1/B_UPDATE_AMPLITUDE_PATH_PROTOCOL.json) |
| TL-E05 | Response-conditioning factor analysis | Identifies target boundary state, target-specific update geometry and parameter–Adam receiving state as the three primary conditioning factors. | [factor-analysis experiment](experiments/gfg_nanogpt_response_factor_analysis_v1/) |
| TL-E06 | CSRG single- and paired-component gating | Establishes distributed functional support, including necessity, substitution, backup, synergy and failure slack. | [CSRG experiment](experiments/gfg_nanogpt_support_redundancy_v1/) |
| TL-E07 | Support reallocation across an actual update | Measures persistent reorganization and support hand-off caused by realized training updates. | [support-reallocation experiment](experiments/gfg_nanogpt_support_reallocation_audit_v1/) |
| TL-E08 | Identity-aligned target margins and readout crossings | Connects internal change to the four observable target transitions through the target-specific readout boundary. | [identity-aligned boundary experiment](experiments/gfg_nanogpt_identity_aligned_margin_crossing_v1/) |
| TL-P01 | Actual-update target-boundary prediction | Tests the predictive sufficiency of the three primary factors before post-update target responses are read. | [prediction experiment](experiments/gfg_nanogpt_actual_update_boundary_v1/) |
| INF-E01 | Frozen-inference projection | Tests exact learned-version use, causal support recruitment, query conditioning, non-additive combination and rollback dependence under frozen inference. | [inference experiment](experiments/gfg_nanogpt_training_learning_inference_projection_v1/) |
| RL-E01 | Reinforcement-learning feedback closure | Tests consequence binding and temporal credit as the relations that close inference outcomes back into effective training formation. | [RL feedback-closure experiment](experiments/gfg_rl_feedback_closure_v1/) |
| RL-E02 | GFG-guided long-delay temporal-credit discovery | Tests whether formation-path retrieval can identify a compact historical candidate set, matched causal replay can distinguish true credit from ancestry, backup and synergy, and the resulting credit relations can form a held-out policy. | [temporal-credit discovery experiment](experiments/gfg_temporal_credit_discovery_v1/) |
| RL-E03 | Long-chain temporal-credit discovery and self-optimization | Replaces direct terminal slot reads with 64-step versioned state formation, validates both the base execution GFG and a credit-discovery meta-GFG, and tests whether formation structure can reduce exact causal-adjudication cost against trace, cache, dependency-DAG and rewired controls. | [long-chain self-optimization experiment](experiments/gfg_temporal_credit_long_chain_self_optimization_v1/) |
| RL-E04 | Stochastic long-chain temporal-credit discovery | Adds occurrence-addressed exogenous inputs to every native state transition, separates conditional action credit from realized stochastic variation through matched replay, estimates expected credit on disjoint random tapes, and tests stochastic-binding, interaction, policy-transfer and full-cost controls. | [stochastic long-chain experiment](experiments/gfg_temporal_credit_stochastic_long_chain_v1/) |

## Additional structural experiment

| ID | Experiment | Main scientific role | Code, protocol and evidence |
|---|---|---|---|
| GF-S01 | Four-stage real-signal formation | Demonstrates that the five-coordinate fact structure and GFG organization remain fixed while concrete domain semantics change across FIR filtering, downsampling, sliding FFT and SVG rendering; a final SVG-space query is traced through the complete path to the participating raw ECG samples. | [multistage signal experiment](experiments/signal_multistage_generated_origin_v1/) |

## Cross-system training--learning falsification

| ID | Experiment | Main scientific role | Code, protocol and evidence |
|---|---|---|---|
| TL-G01 | ResNet/CIFAR-100/SGD-momentum generalization | Changes architecture, modality, task and optimizer together, then retests receiving-state conditioning, finite-amplitude nonlinearity, distributed-support reallocation, target boundaries and held-out F1/F3/F5 transport under a frozen protocol. | [cross-system generalization experiment](experiments/gfg_resnet_cifar_training_learning_generalization_v1/) |
| TL-G02 | DDPM-style U-Net/CIFAR-10/AdamW generalization | Changes the task to generative diffusion residual prediction and the target identity to an image--timestep--noise occurrence, then retests receiving-state conditioning, finite-amplitude nonlinearity, distributed-support reallocation, residual-error boundaries and held-out F1/F3/F5 transport. | [diffusion generalization experiment](experiments/gfg_ddpm_cifar_training_learning_generalization_v1/) |

## Projection dependencies retained for GF-P01

The five native reference paths are retained at:

- `experiments/database_lineage/`
- `experiments/source_map_projection/`
- `experiments/opentelemetry_projection/`
- `experiments/w3c_prov_projection_v1/`
- `experiments/pytorch_autograd_training_lineage_v1/`

GF-P01 invokes these paths separately and compares their independently formed
native/reference views with the projected views. They are not five additional
entries in the 16-experiment manuscript index.
