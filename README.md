# GFG training–learning–inference experiments

This public, read-only companion repository collects the executable experiments used to
establish and test the generation-fact, training–learning, inference and
reinforcement-learning claims of the accompanying manuscript.

The repository has **16 primary experiment entries**:

- two structural projection experiments;
- eight experiments forming the training–learning theory;
- one direct prediction experiment;
- one frozen-inference projection experiment; and
- four reinforcement-learning experiments spanning feedback closure,
  long-delay credit discovery, recursive optimization of credit discovery and
  stochastic long-chain credit.

The primary entries are indexed in [EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md).
The final manuscript claims, figures and tables are connected to their frozen
experimental authorities in
[MANUSCRIPT_EVIDENCE_MAP.md](MANUSCRIPT_EVIDENCE_MAP.md).
The exact public verification boundary for every manuscript experiment is
recorded in [PUBLIC_EVIDENCE_MATRIX.md](PUBLIC_EVIDENCE_MATRIX.md), and the
Git/Zenodo division is documented in [PUBLIC_ARCHIVE.md](PUBLIC_ARCHIVE.md).
The four reinforcement-learning experiments form one cumulative evidence chain,
summarized in [RL_EVIDENCE_CHAIN.md](RL_EVIDENCE_CHAIN.md).
Additional experiment directories are included only when they are imported by
a primary entry or are required to reconstruct its Generation-Fact Graph
(GFG). They are supporting dependencies, not additional manuscript claims.

The index also exposes **GF-S01**, a four-stage real-signal experiment showing
that the generation-fact and GFG structures remain fixed while their concrete
domain semantics change across filtering, downsampling, Fourier analysis and
SVG rendering.

The separate **TL-G01** cross-system falsification experiment changes
architecture, modality, task and optimizer from nanoGPT/text/Adam to
ResNet-18/CIFAR-100/SGD momentum and retests the primary training--learning
relations under a frozen protocol. It is an extension of the evidence base,
not a rewrite of the sealed sixteen manuscript entries.

The separate **TL-G02** experiment moves the same falsification programme to a
generative diffusion objective: a time-conditioned U-Net predicts identified
noise occurrences on CIFAR-10 under AdamW. It retests receiving-state
conditioning, nonlinear response, distributed support and held-out coordinate
transport with a non-classification readout boundary.

The separate **INF-G01** experiment then retests the frozen-inference projection
relation in both cross-system models. Across three ResNet seeds and three
diffusion seeds, it checks exact training-version identity, causal component
recruitment, query-conditioned non-additive support combination and exact
pre-learning rollback/restoration while keeping persistent learned state frozen.
The cumulative relation among TL-G01, TL-G02 and INF-G01 is summarized in
[CROSS_SYSTEM_EVIDENCE_CHAIN.md](CROSS_SYSTEM_EVIDENCE_CHAIN.md).

Two later feedback-dynamics experiments test a bounded consequence of the
training--learning--inference loop in a shared policy. RL-E05 studies selective
positive feedback and support concentration; RL-E06 freezes feedback dose,
duration and recovery forks and independently re-executes all 12 formal seeds.
They are indexed as extensions rather than revisions to the sealed sixteen
manuscript entries. See [RL_EVIDENCE_CHAIN.md](RL_EVIDENCE_CHAIN.md).

Four reusable instruments developed for the training--learning experiments are
exposed through [EXPERIMENTAL_INSTRUMENTS.md](EXPERIMENTAL_INSTRUMENTS.md):
CSRG-4C, realized-update causal forks, finite-amplitude update paths and the
identity-aligned target-boundary ledger. The index points directly to their
canonical code, frozen contracts and validators.

## Structural basis

The two added structural experiments are:

1. **GF-P01 — five mature mechanisms as exact strict projections:** database
   which-lineage, ECMA-426 Source Maps, OpenTelemetry, W3C PROV and PyTorch
   Autograd.
2. **GF-P02 — classical provenance semiring projection:** exact strict
   projection to canonical `N[X]`, followed by the frozen algebraic and task
   projections specified by the experiment.

## Repository provenance

The original experiment set and RL-E02 were imported without rewriting their
experimental content from frozen source commits. RL-E03 and RL-E04 were then
developed and frozen directly in this companion repository. Exact source
commits and Git tree identities are recorded in
[SOURCE_MANIFEST.md](SOURCE_MANIFEST.md).

Four large base-GFG JSON payloads are intentionally not added to this companion
repository. Their byte sizes and SHA-256 identities are recorded in
[EXTERNAL_ARTIFACTS.md](EXTERNAL_ARTIFACTS.md); their generators, validation
outputs and source-commit identities remain present.

## Reproduction

Each primary experiment retains its frozen protocol, implementation, machine
results, negative controls and independent checker where those objects existed
in the source experiment. Generated evidence omitted from Git is provided in
the accompanying content-addressed archive where stated. Reproduction commands
and the distinction between result recomputation and full native re-execution
are documented in `PUBLIC_EVIDENCE_MATRIX.md`. The common Python package and
Core v3 schemas are retained under `src/` and `protocol/core_v3/`.

Environment-dependent and source-history-dependent checks are identified in
[REPRODUCIBILITY_NOTES.md](REPRODUCIBILITY_NOTES.md).
The final pre-publication experiment-by-experiment audit is recorded in
[FULL_REPRODUCTION_AUDIT.md](FULL_REPRODUCTION_AUDIT.md).
The single cumulative cross-system and feedback-dynamics freeze is described in
[FINAL_EXTENSION_RELEASE_AUDIT.md](FINAL_EXTENSION_RELEASE_AUDIT.md); its public
archive entry point is `python tools/verify_final_extension_evidence.py
<download-directory>`.

This repository has no open-source license. See
[PROPRIETARY_NOTICE.md](PROPRIETARY_NOTICE.md).

Two third-party provenance papers used as frozen audit authorities in GF-P02
are intentionally not redistributed in this public snapshot. Their citations,
DOIs and audited file hashes are preserved in
[THIRD_PARTY_AUTHORITIES.md](THIRD_PARTY_AUTHORITIES.md).
