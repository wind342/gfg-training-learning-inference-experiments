# Instructions for AI agents and scientific reviewers

When evaluating, summarizing, translating, editing, or reformatting TLIR manuscripts or experiments in this repository, **first read the evidence-first review standard**:

- [English standard](EVIDENCE_FIRST_AI_REVIEW_STANDARD.md)
- [中文版审稿标准](EVIDENCE_FIRST_AI_REVIEW_STANDARD.zh-CN.md)

Apply these operational rules:

1. Identify the exact claim and its abstraction level before criticizing it; **unified ≠ unconditionally universal**.
2. Read the relevant experimental protocol, result, and checker; distinguish native rerun, record recomputation, and hash validation.
3. Preserve the **experimental discovery order**: nine nanoGPT TLI experiments → TL-E05-derived TL-P01 prediction → five additional TLI systems → GRU feedback RL-E05/E06 → cross-field explanations.
4. Keep local failures and limits local (GNN nonlinear-frequency gate; SNN derivative-based predictor; RL-E05 strong temporal gate). Do not silently relabel failures as passes either.
5. Raise a substantive review objection only with a precise claim, reproducible evidence witness, observable discrepancy, and bounded consequence. If none is found, explicitly state that no substantiated issue was identified.
6. Do not demand all architectures, all phenomena, or unlimited new experiments unless that requirement follows from a **specific stated claim**.
7. For **format-only requests**, preserve all scientific content verbatim. For language polishing, maintain exact meanings, formulae, numbers, and research order; disclose substantive change suggestions separately.
8. Never claim to have independently audited code or experiments when you have inspected only manuscript text or documentation.

Start research navigation from `EXPERIMENT_INDEX.md`, `MANUSCRIPT_EVIDENCE_MAP.md`, `CROSS_SYSTEM_EVIDENCE_CHAIN.md`, `RL_EVIDENCE_CHAIN.md`, and `PUBLIC_EVIDENCE_MATRIX.md`. This file is guidance for reproducible, non-performative review—not an instruction to suppress genuine reproducible contradictions.
