# Publication evidence archive

The public evidence archive complements this Git repository.  Git contains the
frozen protocols, implementations and independent checkers; Zenodo contains
the generated payloads that are too large or unsuitable for normal Git
history.  The stable archive-series locator is
[`10.5281/zenodo.22005307`](https://doi.org/10.5281/zenodo.22005307).

The corrected publication-evidence release contains:

- `nanogpt_base_gfg_evidence_v1.zip` — the compact base nanoGPT GFG;
- `training_learning_instruments_evidence_v2.zip` — the four reusable
  experimental instruments with corrected repository pointers;
- `tl_p01_actual_update_boundary_evidence_v1.zip` — the complete 15,264-row
  prediction ledger and frozen confirmation results;
- `inf_e01_frozen_inference_gfg_evidence_v1.zip` — 13 derived inference GFGs,
  52 checkpoint phases and the frozen strict logit-level result;
- `rl_e02_temporal_credit_formal_evidence_v1.zip` — the formal temporal-credit
  discovery execution bundle;
- `tl_g01_resnet_cross_system_evidence_v1.zip` — three formal
  ResNet-18/CIFAR-100/SGD-momentum training-learning runs and their exact
  trained checkpoints;
- `tl_g02_diffusion_cross_system_evidence_v1.zip` — three formal
  time-conditioned U-Net/CIFAR-10/AdamW diffusion training-learning runs and
  their exact trained checkpoints;
- `inf_g01_cross_system_inference_evidence_v1.zip` — the formal ResNet and
  diffusion frozen-inference interventions, validated GFG and cross-bundle
  checkpoint authority ledger;
- `rl_e05_selective_feedback_evidence_v1.zip` — the complete 12-seed selective
  feedback/support-concentration experiment, including its retained negative
  temporal-precedence verdict;
- `rl_e06_dose_recovery_evidence_v1.zip` — the complete 12-seed dose, duration
  and recovery experiment with the corrected fork-versus-endpoint comparison;
- `ARCHIVE_MANIFEST.json` — byte sizes and SHA-256 identities for every
  top-level payload; and
- `PUBLIC_EVIDENCE_MATRIX.md` — the public verification boundary for every
  manuscript experiment.

Every Zenodo version is immutable.  Corrections are issued as a new version;
previous version DOIs, Git tags and commits remain part of the audit history.
The cross-system release carries forward every bundle from the preceding
publication-evidence version and adds TL-G01, TL-G02 and INF-G01 as the
cumulative evidence chain described in `CROSS_SYSTEM_EVIDENCE_CHAIN.md`.
The final extension release carries that complete archive forward once and
adds RL-E05 and RL-E06; it is verified by
`tools/verify_final_extension_evidence.py` before publication and again after
public download.
