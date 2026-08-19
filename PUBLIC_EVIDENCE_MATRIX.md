# Public evidence and reproduction matrix

This matrix states exactly what a third party can verify from the public Git
repository and the accompanying Zenodo archive.  It deliberately distinguishes
source inspection, result recomputation from frozen machine records, and full
native re-execution.  A protocol or checker is not labelled as a public replay
when its original checkpoints or large tensors are not in the archive.

The archive series is identified by the stable Zenodo concept DOI
[`10.5281/zenodo.22005307`](https://doi.org/10.5281/zenodo.22005307).  Each
published version has its own immutable version DOI and top-level manifest.

| Experiment | Public material | What can be checked publicly | Boundary |
|---|---|---|---|
| GF-P01 | Code, frozen profiles, machine summaries and deletion witnesses in Git | Exact/strict projection decisions and five-coordinate irreducibility witnesses | Three upstream Git-history authority checks require the original source repository, as recorded in `REPRODUCIBILITY_NOTES.md` |
| GF-P02 | Code, machine summaries and frozen third-party authority hashes in Git | Semiring projection and collision-witness decisions | Third-party papers are cited and hashed, not redistributed |
| TL-E01–TL-E02 | Contracts, compact run summaries, ledgers and causal-evaluation records in Git | Reported state-sufficiency counterexamples and intervention decisions | Complete native training histories and checkpoints are not duplicated |
| TL-E03–TL-E08 | Frozen protocols, implementations, validators, the four-instrument source archive and the TL-P01 identity-aligned truth ledger | Method, event boundary, adjudication code, compact recorded results and all 15,264 identity-aligned four-transition truth records | Full native replay requires the source training/CSRG bundles identified by the manifests; TL-E05 retains its disclosed unused global-unseen setup access, and its factor records exclude that run |
| TL-P01 | `tl_p01_actual_update_boundary_evidence_v1.zip` plus the Git checker | All 15,264 target records, hashes, derivative audit, the frozen four-run confirmation split, 4,652/5,088 result, accuracy, balanced accuracy, per-transition recalls and macro recall are recomputed | This is a frozen raw-state re-execution of the established algorithm, not a new confirmation run |
| INF-E01 | `inf_e01_frozen_inference_gfg_evidence_v1.zip` plus the Git checker | Integrity of 13 derived GFGs and their tensor payloads; 52 checkpoint-phase logit-level non-additivity and query-conditioned support-profile results are recomputed | Regenerating the GFGs and rollback forwards requires the original parameter checkpoints |
| RL-E01 | Protocol, implementation, results and compact causal-fork records in Git | Binding/credit fork decisions and reported seed summary | Fresh training requires the documented Python environment |
| RL-E02 | `rl_e02_temporal_credit_formal_evidence_v1.zip` plus Git validators | Formal GFG ledgers, candidate/credit ledgers, checkpoints, policy evaluations and independent checks | The frozen archive preserves the original absolute path labels; validators rebase the runtime root |
| RL-E03–RL-E04 | Code, frozen contracts, committed machine results, validators and tests in Git | Long-chain, exact-credit optimization, stochastic controls and policy-transfer results | Fresh stochastic/training runs require the documented environment |

## Public verification entry points

- TL-P01: `python -m experiments.gfg_nanogpt_actual_update_boundary_v1.INDEPENDENT_CHECKER <extracted-bundle-root>`
- INF-E01: `python -m experiments.gfg_nanogpt_training_learning_inference_projection_v1.PUBLIC_EVIDENCE_CHECKER <extracted-bundle-root>`
- Top-level Zenodo archive: `python tools/verify_publication_evidence_v3.py <download-directory>`

Absolute filesystem strings retained inside frozen source manifests are
provenance labels from the native execution.  They are not required input paths
for the two public result checkers above and have not been rewritten after the
fact.
