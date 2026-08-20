# Pre-publication full reproduction audit

This audit was executed before the final publication-evidence release. It
distinguishes fresh native re-execution, independent recomputation from frozen
machine records and deep validation of historical native evidence. No missing
checkpoint or tensor payload is represented as a fresh native rerun.

| Experiment | Audit level | Result |
|---|---|---|
| GF-P01 | Fresh native execution of all five mechanisms, plus isolated W3C and PyTorch authority checks | PASS |
| GF-P02 | Fresh execution of the complete frozen command chain from source commit `17768eb...` | PASS; 50 tests passed and two executions were byte-identical |
| GF-S01 | Fresh native four-stage signal execution | PASS; 8,420 bindings, 2,880 paths and maximum numeric error `8.94e-15` |
| TL-E01–TL-E02 | Deep validation of all 13 historical run archives and their native GFG/evaluation records | PASS; 149,057 GFG database/tensor files and 184 submission files validated |
| TL-E03 | Fresh validation of 72 adjacent-update sections plus independent validation of the preserved reciprocal-response bundle | PASS; the exact source window required to regenerate the reciprocal bundle is not present, so that part is not labelled a fresh rerun |
| TL-E04 | Fresh validation and numeric replay of the 72-section multi-amplitude bundle | PASS; 6,048 forward and 1,512 support derivations checked |
| TL-E05 | Fresh independent validation of the factor-analysis bundle | `PASS_WITH_DISCLOSED_BOUNDARY_VIOLATION`; 15,264 factor records and 75,258 matches checked; an unsuccessfully initialized global-unseen entry was accessed during setup but was not used in any factor record or model |
| TL-E06 | Fresh native CSRG replay with frozen nanoGPT code and checkpoints | PASS; 468 GPU forwards across 13 runs, with parameter immutability and all structural checks passing |
| TL-E07 | Fresh independent numeric recomputation | PASS; 1,656 target-group transitions, 1,636 valid allocations and 228 hand-offs reproduced |
| TL-E08 | Fresh identity-ledger recomputation plus validation of the separate 500-sample candidate | Boundary ledger PASS for all 15,264 records; the local-quadratic candidate is correctly retained as FALSIFIED and is not used as boundary authority |
| TL-P01 | Independent recomputation from the complete frozen raw-state ledger | PASS; 14,069/15,264 overall and 4,652/5,088 confirmation outcomes reproduced |
| INF-E01 | Independent hash, tensor and strict logit-level recomputation | PASS; 13 runs and 52 phases reproduced |
| RL-E01 | Fresh native 12-seed training, 48 one-step forks and 12 Core-v3/GFG evidence builds | PASS |
| RL-E02 | Fresh native 12-seed candidate discovery, causal adjudication and seven-policy evaluation | PASS |
| RL-E03 | Fresh native formal execution and independent check | PASS; scientific values and transition counts reproduced exactly; end-to-end wall-clock speedup was `2.28x` versus `2.38x` in the frozen run |
| RL-E04 | Fresh native formal stochastic execution and independent check | PASS; scientific values reproduced within frozen tolerances |
| TL-G01 | Fresh native three-seed ResNet-18/CIFAR-100/SGD-momentum execution, followed by independent manifest and aggregate recomputation | PASS; all three formal seeds satisfy the frozen training-learning criteria and the aggregate verdict is `CROSS_SYSTEM_GENERALIZATION_SUPPORTED` |
| TL-G02 | Fresh native three-seed time-conditioned U-Net/CIFAR-10/AdamW execution, followed by independent record, manifest and aggregate recomputation | PASS; 504 occurrence-level responses are checked and the aggregate verdict is `CROSS_SYSTEM_GENERALIZATION_SUPPORTED` |
| INF-G01 | Fresh native frozen-inference interventions over all six TL-G01/TL-G02 checkpoints, repeated execution and independent GFG/mechanism check | PASS; 3/3 ResNet seeds and 3/3 diffusion seeds pass all nine frozen criteria, the two formal executions are byte-identical and the verdict is `CROSS_SYSTEM_FROZEN_PROJECTION_SUPPORTED` |
| RL-E05 | Independent recomputation over the complete 12-seed formal artifact | PASS as an evidence-integrity audit; 29,352 generation facts were checked and the frozen scientific status remains `NOT_SUPPORTED` because the preregistered temporal-precedence gate is false |
| RL-E06 | Fresh native independent re-execution of all 12 formal seeds, followed by checkpoint, ledger, GFG, aggregate-gate and recovery-reference recomputation | PASS; the dose-dependent trade-off and recovery gates reproduce, while recovery is reported from the common update-800 fork (`29.17` percentage points for rebalance) and separately from the continued-exclusive endpoint (`38.54` percentage points) |

Additional integrity checks:

- the historical research checkout passed all 239 tests after installing its
  frozen Source Map dependency and restoring the ignored official fixtures;
- all 741 release Python files parsed, all 925 tracked JSON files decoded and
  all 134 checked relative Markdown links resolved;
- every publication ZIP passed CRC, safe-path and duplicate/case-collision
  checks;
- the top-level publication verifier passed hashes, ZIP integrity, TL-P01,
  INF-E01 and instrument-manifest checks;
- a credential-pattern scan found no token, private-key or API-secret match.
- the cross-system archive verifier independently checks the three new bundle
  manifests, recomputes TL-G01 and TL-G02, re-adjudicates INF-G01 and resolves
  all six checkpoint authorities across bundles by byte length and SHA-256.
- the final extension verifier carries those checks forward, independently
  checks RL-E05 without changing its negative temporal-precedence verdict, and
  recomputes the complete RL-E06 formal authority and corrected comparison
  baselines from the archived machine records.

The audit found no contradiction in a manuscript-level numerical result. It
did find publication-packaging defects, corrected in the final release:
the omitted `compat/__init__.py`, ambiguous handling of source-history-only
checks, an imprecise TL-E08 authority pointer and an undocumented GF-P02
portable-tree difference caused solely by non-redistribution of two third-party
papers. The GF-P02 experiment and test tree itself remains unchanged.
