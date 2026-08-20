# Final cross-system and feedback-dynamics release audit

This audit freezes the extension evidence in one cumulative release. No new
scientific experiment was added during packaging. The audited scope consists
of TL-G01, TL-G02 and INF-G01, followed by RL-E05 and RL-E06 and the correction
that distinguishes recovery from a common update-800 fork from comparison with
the continued-exclusive update-3,200 endpoint.

## Experimental audit

| Evidence | Independent result |
|---|---|
| TL-G01 | PASS; all three ResNet-18/CIFAR-100/SGD-momentum seeds, manifests, checkpoints and aggregate criteria recomputed; verdict `CROSS_SYSTEM_GENERALIZATION_SUPPORTED` |
| TL-G02 | PASS; all three diffusion U-Net/CIFAR-10/AdamW seeds, manifests and 504 occurrence-level responses recomputed; verdict `CROSS_SYSTEM_GENERALIZATION_SUPPORTED` |
| INF-G01 | PASS; all six ResNet/diffusion checkpoint authorities and all nine criteria per seed recomputed; verdict `CROSS_SYSTEM_FROZEN_PROJECTION_SUPPORTED` |
| RL-E05 | PASS as an integrity and adjudication check over 12 seeds and 29,352 generation facts; the frozen scientific status remains `NOT_SUPPORTED` because temporal precedence failed, while the narrower concentration/crowding gates remain positive |
| RL-E06 | PASS after a fresh independent native re-execution of all 12 seeds; all stored diagnostics, state identities, feedback and trajectory hashes, support measurements, final evaluations, GFG validations and aggregate gates matched |

The RL-E06 native audit re-executed 84 condition runs. Every seed preserved
3,200 receipts and 3,201 boundary states for each of five dose conditions, and
2,400 receipts and 2,401 boundary states for each of two recovery conditions.
Across all seeds, 256,788 GFG facts were checked. The independently recomputed
mean exclusive unreinforced-accuracy deficit was `0.390625`, the mean exclusive
support-share excess was `0.09667916595935822`, and the mean rebalance gain over
the continued-exclusive endpoint was `0.38541666666666663`.

## Corrected recovery reference

The numerical endpoints were not changed. The wording was corrected so that
two different comparisons are not conflated:

- from the common update-800 recovery fork, unreinforced-skill accuracy rose
  from `70.31%` to `99.48%`, an observed recovery of `29.17` percentage points;
- relative to the matched continued-exclusive update-3,200 endpoint of
  `60.94%`, the same `99.48%` endpoint is higher by `38.54` percentage points.

The frozen runner, result file and formal evidence hashes were unchanged by
this wording correction.

## Repository and archive controls

- the repository test suite passed with `64 passed, 3 skipped`; the three
  documented skips require source-history objects absent from the public
  companion clone and are unrelated to the extension experiments;
- all Python sources compile, all tracked JSON files parse and all checked
  relative Markdown links resolve;
- each carried-forward archive must match its preceding top-level byte length
  and SHA-256 identity before it can enter the cumulative release;
- each new bundle contains a complete file manifest, rejects unsafe or
  duplicate ZIP paths and is checked against the committed formal-result
  authority;
- the final archive is accepted only if
  `tools/verify_final_extension_evidence.py` passes before publication and
  passes again on files downloaded from the public Zenodo version.

The public RL-E06 bundle stores the complete formal evidence once and includes
the compact independent-check summary. The separate full native replay payload
is intentionally not duplicated because it is an independently regenerated
copy of the archived formal execution rather than a second scientific
authority.
