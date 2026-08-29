# FA-E01 public evidence entry

## Scientific question

Do causal component-gating fingerprints in the existing frozen nanoGPT
histories converge by function across different concrete addition inputs and
different correct outputs?

## Pre-outcome boundary

The source audit established before analysis that all 212 validation examples
instantiate one operation only. FA-E01 can therefore test single-relation
invariance, but cannot test whether different relations remain selectively
separated. The protocol and thresholds were committed at `88b10b2` before the
full analysis was run.

## Evidence used

- 13 independently generated nanoGPT training histories;
- four mechanically registered phases per history;
- native baseline, four single-gate and six pair-gate inference tensors;
- content-addressed training inputs, training targets and validation inputs;
- independently reconstructed validation targets;
- existing validated GFG manifests and generation paths.

No model was trained and no new inference was executed for FA-E01.

## Result

The frozen verdict is **`INCONCLUSIVE`**.

- 13/13 histories were structurally eligible.
- Formed-phase LOO common-direction score: median 0.907752.
- Formed-phase excess over the coordinate-independence null: median 0.006424
  (frozen minimum 0.05).
- Runs above the coordinate-null 95th percentile: 12/13.
- Cross-output practical equivalence: 13/13.
- Formed-minus-pre excess: median -0.003158; positive in 5/13 runs.
- The observed formed-minus-pre contrast was at percentile 0.2184 of the
  frozen phase-label null.
- The formed LOO advantage over output-label permutation was 0.000142 in the
  median.

The evidence contains a strong but mostly static architectural/input-structure
pattern. It does not establish that learning formed a cross-output abstract
addition relation.

## Claim boundary

FA-E01 neither proves nor disproves general functional abstraction. The frozen
evidence contains only one registered relation and cannot test across-relation
selectivity. A new multi-relation experiment would be needed for that claim.

## Files

- `EXISTING_NANOGPT_AUDIT.md` — pre-outcome structural audit;
- `FA_E01_PROTOCOL_FREEZE.md` — frozen protocol and verdict rules;
- `FA_E01_ANALYSIS.py` — primary source-evidence analysis;
- `FA_E01_RESULTS.json` — complete run metrics and null distributions;
- `FA_E01_FINGERPRINTS.npz` — raw, scaled and normalized fingerprints;
- `FA_E01_INDEPENDENT_RECOMPUTE.py` — separate recomputation path;
- `FA_E01_INDEPENDENT_RECOMPUTE.json` — independent PASS receipt;
- `FA_E01_REPORT.md` — complete scientific report;
- `FA_E01_PUBLIC_EVIDENCE_MANIFEST.json` — artifact hashes;
- `FA_E01_PUBLIC_EVIDENCE_CHECKER.py` — quick and deep checks.

All identities and hashes are checked by the supplied checker. Deep mode also
rebuilds the fingerprints, pair classes, leave-one-output-out scores, null
distributions and verdict from the frozen GFG/source objects.
