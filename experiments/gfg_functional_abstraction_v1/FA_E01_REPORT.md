# FA-E01 — frozen-evidence functional-abstraction result

## Outcome

**Frozen verdict: `INCONCLUSIVE`.**

The existing 13 nanoGPT histories do not establish single-relation functional
invariance under the frozen FA-E01 criteria. They contain a highly aligned
ten-coordinate intervention pattern, but the increment above a
coordinate-independence null is small, does not increase from pre-formation to
formed capability, and is almost unchanged when output-group labels are
permuted. Input-overlap controls also retain detectable structure. The result
therefore cannot be interpreted as evidence that learning formed an abstract
addition relation.

This is not a negative verdict on functional abstraction in general. The
existing task contains only one registered operation, so it cannot test
across-relation selectivity. That structural limitation was identified before
outcome analysis in `EXISTING_NANOGPT_AUDIT.md`.

## Frozen evidence and eligibility

- Repository base: `36dab5ce347dbbdac157ef23205f556606d18294`
- Protocol commit: `88b10b2`
- Historical runs: 13
- Eligible runs: 13/13
- Historical phases per run: pre-formation, formed, decline and recovered
- Validation examples per run-phase: 212
- Correct-output groups: 23
- Primitive causal coordinates: four single-component gates and six
  pair-component gates
- New training or inference: none
- Required source, manifest, validation and tensor hashes: all passed
- Independently reconstructed validation targets: 13/13 exact
- Direction-eligible examples in every formed run: 100%

## Primary results

| Frozen criterion | Result | Required for support |
|---|---:|---:|
| Formed-phase median LOO score | 0.907752 | descriptive |
| Formed-phase median coordinate-null excess | 0.006424 | at least 0.05 |
| Formed runs above coordinate-null 95th percentile | 12/13 | at least 9/13 |
| Formed cross-output equivalence runs | 13/13 | at least 9/13 |
| Median formed-minus-pre excess | -0.003158 | at least 0.05 |
| Positive formed-minus-pre runs | 5/13 | at least 9/13 |
| Robustness checks with an opposite conclusion | 7/13 | 0 |

The 12/13 percentile result shows that the ten intervention coordinates are
not independent at the example level. Its practical magnitude is only 0.0064,
however, and the corresponding structure is already present before capability
formation. The phase-label permutation placed the observed formed-minus-pre
median at percentile 0.2184, with a 95% null interval of
[-0.009636, 0.009055].

## Run-level historical contrast

| Run | Pre excess | Formed excess | Formed-pre | Formed percentile |
|---|---:|---:|---:|---:|
| `entry-362ded584a953f360aec` | 0.003524 | 0.006266 | 0.002742 | 1.000 |
| `entry-481b86f81d58d496a687` | 0.016761 | 0.008373 | -0.008388 | 1.000 |
| `entry-4ed462761347d6b87e61` | 0.024268 | 0.012635 | -0.011634 | 1.000 |
| `entry-54ba9566f731754a0e3f` | 0.017178 | 0.009142 | -0.008036 | 1.000 |
| `entry-5bb1186bc27eb82111fb` | 0.007180 | 0.004069 | -0.003111 | 1.000 |
| `entry-786d0a3628f6f791399f` | 0.019248 | 0.007403 | -0.011845 | 1.000 |
| `entry-7a22f51938059541de98` | 0.005888 | 0.002730 | -0.003158 | 0.977 |
| `entry-7c1a3094f8acf9cf4bb0` | 0.006578 | 0.010516 | 0.003939 | 1.000 |
| `entry-8fa6576fc7128f93a228` | 0.004839 | 0.006424 | 0.001584 | 1.000 |
| `entry-abc8f864a49ee0e056f4` | 0.011828 | 0.006136 | -0.005692 | 1.000 |
| `entry-d5b80ca9b9cd18fa343f` | 0.007125 | 0.008150 | 0.001024 | 1.000 |
| `entry-d8b1bf9cd00ddf314725` | 0.017189 | 0.001608 | -0.015581 | 0.856 |
| `entry-f34e7e61444c90976b36` | 0.006178 | 0.006354 | 0.000177 | 1.000 |

## Phase profile

| Phase | Median LOO | Median null excess | Median accuracy | Median fingerprint RMS |
|---|---:|---:|---:|---:|
| Pre-formation | 0.905185 | 0.007180 | 0.811321 | 11.839314 |
| Formed | 0.907752 | 0.006424 | 0.957547 | 11.498403 |
| Decline | 0.530310 | -0.009791 | 0.193396 | 6.186024 |
| Recovered | 0.900281 | 0.005500 | 1.000000 | 16.090186 |

The decline phase has a much weaker pattern, but the frozen primary historical
contrast was formed minus pre-formation. That contrast is absent. The phase
profile is therefore descriptive and does not rescue the primary claim.

## Pair and null controls at the formed phase

| Class/control | Median cosine |
|---|---:|
| Main different-output unmatched | 0.824253 |
| Same-output different-instance | 0.854449 |
| Commuted | 0.888009 |
| Shared-left | 0.865004 |
| Shared-right | 0.831710 |
| Fully unmatched, matched to shared-left frequencies | 0.744881 |
| Fully unmatched, matched to shared-right frequencies | 0.743330 |

The main class met the frozen 0.10 practical-equivalence rule in all 13 runs.
This alone is insufficient because:

1. the median LOO advantage over output-label permutation was only 0.000142;
2. shared-left exceeded its input-identity null by 0.033109 in the median and
   did so above the null 95th percentile in 13/13 runs;
3. shared-left exceeded its frequency-matched fully unmatched control by
   0.109505 in the median (13/13 positive runs);
4. commuted pairs exceeded their input-identity null by 0.054918 in the median.

Thus, much of the large raw cosine is compatible with stable architectural and
input-identity structure rather than a learned output-independent relation.

## Component decomposition

All 13 formed runs were classified as `distributed`. Median formed-phase
quantities were:

- single-coordinate squared-magnitude share: 0.297640;
- single-effect RMS: 9.835376;
- pair-effect RMS: 12.465477;
- non-additive interaction RMS: 6.745955.

The evidence is therefore not reducible to one dominant component, but a
distributed intervention response is not by itself proof of abstraction.

## Interpretation and claim boundary

The strict result is:

> Existing frozen nanoGPT evidence is insufficient to identify a learned,
> single-relation functional invariance across concrete inputs and outputs.

What is supported is narrower:

- component-gating responses have a stable, distributed multivariate shape;
- that shape is detectable across outputs and weakens sharply in decline;
- most of the cross-output agreement is already present before formation and
  survives output-label permutation;
- concrete input overlap contributes additional detectable organization.

No claim is made about full functional abstraction, human-like abstraction or
general relation learning. A decisive experiment would require at least two
independently registered relations and a frozen test of within-relation
convergence together with across-relation selectivity.

## Reproduction

Primary analysis:

```powershell
python -m experiments.gfg_functional_abstraction_v1.FA_E01_ANALYSIS
```

Independent recomputation from source GFG objects and native gate tensors:

```powershell
python -m experiments.gfg_functional_abstraction_v1.FA_E01_INDEPENDENT_RECOMPUTE
```

Public evidence check:

```powershell
python -m experiments.gfg_functional_abstraction_v1.FA_E01_PUBLIC_EVIDENCE_CHECKER
python -m experiments.gfg_functional_abstraction_v1.FA_E01_PUBLIC_EVIDENCE_CHECKER --deep
```

The full run-level metrics and frozen null distributions are retained in
`FA_E01_RESULTS.json`; raw, scaled and unit-normalized fingerprints are retained
in `FA_E01_FINGERPRINTS.npz`.
