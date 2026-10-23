# Formal results

## Outcome

**Frozen joint verdict: `MOE_UNIFIED_DYNAMICS_SUPPORTED`.**

The experiment changed the tested organization to a deterministic top-2 routed
Vision Mixture-of-Experts classifier trained from random initialization on
CIFAR-100 with AdamW. Three independently seeded 30-epoch runs produced final
test accuracies of 46.69%, 46.73% and 46.27%. All seven frozen TL-G03 tests and
all frozen INF-G02 tests passed.

## Integrity

| Check | Formal result |
|---|---:|
| validated compact GFGs | 3/3 |
| maximum reconstructed AdamW-update error | 1.192e-07 |
| maximum alpha=0 endpoint error | 0 |
| maximum alpha=1 native-endpoint error | 0 |
| maximum repeated-support error | 0 |
| persistent model and optimizer state frozen during inference | 3/3 |
| independent checker | PASS |

Each run established 49 complete atomic generation facts across 34 concrete
occurrences. The formal runtime contained 35 files occupying approximately
13.56 MiB.

## Receiving state and nonlinear response

All 36 parameter receiving-state exchanges exceeded the frozen response-NRMSE
threshold of 0.05. All 18 Adam-memory exchanges exceeded the separate frozen
threshold of 0.02. Thus the realized update alone did not determine its target
response in the executed MoE system: both the receiving parameter state and the
Adam memory changed the response.

Across 1,344 identity-aligned target-update records, 480 responses (35.71%)
met the frozen finite-amplitude nonlinearity criteria. The morphology ledger
contained 989 near-linear, 86 saturating, 140 accelerating, 53 turnback and 76
sign-reversal responses. The result establishes that nonlinear paths occur; it
does not claim that every response is strongly nonlinear.

## Routed distributed functional support

Expert-coalition gating identified distributed support in 94.72% of evaluated
target-update records. Support reallocation exceeded 0.01 in 80.43%, and the
primary supporting expert changed in 9.97%.

Routing retained exactly two non-zero experts per token. The maximum routing-
weight sum error was 1.192e-07, the largest query-profile L1 difference was
1.9516, the mean update-induced routing-profile reallocation was 0.0317 and the
mean top-2 identity-change fraction was 3.42%.

## Target-boundary prediction

| Predictor | Accuracy | Four-way macro recall |
|---|---:|---:|
| unchanged boundary | 98.66% | 50.00% |
| first directional term | 99.55% | 96.93% |
| first + second directional terms | 99.63% | 96.95% |

The quadratic predictor's transition recalls were 99.01% for remaining
correct, 88.89% for correct-to-wrong, 99.90% for remaining wrong and 100% for
wrong-to-correct. Only nine records occurred in each of the two boundary-
changing classes, so the rare-transition recalls must be interpreted with that
sample-size limitation.

## Run-isolated coordinate transport

Every held-out seed was predicted only from the other two complete runs.

| Coordinate family | Accuracy | Four-way macro recall |
|---|---:|---:|
| F1 | 98.14% | 49.65% |
| F1 + F3 | 99.18% | 77.70% |
| F1 + F5 | 98.29% | 49.76% |
| F1 + F3 + F5 | 99.03% | 74.79% |

F3, the target-specific geometry of the actual update, supplied the dominant
predictive gain. The complete three-coordinate representation exceeded F1 by
25.13 percentage points of macro recall and passed the frozen prospective
criterion. The present low-dimensional F5 summary did not add stable gain
beyond F1+F3. This does not negate the 18/18 native Adam-memory exchanges: it
separates causal optimizer-memory relevance from the sufficiency of this
particular practical F5 representation.

The target-outcome, F3-geometry and F5-Adam permutation controls reduced
four-way macro recall to 49.84%, 55.48% and 66.45%, respectively, compared with
74.79% for correctly paired F1+F3+F5 coordinates.

## Frozen inference projection

All three formal seeds passed exact repeat inference, exact repeat routing,
persistent-state preservation, deterministic top-2 routing, recruitment of all
four experts, query-conditioned routing, causal single-expert gating,
non-additive expert combination, learned expert/router version dependence and
exact restoration.

Repeated outputs and routes were bit-exact, and every rollback restoration
returned to zero output error. Across all seeds, the smallest trained-to-initial
expert/router rollback effect was 5.9756 logits, the smallest complete
single-expert gating effect was 5.8812 logits and the smallest paired-expert
interaction was 1.3360 logits. Each expert's minimum utilization across the
formal seeds was at least 33.86%.

## Claim boundary

The result transports the tested training--learning relations and frozen-
inference projection relation to the executed deterministic top-2 Vision-MoE/
CIFAR-100/AdamW system. All expert branches were evaluated and non-selected
branches were assigned exactly zero routing weight, which is mathematically
equivalent to the tested deterministic top-2 routed function but is not a claim
about sparse-kernel performance. The experiment does not prove that every MoE
implementation or every neural system must satisfy the complete theory.
