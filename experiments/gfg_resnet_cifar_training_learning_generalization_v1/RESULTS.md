# Formal results

## Outcome

**Frozen verdict: `CROSS_SYSTEM_GENERALIZATION_SUPPORTED`.**

The experiment changed architecture, modality, task and optimizer memory at
the same time: nanoGPT/text/next-token prediction/Adam was replaced by
ResNet-18/images/CIFAR-100 classification/SGD momentum. Three independently
seeded 60-epoch runs produced final CIFAR-100 test accuracies of 76.03%,
76.44% and 76.22%. All six frozen mechanism tests passed.

## Integrity

| Check | Formal result |
|---|---:|
| validated compact GFGs | 3/3 |
| maximum reconstructed SGD-update error | 5.958e-08 |
| maximum alpha=0 endpoint error | 0 |
| maximum alpha=1 native-endpoint error | 0 |
| maximum repeated-inference error | 0 |
| independent checker | PASS |
| formal runtime footprint | 274,059,252 bytes |

The runtime footprint was approximately 261 MiB, far below the frozen 25 GiB
cap. The GFG therefore remained a compact scientific record rather than a
dense trace of every framework operation.

## Receiving state and nonlinear response

All 42 parameter receiving-state exchanges exceeded the frozen response-
difference threshold. Eighteen of 21 momentum-state exchanges exceeded the
separate optimizer-memory threshold. Thus the realized update alone did not
determine its target response: both the receiving parameter state and, in most
registered exchanges, the SGD momentum memory changed the response.

Across 1,536 identity-aligned target-update records, 495 responses (32.23%)
were classified as finite-amplitude nonlinear. The frozen morphology ledger
contained 1,033 near-linear, 101 saturating, 179 accelerating, 80 turnback and
143 sign-reversal responses. This supports finite-amplitude nonlinearity
without claiming that every target response is strongly nonlinear.

## Distributed functional support

Residual-stage gating identified distributed support in 98.63% of evaluated
target-update records. Support reallocation exceeded 0.01 in 91.41%, and the
primary supporting stage changed in 8.85%. These results transport the
distributed-support and persistent-reorganization relation from Transformer
components to residual stages in an image classifier.

## Target-boundary prediction

| Predictor | Accuracy | Four-way macro recall |
|---|---:|---:|
| unchanged boundary | 91.34% | 50.00% |
| first directional term | 99.35% | 97.64% |
| first + second directional terms | 99.28% | 97.99% |

The quadratic predictor's four transition recalls were 99.36% for remaining
correct, 98.33% for correct-to-wrong, 99.74% for remaining wrong and 94.52%
for wrong-to-correct. These are target-boundary results, not curve-fit scores.
The coexistence of nonlinear full paths and high endpoint-boundary prediction
means that a coarse boundary sign can remain predictable even when the entire
finite-amplitude curve is not globally represented by one fixed local law.

## Run-isolated coordinate transport

Every held-out seed was predicted only from the other two complete runs.

| Coordinate family | Accuracy | Four-way macro recall |
|---|---:|---:|
| F1 | 91.67% | 58.11% |
| F1 + F3 | 96.48% | 80.72% |
| F1 + F5 | 91.54% | 59.67% |
| F1 + F3 + F5 | 96.16% | 80.06% |

F3, the target-specific geometry of the actual update, supplied the dominant
predictive gain. F5 supplied a small gain over F1 alone, but the implemented
F5 summary did not add stable predictive improvement beyond F1+F3. This does
not negate the native momentum-state exchange result: it separates causal
optimizer-memory relevance from the sufficiency of the present practical F5
representation.

The target-outcome, F3-geometry and F5-momentum permutation controls reduced
four-way macro recall to 57.95%, 59.49% and 74.99%, respectively, compared
with 80.06% for the correctly paired F1+F3+F5 coordinates. The held-out gain
therefore depended on correct target, update and receiving-memory identities,
not merely on adding feature dimensions.

## Claim boundary

This result establishes cross-system transport to the executed
ResNet-18/CIFAR-100/SGD-momentum system. It is strong evidence that the
training--learning relations are not peculiar to nanoGPT, text, Transformers
or Adam. It is not, by itself, a proof that every architecture, optimizer and
environment must obey the complete theory.
