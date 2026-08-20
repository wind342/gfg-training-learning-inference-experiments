# Formal results

## Outcome

**Frozen verdict: `CROSS_SYSTEM_GENERALIZATION_SUPPORTED`.**

This experiment moves the studied formation relations to a generative system:
a time-conditioned U-Net trained by the diffusion epsilon-prediction objective
on CIFAR-10 with AdamW. The evaluated target is an identified image--timestep--
noise occurrence. Correctness is a residual-error boundary against seven frozen
competitor residuals, rather than a class-label boundary. All seven frozen
tests passed across three complete runs.

## Integrity

| Check | Formal result |
|---|---:|
| validated compact GFGs | 3/3 |
| maximum reconstructed AdamW-update error | 1.490e-08 |
| maximum alpha=0 endpoint error | 0 |
| maximum alpha=1 native-endpoint error | 0 |
| maximum repeated-inference error | 0 |
| independent checker | PASS |
| formal runtime footprint | 36,789,331 bytes |

The runtime footprint was approximately 35.1 MiB, far below the frozen 25 GiB
cap. The GFG contains registered scientific occurrences rather than a dense
trace of every tensor operation.

## Receiving state and nonlinear response

All 18 parameter receiving-state exchanges and all 18 AdamW-memory exchanges
exceeded their frozen response-difference thresholds. The same update geometry
therefore did not determine a unique target response independently of the
receiving parameter and optimizer-memory state.

Across 504 identity-aligned target-update records, 97 responses (19.25%) were
classified as finite-amplitude nonlinear. The frozen morphology ledger
contained 407 near-linear, 38 saturating, 34 accelerating, 9 turnback and 16
sign-reversal responses. This establishes finite-amplitude nonlinearity in the
executed diffusion system without claiming that every response is strongly
nonlinear.

## Distributed functional support

Coalition gating across the high-resolution skip, low-resolution skip,
bottleneck and decoder-refinement components identified distributed support in
76.59% of evaluated target-update records. Support reallocation exceeded 0.01
in 97.02%, and the primary supporting component changed in 8.13%. The support
relation therefore transports to a generative U-Net whose internal organization
and output target differ from both nanoGPT and a discriminative ResNet.

## Target-boundary prediction

| Predictor | Accuracy | Four-way macro recall |
|---|---:|---:|
| unchanged boundary | 89.88% | 50.00% |
| first directional term | 99.40% | 98.96% |
| first + second directional terms | 99.60% | 97.98% |

The quadratic predictor's transition recalls were 100.00% for remaining
correct, 95.24% for correct-to-wrong, 100.00% for remaining wrong and 96.67%
for wrong-to-correct. The first-order predictor had slightly higher four-way
macro recall, while the second-order predictor had slightly higher overall
accuracy. These are endpoint-boundary results, not claims that either local
formula reconstructs every complete nonlinear response curve.

## Run-isolated coordinate transport

Each complete seed was predicted only from the other two runs.

| Coordinate family | Accuracy | Four-way macro recall |
|---|---:|---:|
| F1 | 89.68% | 52.81% |
| F1 + F3 | 94.84% | 75.48% |
| F1 + F5 | 88.89% | 50.92% |
| F1 + F3 + F5 | 92.26% | 63.59% |

F3, the target-specific geometry of the actual update, supplied the dominant
predictive gain. The complete F1+F3+F5 representation exceeded F1 on every
held-out run and by 10.78 percentage points in pooled four-way macro recall,
satisfying the frozen prediction criterion. However, it did not improve on
F1+F3. The current low-dimensional F5 summary is therefore not an optimally
integrated predictive representation, even though native AdamW-memory exchange
establishes that optimizer memory causally changes the response.

The target-outcome, F3-geometry and F5-memory permutation controls reduced
four-way macro recall to 56.84%, 51.02% and 60.84%, respectively, compared with
63.59% for correctly paired F1+F3+F5 coordinates. The smallest control gap is
for F5, consistent with the limited predictive sufficiency of its present
summary rather than with absence of a native memory effect.

## Claim boundary

This result transports the tested training--learning formation relations to
the executed DDPM-style U-Net/CIFAR-10/AdamW system. Together with the separate
ResNet/CIFAR-100/SGD-momentum experiment, it shows that the evidence is not tied
to language, next-token prediction, Transformers, classification, a single
optimizer-memory form or a single kind of target boundary. It does not by
itself prove an unconditional law for every neural architecture, optimizer or
data-generating process.
