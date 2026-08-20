# ResNet--CIFAR cross-system training--learning generalization protocol v1

## Scientific question

Does the training--learning mechanism established in nanoGPT remain observable
when architecture, modality, task and optimizer are all changed to a ResNet-18
image classifier trained on CIFAR-100 with SGD and momentum?

This is a falsification-first cross-system experiment. A positive result can
establish transport to this materially different executed system; it cannot by
itself establish an unconditional law for every neural network.

## Frozen system

- model: ResNet-18 adapted to 32 x 32 images, trained from random initialization;
- task: 100-class CIFAR-100 image classification;
- optimizer: SGD with momentum and weight decay;
- evaluation target: one identified test image and its correct class;
- target boundary: correct-class logit minus the largest competing-class logit;
- four observable transitions: remain correct, correct to incorrect, remain
  incorrect and incorrect to correct;
- support components: the four residual stages of ResNet-18;
- primary run seed: `20260820`;
- confirmation seeds: frozen in `MODEL_CONTRACT.json` and run only after the
  primary execution passes all integrity checks.

No pretrained parameters are permitted.

## Event boundary and admissible information

For one registered training occurrence, the pre-response record is sealed after
the optimizer has formed the actual parameter update but before post-update
evaluation targets are computed.

The following information is admissible before sealing:

- the complete pre-update parameter and buffer state;
- the complete pre-update SGD momentum state;
- the identified training batch and loss;
- the actual formed parameter update;
- identified evaluation inputs and correct classes;
- quantities computed on the pre-update model by automatic differentiation;
- hashes and occurrence identities required to bind these objects.

The following are forbidden before a prediction or prospective adjudication is
sealed:

- post-update logits, margins, predictions or transition labels;
- any response at update amplitude greater than zero;
- post-update support gates or support allocation;
- future optimizer state, future batches or future checkpoints;
- run identifiers, absolute training step or phase labels as predictive
  features;
- target selection based on a post-update result.

Evaluation targets are selected only from frozen identities and pre-update
margin strata. No target may be removed after its result is read.

## Frozen tests

### R1. Receiving-state conditioning

For registered states A and B, apply the same already formed update to both
receiving parameter states and compare target-level margin responses. Execute
the reciprocal comparison with the update formed at B. Skip branches preserve
the receiving states. A response difference must be measured relative to the
corresponding skip branch.

Separately, hold parameters and the training batch fixed while exchanging the
SGD momentum receiving state. This tests whether optimizer memory changes the
actual update that is formed; it is not conflated with the fixed-update
receiving-state test.

### R2. Finite-amplitude nonlinear response

Apply one exact realized parameter update to the identical pre-update state at

```text
alpha in {0, 0.125, 0.25, 0.5, 0.75, 1}.
```

Record each identified target's complete margin path. Compare the endpoint with
the first- and second-order directional expansions computed entirely at
`alpha=0`. Saturating, accelerating, turnback and sign-reversal morphologies are
derived from the sealed path by frozen numerical rules.

### R3. Primary response coordinates

Adapt the three coordinate families without importing nanoGPT-specific fields:

- F1: the target's current boundary state;
- F3: the geometry of the actual update with respect to that target, obtained
  from pre-update directional derivatives and blockwise update geometry;
- F5: the parameter--SGD-momentum receiving state and its interactions with the
  formed update.

Report the ablations F1, F1+F3, F1+F5 and F1+F3+F5 under complete run isolation.
Any learned preprocessing or hyperparameter choice must be fitted within the
training runs of a split. Post-update outcomes are used only as labels after the
event boundary has been sealed.

### R4. Distributed support and persistent reorganization

At the same identified pre- and post-update states, execute all sixteen masks of
the four registered residual stages. Derive exact four-player Shapley support
for the correct-class margin, single-stage necessity, pair interaction and
support concentration. Reversible stage gates alter only the registered
residual contribution and do not update model state.

Compare the complete support allocation before and after the actual update.
The observable target transition is adjudicated only from the ungated target
margin boundary, not from a separately trained classifier.

## Integrity and negative controls

- repeated ungated inference at one frozen state must be numerically identical;
- `alpha=0` must reconstruct the sealed pre-update state;
- `alpha=1` must reconstruct the native post-update parameter state;
- skip branches must preserve their receiving state;
- all target, state, batch and update identities must remain aligned;
- label permutation is a negative control for target identity;
- update permutation is a negative control for F3;
- momentum permutation is a negative control for F5;
- a time/step-only baseline is prohibited as a primary model and may be reported
  only as a leakage audit;
- exact output manifests and independent recomputation are required.

## Resource boundary

The complete experiment, including CIFAR-100, checkpoints, compact GFG evidence
and results, must remain below 25 GiB. Dense per-operation framework traces and
duplicate model snapshots are prohibited. The GFG records only registered
scientific occurrences, hashes, bindings and compact outcomes required by the
frozen tests.

## Verdicts

The experiment returns one of:

- `CROSS_SYSTEM_GENERALIZATION_SUPPORTED`;
- `CROSS_SYSTEM_GENERALIZATION_PARTIALLY_SUPPORTED`;
- `CROSS_SYSTEM_GENERALIZATION_NOT_SUPPORTED`;
- `INTEGRITY_FAILURE`.

The primary criterion is not a single accuracy number. A supported verdict
requires all integrity checks, receiving-state-conditioned response, measurable
finite-amplitude nonlinearity, distributed support reallocation connected to
target-boundary outcomes, and positive held-out predictive contribution from
the adapted coordinate families. Failures and counterexamples remain in the
evidence set.
