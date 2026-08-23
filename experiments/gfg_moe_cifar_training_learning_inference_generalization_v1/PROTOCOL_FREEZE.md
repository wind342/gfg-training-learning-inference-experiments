# Frozen protocol: MoE cross-system validation (TL-G03 and INF-G02)

## Scientific question

Do the training--learning relations and frozen-inference projection relation previously
identified in nanoGPT, ResNet/CIFAR and diffusion/CIFAR survive a move to a routed
Mixture-of-Experts system, or does expert routing expose a counterexample?

This is a falsification-first cross-system test. A positive result is scoped to the
executed top-2 Vision-MoE/CIFAR-100/AdamW system and is not an unconditional universality
claim.

## Frozen system

- Randomly initialized image classifier on CIFAR-100.
- A deterministic top-2 router assigns every spatial token to two of four identified
  MLP experts. There is no router noise, capacity dropping or stochastic token loss.
- Expert identities, routing weights and concrete routing occurrences are preserved.
- AdamW supplies parameter--optimizer state and memory.
- Formal seeds are `20260910`, `20260911` and `20260912`.
- Formal hyperparameters and registered epochs are fixed in `MODEL_CONTRACT.json`.

All four expert functions are evaluated in the instrumented reference implementation,
but only the two selected by the router receive non-zero mixture weight. This preserves
the exact top-2 routed function while making coalition interventions deterministic and
free of kernel-dependent dispatch effects.

## TL-G03: training--learning tests

At the first native training batch of every registered epoch, the experiment seals the
pre-update parameter state, pre-update Adam memory, identified training batch, gradients
and the actual native update before any target response is inspected.

The following tests are fixed before execution:

1. **Actual-update identity and formula audit.** The recorded update must equal the
   native AdamW update within the numeric tolerance, and alpha 0/1 replay must reproduce
   the sealed pre-state and native post-state.
2. **Receiving-parameter-state exchange.** Corresponding realized updates are applied
   to different historical receiving states. At least the frozen fraction of comparisons
   must exceed the frozen response-NRMSE threshold.
3. **Receiving-optimizer-memory exchange.** Current gradients are recomputed under a
   different historical Adam memory while parameter state is fixed. The response must
   differ at the frozen rate and threshold.
4. **Finite-amplitude response.** The same actual update is replayed at
   `alpha = {0, .125, .25, .5, .75, 1}`. Nonlinearity is judged by the frozen chord
   criteria, not by visual inspection.
5. **Three-coordinate held-out prediction.** Run-isolated nearest-neighbour prediction
   compares target-boundary state alone with target boundary plus target-specific update
   geometry and parameter--Adam receiving state. Target-outcome, update-geometry and
   Adam-memory permutations are negative controls. No post-response quantity is admitted.
6. **Distributed expert support.** All 16 coalitions of the four identified experts are
   evaluated before and after the actual update. Exact four-player Shapley allocations,
   pair interactions, support multiplicity, reallocation and primary-support changes are
   computed.
7. **Routing audit.** Every token must have exactly two non-zero normalized routing
   weights. Query-conditioned routing profiles and their changes across training are
   recorded as observations, not supplied as outcome labels.

Failure of any relation remains a recorded counterexample. Integrity failure stops the
confirmation seeds; mechanism failure does not.

## INF-G02: frozen-inference tests

The final exact trained checkpoint and exact pre-learning checkpoint of every formal seed
are used. Inference is run without optimizer steps or persistent buffer updates.

Each seed must pass:

1. exact trained-version identity and exact repeat inference;
2. byte-identical model and optimizer state before and after inference;
3. deterministic top-2 routing and non-trivial use of all identified experts across the
   frozen query set;
4. query-conditioned routing/support profiles;
5. output changes under expert coalition gates;
6. non-additive expert pair interactions;
7. output changes when each expert or the router is rolled back to its exact pre-learning
   version; and
8. exact output restoration after restoring the trained versions.

The inference claim fails if the learned state changes persistently, if queries do not
condition routed support, if trained component versions have no causal effect, or if
restoration is not exact within the frozen tolerance.

## Generation-fact and identity requirements

The compact GFG records identified training sources, receiving parameter and Adam states,
actual updates, finite-amplitude replays, expert coalitions, routing audits, frozen
inferences and version rollbacks. Every registered occurrence binds complete
`(u, tau, omega, z, rho)` facts. Independent checkers recompute the frozen verdicts from
machine results and verify manifests and graph incidence.

## Verdicts

`TL-G03` and `INF-G02` receive separate verdicts. The joint MoE result is supported only
when integrity passes and both frozen verdicts are supported in all three formal seeds.
Partial and negative outcomes are retained without reinterpretation.
