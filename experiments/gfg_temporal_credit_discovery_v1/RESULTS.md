# Formal result

## Outcome

All nine preregistered scientific gates passed on twelve formal seeds, and the
independent checker passed every reconstruction, isolation, hash, replay and
metric check.

The executed result supports the following bounded statement:

> In this observable, identity-preserving and exactly replayable 64-step
> terminal-consequence system, GFG formation-path retrieval converted a
> long-delay temporal-credit search from a 64-action chronology into nine
> execution-specific candidates.  Matched causal replay then separated causal
> credit from mere formation ancestry, recovered signed and non-additive credit
> relations exactly, and those relations formed a held-out policy equivalent
> to the hidden credit oracle and better than relation-free history controls.

## Candidate retrieval and causal adjudication

Each episode contained six hidden functional actions and three actions that
were real ancestors of the terminal report but had no causal effect on its
scalar consequence.  GFG retrieval did not label or remove these passengers:

| result | value |
|---|---:|
| chronological actions | 64 |
| GFG formation candidates | 9 |
| retained history | 14.0625% |
| functional-action recall | 100% |
| candidate precision before forks | 66.67% |
| candidate F1 before forks | 0.80 |

Matched forks and exact candidate-set Shapley adjudication then produced, on
every formal seed:

- causal-credit F1: **1.00**;
- signed-credit accuracy on oracle-nonzero actions: **1.00**;
- pair-interaction F1: **1.00**;
- passenger credit: **zero**.

This distinction is essential.  GFG ancestry alone had complete candidate
recall but did not establish causal credit.

## Held-out policy formation

All methods used the same policy architecture, the same number of training
episodes and seeded initial parameters.  All fork-based non-oracle methods
received the same nine-candidate replay budget.

| method | candidate recall | terminal success | functional-action accuracy | mean terminal consequence |
|---|---:|---:|---:|---:|
| GFG + matched forks | **1.000** | **0.9648** | **0.9908** | **1.9534** |
| hidden oracle + matched forks | 1.000 | 0.9648 | 0.9908 | 1.9534 |
| relation-free trace decomposition + forks | 0.9142 | 0.8175 | 0.9568 | 1.7690 |
| GFG ancestry without forks | 1.000 | 0.2741 | 0.7005 | 0.9995 |
| temporal recency + forks | 0.1340 | 0.0701 | 0.5832 | 0.6966 |
| terminal consequence across all actions | 1.000 | 0.0547 | 0.5273 | 0.5824 |
| rewired GFG + forks | 0.000 | 0.0310 | 0.5014 | 0.5338 |

GFG terminal success was 0.9648 (95% CI 0.9408--0.9889).  The relation-free
trace decomposition achieved 0.8175 (95% CI 0.7877--0.8473).  The paired GFG
advantage was 0.1474 (95% CI 0.1160--0.1787).  GFG and the hidden oracle
produced identical held-out results.

The rewiring result rules out node payloads, action counts and replay alone as
an explanation.  The ancestry-only result rules out treating every formation
ancestor as credit.  The trace decomposition was a nontrivial comparator: it
recovered 91.42% of hidden functional actions and reached 81.75% success, but
still did not match identity-preserving GFG retrieval.

## Cost

The result does not make causal adjudication free.  With nine formation
candidates, exact Shapley replay used 8,388,608 counterfactual environment
transitions per formal seed.  The hidden six-action oracle required 1,048,576.
Thus GFG eliminated most of the 64-step search space, but its three causal
passengers still imposed an eight-fold exact-Shapley cost relative to the
unavailable oracle.  This cost is retained as part of the result, not removed
from the comparison.

The implementation used a semantics-preserving terminal projection to execute
the deterministic fork volume efficiently.  The independent checker compared
that projection with full native episode execution and obtained exact equality.

## Scientific interpretation

RL-E01 established that predeclared consequence binding and temporal credit
change actual gradients, parameter updates and future policy.  This experiment
adds the missing discovery step.  It shows, in the executed system, that:

1. a terminal result can retrieve candidate historical decisions through its
   actual generation relations rather than temporal distance;
2. formation ancestry and causal credit are empirically separable;
3. matched forks can establish signed credit and backup/synergy interactions;
4. causal credit and learning credit are distinct--effect magnitude need not
   equal learning priority;
5. validated credit relations can form a successful subsequent policy.

The experiment therefore supports temporal credit as a two-stage formation
problem:

\[
\text{generation-relation identification}
+
\text{causal credit adjudication},
\]

followed by receiving-state-dependent formation of an actual learning update.

## Limits

This is a controlled long-horizon environment, not evidence that arbitrary
real environments are fully observable, restorable or safely intervenable.
It does not establish a universal replacement for policy gradients, RUDDER,
Temporal Value Transport, HCA or COCOA.  It also does not yet test an established
large environment such as Craftax.  Exact Shapley replay scales exponentially
with the candidate count, so approximate adjudication will be necessary at
larger scale.

The next scientifically necessary test is therefore external scaling, not a
stronger claim from this same environment: preserve the same retrieval and
fork logic in an established stochastic long-horizon task, compare against
implemented contemporary credit-assignment baselines and count every replayed
transition.

## Verification

- formal aggregate SHA-256:
  `c1bf98a259e842594585bf373c50dd45a187607d5f33f01e2fcc026d1f168a3b`;
- independent check SHA-256:
  `4fec707d93f009daaf5bbdba09fb12df1438295b5e90e46366abefcaf7f59043`;
- independent status: **PASS**;
- runtime tests: **4 passed**.
