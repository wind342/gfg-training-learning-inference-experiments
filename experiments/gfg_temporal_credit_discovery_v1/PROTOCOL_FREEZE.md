# GFG-guided long-delay temporal-credit discovery protocol v1

## Scientific question

The experiment asks whether an identity-preserving Generation-Fact Graph can
do more than archive a delayed trajectory.  Specifically, can it retrieve a
small, high-recall set of historical action occurrences from the actual
formation path of a terminal consequence, after which matched causal replay
adjudicates rather than assumes their credit, and can those validated credit
relations form a better subsequent policy than history-only alternatives?

The tested transformation is:

\[
\text{terminal consequence}
\rightarrow\text{formation-path candidates}
\rightarrow\text{matched action forks}
\rightarrow\text{validated credit}
\rightarrow\text{policy formation}.
\]

Formation ancestry is not accepted as causal credit.  A base GFG is forbidden
from containing `credited_to_action`, `causes_consequence`, `necessary`,
`backup`, `substitution` or `synergy` labels.

## Long-delay environment

Each episode contains 64 binary decisions.  Nine opaque event occurrences
form state slots later consumed by the terminal report.  Six of these slots
enter the terminal scalar consequence; three are genuine formation ancestors
of the report but are causal passengers for that consequence.  The six
functional decisions contain a necessary route, alternative source paths,
two jointly required catalyst decisions and a final decision.  Their positions
vary deterministically by episode.  The environment emits no intermediate
reward or component reward.

Only the hidden evaluator knows the target action functions and the four
terminal criteria.  Neither the base GFG nor the learner receives those
labels.  All methods receive the same cue, chronological action ledger,
opaque event codes and scalar terminal consequence.

## Base GFG and candidate retrieval

For every real episode, synchronous capture establishes action sources,
concrete action occurrences, formed slot facts, a terminal occurrence and the
terminal consequence fact.  `program_order` is retained but is explicitly
excluded from candidate retrieval.  Retrieval follows only native formation
edges backwards from the terminal consequence to historical action sources.

The participant-visible event codes are opaque identities.  They do not name
route, source, catalyst, passenger or final roles.  The graph therefore gives
actual identity-preserving formation relations, not the hidden answer.

## Causal adjudication

For each candidate action occurrence, replay restores the exact pre-episode
cue and schedule, preserves every other action, changes only the selected
binary action and re-executes the native environment to the terminal
consequence.  Exact Shapley effects over the candidate set allocate effects in
the presence of backup and synergy.  Pair forks separately record non-additive
interaction:

\[
I_{ij}=\Delta_{ij}-\Delta_i-\Delta_j.
\]

The replay ledger and its complete cost are retained.  Credit is a derived
result of these new executions, never an edge copied from base ancestry.

## Comparators

All candidate methods receive the same replay budget:

- `gfg_forks`: base-GFG formation candidates followed by matched forks;
- `trace_decomposition_forks`: a ridge return-decomposition model fitted only
  to chronological payloads, followed by the same forks;
- `temporal_recency_forks`: the most recent actions followed by the same forks;
- `rewired_gfg_forks`: relation-rewired graph candidates followed by forks;
- `oracle_forks`: hidden functional action identities followed by forks, used
  only as an upper bound.

Two additional controls separate retrieval from adjudication:

- `gfg_ancestry_only`: formation ancestors are used without causal forks;
- `terminal_all_actions`: the same terminal consequence is assigned across
  the whole chronological action ledger.

## Learning test

Validated signed action effects are converted into action targets.  Positive
credit reinforces the executed action; negative credit reinforces its matched
alternative.  Causal magnitude remains in the evidence ledger but is not
silently equated with learning priority: every validated nonzero relation
contributes one learning unit.  Every method trains the same policy architecture
from the same number of episodes and starts from the same seeded parameter
distribution.  Held-out cues, schedules and episodes are used for evaluation.

The main outcomes are terminal success, mean terminal consequence and
functional-action accuracy.  Candidate recall, precision, history reduction,
credit sign, interaction recovery, counterfactual transition count and
wall-clock time are reported separately.

## Falsification

The hypothesis fails in this executed system if any of the following occurs:

1. GFG retrieval misses a material fraction of oracle-relevant actions;
2. it retains almost the entire history;
3. a relation-rewired graph performs equivalently;
4. the chronological trace comparator matches it at the same fork budget;
5. ancestry without adjudication matches adjudicated credit despite passenger,
   backup and synergy cases;
6. any apparent learning advantage disappears after replay cost is counted;
7. retrieved and causally validated credit does not improve held-out policy
   formation.

Success is limited to the executed, observable and exactly replayable system.
It would not establish that arbitrary real environments admit matched forks.
