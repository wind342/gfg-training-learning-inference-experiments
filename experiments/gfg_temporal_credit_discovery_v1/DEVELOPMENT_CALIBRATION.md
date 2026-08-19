# Development calibration

Development preceded source and threshold freezing and used only seeds 1729,
2718 and 31415.  No formal seed was read during calibration.

## First development execution

The first implementation used the absolute counterfactual effect as the
learning weight.  GFG candidate retrieval and causal adjudication were exact,
but the downstream policy result did not preserve that ordering: GFG and the
hidden oracle reached mean terminal success 0.6367, while the relation-free
trace decomposition reached 0.7083.  The result was retained rather than
discarded:

- aggregate SHA-256:
  `155ee77a0a6b210ba19fb2afaf2d7b86d3ea7cf94f3a02dc3e00d681d3e28406`;
- artifact:
  `external://gfg-temporal-credit-discovery-v1/development-v1/AGGREGATE_RESULT.json`.

Inspection showed a conceptual error rather than a failed credit relation.
The implementation had silently equated *causal effect magnitude* with
*learning priority*.  Terminal components with smaller numerical contribution
were therefore under-trained even when their causal relation was perfectly
identified.  This is precisely the distinction between causal credit and
learning credit posed by the experiment.

## Frozen correction

The corrected learning rule retains the signed causal magnitude in the
evidence ledger but assigns one learning unit to every validated nonzero
credit relation.  Positive credit reinforces the executed action and negative
credit reinforces its matched alternative.  The correction was made before
the formal contract, thresholds or source hashes were frozen.

Repeated development execution produced:

| method | candidate recall | terminal success | functional-action accuracy |
|---|---:|---:|---:|
| GFG + matched forks | 1.000 | 0.850 | 0.962 |
| hidden oracle + matched forks | 1.000 | 0.850 | 0.962 |
| relation-free trace decomposition + forks | 0.921 | 0.697 | 0.920 |
| GFG ancestry without forks | 1.000 | 0.165 | 0.624 |
| rewired GFG + forks | 0.000 | 0.035 | 0.501 |

The final development aggregate is stored at
`external://gfg-temporal-credit-discovery-v1/development-v3/AGGREGATE_RESULT.json`
with SHA-256
`aa13bfb9a65ddac2d18cb7fa260ce4b89ed36781eae312fce627b48586832fe1`.

Only after this execution were the formal gates and all executable source
hashes written to `CONTRACT_FREEZE.json`.
