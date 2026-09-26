# Recovery-aware functional update governor v4

Base: `da806c1db4436d50a3ef40b06e6a3d85492aad7b`. New code only; v3 and frozen
research identities remain unchanged. This is a policy extension: v3 deliberately
used current capability constraints, whereas v4 additionally consumes history.

## Selection

The unchanged v3 scheduler supplies group states and excursion anchors. The
governor previews the current full measurement's scheduler refresh before deciding;
the runtime commits the same refresh and checks exact equality. This avoids using
stale history or accidentally dropping a newly observed/recovered group.

- Hard constraints: currently >=90% groups, excluding unresolved DEGRADED groups.
- Recovery debt: previously protected groups still in a DEGRADED excursion.
- Current hard constraints keep v3's violation-count/worst/total-deficit order.
- Within the best hard-constraint pool, maximize overall count and macro capability.
  Native wins count/macro ties; debt is a tie-break among remaining alternatives.
- Without hard constraints but with debt: maximize exact overall correct count,
  then degraded-group macro capability, fewer worsening groups, more recovered
  groups, and overall macro capability. Native wins ties on those quantities.
- Without either: reuse v3 formation selection, including its native tie preference.

No tolerance, lambda, TTL, new alpha, stale release, or target-level veto. Recovery
count is a predicted >=90% count, not a claim that actual anchor re-arming occurred.
Actual debt clears only under the original scheduler capability/q10 re-arm rule.
Remaining ties between nonnative formation actions use the existing formation/q10
progress and distance-to1. Candidate directions/derivatives are not changed.

## Fixed regression results

| Case | First >=99% | Final accuracy | Wall | Full calls |
|---|---:|---:|---:|---:|
| SAFE | 123 | 99.0536% | 9.38 s | 3 |
| NEAR | 182 | 99.6845% | 15.84 s | 12 |

Both alpha sequences and all recorded evaluation accuracies equal v3 exactly.
Wall differences are repeat timing variation, not evidence of a policy speedup.
Same seed201/rates/200-update limit, one attempt per case. No post-result retuning.
The old modular-addition task is retained only for regression checks.

At historical NEAR step76, debt groups4 and13 are now explicitly included in
RECOVERY_AWARE_FORMATION. All five actions predict zero accuracy on both groups;
alpha2 remains selected on the remaining criteria. Changing this decision was
not a required outcome. A synthetic equal-overall alternative that restores debt
does change the selection, and an alternative with uniquely higher overall count
is not vetoed merely for failing to repay debt.

42 tests pass: all34 v3 tests plus8 history/recovery/runtime checks. `report.py`
recomputes scheduler histories, candidate decisions, raw evaluation accuracies and
source hashes. Raw files remain outside Git under
`E:/gfg-evidence/functional-update-governor-v4/`; compact records are in `results/`.
The report requires these external raw files and is create-only.

```text
python -m experiments.gfg_functional_update_governor_v4.tests
python -m experiments.gfg_functional_update_governor_v4.smoke safe
python -m experiments.gfg_functional_update_governor_v4.smoke near
python -m experiments.gfg_functional_update_governor_v4.report
```

Real-task work is isolated in `../gfg_governor_fewrel_v1/`. It does not modify the
modular-addition runs or convert them into a new scientific benchmark.
