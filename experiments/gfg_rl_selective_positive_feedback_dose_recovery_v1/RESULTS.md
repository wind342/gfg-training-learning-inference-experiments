# RL-E06 formal results

## Frozen verdict

The preregistered dose--duration--recovery hypothesis is **SUPPORTED** in the
executed shared finite GRU. Every frozen decision gate passed and all 12 formal
seeds were retained.

All branches began from byte-identical parameter--AdamW states after all four
skills reached exact mastery. Updated dose branches used equal budgets of 3,200
updates and 192,000 episodes per branch. The frozen branch saw the same
exclusive episodes and sampling uniforms without persistent state updates.
Every reward was the exact terminal result of the complete two-action chain;
there was no learned or proxy reward, negative reward or entropy bonus.

## Dose and endpoint results

| Quantity | Formal result |
| --- | ---: |
| Formal seeds retained | 12/12 |
| Seeds with positive support--dose association | 12/12 |
| Mean support--dose Spearman correlation | 1.000 |
| Seeds with negative other-skill margin--dose association | 12/12 |
| Mean other-skill margin--dose Spearman correlation | -1.000 |
| Exclusive reinforced-skill support-share excess over balanced | 9.67 percentage points |
| Exclusive other-skill accuracy deficit versus balanced | 39.06 percentage points |
| Reinforced-skill accuracy in every dose branch | 100.00% |
| Balanced final accuracy of every skill | at least 95%; observed 100.00% |

The relation is not confined to a discrete endpoint failure. At update 3,200,
the balanced, mild, high and exclusive branches had mean reinforced-skill
support shares of 25.54%, 26.47%, 27.50% and 35.20%, respectively. The
corresponding mean margins of the other three skills were 9.474, 9.130, 8.493
and 3.962. Mild and high feedback still retained 100% endpoint accuracy, so the
continuous margin already detected dose-ordered internal erosion before it
necessarily crossed a discrete correctness boundary.

Exclusive feedback also showed a duration relation. From updates 100, 400,
800, 1,600 and 3,200, reinforced-skill support share rose from 28.00% to 30.33%,
31.78%, 33.38% and 35.20%, while other-skill mean accuracy fell from 88.89% to
76.56%, 70.31%, 65.80% and 60.94%. The effect therefore continued well beyond
the 800-update endpoint used in RL-E05.

## Recovery results

Both recovery branches forked from the exact exclusive state at update 800,
where the other skills averaged 70.31% accuracy.

| Quantity | Rebalanced recovery | Other-skills-only repair |
| --- | ---: | ---: |
| Added updates | 2,400 | 2,400 |
| Final other-skill accuracy | 99.48% | 99.83% |
| Gain relative to exclusive fork | 38.54 percentage points | 38.89 percentage points |
| Final formerly reinforced-skill accuracy | 100.00% | 89.06% |
| Seeds with support movement back toward balanced | 12/12 | 12/12 |

Rebalanced recovery moved the support profile back toward the balanced branch
and restored nearly all other-skill performance while retaining the reinforced
skill at 100% in every seed. Training only the other three skills restored them
slightly more completely but reduced the formerly reinforced skill to 89.06%
on average. This symmetric trade-off is stronger evidence for feedback-directed
support reorganization than an irreversible-damage explanation.

## Evidence integrity

Each formal seed contains 17,600 real optimizer updates across the four dose
and two recovery branches; across 12 seeds this is 211,200 updates. Every seed's
validated experiment GFG contains 21,399 complete generation facts, or 256,788
facts in total. The independent checker re-executed all 12 seeds from scratch,
verified every stored receipt and boundary trajectory, recomputed all support
measurements, checked the GFG contracts and independently recalculated the
decision gates. Its status is **PASS**.

## Scientific conclusion

In the executed shared GRU, the concentration and duration of correct positive
feedback causally controlled both the allocation of distributed functional
support and the reliability of unreinforced capabilities. The same feedback
loop that strengthened the selected skill progressively weakened other skills,
while redistributing feedback reversed the support and behavioural trade-off.
This establishes a bounded double-edged feedback mechanism. It does not prove
that every reinforcement-learning system must exhibit the effect, nor that the
measured support shares constitute a conserved finite resource.
