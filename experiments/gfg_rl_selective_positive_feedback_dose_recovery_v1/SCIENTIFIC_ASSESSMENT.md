# Scientific assessment

## Did feedback concentration act as a causal coordinate?

Yes, in the executed shared finite GRU. All updated branches started from the
same mastered parameter--AdamW state, used equal episode and update budgets,
and differed only in how exact positive feedback was distributed across the
four skills. In all 12 formal seeds, increasing concentration strictly increased
the reinforced skill's functional-support share and strictly decreased the
other skills' mean margin. The frozen exposure control preserved its initial
state exactly.

## Was this merely a wrong-reward or proxy-objective failure?

No. Reward authority was the exact terminal two-action task rule. There was no
reward model, proxy reward, negative reward, entropy bonus or intentionally
misbound consequence. The experiment therefore isolates a different failure
mode: even correct feedback can be too narrowly distributed for a shared
learned system.

## Did the experiment separate internal change from visible capability loss?

Yes. At the final dose endpoint, mild and high branches still achieved exact
discrete accuracy for the three other skills, while their mean margins were
already lower in strict dose order. The continuous target boundary state thus
detected graded erosion before the external correctness result necessarily
changed. Exclusive feedback then crossed many of those boundaries as its
duration increased.

## Is the effect reversible?

Largely. Rebalancing feedback after the exact exclusive update-800 state
raised other-skill accuracy from 70.31% to 99.48%, an observed recovery of
29.17 percentage points, and retained the reinforced skill at 100% in every
seed. Its endpoint was 38.54 points above the matched branch that continued
exclusive feedback from update 800 to update 3,200 and ended at 60.94%. The
support profile moved toward the balanced reference in 12/12 seeds. This
contradicts an explanation based only on permanent damage or unlucky
initialization.

The other-skills-only repair is informative in the opposite direction: it
restored the other skills to 99.83% but reduced the formerly reinforced skill
to 89.06%. Feedback direction therefore reorganized the shared functional
system both ways.

## What does this add to the training--learning--inference theory?

Inference projects functional support already formed by learning. When one
class of projected success repeatedly supplies the next training action, the
resulting positive feedback can further reorganize shared support toward that
class. This explains both sides of the loop: focused feedback can efficiently
strengthen a selected capability, but the same mechanism can reduce the margin
and eventual boundary reliability of capabilities omitted from feedback.

## What is not established?

- that all reinforcement-learning algorithms, architectures or environments
  inevitably exhibit this trade-off;
- that support is literally conserved or that one capability physically
  consumes a fixed quantity from another;
- that every form of model collapse is produced by this mechanism;
- that modularization, rehearsal, replay or explicit anti-interference methods
  cannot prevent the effect;
- that 3,200 updates identify a universal timescale.

The strongest supported wording is: in the executed shared policy, correct but
narrow positive feedback created a dose- and duration-dependent functional
imbalance, and redirecting feedback causally reversed it.
