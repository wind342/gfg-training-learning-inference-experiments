# RL-E05 development calibration

Development calibration was restricted to seeds `20260851` and `20260852` and
was completed before the formal seed list was executed.  It was used to choose
a numerically stable shared bottleneck and to ensure that all four capabilities
could first be formed with positive margin.  It was not admitted as formal
evidence.

The calibration crossed hidden sizes 8, 12 and 16 with learning rates 0.001 and
0.003.  Every configuration reached exact four-capability mastery before the
feedback branches were created.  The frozen formal configuration uses the
middle hidden size (12) and lower learning rate (0.001), rather than the
configuration with the largest development effect.  Eight hundred feedback
updates are retained because they expose both early internal transitions and
later boundary outcomes without numerical divergence.

Development showed that measurements at updates 10, 25, 50, 75, 100, 200, 400
and 800 resolve the proposed temporal ordering.  These checkpoints, all
formal seeds, the exact support functional and all decision gates were then
frozen in `MODEL_CONTRACT.json` before formal execution.

The raw development output is retained outside Git and is explicitly excluded
from formal aggregation.
