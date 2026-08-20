# RL-E06 development calibration

The two development-only seeds were executed before the formal contract was
frozen.  They were not reused as formal evidence.

Both development seeds showed a strictly positive Spearman association between
feedback concentration and capability-0 support share (`1.0`) and a strictly
negative association between feedback concentration and the mean margin of
capabilities 1--3 (`-1.0`).  The mean exclusive-versus-balanced unreinforced
accuracy deficit was `0.302083`; the mean capability-0 support-share excess was
`0.093998`.  Rebalancing feedback restored the complete unreinforced accuracy
deficit in both seeds while retaining capability-0 accuracy at `1.0`, and moved
the support state closer to the balanced authority in both seeds.

The formal thresholds were deliberately frozen below these observed development
effects: 9/12 directional dose associations, a mean unreinforced accuracy deficit
of `0.15`, a mean support-share excess of `0.04`, recovery of at least `0.10` in
8/12 seeds and `0.15` on average, capability-0 retention of `0.95`, and support
reversal in 8/12 seeds.  No formal seed was inspected before these thresholds and
the frozen-file hashes were sealed.

