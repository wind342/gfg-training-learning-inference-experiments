# Optimizer capability benchmark results

All counts below EXCLUDE histories already at the stated threshold. Unreached runs cost 200 updates; times are construction + synchronized update compute, median of three exact native replays.

## First 95%

| Method | Reached / eligible | Mean restricted updates | Mean restricted compute seconds |
|---|---:|---:|---:|
| adamw | 6/6 | 17.000 | 0.181127 |
| sophia | 5/6 | 86.000 | 1.051563 |
| shampoo | 6/6 | 44.000 | 2.908398 |
| sls | 5/6 | 108.833 | 4.858284 |
| controller | 6/6 | 20.000 | 0.514551 |

## First 99%

| Method | Reached / eligible | Mean restricted updates | Mean restricted compute seconds |
|---|---:|---:|---:|
| adamw | 6/7 | 73.571 | 0.679249 |
| sophia | 5/7 | 156.571 | 1.870323 |
| shampoo | 2/7 | 169.714 | 12.310897 |
| sls | 1/7 | 171.714 | 7.607362 |
| controller | 6/7 | 67.286 | 1.554637 |

## Scope

- Historical trained-weight continuation with fresh optimizers and zero decay, not original receiving state.
- Control and evaluation targets coincide; no held-out-target claim.
- Four LR trials, LOO history selection; not best possible tuning.
- First hit, not sustained capability. Timing on one small model and one GPU.

## Audit

420 trajectory ledgers, 33960 recorded updates; all selected trajectories repeated three times with exact native hashes, boundaries and choices.

Memory peaks (bytes): {'peak_allocated_bytes': 705261056, 'peak_reserved_bytes': 748683264, 'peak_process_rss_bytes': 2101858304}.
Raw artifacts: 6814007534 bytes.
