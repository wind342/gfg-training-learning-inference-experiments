# Per-history first 99% and cost accounting

Each cell is first-hit updates / median synchronized compute seconds. A dash means not reached within 200 updates; that full cost remains in aggregate results. Histories initially >=99% are omitted here.

| History | AdamW | Sophia-G | Shampoo | SLS | Controller |
|---|---:|---:|---:|---:|---:|
| entry-481b86f81d58d496a687 | 36 / 0.3735s | 181 / 2.1842s | 110 / 7.8045s | not reached / 8.7167s | 50 / 1.1753s |
| entry-786d0a3628f6f791399f | 125 / 1.1612s | 158 / 1.9289s | not reached / 14.5873s | not reached / 9.1197s | 55 / 1.2885s |
| entry-7a22f51938059541de98 | 35 / 0.3517s | 87 / 0.9840s | not reached / 14.5399s | not reached / 9.3790s | 41 / 1.0134s |
| entry-8fa6576fc7128f93a228 | 24 / 0.2440s | 122 / 1.4391s | 78 / 5.4674s | not reached / 8.5863s | 22 / 0.5389s |
| entry-d5b80ca9b9cd18fa343f | not reached / 1.7171s | not reached / 2.4148s | not reached / 14.6065s | not reached / 8.6161s | not reached / 4.4856s |
| entry-d8b1bf9cd00ddf314725 | 36 / 0.3346s | not reached / 2.3565s | not reached / 14.6708s | 2 / 0.1280s | 26 / 0.5908s |
| entry-f34e7e61444c90976b36 | 59 / 0.5726s | 148 / 1.7847s | not reached / 14.4998s | not reached / 8.7056s | 77 / 1.7899s |

## Selected initial learning rates

| History | AdamW | Sophia-G | Shampoo | SLS | Controller |
|---|---:|---:|---:|---:|---:|
| entry-481b86f81d58d496a687 | 0.001 | 0.0001 | 0.001 | 1.0 | 0.003 |
| entry-786d0a3628f6f791399f | 0.001 | 0.0003 | 0.001 | 1.0 | 0.003 |
| entry-7a22f51938059541de98 | 0.001 | 0.0001 | 0.003 | 1.0 | 0.001 |
| entry-8fa6576fc7128f93a228 | 0.001 | 0.0001 | 0.001 | 1.0 | 0.003 |
| entry-d5b80ca9b9cd18fa343f | 0.001 | 0.0003 | 0.001 | 1.0 | 0.003 |
| entry-d8b1bf9cd00ddf314725 | 0.001 | 0.0003 | 0.003 | 1.0 | 0.003 |
| entry-f34e7e61444c90976b36 | 0.001 | 0.0003 | 0.001 | 1.0 | 0.003 |

## All four-configuration search costs

These are additional experimental/tuning costs, not the cost of executing one selected controller.

| Method | Updates | Compute seconds | Evidence wall seconds |
|---|---:|---:|---:|
| adamw | 4141 | 35.478 | 65.300 |
| sophia | 4258 | 46.794 | 76.698 |
| shampoo | 4603 | 332.100 | 370.738 |
| sls | 4811 | 200.706 | 237.221 |
| controller | 2731 | 63.980 | 91.506 |

## End-to-end wall to first 99% or cap

Includes auditing and disk I/O, so it is not pure optimizer speed. Excludes source loading and synthetic warmup.

| Method | Mean restricted wall seconds |
|---|---:|
| adamw | 1.3314 |
| sophia | 3.2444 |
| shampoo | 13.8370 |
| sls | 9.1667 |
| controller | 2.1975 |

Electricity consumption was not measured; do not infer energy savings from these timings.
