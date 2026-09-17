# Online functional amplitude controller

ONLINE_FUNCTIONAL_CONTROLLER_NOT_SUPPORTED

Frozen primary: run-mean normalized AUC improvement >= 1 pp and >= 9/12 winning runs against BOTH controls.

| Comparator | Mean AUC delta (pp) | AUC W/T/L | Mean final delta (pp) | Final W/T/L |
|---|---:|---|---:|---|
| native | 1.810142 | 9/0/3 | -0.078616 | 2/8/2 |
| fixed | 0.298742 | 10/1/1 | 1.808176 | 5/4/3 |

## Per-run comparisons

| Run | Source step | AUC vs native (pp) | AUC vs fixed (pp) | Final vs native (pp) | Final vs fixed (pp) |
|---|---:|---:|---:|---:|---:|
| entry-481b86f81d58d496a687 | 7869 | -0.259434 | 1.922170 | 0.000000 | 1.415094 |
| entry-4ed462761347d6b87e61 | 2986 | 2.287736 | 0.023585 | 0.000000 | 0.000000 |
| entry-54ba9566f731754a0e3f | 7793 | 3.537736 | 0.530660 | 0.000000 | 21.226415 |
| entry-5bb1186bc27eb82111fb | 4097 | 1.933962 | 0.047170 | 0.000000 | 0.000000 |
| entry-786d0a3628f6f791399f | 2261 | 0.625000 | 0.176887 | -0.471698 | 0.471698 |
| entry-7a22f51938059541de98 | 2667 | 0.094340 | 0.731132 | 0.000000 | 0.000000 |
| entry-7c1a3094f8acf9cf4bb0 | 8288 | 3.632075 | 0.000000 | 0.000000 | 0.000000 |
| entry-8fa6576fc7128f93a228 | 2177 | -0.377358 | 0.365566 | 0.000000 | -0.471698 |
| entry-abc8f864a49ee0e056f4 | 1478 | -0.625000 | 0.271226 | -1.415094 | -0.471698 |
| entry-d5b80ca9b9cd18fa343f | 7729 | 5.176887 | 0.518868 | 0.471698 | 0.943396 |
| entry-d8b1bf9cd00ddf314725 | 2075 | 5.436321 | -2.665094 | 0.471698 | -1.886792 |
| entry-f34e7e61444c90976b36 | 1804 | 0.259434 | 1.662736 | 0.000000 | 0.471698 |

Alpha counts: {'0.25': 1224, '0.5': 257, '0.75': 175, '1.0': 744}

Measured training+selection cost ratio vs native: 2.547x.
Measured wall-time ratio including evidence: 1.814x.

Transitions are descriptive counts at the 10-step evaluation grid, not unobserved exact crossing times.
No new capability-formation threshold was introduced. These selected checkpoints need not begin before formation.
Same control/evaluation targets; no held-out-generalization, generic optimizer, or wall-budget-efficiency claim.
