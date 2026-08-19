# External content-addressed base-GFG payloads

The following generated JSON payloads are not tracked in this companion
repository because they duplicate large base-GFG evidence already fixed by
source commit `0b03a0b65b24dfce00e6f70610efa6b566c6bd3b`.

| Relative path | Bytes | SHA-256 |
|---|---:|---|
| `experiments/nanogpt_training_generation_fact_graph_v1/artifacts/core_v3_snapshot.json` | 66,527,339 | `b0f64e31f888daaba22ecff91e577458c6933d9d9728f30309d71ee579ae93cb` |
| `experiments/nanogpt_training_generation_fact_graph_v1/artifacts/complete_generation_fact_graph.json` | 20,288,833 | `23b3f65e9a89c49c63afccc497ca0d276ad2447bd0b6e440f2e7d7fe5c000121` |
| `experiments/nanogpt_training_generation_fact_graph_v1/artifacts/runtime_receipts.json` | 8,595,465 | `4eae89dc3dac7d073bf959e5477396a7b88219bb7063e5394d3b714c74bc816a` |
| `experiments/nanogpt_training_generation_fact_graph_v1/artifacts/runtime_receipts_checkpoint.json` | 4,687,531 | `004969bdd8a18ef643d003e876326e49c92db0b68f41ed95c9bb056a0e846d57` |

Their generation code, compact run manifest, validation decision and graph
visualizations remain in
`experiments/nanogpt_training_generation_fact_graph_v1/`. A restored payload
must match both the source commit and the SHA-256 value above before it is used
as evidence.

## RL-E02 formal execution bundle

RL-E02 produced 134 external runtime files totalling 81.72 MiB. The bundle
contains the per-seed GFG ledgers, candidate and causal-credit ledgers, sealed
policy checkpoints and policy-evaluation records. It is not tracked here.
The two compact adjudication entry points are:

| Logical locator | SHA-256 |
|---|---|
| `external://gfg-temporal-credit-discovery-v1/formal/AGGREGATE_RESULT.json` | `c1bf98a259e842594585bf373c50dd45a187607d5f33f01e2fcc026d1f168a3b` |
| `external://gfg-temporal-credit-discovery-v1/formal/INDEPENDENT_CHECK.json` | `4fec707d93f009daaf5bbdba09fb12df1438295b5e90e46366abefcaf7f59043` |

A restored bundle must preserve these hashes. A fresh run may use another
filesystem location, but its path must remain outside the tracked repository.
