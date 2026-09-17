# Functional online amplitude controller v1

A separate prospective online continuation of functional step-size smoke v1.
No earlier frozen scientific result or tag is changed.

Read PROTOCOL_FREEZE.md first. This experiment compares native AdamW, fixed
quarter displacement and the unchanged smoke functional selector on 12 existing
source states, 200 steps each, with evaluations every 10 steps. The control
targets are also the evaluation targets; there is no new held-out-generalization
claim. Full source identities and batch opportunities are in RUN_ASSIGNMENT.json.

## Execution

From repository root, using the recorded CUDA Python environment:

```powershell
python -B -m experiments.gfg_nanogpt_functional_online_controller_v1.preflight audit
python -B -m experiments.gfg_nanogpt_functional_online_controller_v1.self_test
python -B -m experiments.gfg_nanogpt_functional_online_controller_v1.preflight freeze
python -B -m experiments.gfg_nanogpt_functional_online_controller_v1.runner
python -B -m experiments.gfg_nanogpt_functional_online_controller_v1.analysis
python -B -m experiments.gfg_nanogpt_functional_online_controller_v1.independent_checker
python -B -m experiments.gfg_nanogpt_functional_online_controller_v1.package
```

Preparation and results use exclusive creation. Existing partial/failed branches
are never overwritten or silently restarted. Completed branches may be skipped
when continuing execution under the same frozen protocol. A partial failure
requires explicit investigation, with the failed evidence retained. Changing
paths for another machine requires a separate execution identity, not silently
editing this recorded run's frozen context.

Full parameter/optimizer/gradient tensors and GFG databases are external under
the path in EXPERIMENT_CONTEXT.json; MANIFEST.json records all SHA-256 identities.
Compact ledgers and reports are distributed separately from these larger tensors.
This Git extension does not claim its raw tensor payloads are in the existing
Zenodo research archive.
