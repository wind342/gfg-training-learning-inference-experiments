# SFP official minimal demo: incomplete attempt

Status: **FAILED BEFORE TRAINING**. This directory preserves the observation
wrapper and the outcome reached before the user paused the experiment. It does
not contain a completed SFP benchmark or a governor adaptation.

Official source: https://github.com/paradigmxyz/sfp at
`e4ef486c169dcc95af52dc504a8fcebcb26f9959` (MIT).
Model: `HuggingFaceTB/SmolLM2-135M-Instruct` at
`12fd25f77366fa6b3b4b768ec3050bf629380bac` (Apache 2.0).
External source files, model weights and raw attempt records remain outside Git.

`run.py` calls the pinned official `forget.main`, pins the local model and seed42,
and observes outputs, training batches and losses without changing the questions,
substring scorer or training decisions. The official minimal demo has ten math
questions; it is not the complete multi-task SFP evaluation. It does not evaluate
new code capability, and training loss would not substitute for that measurement.

## Observed attempt-1 result

- Before training: 0/10 under the unchanged official prompts and substring scorer;
  all ten generated continuations were empty. This creates a floor for measuring
  further score decline; it is not a general assessment of the model's ability.
- Training updates: **0**. No after-training score exists.
- Failure: `DatasetNotFoundError: Revision '12e9221' doesn't exist for dataset
  'google-research-datasets/mbpp' on the Hub.` This revision is pinned by the
  official `data.py`. A separate read-only Hub revision request returned HTTP404.
- No data-pin repair, replacement questions, prompt changes or selected rerun
  were performed. No forgetting or retention conclusion is supported.

Raw records: `E:/gfg-evidence/governor-sfp-v1/attempt-1/`.
SHA-256 identities of those unchanged records:

| File | SHA-256 |
|---|---|
| CONFIG.json | c7e4d966367e2d06723cf0761358440c31d4bca0020c8fa360c098c580ce0752 |
| RESULT.json | cf2a50748d005233ea915cc24004a1092731753bf0eeab81a0ef6bf3e5a62968 |
| before.json | b7bb0e8768f4a36aa5a176a5d6a1b3e8ae9f1ae7f5f7a4485030643ccdb97728 |
| RUN.log | a2d60deaf59b0cbf07ce289fe8aedb80df18b137e28b50f3fe6d86f4c1ee0a2c |

The wrapper is create-only and refuses to overwrite attempt-1. Its existence is
not evidence of a successful run. Current task interfaces use generated answers,
programs or response constraints; a fixed classification boundary must not be
silently substituted for those official scoring rules.
