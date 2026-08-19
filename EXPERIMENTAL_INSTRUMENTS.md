# Experimental instruments

This index exposes the four reusable experimental instruments developed for
the training--learning study.  The canonical implementations remain in the
experiment packages that execute and validate them; they are not duplicated
under a second source tree.

## CSRG-4C

CSRG-4C measures how four registered residual components support each target.
At one restored checkpoint it executes two baselines, four single-component
gates and six paired-component gates, then derives necessity, backup,
effective support, concentration and failure slack.

- frozen capture contract:
  [`capture_contract_v2.json`](experiments/gfg_nanogpt_support_redundancy_v1/capture_contract_v2.json)
- gated CUDA execution:
  [`runtime.py`](experiments/gfg_nanogpt_support_redundancy_v1/runtime.py)
- support construction:
  [`builder.py`](experiments/gfg_nanogpt_support_redundancy_v1/builder.py)
- generation-fact construction:
  [`support_gfg.py`](experiments/gfg_nanogpt_support_redundancy_v1/support_gfg.py)
- independent archive verification:
  [`verify_archive.py`](experiments/gfg_nanogpt_support_redundancy_v1/verify_archive.py)

## Realized-update causal forks

The causal-fork instrument restores an exact receiving state and executes
skip, native full-update and reciprocal parameter/optimizer-update branches.
Branch effects are measured relative to the corresponding skip state so that
the realized update can be separated from the state receiving it.

- branch execution:
  [`branches.py`](experiments/gfg_nanogpt_stepwise_support_transition_v1/branches.py)
- reciprocal state/update experiment:
  [`reciprocal.py`](experiments/gfg_nanogpt_stepwise_support_transition_v1/reciprocal.py)
- frozen reciprocal protocol:
  [`RECIPROCAL_MATCHED_PAIR_PROTOCOL_V2.json`](experiments/gfg_nanogpt_stepwise_support_transition_v1/RECIPROCAL_MATCHED_PAIR_PROTOCOL_V2.json)
- validation:
  [`reciprocal_validator.py`](experiments/gfg_nanogpt_stepwise_support_transition_v1/reciprocal_validator.py)
- generation-fact construction:
  [`reciprocal_gfg.py`](experiments/gfg_nanogpt_stepwise_support_transition_v1/reciprocal_gfg.py)

## Finite-amplitude update paths

This instrument applies one exact realized parameter update to the same
receiving state at a frozen amplitude grid.  It records the complete numeric
and categorical response path, executes CSRG-4C at registered states and
compares finite-amplitude outcomes with local response approximations.

- frozen amplitude protocol:
  [`B_UPDATE_AMPLITUDE_PATH_PROTOCOL.json`](experiments/gfg_nanogpt_stepwise_support_transition_v1/B_UPDATE_AMPLITUDE_PATH_PROTOCOL.json)
- path execution and response construction:
  [`amplitude_path.py`](experiments/gfg_nanogpt_stepwise_support_transition_v1/amplitude_path.py)
- validation:
  [`amplitude_path_validator.py`](experiments/gfg_nanogpt_stepwise_support_transition_v1/amplitude_path_validator.py)
- generation-fact construction and validation:
  [`amplitude_path_gfg.py`](experiments/gfg_nanogpt_stepwise_support_transition_v1/amplitude_path_gfg.py)
  and
  [`amplitude_path_gfg_validator.py`](experiments/gfg_nanogpt_stepwise_support_transition_v1/amplitude_path_gfg_validator.py)

## Identity-aligned target-boundary ledger

The boundary-ledger instrument aligns every evaluation target by preserved
identity, reconstructs its correct-versus-competitor margins and seals the
pretarget material before reading the complete-update endpoint.  It then
adjudicates remaining-correct, correct-to-wrong, remaining-wrong and
wrong-to-correct transitions without positional or value-based identity
matching.

- projection, sealing, adjudication and validation:
  [`experiment.py`](experiments/gfg_nanogpt_identity_aligned_margin_crossing_v1/experiment.py)
- upstream response execution and evidence:
  [`native_response_500.py`](experiments/gfg_nanogpt_stepwise_support_transition_v1/native_response_500.py)
- upstream evidence validation and GFG construction:
  [`native_response_500_validator.py`](experiments/gfg_nanogpt_stepwise_support_transition_v1/native_response_500_validator.py),
  [`native_response_500_gfg.py`](experiments/gfg_nanogpt_stepwise_support_transition_v1/native_response_500_gfg.py) and
  [`native_response_500_gfg_validator.py`](experiments/gfg_nanogpt_stepwise_support_transition_v1/native_response_500_gfg_validator.py)

The associated scientific experiments are indexed as TL-E03, TL-E04, TL-E06
and TL-E08 in [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md).
