# Functional step-size smoke test (exploratory)

This isolated experiment scales an already realized AdamW update. It does not
train a model, run new inference, change an update direction, or modify any
previous experiment. The parent is the frozen research commit
`36dab5ce347dbbdac157ef23205f556606d18294`.

The 72-section native-update corpus is used, not the special B-to-A transplanted
update in `B_UPDATE_AMPLITUDE_PATH_PROTOCOL.json`. Every section retains all 212
targets; the outer statistical unit is one of the 12 historical runs.

## Reuse and information boundary

The cached `first_gap`, `joint_rotation`, and `hidden_curvature` are produced by
`gfg_nanogpt_joint_rotation_branch_v1.analysis.compile_coordinates`, also called
by TL-P01. The exact double-precision starting gaps are taken from the companion
competitor-coordinate cache (the joint cache stores a float32 copy of these).
The original `load_named_array` validator is reused for raw tensor identities.
No probe, KNN, fitting, numerical coefficient search, or new derivative execution
is introduced. Alpha 1 is cross-checked against all archived TL-P01 predictions
after the selection has been sealed.

The original response compiler loads outcome fields, so it is deliberately NOT
called in Stage A. A read allowlist instead admits only coordinate artifacts and
identity metadata. The selector process cannot read response-section NPZs,
prediction ledgers, or other outcome files. Stage A terminates after sealing;
Stage B is a separate process and verifies the seal before opening outcomes.

## Run

Requires Python >=3.12 and NumPy; no GPU or training dependencies are needed.
From the repository root:

```powershell
python -B -m experiments.gfg_nanogpt_functional_step_size_smoke_v1.runner --stage all
python -B -m experiments.gfg_nanogpt_functional_step_size_smoke_v1.independent_checker
python -B -m unittest experiments.gfg_nanogpt_functional_step_size_smoke_v1.test_smoke
```

Defaults use the existing local evidence locations. For relocated evidence pass
`--reports-root`, `--response-root`, `--boundary-root` and `--output-root`; run
`runner --help` for their meanings. Output files are created exclusively and are
never overwritten. To reproduce, use a new output directory. Frozen source
paths embedded in inventory metadata must remain accessible for raw identity
checking. Missing files cause a hard failure, not an exclusion or replacement.

Only this new experiment directory may be committed. No root manifest, frozen
tag, published evidence bundle, or Zenodo version is changed. Local cached
source evidence is read-only; its hashes are checked again after adjudication.

See `PROTOCOL_FREEZE.md` for the fixed rule and `RESULTS.md` for actual results.
