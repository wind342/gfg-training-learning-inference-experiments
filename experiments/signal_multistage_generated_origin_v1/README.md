# Multi-stage signal generation through GeneratedOrigin

This experiment uses a real public ECG signal and executes one fixed pipeline:

```text
MIT-BIH record 100, MLII samples
  -> nine-tap FIR filtering
  -> factor-four downsampling
  -> sliding 32-point real FFT
  -> deterministic SVG spectrogram
```

Every stage establishes Core v3 generation facts synchronously. A result
support from one stage is reintroduced into the next stage only through a
`GeneratedOrigin` whose payload names the prior support and its producer
operation. Samples removed by downsampling and the incomplete FFT tail are
recorded as `ExplicitDisposition` outcomes.

The final query is a rectangle in SVG coordinates. `QueryEngine` selects the
rendered spectrum cells. A snapshot-only traversal then follows three
`GeneratedOrigin` bridges back to the precise raw ECG samples. An independent
reference uses NumPy convolution and an explicit discrete Fourier transform to
compute values and the complete expected path multiset.

## Structural role

Filtering, downsampling, Fourier analysis and SVG rendering use different
domain objects, operations, support spaces and relation roles.  They do not
require different generation-fact schemas.  Every stage uses the same atomic
structure `f=(u,tau,omega_bar,z;rho)` and the same GFG organization; the
stage-specific meanings are supplied by the frozen capture protocol,
transformation registry and support-space predicates.  The executed pipeline
therefore demonstrates fixed structural roles with changing concrete domain
semantics, rather than a graph tied to one domain vocabulary.

## Public input

- Dataset: MIT-BIH Arrhythmia Database v1.0.0
- Record: 100
- Channel: MLII
- Source: <https://physionet.org/content/mitdb/1.0.0/>
- DOI: <https://doi.org/10.13026/C2F305>
- License: Open Data Commons Attribution License v1.0
- Citation: Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia
  Database*. IEEE Engineering in Medicine and Biology Magazine 20(3):45-50,
  2001.

The downloader verifies the exact sizes and SHA-256 digests frozen in
`contracts/input_manifest.json`. Original bytes are stored only under
`data_private/`, which is ignored by Git.

## Run

```console
python -m experiments.signal_multistage_generated_origin_v1.run_experiment
```

The result bundle is written to
`results/signal_multistage_generated_origin_v1/`.

## Claim boundary

The experiment tests exact multi-stage formation paths and cross-space
selective access for the frozen real-data workload. It does not test
arrhythmia diagnosis, clinical decision-making, arbitrary filters, arbitrary
FFT configurations, signal reconstruction, or universal biomedical
provenance.
