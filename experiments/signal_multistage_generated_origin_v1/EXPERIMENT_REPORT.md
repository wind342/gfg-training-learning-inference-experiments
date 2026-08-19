# Multi-stage signal GeneratedOrigin experiment

Final status:
`MULTISTAGE_SIGNAL_GENERATION_FACTS_SUPPORTED`.

The experiment downloaded record 100 from the public MIT-BIH Arrhythmia
Database v1.0.0 and selected 512 MLII samples at 360 Hz. The immutable
`100.dat` input has SHA-256
`b2ea3c250e56e48f4b7b90697832b8ecd1afa1e0bb31f2dcfea4ed6e1075a639`.

The real generator executed four stages:

```text
raw ECG -> nine-tap FIR -> factor-four downsample
        -> 32-point sliding FFT -> SVG spectrogram
```

The validated snapshot contains 512 registered sources, 1,117 concrete
occurrences, 732 GeneratedOrigins, 834 outcome supports, 392
ExplicitDispositions and 8,420 GenerationBindings. All 8,420 bindings have
formal primary relation evidence and successful operation closure.

A rectangle in the final SVG selected ten spectrum cells. Snapshot-only
traversal followed three GeneratedOrigin bridges and returned 197 raw ECG
samples through 2,880 multiplicity-preserving paths. The answer was exactly
equal to the independent mathematical reference. Its canonical path multiset
SHA-256 is
`f99609b8bf9dd7f0397c35b7a2331c93ddfc66182658cd013ea12de41bfc3049`.

Capture-on and capture-off SVG bytes are identical with SHA-256
`a83b8cdcbb1ae4e60e331ed0c841c597538480e7f25c3e3bfd6ea1e35779f7cf`.
The largest numeric difference from the independent explicit-DFT reference is
`8.93729534823251e-15`.

Negative controls establish that:

- there is no direct raw-source-to-final-cell shortcut;
- Cartesian expansion would add 640 false final-cell/source pairs;
- a tampered GeneratedOrigin is rejected with `HASH_OR_ID_MISMATCH`;
- no source identity is encoded in the SVG output.

Two complete executions produced byte-identical scientific summaries with
SHA-256
`202a7ba75e9c48e153fcfd3f563334188e4aefed91551da10c6b70dd1f4a285d`
and byte-identical SVG output.

The result establishes the declared real-data pipeline only. It does not make
an arrhythmia-diagnosis, clinical-decision, inverse-reconstruction or universal
signal-processing claim.
