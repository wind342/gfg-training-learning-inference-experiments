# Blockwise backend integration gate

This is a separate extension; the original phase-1 sources, seal and outcomes
remain untouched. First require the original offline audit PASS and every
blockwise h/h'/h'' diagnostic exact. Then seal this explicit blockwise backend
before any integrated execution. No tolerance allowance or outcome-dependent
fallback. A mismatch fails this backend rather than selecting a different path.

Repeat all 72 audit/runtime trajectories using BlockwiseController. AUDIT mode
also computes global-JVP arrays for equality checks; RUNTIME computes only the
blockwise arrays. The same independent raw-artifact auditor is rerun in a new
directory. Native candidate, coordinates, minima, alpha, post-parameters, logits,
capability crossings and final optimizer must match the original d17f451 oracle.

Scope: nanoGPT-compatible decoder with no stochastic inference-time layers,
standard embedding+position addition, sequential blocks, final layer norm,
and the tied bias-free readout. Generic proposal storage and boundary reduction
do not depend on this adapter. Other architectures require their own adapters.
No distributed, offload or pretrained-model claim. No 70M/160M training here.
