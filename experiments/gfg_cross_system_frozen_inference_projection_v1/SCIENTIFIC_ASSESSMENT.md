# Scientific assessment

The result supports the claim that inference is a frozen projection of
training--learning formation in both executed cross-system settings. It is not
merely the observation that outputs depend on trained parameters. Exact
training versions were identified, their distributed components were observed
to participate in frozen executions, the recruited support depended on the
current query and combined non-additively, and exact component-version rollback
changed complete outputs before exact restoration recovered them.

This materially strengthens the original nanoGPT result. The relation now holds
in an autoregressive language model with Adam, a discriminative residual image
classifier with SGD momentum, and a generative diffusion U-Net with AdamW. The
diffusion arm also separates transient iterative inference state from persistent
learned state: a sampler may evolve its current sample while the learned
parameter--optimizer state remains frozen.

The strongest warranted conclusion is therefore cross-system support for the
same mechanistic relation across these three executed architectures, objectives,
modalities and optimizers. The experiment does not rely on an architecture-
specific attention mechanism, a classification-only readout or a single
optimizer.
