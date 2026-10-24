# TLIR v8 LaTeX manuscript

This folder contains the complete LaTeX source and publication figures for the v8 manuscript **Unified Training–Learning–Inference–Reinforcement Dynamics Across Neural Systems**.

## Files

- `short_paper.tex` — IEEE two-column LaTeX manuscript (includes its bibliography).
- `figures/figure1_functional_chain.pdf` — four peer roles of TLIR.
- `figures/figure2_six_system_evidence.pdf` — six-system TLI experimental evidence.
- `figures/draw_figures.py` — Python/Matplotlib source to regenerate the figure PDFs.

## Compile

From this directory, with a TeX installation that includes IEEEtran:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error short_paper.tex
```

On Overleaf, select `short_paper.tex` as the main document and use pdfLaTeX. The two vector-PDF images are already included, so running Python is **not** required for Overleaf compilation.

To regenerate figures locally with Matplotlib:

```bash
python figures/draw_figures.py
```

This manuscript is separate from the repository's frozen experiments and does not alter experimental protocols, code, or evidence.
