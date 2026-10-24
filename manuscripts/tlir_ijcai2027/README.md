# TLIR — IJCAI 2027 double-blind LaTeX formatting draft

An additive formatting-only adaptation of `../tlir_v8/short_paper.tex`. The IEEE original is unchanged. The paper's theoretical statements, equations, experimental results, main tables and figures remain in place. IEEE-specific class and manual text/float spacing changes are removed. The bibliography is alphabetized with IJCAI-style author/year labels, while preserving each source's bibliographic details.

## Contents

- `short_paper.tex` — IJCAI-style anonymized manuscript source.
- `figures/figure1_functional_chain.pdf` — unchanged original vector figure.
- `figures/figure2_six_system_evidence.pdf` — unchanged original vector figure.

## Compile (official style required)

Download the official IJCAI 2027 author kit from https://www.ijcai.org/authors_kit (official zip: https://ijcai.org/sites/default/files/Guidelines/FormattingGuidelines-IJCAI-27.zip).

Place the **unmodified** `ijcai27.sty` from that kit in this directory before compiling:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error short_paper.tex
```

The 2027 ZIP could not be obtained in this conversion environment, so no nonofficial imitation style is included. If the official kit contains different package or metadata instructions, those instructions take precedence.

## Final submission checks

- Official IJCAI 2027 style, US Letter, two columns, and no margin/font reductions.
- No more than **7 body pages + 2 references/ethics pages**. Figures and tables count toward the 7.
- Double-blind anonymity: author names, institutions, acknowledgements, self-identifying links and identifying PDF metadata absent.
- Anonymize any separate code or evidence supplementary material.
- Build and visually verify the PDF using the **official 2027 kit**.

**Status:** LaTeX formatting and figure copying completed. Official-kit compilation and measured page-count/visual review remain unverified. The body is not substantively shortened merely to assume a page count.
