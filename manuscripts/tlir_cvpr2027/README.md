# TLIR — CVPR 2027 review submission package

This project was compiled from the official CVPR 2027 author kit (`author-kit-CVPR2027-v1-latex.zip`) and includes the official, unmodified `cvpr.sty` and `ieeenat_fullname.bst` supplied with that kit. `main.tex` is the anonymous review-mode manuscript.

Contents:
- `main.tex` — manuscript
- `main.bib` — BibTeX sources (26 references)
- `main.bbl` — compiled bibliography (provided to ease reproducibility)
- `cvpr.sty`, `ieeenat_fullname.bst`, `preamble.tex` — official template components
- `figures/figure1_functional_chain.pdf` — vector figure with straight horizontal forward arrows and vertical right-side feedback arrow
- `figures/figure2_six_system_evidence.pdf` — original second vector figure

Compile in this directory with `latexmk -pdf -bibtex main.tex` (pdfLaTeX + BibTeX). If you use an environment without a functioning BibTeX binary, the included `main.bbl` lets you run `pdflatex main.tex` multiple times.

The document is eight US Letter pages total. The LLM Usage Statement and bibliography appear in the final pages. Review-mode line numbers and double-blind author block are included. The authors must complete OpenReview profiles and register the submission by the official abstract/enrollment deadline.

Source of the official kit: https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines
