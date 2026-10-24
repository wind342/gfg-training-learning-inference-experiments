# TLIR — IJCAI 2027 Official-Style Manuscript

This archive contains the anonymized TLIR manuscript compiled with the unmodified `ijcai27.sty` supplied by the author (official IJCAI 2027 author kit).

- `short_paper.tex`: source, 10-point official IJCAI style; line-numbered blind review version.
- `ijcai27.sty`: original official IJCAI 2027 style file (unchanged).
- `figures/`: two original vector figures retained from the TLIR v8 paper PDF.

Run `latexmk -pdf -interaction=nonstopmode -halt-on-error short_paper.tex`. This manuscript includes 26 author-year inline bibliography entries; no external BibTeX pass is needed.

Validation: compiled to **7 pages total**, US Letter, line-numbered, anonymous; scientific sections and experimental results retained. Abstract shortened to meet the 200-word limit; Table 1 reorganized for legibility at IJCAI font size; tables have captions below per IJCAI guidance. References start on page 6. Existing IEEE source remains unchanged.

For conference submission, review any separate supplementary files to ensure double-blind anonymity, and ensure the final manuscript and registration information match the intended IJCAI 2027 track.
