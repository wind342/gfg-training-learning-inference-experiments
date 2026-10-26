"""Verify Science formatting without modifying original scientific findings."""
from pathlib import Path
import hashlib, json, re

ROOT = Path(__file__).resolve().parent
original = (ROOT / "original_source/main.tex").read_text(encoding="utf-8")
science = (ROOT / "main.tex").read_text(encoding="utf-8")
bib = (ROOT / "references.bbl").read_text(encoding="utf-8")
FLOAT = r"\\begin\{(figure\*?|table)\}.*?\\end\{\1\}"

def normalize(x):
    x = re.sub(FLOAT, "", x, flags=re.S)
    x = x.replace(r"\clearpage", "").replace(r"\newpage", "")
    return re.sub(r"\s+", " ", x).strip()

o = original.split(r"\section{Introduction}", 1)[1].split(
    r"\noindent\textbf{Evidence availability and machine audit.}", 1)[0]
s = science.split(r"\end{abstract}", 1)[1].split(
    "\\clearpage\n\\begin{figure}", 1)[0]
s = re.sub(r"^\s*\\noindent\s*", "", s, count=1)
s, count = re.subn(
    r"\\subsection\{Experimental Instruments and Evidence Control\}.*?"
    r"(?=\\subsection\{Training: Realized Updates and Receiving States\})",
    "", s, flags=re.S)
assert count == 1
s = s.replace(r"Figure~\ref{fig:chain} illustrates the four linked dynamical roles.", "")
s = s.replace(r"Figure~\ref{fig:systems} summarizes the six within-system TLI experiment chains.", "")
s = s.replace("The core TLIR relations were established experimentally before this cross-field comparison", "The core TLI relations were established experimentally before this cross-field comparison")
s = s.replace("within the main text", "within the conference page limit")
s = s.replace("examined in the reinforcement section",
              r"examined in Section~\ref{sec:reinforcement}")
historical_start = r"\section{Unified Explanations of Positive and Negative Findings}"
historical_end = r"\section{Conclusion}"
def without_historical_section(text):
    assert text.count(historical_start) == 1
    assert text.count(historical_end) == 1
    return text.split(historical_start, 1)[0] + historical_end + text.split(historical_end, 1)[1]

# The author authorized the historical interpretation/extrapolation Section only.
# Validate every other scientific paragraph verbatim against the frozen CVPR source.
assert normalize(without_historical_section(o)) == normalize(without_historical_section(s)), (
    "Unexpected modification outside the authorized unification section")


equations = lambda x: re.findall(r"\\begin\{equation\}.*?\\end\{equation\}", x, re.S)
assert len(equations(original)) == 10
assert equations(original) == equations(science)

abstract = science.split(r"\begin{abstract}", 1)[1].split(
    r"\end{abstract}", 1)[0].strip()
assert len(abstract.split()) <= 125 and "\n\n" not in abstract
assert "learning persistently reorganizes distributed functional support, with target-specific readout-boundary crossings determining capability transitions" in abstract
assert "learning reorganizes distributed functional support across capability boundaries" not in abstract
assert "The core TLIR relations were established experimentally before this cross-field comparison" in science
assert abstract.startswith("Although training, learning, inference, and reinforcement have been extensively studied, their distinct dynamical roles and connections have lacked a unified experimental characterization.")
for phrase in ["Nine nanoGPT experiments", "91.43", "91.49", "5,088",
               "five additional", "seventh system", "70.31", "99.48"]:
    assert phrase in abstract, phrase
assert "zenodo.org" not in science and "doi.org/10.5281/zenodo" not in science
assert r"\title{\bfseries Unified dynamics of training, learning, inference, and reinforcement in neural networks}" in science
assert "pdftitle={Unified dynamics of training, learning, inference, and reinforcement in neural networks}" in science
assert r"\author{Mian Wang\\" in science
assert "pdfauthor={Mian Wang}" in science
assert r"\section{Introduction}" not in science
assert r"\renewcommand\refname{References and Notes}" in science
assert science.index(r"\input{references.bbl}") < science.index(
    r"\section*{Acknowledgments}")

for phrase in [
    r"\paragraph*{Funding:}", "This research received no funding.",
    r"\paragraph*{Author contributions:}",
    r"\paragraph*{Competing interests:}",
    "There are no competing interests to declare.",
    r"\paragraph*{Data, code and materials availability:}",
    "organized by the scientific claims of this paper",
    "rapid machine-assisted auditing",
    "paper-to-evidence map",
    "independent verification procedures",
    "https://github.com/wind342/tlir-research-evidence/",
    "public evidence matrix",
    r"\paragraph*{Use of generative artificial intelligence:}",
    "AI systems invented and implemented the four experimental instruments",
    "designed, implemented, initiated, monitored, and executed the reported experiments",
    "OpenAI Codex and ChatGPT"
]:
    assert phrase in science, "Missing declaration: " + phrase

def floats(x):
    result = {}
    for match in re.finditer(FLOAT, x, re.S):
        t = match.group(0)
        label = re.search(r"\\label\{([^}]+)\}", t)
        caption = re.search(r"\\caption\{([^\n]*)\}", t)
        table = re.search(r"\\begin\{tabular\}.*?\\end\{tabular\}", t, re.S)
        image = re.search(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", t)
        assert label and caption
        result[label.group(1)] = (
            caption.group(1),
            table.group(0) if table else None,
            image.group(1) if image else None
        )
    return result

of = floats(original)
sf = floats(science)
assert of.keys() == sf.keys() and len(sf) == 6
for key, (caption, table, image) in of.items():
    updated_caption, updated_table, updated_image = sf[key]
    assert image == updated_image, key
    if key == "tab:systems":
        assert table.replace(r"\newline", " ") == updated_table
    else:
        assert table == updated_table, key
    if key == "fig:tlir-evidence":
        assert updated_caption.startswith(caption[:-1])
        for panel in "a1 a2 a3 a4 b1 b2 b3 c1 c2 c3 d1 d2 d3 d4".split():
            assert "(" + panel + ")" in updated_caption
    else:
        assert caption == updated_caption, key

refs = re.findall(r"\\bibitem\{([^}]+)\}", bib)
main_without_floats = re.sub(FLOAT, "", science, flags=re.S)
used = list(dict.fromkeys(
    k.strip() for m in re.finditer(r"\\cite(?:\[[^]]*\])?\{([^}]+)\}",
                                  main_without_floats)
    for k in m.group(1).split(",")
))
assert len(refs) == 27 and refs == used, ("Citation mismatch", refs, used)
new_bib = (ROOT / "additional_references.bib").read_text(encoding="utf-8")
assert "10.18653/v1/N19-1357" in new_bib
assert "10.18653/v1/D19-1002" in new_bib
assert "jain2019" in used and "wiegreffe2019" in used
assert "We invite reviewers and independent researchers" in science
assert "derivation-before-search" in science
assert "Its stronger preregistered temporal-precedence criterion was not met." in science
assert not re.search(r"\bet\s*al\.", bib, re.I)
assert r"\emph{" in bib and r"\textbf{" in bib
assert "geiger2025" not in bib
assert (ROOT / "main.bib").read_bytes() == (
    ROOT / "original_source/main.bib").read_bytes()

report = json.loads((ROOT / "verification.json").read_text())
for name, expected_sha in report["figure_files_sha256_identical"].items():
    actual_sha = hashlib.sha256((ROOT / "figures" / name).read_bytes()).hexdigest()
    assert actual_sha == expected_sha, name
assert "developed the atomic generation-fact model, the Generation-Fact Graph" in science
assert "invented and implemented the four experimental instruments" in science
assert "assistance with experimental-protocol design" not in science
method_section = science.split(r"\subsection{Experimental Instruments and Evidence Control}",1)[1].split(r"\subsection{Training: Realized Updates and Receiving States}",1)[0]
assert "AI" not in method_section and "Codex" not in method_section and "ChatGPT" not in method_section
assert method_section.strip().startswith("These experiments required instruments that were developed as the measurement questions arose")
assert science.count(r"\clearpage") == 0 and science.count(r"\newpage") == 1

print("PASS: scientific paragraphs, all ten equations, and six original figures/tables.")
print(f"PASS: Science abstract {len(abstract.split())} words; 27 formatted references.")
print("PASS: funding, authorship, competing interest, data/DOI and AI disclosures.")
