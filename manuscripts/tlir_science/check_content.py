"""Audit format-only changes against the frozen CVPR source without third-party packages."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parent
original = (ROOT / "original_source/main.tex").read_text(encoding="utf-8")
science = (ROOT / "main.tex").read_text(encoding="utf-8")
FLOAT = r"\\begin\{(figure\*?|table)\}.*?\\end\{\1\}"


def without_layout(tex):
    body = tex.split(r"\begin{document}", 1)[1]
    body = re.sub(FLOAT, "", body, flags=re.S)
    body = re.sub(
        r"\{\s*\\small\s*\\bibliographystyle\{ieeenat_fullname\}\s*"
        r"\\bibliography\{main\}\s*\}",
        "",
        body,
    )
    body = body.replace(r"\input{references.bbl}", "").replace(r"\clearpage", "")
    return body


def canonical(value):
    return re.sub(r"\s+", " ", value).strip()


def strip_approved_changes(tex):
    tex = tex.replace("Figure~\\ref{fig:chain} illustrates the four linked dynamical roles.", "")
    tex = tex.replace(
        "Figure~\\ref{fig:systems} summarizes the six within-system TLI experiment chains.",
        "",
    )
    tex = tex.replace("within the main text", "within the conference page limit")
    tex = tex.replace(
        "examined in the reinforcement section",
        r"examined in Section~\ref{sec:reinforcement}",
    )
    tex = re.sub(
        r"\\section\*\{Funding and Competing Interests\}.*?"
        r"(?=\\section\*\{LLM Usage Statement\})",
        "",
        tex,
        flags=re.S,
    )
    return tex


assert canonical(strip_approved_changes(without_layout(science))) == canonical(
    without_layout(original)
), "Unexpected scientific text change"

expected_funding = r"\noindent\textbf{Funding.} This research received no funding."
expected_competing = (
    r"\noindent\textbf{Competing interests.} The author declares no competing interests."
)
assert expected_funding in science and expected_competing in science
assert "within the conference page limit" not in science
assert r"\setcounter{secnumdepth}{0}" in science
assert r"\captionsetup[figure]{name=Fig.}" in science
assert r"\renewenvironment{abstract}{\par\noindent\textbf{Abstract}\par\bfseries}{\par\clearpage}" not in science


def floats(tex):
    result = {}
    for match in re.finditer(FLOAT, tex, flags=re.S):
        content = match.group(0)
        label_match = re.search(r"\\label\{([^}]+)\}", content)
        caption_match = re.search(r"\\caption\{([^\n]*)\}", content)
        assert label_match and caption_match, "Float missing label or caption"
        label = label_match.group(1)
        table = re.search(r"\\begin\{tabular\}.*?\\end\{tabular\}", content, flags=re.S)
        image = re.search(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", content)
        result[label] = {
            "caption": caption_match.group(1),
            "table": table.group(0) if table else None,
            "image": image.group(1) if image else None,
        }
    return result


of = floats(original)
sf = floats(science)
assert of.keys() == sf.keys() and len(of) == 6
for key in of:
    assert of[key]["table"] == sf[key]["table"], ("Changed table cells", key)
    assert of[key]["image"] == sf[key]["image"], ("Changed figure source", key)
    if key == "fig:tlir-evidence":
        original_caption = of[key]["caption"]
        updated_caption = sf[key]["caption"]
        assert updated_caption.startswith(original_caption[:-1])
        assert all(f"({label})" in updated_caption for label in (
            "a1", "a2", "a3", "a4", "b1", "b2", "b3", "c1", "c2", "c3",
            "d1", "d2", "d3", "d4",
        ))
    else:
        assert of[key]["caption"] == sf[key]["caption"], ("Caption changed", key)

body = re.sub(FLOAT, "", science, flags=re.S)
figure_order = [r"fig:chain", r"fig:tlir-evidence", r"fig:systems"]
figure_positions = [body.index(r"\ref{" + k + "}") for k in figure_order]
assert figure_positions == sorted(figure_positions)
for table in ("tab:systems", "tab:prediction", "tab:dose"):
    assert r"\ref{" + table + "}" in body

equations = lambda text: re.findall(
    r"\\begin\{equation\}.*?\\end\{equation\}", text, flags=re.S
)
assert len(equations(original)) == 10
assert equations(original) == equations(science)


def bibitems(tex):
    found = list(re.finditer(r"\\bibitem(?:\[[\s\S]*?\])?\{([^}]+)\}", tex))
    end = tex.rfind(r"\end{thebibliography}")
    return {
        match.group(1): tex[match.start(): (found[i+1].start() if i+1 < len(found) else end)].strip()
        for i, match in enumerate(found)
    }, [match.group(1) for match in found]


original_bib = (ROOT / "original_source/main.bbl").read_text(encoding="utf-8")
current_bib = (ROOT / "references.bbl").read_text(encoding="utf-8")
old_map, old_order = bibitems(original_bib)
new_map, new_order = bibitems(current_bib)
cite_order = list(dict.fromkeys(
    key.strip()
    for match in re.finditer(r"\\cite(?:\[[^]]*\])?\{([^}]+)\}", body)
    for key in match.group(1).split(",")
))
assert new_order == cite_order, "References are not in first-citation order"
assert set(old_order) - set(new_order) == {"geiger2025"}
assert len(new_order) == 25
assert all(new_map[key] == old_map[key] for key in new_order)
assert (ROOT / "original_source/main.bib").read_bytes() == (ROOT / "main.bib").read_bytes()
assert r"\begin{thebibliography}{25}" in current_bib

title = re.search(r"\\title\{([^\n]*)\}", original).group(1)
assert r"\title{\bfseries " + title + "}" in science
expected_author = r"""\author{wangmian\\
{\small Wuhan Polytechnic University, Wuhan, China}\\
{\small Corresponding author. Email: \href{mailto:wangmian945@gmail.com}{wangmian945@gmail.com}}}"""
assert expected_author in science
assert "pdfauthor={wangmian}" in science

report = json.loads((ROOT / "verification.json").read_text(encoding="utf-8"))
for name, expected_sha in report["figure_files_sha256_identical"].items():
    figure_file = ROOT / "figures" / name
    assert hashlib.sha256(figure_file.read_bytes()).hexdigest() == expected_sha, name

print("PASS: original scientific prose, abstract, ten equations, tables and image bytes preserved.")
print("PASS: approved figure callouts, fourteen-panel legend and declarations present.")
print("PASS: 25 cited references ordered by appearance; unused Geiger entry remains in original backup.")
print("NOTE: previous compiled TLIR_Science_Final.pdf is stale until build.py is rerun.")
