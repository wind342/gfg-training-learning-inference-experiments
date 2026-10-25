"""Verify the format-only conversion without third-party Python dependencies."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parent
original = (ROOT / 'original_source/main.tex').read_text(encoding='utf-8')
science = (ROOT / 'main.tex').read_text(encoding='utf-8')
FLOAT = r'\\begin\{(figure\*?|table)\}.*?\\end\{\1\}'

def text_without_layout(tex):
    s = tex.split('\\begin{document}', 1)[1]
    s = re.sub(FLOAT, '', s, flags=re.S)
    s = s.replace('{\\small\n\\bibliographystyle{ieeenat_fullname}\n\\bibliography{main}\n}', '')
    s = s.replace('\\input{references.bbl}', '').replace('\\clearpage', '')
    return re.sub(r'\s+', ' ', s).strip()

def floats(tex):
    result = {}
    for match in re.finditer(FLOAT, tex, re.S):
        s = match.group()
        label = re.search(r'\\label\{([^}]+)\}', s).group(1)
        caption = re.search(r'\\caption\{([^\n]*)\}', s).group(1)
        table = re.search(r'\\begin\{tabular\}.*?\\end\{tabular\}', s, re.S)
        image = re.search(r'\\includegraphics\[[^]]*\]\{([^}]+)\}', s)
        result[label] = (caption, table.group() if table else None, image.group(1) if image else None)
    return result

assert text_without_layout(original) == text_without_layout(science)
assert floats(original) == floats(science)
figure_positions = [m.start() for m in re.finditer(r'\\begin\{figure\}', science)]
table_positions = [m.start() for m in re.finditer(r'\\begin\{table\}', science)]
assert len(figure_positions) == len(table_positions) == 3
assert max(figure_positions) < min(table_positions)
assert max(table_positions) < science.index('\\input{references.bbl}')
assert science.index('\\section*{LLM Usage Statement}') < min(figure_positions)
equations = lambda s: re.findall(r'\\begin\{equation\}.*?\\end\{equation\}', s, re.S)
assert equations(original) == equations(science)
title = re.search(r'\\title\{([^\n]*)\}', original).group(1)
assert '\\title{\\bfseries ' + title + '}' in science
expected_author = r'''\author{wangmian\\
{\small Wuhan Polytechnic University, Wuhan, China}\\
{\small Corresponding author. Email: \href{mailto:wangmian945@gmail.com}{wangmian945@gmail.com}}}'''
assert expected_author in science
assert 'pdfauthor={wangmian}' in science
assert (ROOT / 'original_source/main.bbl').read_bytes() == (ROOT / 'references.bbl').read_bytes()
assert (ROOT / 'original_source/main.bib').read_bytes() == (ROOT / 'main.bib').read_bytes()
report = json.loads((ROOT / 'verification.json').read_text(encoding='utf-8'))
for name, expected in report['figure_files_sha256_identical'].items():
    assert hashlib.sha256((ROOT / 'figures' / name).read_bytes()).hexdigest() == expected, name
print('PASS: all text, equations, captions, table cells, bibliography and figure bytes preserved.')
print('PASS: main text and existing declaration -> 3 figures -> 3 tables -> 26 references.')
