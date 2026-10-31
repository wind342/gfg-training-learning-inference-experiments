"""Build and check the manuscript using Python 3 and pdfLaTeX."""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
compiler = shutil.which('pdflatex')
if not compiler:
    raise SystemExit('pdfLaTeX is not on PATH. Install TeX Live or MiKTeX, or use Overleaf.')
subprocess.run([sys.executable, str(ROOT / 'check_content.py')], cwd=ROOT, check=True)
build = ROOT / 'build'
build.mkdir(exist_ok=True)
for i in range(1, 4):
    command = [compiler, '-no-shell-escape', '-interaction=nonstopmode', '-halt-on-error', '-output-directory=build', 'main.tex']
    result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (build / f'pass_{i}.txt').write_bytes(result.stdout)
    if result.returncode:
        print(result.stdout.decode(errors='replace')[-5000:])
        raise SystemExit(result.returncode)
log = (build / 'main.log').read_text(errors='replace')
failures = ['Overfull', 'undefined citations', 'undefined references', 'Missing character', 'Fatal error', 'Label(s) may have changed']
for failure in failures:
    if failure in log:
        raise SystemExit(f'Build check failed: {failure}. See build/main.log.')
pdf = ROOT / 'TLIR_Science_Final.pdf'
shutil.copyfile(build / 'main.pdf', pdf)
print(f'PASS: three pdfLaTeX passes, resolved references, no overflow or missing characters.\n{pdf}')
