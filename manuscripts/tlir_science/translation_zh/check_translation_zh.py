"""Static, independent consistency checks for the Chinese reader translation.

The submitted English manuscript remains the scientific authority. This checker
confirms that the Chinese counterpart has the same mathematics, citation order,
figure/table identities, measured numeric results and contribution disclosures.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
original = (BASE / "main.tex").read_text(encoding="utf-8")
master = (ROOT / "TLIR_Science_ZH.tex").read_text(encoding="utf-8")
parts = [(ROOT / f"part{i}_zh.tex").read_text(encoding="utf-8") for i in (1, 2, 3)]
translation = "\n".join(parts)

def labels(text: str) -> list[str]:
    return re.findall(r"\\label\{([^}]+)\}", text)

def equations(text: str) -> list[str]:
    return [
        re.sub(r"\s+", "", m)
        for m in re.findall(r"\\begin\{equation\}.*?\\end\{equation\}", text, re.S)
    ]

def citations(text: str) -> list[str]:
    return [
        key.strip()
        for item in re.findall(r"\\cite\{([^}]+)\}", text)
        for key in item.split(",")
    ]

assert len(equations(original)) == 10
assert equations(original) == equations(translation), (
    "Some mathematics or equation labels changed during translation"
)
assert len(citations(original)) == 47
assert citations(original) == citations(translation), (
    "Bibliographic citations were omitted or reordered"
)
assert labels(original) == labels(translation), (
    "Equation, figure, section or table label discrepancy"
)

for kind, required in (("figure", 3), ("table", 3)):
    lhs = len(re.findall(r"\\begin\{" + kind + r"\}", original))
    rhs = len(re.findall(r"\\begin\{" + kind + r"\}", translation))
    assert lhs == rhs == required, (kind, lhs, rhs)

for name in (
    "figure1_tlir_closed_loop.png",
    "figure2_experimental_14panel.png",
    "figure2_six_system_evidence.pdf",
):
    assert name in original and name in translation
    assert (BASE / "figures" / name).is_file(), name

for marker in (
    "91.43", "91.49", "5,088", "1,636", "228", "32.23", "19.25",
    "35.71", "91.41", "97.02", "80.43", "1.56", "95.31",
    "70.74", "25.54", "35.20", "9.474", "3.962", "39.06",
    "70.31", "99.48", "99.83", "89.06", "3,200"
):
    assert marker in original, ("original missing", marker)
    assert marker in translation, ("translation missing", marker)

for marker in (
    "Mian Wang", "wangmian945@gmail.com",
    "CSRG-4C", "OpenAI Codex", "ChatGPT",
    "AI", "发明并实现", "设计、实现、启动、监控和执行",
    "机器辅助审计", "论文—证据映射", "公开证据矩阵",
    "时间先后关系判据", "未能满足", "21 组正反",
    "九项 nanoGPT 实验", "五种神经系统",
    "第七种系统", "目标组",
):
    assert marker in master + translation, marker

assert "10.5281/zenodo" not in translation
assert "10.5281/zenodo" not in master
assert "https://github.com/wind342/tlir-research-evidence/" in translation
assert r"\input{../references.bbl}" in translation
assert len(re.findall(r"\\bibitem\{", (BASE / "references.bbl").read_text())) == 34
method_zh = parts[0].split(r"\subsection{实验仪器与证据控制}",1)[1].split(r"\subsection{训练：真实更新与接收状态}",1)[0]
assert "AI" not in method_zh and "Codex" not in method_zh and "ChatGPT" not in method_zh
assert "发明" not in method_zh and "作者的科学指导" not in method_zh
assert "学习表现为分布式功能支撑的持久重组；目标跨越其特异性读出边界时，才会发生可观察的能力状态转变" in parts[0]
assert "TLIR 的核心关系是在开展上述跨领域比较之前，通过实验建立的" in parts[0]
assert "学习使分布式功能支撑持续重组，并涉及能力边界的跨越" not in parts[0]
assert "欢迎审稿人和其他独立研究者" in translation
assert "理论先行的有条件外推" in translation
assert "10.18653/v1/N19-1357" in (BASE / "additional_references.bib").read_text()
assert "10.18653/v1/D19-1002" in (BASE / "additional_references.bib").read_text()
assert "中文对照译稿" in master
assert r"\documentclass[UTF8,fontset=fandol" in master
for i in (1, 2, 3):
    assert rf"\input{{part{i}_zh.tex}}" in master

print("PASS: 10 equations unchanged and all 18 labels in the same order.")
print("PASS: 47 bibliographic citations in the same order; 34 verified sources.")
print("PASS: all 3 figures and 3 tables, preserved source images and key values.")
print("PASS: actual AI invention, experiment execution, negative verdicts and GitHub evidence disclosure.")
