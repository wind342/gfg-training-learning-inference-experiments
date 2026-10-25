# TLIR Science LaTeX 稿件工程

分支：`codex/tlir-science-latex`  
目录：`manuscripts/tlir_science/`

本工程保留原八页 CVPR 科学论文的摘要、原有理论论证、实验数值和 10 个公式，并在第二节开头增补 162 词的原创实验仪器简介。2026-10-26 根据用户明确要求，完成期刊投稿版的格式调整，并加入已确认的 Funding、Competing interests 声明。**未改写摘要、研究发现顺序、四个动力学定义、原有实验段落或实验结论。**

## 本轮格式修改

- 单栏、12 pt、US Letter、1.5 倍行距的 Science 投稿版排版继续保留。
- 取消摘要后强制另起一页及标题编号。
- 删除 `conference page limit` 会议版遗留文字，增加 Fig. 1 与 Fig. 3 正文交叉引用。
- Fig. 2 的图注补齐 14 个子图的简短说明；三幅原图的文件内容完全不变。
- 参考文献按正文首次引用顺序编号：投稿版保留 25 条实际被引文献。未引用的 Geiger 2025 原条目仍保留在 `original_source/main.bbl` 和 `main.bib`，并未删除原始证据。
- 经作者确认，在正文末加入：
  - **Funding:** This research received no funding.
  - **Competing interests:** The author declares no competing interests.
- 保留 LLM Usage Statement、证据仓库链接及原作者信息。

## 本次仪器简介与排版修复

- 在第二节最前面新增 **Experimental Instruments and Evidence Control**（162 英文词），简要介绍 GFG、真实更新因果分叉、有限幅度路径、CSRG-4C 和身份对齐目标边界账本。明确仪器随测量问题逐步发展，并未预先假设理论成立。
- 清除图表之间的 5 处强制 `\\clearpage`；保留正文结束和文献表开始两处必要分页，让浮动体自行紧凑布局。
- 撤销 Table 1 中 `Diffusion U-Net` 与 `(AdamW)` 之间的强制换行，保留原始表格数据。
- 页面 QA 已由 GitHub Actions 编译后生成 `PAGE_QA.json`，记录逐页文本量并检测空白页；结果 PASS，18 页。

## 文件与溯源

- `main.tex`：当前可编辑的期刊排版稿。
- `references.bbl`：按首次引用顺序整理的参考文献；编译时直接引入，不运行 BibTeX。
- `figures/`：三幅原图，字节未修改。
- `original_source/`：冻结 CVPR 原始来源，用于逐项核对。
- `check_content.py`：验证科学文本、公式、图像哈希、表格与引文顺序。
- `verification.json`：最新源文件验证范围和当前 PDF 编译状态。
- `SUBMISSION_NOTES.md`：投稿元数据与仍未处理的材料。
- `formatting_changes.diff`：先前从 CVPR 到 Science 的转换差异快照（并非这次更新后的完整差异）。

**最终 PDF 已重新编译并提交：`TLIR_Science_Final.pdf`（18 页，Git blob SHA `eb84926d1a0179df2f2af0f0d33589890a2afbdd`）。`PAGE_QA.json` 已通过逐页检查，无空白或近乎空白的页面；Table 3 不再单独占据空旷页面。**

## 复核与生成 PDF

Overleaf：上传完整工程，并将顶层 `main.tex` 设为主文件；使用 pdfLaTeX。

本地执行：

```bash
python check_content.py
python build.py
```

`build.py` 将对内容进行检查、执行三次 pdfLaTeX 编译，并生成新的 `TLIR_Science_Final.pdf`。

## 仍待正式投稿前确认

期刊实际所需的独立补充材料、稿件类型，以及投稿系统要求的作者贡献等事项尚未在本轮补写或伪造。这里进行的是格式修订，不是重新撰写 Science 版科学正文。
