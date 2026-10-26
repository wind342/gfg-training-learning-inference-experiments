# TLIR Science LaTeX 稿件工程

分支：`codex/tlir-science-latex`  
目录：`manuscripts/tlir_science/`

本工程保留原八页 CVPR 科学论文的理论论证、实验数值和 10 个公式，新增 163 词实验仪器小节，并整理了 Science 期刊投稿所需的摘要与声明。**没有修改原有实验顺序、理论定义、实验段落和结果。**

## 本轮格式修改

- 单栏、12 pt、US Letter、1.5 倍行距的 Science 投稿版排版继续保留。
- 取消摘要后强制另起一页及标题编号。
- 删除 `conference page limit` 会议版遗留文字，增加 Fig. 1 与 Fig. 3 正文交叉引用。
- Fig. 2 的图注补齐 14 个子图的简短说明；三幅原图的文件内容完全不变。
- 参考文献按正文首次引用顺序编号：投稿版保留 25 条实际被引文献。未引用的 Geiger 2025 原条目仍保留在 `original_source/main.bbl` 和 `main.bib`，并未删除原始证据。
- 经作者确认，在正文末加入：
  - **Funding:** This research received no funding.
  - **Competing interests:** There are no competing interests to declare.
- 保留 AI 使用情况的完整事实性披露、证据仓库链接及原作者信息；投稿版将该声明移入 Acknowledgments。

## 本次仪器简介与排版修复

- 在第二节最前面新增 **Experimental Instruments and Evidence Control**（163 英文词），简要介绍 GFG、真实更新因果分叉、有限幅度路径、CSRG-4C 和身份对齐目标边界账本。仪器正文只介绍实验方法及测量功能，不出现 AI 创造、实验设计或执行归属；相关事实**仅在文末贡献与 AI 使用声明中披露**。
- 清除图表之间的 5 处强制 `\\clearpage`；正文末保留 `\\clearpage`、参考文献前使用 `\\newpage`，避免小表独占页面。
- 撤销 Table 1 中 `Diffusion U-Net` 与 `(AdamW)` 之间的强制换行，保留原始表格数据。
- 之前版本的 `PAGE_QA.json` 检查为 PASS（18 页）；当前投稿版摘要与文献修改后会重新生成页面报告。

## 投稿前最后整理（2026-10-26）

- 将摘要由原来的约 187 词整理为 **125 词、单段**的 Science Research Article 投稿版。全文论证、九项实验顺序、定量结果和理论定义不改；较长原摘要仍在 `original_source/main.tex`。
- 去掉正式稿中 Introduction 标题，符合 Science LaTeX 模板第一段不加标题的设置。
- 按 Science 模板重新排列：**正文 → 三图三表 → References and Notes → Acknowledgments**。声明位于 Acknowledgments 下，含 Funding、Author contributions、Competing interests、Data/code/materials availability、AI 使用。
- 25 条参考文献保留首次引用编号，扩展原来省略为 `et al.` 的作者并统一为 Science 风格；原始 `main.bib` 保持冻结。
- 在数据可用性声明中仅正式引用 GitHub 证据仓库，重点说明按科学主张组织、可机器快速审计；不将不完整覆盖 MoE/GNN 等的 Zenodo DOI 作为全文证据入口，不创建重复的 Supplementary Materials。
- `SUBMISSION_METADATA.md` 收录一句话总结和投稿类型建议。**Cover Letter 尚未撰写**；作者贡献声明已依据作者提供的旧稿核对，并修正四套仪器的 AI 发明归属。
- 当前有 3 幅图、3 张表；建议向 Science 投稿系统申请考虑 **extended online Research Article**，不能提前声称获得编辑批准。

## 全文审读与中文版

已完成当前英文 Science 主稿的逐节检查，并建立完整中文对照译稿：

- [**中文 PDF 与译稿入口**](translation_zh/README_ZH.md)（包含中文 PDF、LaTeX 源文件和逐页 QA）。
- [**中文全文审读记录**](translation_zh/FULL_TEXT_REVIEW_ZH.md)：核对九项 nanoGPT 实验、预测、五系统验证、GRU 闭环、公式、图表和真实 AI 贡献。
- 中英文 **10 个公式、34 次正文引文、3 图、3 表及交叉引用标签**对应一致。CSRG-4C 组级支撑措辞已按照冻结协议修正。
- 中文译稿为阅读和核对用途，不是正式英文投稿文件；原始图内英文标签不修改。

## 本轮精准概念纠正

- 摘要恢复训练、学习与能力转变的严格区分：学习是**分布式功能支撑持久重组**；能力状态变化发生在目标特异性读出边界跨越时。中英文同步修正。
- 引言末尾的核心关系名称按作者要求由 **TLI 改为 TLIR**。
- 本轮除此两处概念修正外不修改任何其他科学正文、公式、图表、实验数字与贡献声明。

## 文件与溯源

- `main.tex`：当前可编辑的期刊排版稿。
- `references.bbl`：按首次引用顺序整理的参考文献；编译时直接引入，不运行 BibTeX。
- `figures/`：三幅原图，字节未修改。
- `original_source/`：冻结 CVPR 原始来源，用于逐项核对。
- `check_content.py`：验证科学文本、公式、图像哈希、表格与引文顺序。
- `verification.json`：最新源文件验证范围和当前 PDF 编译状态。
- `SUBMISSION_NOTES.md`：投稿步骤与待确认事项。
- `SUBMISSION_METADATA.md`：一句话总结、稿件类型建议及证据 DOI。
- `formatting_changes.diff`：先前从 CVPR 到 Science 的转换差异快照（并非这次更新后的完整差异）。

**最终投稿版 PDF 已重新编译通过：`TLIR_Science_Final.pdf`，18 页；Git blob SHA：`88b4962290d11c046a4fa79e30cbb49a763201db`。逐页检查为 PASS，无空白或近乎空白页。稿件的数据/代码声明现在仅引用按科学主张组织、支持机器快速审计的 GitHub 证据仓库。**

## 复核与生成 PDF

Overleaf：上传完整工程，并将顶层 `main.tex` 设为主文件；使用 pdfLaTeX。

本地执行：

```bash
python check_content.py
python build.py
```

`build.py` 将对内容进行检查、执行三次 pdfLaTeX 编译，并生成新的 `TLIR_Science_Final.pdf`。

## 仍待正式投稿前确认

未另外制作补充材料文件，论文的证据入口仅为公开 GitHub 证据仓库。作者贡献及 AI 使用声明依据作者提供的真实分工调整；Cover Letter 已按要求延后。
