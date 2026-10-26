# 投稿信息与原文保留说明

排版和 LaTeX 工程已完成。以下信息在本次给定稿件中缺失或不能确认，未擅自写进论文。

| 项目 | 当前处理 |
|---|---|
| 作者、单位、通讯邮箱 | 已按用户明确提供的信息写入 wangmian、Wuhan Polytechnic University、wangmian945@gmail.com；学校英文名已核对官网章程。未添加未提供的学院或系。 |
| 资金来源、项目编号 | 用户已确认无资金资助，已在主稿中加入 Funding 声明。 |
| 作者贡献 | 未提供各作者贡献，不代填。 |
| 利益冲突 | 用户已确认无利益冲突，已在主稿中加入 Competing interests 声明。 |
| 致谢 | 未提供，不新增。 |
| 数据、代码与材料可用性 | 原稿的证据仓库链接及说明保持不变；本轮仅调整格式与经用户确认的声明，未重新运行实验。 |
| 补充材料 | 未提供拟投稿的补充材料文件或确定的目录，不凭空创建 Supplementary Materials 内容。 |
| 文章类型与篇幅 | 仅按 Science LaTeX 投稿模板排版；未选择或替作者确认投稿类型，未删减文字、图表或文献以满足篇幅限制。 |

前期转换曾优先遵循“其它一个字不许改”的约束，因此保留原文：Abstract、Introduction、References 等原有标题；原章节编号；文献条目的全部文字和编号。原稿的 LLM Usage Statement 保持原文和其在正文之后的位置，不擅自改写成其他声明。作者栏是本次用户明确授权的唯一文字替换，替换了旧匿名作者文字。

学校英文名依据：[武汉轻工大学官网《学校章程》第二条](https://www.whpu.edu.cn/xxgk/xxzc.htm)，明确为 Wuhan Polytechnic University；[学校国际交流与合作处英文简介](https://international.whpu.edu.cn/en/About_WHPU.htm)也使用该名称。

本项目使用单栏、12 pt Times 字体、US Letter、1 英寸页边距及模板行距；顺序已修正为正文及原有声明、图、表、参考文献。它是完整可编译的、保持原文的投稿排版工程，不等于已经补齐全部投稿信息。

依据：[Science and Science-Family Journals Template](https://www.overleaf.com/latex/templates/science-and-science-family-journals-template/ptkzkvxkznbh)，其中明确列出作者、单位、通讯作者以及资金、贡献、利益冲突和可用性声明位置。模板来源指向 [Science LaTeX 指南](https://www.science.org/content/page/preparing-manuscripts-using-latex)；该官方页面此前访问返回 403。

## 本轮已处理的格式事项（2026-10-26）

- 摘要及科学正文保留；删除“conference page limit”会议遗留表述。
- 删除强制的摘要后分页；章、节标题不显示数字；添加 Figure 1 和 Figure 3 的正文引用。
- Figure 2 图注增加 14 个子图的简要对应解释；原图文件完全不改。
- 参考文献按首次引用重新编号；未被引用的 Geiger 2025 条目从当前投稿文献表移出，但保留在 `original_source/main.bbl` 与 `main.bib` 中。
- Funding 与 Competing interests 声明已按用户确认加入。
- 旧版本 PDF 已在前轮从 GitHub Actions 编译更新；本次新增仪器介绍和浮动体优化后已经重新编译 PDF，页面质量检查 PASS（18 页、无空白页）。

## 新增仪器简介与页面修复

- 新增 162 词 **Experimental Instruments and Evidence Control**，说明 GFG 和四套原创仪器，保留发展顺序。
- 合并图表浮动排版，移除 5 处逐项 `\\clearpage`，正文结束保留一次 `\clearpage`，文献前使用 `\newpage`，避免最后小表独占一页。
- 取消 Table 1 中 Diffusion U-Net 优化器条目的显式换行；数据和其他表格文字不变。
- 其余实验段落、原摘要、10 公式和原图保持原样；新增小节仅介绍已存在的实验方法。


## Science 投稿项目处理结论（2026-10-26）

- **摘要：** 投稿版为 116 词，单段；原版可从 `original_source/main.tex` 恢复。
- **一句话总结：** `SUBMISSION_METADATA.md` 已单独准备，作为可直接复制到投稿表单的建议文本。
- **作者贡献：** 已在 Acknowledgments 中编写单作者的贡献说明草稿，作者本人需要审定措辞和实际角色；不把 AI 认定为作者。
- **资金 / 利益冲突：** 按用户确认保留在规范的 Acknowledgments 小标题下。
- **参考文献：** 当前 25 条按首次引用排列，统一 Science 风格并补齐文献作者名单；`main.bib` 是原始溯源数据，并非当前渲染参考文献的权威文件。
- **证据：** Data, code and materials availability 仅使用 GitHub 证据仓库入口，突出证据依论文科学主张组织、支持机器快速审计，并保留 `PUBLIC_EVIDENCE_MATRIX.md` 对可核验范围及缺失原始运行文件的准确说明；不将覆盖不全的 Zenodo 归档表述为整篇论文的证据权威。
- **投稿类型：** 现有三图三表共六个展示项；优先考虑 Science Research Article 扩展在线格式，具体以投稿系统与编辑判定为准。
- **Supplementary Materials：** 不制造重复的附件。公开证据矩阵中已有的原始权威文件可用性边界应保留。
- **Cover Letter：** 按作者要求最后处理，本轮不写。

## 投稿版 PDF 最终生成结果

- GitHub Actions 编译通过；当前 `TLIR_Science_Final.pdf` 为 **18 页**，文件 Git blob SHA `70d5ad5d82b7cecf18574080b1201394f27080c5`。
- `PAGE_QA.json` 逐页文本检查为 **PASS**，空白或近乎空白页 0；本次编译未报告 Overfull 或 Float too large 问题。
- 封面信仍按作者要求留待最后处理；作者贡献声明为待本人核定的草稿。

## 作者贡献与 AI 科研代理的真实归属（2026-10-26）

- 以作者上传的旧稿《Training, learning and inference: unified dynamics of neural systems》末尾 `AI-assisted scientific execution` 和 `Author contributions` 为事实来源。
- **明确纠正一项原旧稿的错误：四套实验仪器（CSRG-4C、真实更新因果分叉、有限幅度更新路径、身份对齐目标边界账本）由 AI 科研代理发明、设计和实现，而非人类作者发明。**
- 作者明确确认全部报告实验的设计、实现和执行均由 AI 完成；AI 声明不再使用弱化实际作用的 “assistance with experimental-protocol design”。
- 保留旧稿确定的人类责任：研究构想、生成事实及 GFG 理论基础、科研问题与研究边界、主要理论假设和概念转换、科学决策、证据审查与结论责任。
- 数值结果由真实程序和执行环境产生，不是语言模型编写出来的数字。
- 只修正具体贡献归属；不改变论文核心实验结果、公式、摘要、图表和证据仓库链接。Cover Letter 仍延后。
