# TLIR Science 论文完整中文翻译版

这是一份用于作者通读、检查术语和科学叙事的 **完整中文对照译稿**，不是正式向 Science 提交的中文论文。正式论文仍以同一分支的 [英文 `main.tex`](../main.tex) 和英文 PDF 为准。

**正文仪器介绍中的 AI 发明归属已撤回，仅在文末贡献声明披露。**

## 最终成品

**中文对照 PDF 共 14 页，XeLaTeX 编译及逐页自动检查均通过：没有空白页、文字溢出或缺失字形警告。** 已浏览全部页面联系图，并逐页放大检查标题/摘要页、14 子图页、六系统验证矩阵及表格页、最终 AI 贡献声明页。Git blob SHA：`9b36b401fb2d5babdc28fa3079e49c4151e696ea`。

## 文件

- **[中文 PDF（完整翻译）](TLIR_Science_ZH.pdf)**：正文、十个原有公式、三幅原图、三张表、三幅图的完整中文图注、27 条参考文献和所有投稿声明。
- **[中文 LaTeX 主文件](TLIR_Science_ZH.tex)**：调用 `part1_zh.tex`、`part2_zh.tex`、`part3_zh.tex`；使用 XeLaTeX 与中文 Fandol 字体编译。
- **[完整审读记录](FULL_TEXT_REVIEW_ZH.md)**：英文正文与证据逐节核查，以及术语修正和可公开复核边界。
- **[自动对照检查程序](check_translation_zh.py)**：比较中英文公式、引文调用、标签、图表、关键数字与 AI 真实贡献归属。
- **[逐页 PDF 检查报告](PAGE_QA_ZH.json)**：编译页数、空白页、字体缺失或溢出检查。
- `qa_previews/`：为审读而生成的低清预览和部分页面图像的 Base64 文本，不是投稿材料。

## 翻译规则

1. 译文在学术表述上忠实于当前 Science 英文主稿，不加入新实验或修改结果。
2. nanoGPT 九项 TLI 实验、TL-P01 预测、五种其他架构、第七种 GRU 反馈闭环的**发现顺序保持不变**。
3. 原有 **10 个公式、38 次参考文献引用、3 图、3 表以及 18 个交叉引用标签**逐一对应，27 篇文献保持英文正式条目以便检索。
4. 所有原始图像文件保持原样，包括图中英文坐标与标签；相应中文图注已全部翻译。
5. ​实验仪器章节仅介绍仪器的功能与测量方法，**不插入 AI 发明或设计归属**。人类与 AI 的科研分工按作者要求只在文末贡献与 AI 使用声明中如实披露：AI 发明和实现四套实验仪器，并设计与执行实验。
6. 证据只引用完整的 [TLIR GitHub 证据仓库](https://github.com/wind342/tlir-research-evidence/tree/codex/tlir-paper-evidence-navigation-v1)，不把覆盖范围有限的其他归档表述为全论文的证据入口。

## 译稿中读者应特别注意

- Section 2 的 CSRG-4C 测量统一采用“注册目标组的功能支撑”表述，对应冻结协议中 23 个目标组的组级 q10 边界量。
- Section 4 保留了 RL-E05 严格时间判据未通过的结论。
- Section 5 的 21 组历史正反比较属于文献分析，不能被误解为 21 项新的原生实验；配套证据仓库已补充[完整 21 组矩阵](https://github.com/wind342/tlir-research-evidence/blob/codex/tlir-paper-evidence-navigation-v1/TLIR_HISTORICAL_STRESS_TEST_21.md)。

## 第五节同步更新

英文第五节现以四组代表性历史正反案例和三项有条件外推组织全文，中文译稿已逐段同步，并保留审稿人与独立研究者继续提出可证伪预测的公开邀请。另可查阅[完整 21 组历史矩阵](https://github.com/wind342/tlir-research-evidence/blob/codex/tlir-paper-evidence-navigation-v1/TLIR_HISTORICAL_STRESS_TEST_21.md)与[14 项 theory-first 研究记录](https://github.com/wind342/tlir-research-evidence/blob/codex/tlir-paper-evidence-navigation-v1/TLIR_THEORY_FIRST_14_DERIVATIONS.md)。中英文共对应 10 个原编号公式、38 次引用键、27 条正式参考文献。
