# TLIR Science LaTeX 最终工程

已修正为正文及原有声明 → 图 → 表 → 参考文献。作者栏按用户授权写入 wangmian、Wuhan Polytechnic University, Wuhan, China 和 wangmian945@gmail.com；除此之外未改写论文。原始 PDF、先前交付版本和 GitHub 分支均未修改。

源仓库：<https://github.com/wind342/gfg-training-learning-inference-experiments/tree/tlir-cvpr2027-format/manuscripts/tlir_cvpr2027>

锁定提交：`effbe99b0b8299bc1e457a309339dc85d722c1f8`。

## 版式

参考 Science 作者 LaTeX 模板的单栏、12 pt Times 字体、US Letter 纸张、1 英寸页边距及 `\linespread{1.5}` 行距。图表集中置于文后，表题移至表格上方；文内文献编号使用斜体圆括号。参考模板：<https://www.overleaf.com/latex/templates/science-and-science-family-journals-template/ptkzkvxkznbh>，其来源指向 <https://www.science.org/content/page/preparing-manuscripts-using-latex>（该官方页面访问返回 403）。本地 newtx 依赖缺少 xpatch，采用模板列出的 mathptmx 字体替代方案。

“其它一个字不许改”优先于模板中的文字或结构建议，因此保留 Abstract、Introduction、References 等原有标题、章节编号、所有声明以及原参考文献条目的全部文字和编号。作者信息是用户后续明确授权的文字变更；未添加未知的资金或其它声明，未缩短摘要或正文。仍需作者提供的真实投稿信息和模板差异列在 `SUBMISSION_NOTES.md`。

CVPR 页眉和边栏审稿行号作为模板元素移除；分页、字体、标号外观和引用回链随新版式调整。学校英文名已通过 [学校官网章程第二条](https://www.whpu.edu.cn/xxgk/xxzc.htm) 核对。作者姓名按用户提供的连写小写形式保留，未擅自拆分或倒置姓名。

## 编译

Overleaf：上传整个 ZIP 工程，选择顶层 `main.tex` 为主文件，编译器选择 pdfLaTeX。

本地：安装 Python 3 和 TeX Live 或 MiKTeX 后运行：

```text
python build.py
```

脚本先检查内容，再编译三次并检查日志，输出 `TLIR_Science_Final.pdf`。中间文件和编译日志保存在 `build/`。也可直接使用 pdfLaTeX：

```text
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

参考文献直接载入冻结的 `references.bbl`；不要运行 BibTeX 重新生成文献，以免自动丢弃原稿中未在正文引用的第 7 条 Geiger 文献。`main.bib` 仅随包保留。原分支自带成品 PDF 有 25 条，而用户指定 PDF 和原 `main.bbl` 有 26 条，本副本保留 26 条。

## 内容核对

- 用原 `main.tex` 和原 `main.bbl` 重编译 CVPR 稿，提取出的全文与用户指定 PDF 完全一致。
- 正文源字符串在仅忽略空白和明确的版式命令后完全一致。
- 10 个公式源代码、3 幅图的图注、3 张表的表题及单元格内容完全一致。
- 3 幅图片文件逐字节保持，`main.bib`、`references.bbl` 逐字节保持。
- 新版 19 页；已完成全文渲染检查，无编译报错、溢出、缺字或未定义引用；姓名、学校、邮箱及邮件链接均已核对。
- 原文件哈希在任务结束时重新核对未变化。

`verification.json` 为核对结果，`formatting_changes.diff` 为包含作者栏授权变更的完整改动清单，`source_manifest.json` 为来源与哈希。`original_source/` 仅供比较，正式编译入口始终为顶层 `main.tex`。运行 `python check_content.py` 可重新执行源内容及图表顺序核对。
