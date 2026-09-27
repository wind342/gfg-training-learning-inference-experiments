# Banking77 nonlinear V2: exact-check capped local candidates

**v2 修复了 v1 过早回退：两档 task2 smoke 通过，随后按冻结协议完成两档11-task。** v1全部源码与证据保持原样。新版本允许有限且满足trust ball的内层上限候选进入实际FP32检查；实际floor安全则first-safe提交，否则重新线性化并继续原8轮外循环。没有放宽floor，没有old CE backward，也没有重新推进AdamW moments。

**但正式能力验证没有成功达到“接近ER”的目标。** 3x最终All为12.73%，ER为35.75%；1x则为21.75%，ER为9.38%。更多训练预算在这条轨迹上反而更差。主要暴露的问题是任务入口扩类时旧能力先丢失、之后K随之降低；3x task10又在零训练update的扩类边界丢失212个旧正确controls。任务内floor=0 violations并没有覆盖这些损失。

这是不同的 actuator 停止规则与新轨迹。未充分收敛的局部候选明确标记为近似QP结果，不能称为精确最优projection或byte-identical工程提速。rho、reserve、200内循环上限、数值容差和数据预算均未改变。

## 最终测试结果

Old=最终70个旧intent的macro，New=最后7个intent的macro，All=77个intent的macro。test只在两条完整新轨迹封存并审计后评价；此前旧Banking77 test结果已被观察，属于后续实验，不能宣称未见过此benchmark。Native、ER、RanPAC为同输入/同模型协议的已封存旧基线，哈希验证复用，没有重新选择结果。

| 方法 | Old % | New % | All % | Old backward examples | Old forward/eval G | Cutter估计 G | 已覆盖总量 G | Algorithm wall s | 方法持久状态 MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native_210 | 0.00 | 71.43 | 6.49 | 0 | 0.00 | 0.00 | 33.44 | 5.21 | 2.60 |
| er_210 | 3.21 | 71.07 | 9.38 | 33,600 | 14.54 | 0.00 | 63.84 | 6.65 | 4.76 |
| jacobian_210 | 17.54 | 63.93 | 21.75 | 0 | 1958.72 | 602.26 | 2594.41 | 169.97 | 18.00 |
| native_630 | 0.00 | 89.64 | 8.15 | 0 | 0.00 | 0.00 | 100.31 | 19.79 | 2.60 |
| er_630 | 30.71 | 86.07 | 35.75 | 100,800 | 43.61 | 0.00 | 191.51 | 19.84 | 4.76 |
| jacobian_630 | 5.86 | 81.43 | 12.73 | 0 | 5212.39 | 828.57 | 6141.27 | 362.08 | 18.00 |
| ranpac | 83.32 | 87.50 | 83.70 | 0 | 10.75 | 0.00 | 6121.95 | 21.99 | 843.04 |

FLOPs以G计。已覆盖总量包含dense multiply-add下界、cutter估计或RanPAC经典求解算术；未完整计入GELU、部分逐元素/归约、优化器或CV eigensolver指令，不是精确硬件Total FLOPs。旧controls的解析margin Jacobian计入cutter，并非“没有旧数据导数计算”。Old forward/eval对controller指actual endpoint；RanPAC是释放旧controls的RP变换。

共同的frozen BERT特征预计算另需 26.184s、223.830T dense FLOPs；预训练参数 440.450MB、训练feature tensor 30.729MB。上表方法状态不重复这些公共项。

方法持久状态：MLP含head和AdamW；ER加旧feature bank；controller加CPU H/abs(H)/input Gram/IDs/labels与GPU bank buffers；RanPAC含G/Q、RP矩阵与classifier。临时Jacobian因子、Gram/求解工作区、shadow head和峰值分配不在该静态列中，相关测量保存在各task CHECKS。

旧基线wall为之前封存运行；新controller为本次带同步分区计时运行。此表不是同轮无插桩速度排名。算法时间包含current训练、controller与任务入口，不含证据hash/序列化、观察性训练指标、独立回放、缓存预计算和最终test。共享task1使用对应Native的封存时长。

## 修复前后的task2 smoke

New train为排除control bank的当前任务训练macro，不是test。通过门槛保持v1原样，1x/3x同时预声明，均未根据结果放宽。

| 版本/预算 | New train % | Unsafe修复率 | Rollback | Floor violation | Algorithm wall s | Gate |
|---|---:|---:|---:|---:|---:|---|
| v1 1x | 47.85 | 68.27% | 66 | 0 | 7.053 | FAIL |
| v2 1x | 58.84 | 100.00% | 0 | 0 | 7.950 | PASS |
| v1 3x | 69.89 | 61.65% | 237 | 0 | 15.156 | FAIL |
| v2 3x | 76.55 | 100.00% | 0 | 0 | 17.699 | PASS |

v1在cap直接回退；独立事后检查发现其51/66与237/237个上限候选实际上floor-safe。v2重新完整训练与审计，没有直接把这些旧候选计成新成绩。修复使更多更新执行，耗时未因此下降。

## 扩类边界损失：floor=0 violations不能等同于跨任务不遗忘

按原协议，K在seen-class expansion之后计算。共享的第一层改变后，尚未训练的输出行也可能对当前hidden activation产生较大logit；新类别进入竞争时，即使没有任何训练update，旧control也会立刻变错。这些损失发生在任务内floor建立之前。独立EXPANSION_AUDIT固定head逐一对照扩类前后，验证了这条边界。

| 预算 | 进入task | 扩类前旧正确数 | 扩类后K总和 | 扩类损失 | K=0 intents / old intents |
|---|---:|---:|---:|---:|---:|
| 1x | 2 | 56 | 56 | 0 | 0/7 |
| 1x | 3 | 91 | 85 | 6 | 0/14 |
| 1x | 4 | 125 | 49 | 76 | 6/21 |
| 1x | 5 | 86 | 86 | 0 | 6/28 |
| 1x | 6 | 129 | 115 | 14 | 11/35 |
| 1x | 7 | 158 | 19 | 139 | 33/42 |
| 1x | 8 | 60 | 60 | 0 | 33/49 |
| 1x | 9 | 112 | 112 | 0 | 33/56 |
| 1x | 10 | 159 | 159 | 0 | 33/63 |
| 1x | 11 | 212 | 212 | 0 | 33/70 |
| 3x | 2 | 57 | 57 | 0 | 0/7 |
| 3x | 3 | 108 | 108 | 0 | 0/14 |
| 3x | 4 | 155 | 151 | 4 | 1/21 |
| 3x | 5 | 200 | 99 | 101 | 3/28 |
| 3x | 6 | 153 | 0 | 153 | 35/35 |
| 3x | 7 | 63 | 63 | 0 | 35/42 |
| 3x | 8 | 116 | 116 | 0 | 35/49 |
| 3x | 9 | 170 | 167 | 3 | 35/56 |
| 3x | 10 | 221 | 9 | 212 | 60/63 |
| 3x | 11 | 67 | 63 | 4 | 61/70 |

特别是3x task6，旧control从153个正确降至0，所有35个旧intent的K都变成0，因此该task不触发cutter。此后任务内floor安全不能恢复这次入口损失。该结果保留且未修改扩类/K语义来隐藏；解决它需要另立跨类扩展的保护协议，不能冒充本版cutter工程修复。

## 正式controller行为与工程成本

以下统计tasks2–11；task1没有旧floor/cutter。Native-safe立即提交且不算Jacobian；correct→wrong允许被同intent的wrong→correct替代。K按每个task入口固定，跨task可重算，不做任务内ratchet。floor只保证有限control bank，不等于旧test能力不下降。

| 预算 | Scheduled | Native safe | Rollback | Floor violations | Jacobian batches/outer rounds | Pair evaluations | Inner sweeps | Exact endpoints | 近似局部解安全提交 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x | 2,100 | 2 | 0 | 0 | 4,276 | 301,513 | 711,686 | 8,486 | 1,164 |
| 3x | 6,300 | 658 | 200 | 0 | 10,779 | 433,762 | 1,746,862 | 23,389 | 3,191 |

| 预算 | Endpoint s | Factors s | Gram s | Reserve s | Hildreth+ball s | 参数CPU几何 s | 重建 s | Materialize s | Control s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x | 10.647 | 4.389 | 1.191 | 4.195 | 81.025 | 7.117 | 14.952 | 12.067 | 160.260 |
| 3x | 29.087 | 8.792 | 1.720 | 9.071 | 141.443 | 19.729 | 32.754 | 32.348 | 334.020 |

Factors/Gram/Reserve是jacobian_gram_guard的子区间，不能与父区间重复相加。解析Jacobian采用结构化因子和小Gram，生产路径不展开pair×216653矩阵。输入/abs(input)、labels/IDs、relation rows、input Gram一次缓存；随参数变化的激活/Jacobian仍每轮重算。固定buffer256 input/hidden/output预分配，不逐tile填充复制；GELU仍分配小临时激活。没有transition screening或旧warm-QP替换。

SUMMARY保存norm ratio/cosine、selected/active规模、成员替换、linearization误差、cap检查与剩余QP残差、control bank reads等详细统计；逐step记录完整选择/active pairs、Gram/RHS/dual/head/moment哈希。不要将approximate-local commit解释成严格局部QP已收敛。

RanPAC固定RP=10000，独立累积统计式classifier adaptation，没有3x重复训练或old CE backward。充分统计为806.16MB；其表现不证明可训练MLP也能用相同统计量替代所有训练。ER每步16 current+16 old统一CE；old replay reads与backward examples、拆分估计见SUMMARY。

## 验证与复现

解析Jacobian/autograd、结构化Gram/apply/reconstruct、12个带球QP oracle、不可行球、fixed256、shadow/in-place AdamW和强制cap-safe提交数值检查全部通过。task2的全部840步用独立in-place AdamW程序回放，head/moments/counter/决策与独立functional endpoint完全一致。正式tasks2–11的全部8400步检查floor、native-safe无Jacobian、提交来源、trust-ball、checkpoint链与最终optimizer counter；这部分为证据链审计，不冒称重复执行全部formal轨迹。

所有旧原始authority与v1源码/证据逐文件哈希不变。BERT模型文件hash复核，使用已验证CLS缓存。新数据源、seed、类序和reserved control bank均未改变；旧源码不作为动态训练依赖。

创建独立新工作副本，设置 GFG_JACOBIAN_V2_ROOT 为不存在目录；按顺序执行本包 preflight、numerics、run、audit、formal、seal_audit、expansion_audit、evaluate、report（python -B -m experiments.gfg_banking77_jacobian_hildreth_v2.<module>）。输出create-only，不覆盖封存目录。外部数据/cache/checkpoints位置及hash见ARTIFACT_MANIFEST；compact结果不包含原始文本或完整模型。

基线提交 aaf66dbfb8f28c890b2645f22992226becc9f273。本目录为v2；相邻v1目录保留失败smoke及其诊断。
