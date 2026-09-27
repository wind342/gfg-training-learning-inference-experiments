# Banking77 nonlinear V2 Jacobian-Hildreth v1

**task2 smoke 未通过预注册门槛，已停止 tasks3–11。** 两档轨迹均保持有限 control bank 的 floor violation=0，旧 CE backward=0；但达到内循环上限后回退过多。没有改变 rho、reserve、容差、上限或 floor，也没有开展新的正式 ER/RanPAC 比较。

基线提交 `aaf66dbfb8f28c890b2645f22992226becc9f273`。所有新增文件位于独立目录，旧证据逐文件哈希复核。此前 Banking77 test 结果已被观察，本实验明确属于后续实验。本次未反序列化、评价 test；完整性审计仅读取旧文件字节计算哈希。

## task2 学习与安全

以下 Old/New/All 是**已到达任务的训练集 macro accuracy，排除 control bank**，不是测试集泛化能力。Native 对照的全部 210/630 个 head 和 moments 与原封存 Native 轨迹逐步一致。

| 方法 | Old train % | New train % | All train % | Algorithm wall s |
|---|---:|---:|---:|---:|
| Native 1x | 0.00 | 61.20 | 30.60 | 0.640 |
| Nonlinear V2 1x | 56.30 | 47.85 | 52.07 | 7.053 |
| Native 3x | 0.00 | 85.72 | 42.86 | 2.478 |
| Nonlinear V2 3x | 70.90 | 69.89 | 70.40 | 15.156 |

| 步数 | Unsafe native | 无回退修复 | Rollback | Floor violations | 触发步 executed/native norm 中位数 | Gate |
|---|---:|---:|---:|---:|---:|---|
| 210 | 208/210 | 142 (68.27%) | 66 (31.43%) | 0 | 0.8765 | FAIL |
| 630 | 618/630 | 381 (61.65%) | 237 (37.62%) | 0 | 0.8921 | FAIL |

固定门槛：至少20次 unsafe，修复率≥90%，总回退≤10%，新任务训练 macro≥50%，CE降至入口的90%以下，触发步执行位移比例中位数≥0.1，所有安全/状态检查通过。210步还未达到50%新任务训练 macro；630步学习指标通过，但修复率与回退率未过门槛。失败轨迹完整保留。

## 失败原因：严格内层停止条件挡住了实际安全的候选

全部303次回退均为200次内循环上限，没有 outer cap 失败。失败局部求解的球投影命中数均为0，不能将此次失败归因于 trust region 太小。

事后只读诊断把到达上限的候选 materialize，再用独立 FP32 endpoint 检查，**从未把这些诊断候选提交到训练**：

- 210步：66次上限回退，其中51个候选实际 floor-safe。
- 630步：237次上限回退，237个候选全部实际 floor-safe。
- 失败时线性不等式残差中位数分别约6.06e-7、2.65e-7；冻结接受标准为1e-8，另有完整周期位移标准。
- 各预算的首个与最大残差失败局部问题共4例，独立 SciPy SLSQP 均找到满足冻结 primal 容差的解，位移仅为 rho 的约20%–25%。这是局部可行性诊断，不是替代 solver 的训练速度实验。

因此这版失败主要暴露了“求解到严格收敛才允许 exact recheck”的代价。不能由这些事后候选推算修改后的完整学习轨迹或宣称已解决。下一版若开展，应独立预注册 solver 达上限时的真实 endpoint 检查/停止规则，保留原 floor，并重新完整验证。本版不改协议、不补跑挑结果。

## 工程实现与实测成本

保留原 actual native → relation floor → unsafe cutter → actual recheck → commit 工作链。解析 Jacobian 用 GELU 导数和低维因子构造 Gram；生产代码没有 pair×216653 的显式 Jacobian。输入、abs(input)、labels、IDs、relation索引和输入内积每 task 缓存一次。随参数变化的激活与 Jacobian 每轮重新计算，所有 active pairs 使用同一个线性化点。

endpoint 固定256行，input/hidden/output缓冲预分配，输入只填充一次；GELU 激活仍有一个小型临时张量，不宣称消除了全部库内或算子分配。native-safe 不计算 Jacobian。旧 controls 从未进入 CE loss 或 autograd backward；解析 margin Jacobian 属于旧数据导数工作，成本单独记录。

以下为单次、带同步分区计时的 smoke；不是去除 profiling 开销后的独立性能 benchmark，不报告最佳值或置信区间。算法 wall 包含 current 更新、controller、controller入口；hash、证据写入、训练指标观察和独立审计不计入。分区有嵌套，不能把所有字段直接相加。

| 步数 | Endpoint s | Jacobian factors s | Gram s | Reserve s | Hildreth+ball s | 参数CPU几何 s | 重建 s | Materialize s | Control wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 210 | 0.285 | 0.126 | 0.026 | 0.211 | 3.050 | 0.534 | 0.629 | 0.267 | 6.362 |
| 630 | 1.281 | 0.282 | 0.043 | 0.431 | 4.046 | 2.059 | 0.901 | 1.066 | 12.888 |

FLOPs 单位 G。current forward/backward、endpoint 为 dense multiply-add 下界；cutter 包含解析导数、directional apply、Gram、Hildreth、重建的已标记估计。GELU超越函数、部分逐元素运算、归约和 optimizer 指令未全部计数。最后一列只是已覆盖项合计，**不是精确 Total hardware FLOPs**。Native 的入口 old-bank observer 不计入 Native 方法成本。

| 方法 | Current forward G | Current backward G | Old CE backward G | Endpoint G | Cutter估计 G | 已覆盖项合计 G |
|---|---:|---:|---:|---:|---:|---:|
| Native 1x | 1.454 | 1.586 | 0 | 0.000 | 0.000 | 3.040 |
| Nonlinear V2 1x | 1.454 | 1.586 | 0 | 72.767 | 13.894 | 89.700 |
| Native 3x | 4.361 | 4.758 | 0 | 0.000 | 0.000 | 9.119 |
| Nonlinear V2 3x | 4.361 | 4.758 | 0 | 181.861 | 16.481 | 207.461 |

| 步数 | Head bytes | AdamW state bytes | Bank GPU buffers bytes | Bank CPU cache bytes | 全进程测得峰值GPU bytes |
|---|---:|---:|---:|---:|---:|
| 210 | 866,612 | 1,733,240 | 1,127,984 | 900,480 | 119,901,696 |
| 630 | 866,612 | 1,733,240 | 1,127,984 | 900,480 | 119,034,880 |

Bank字节为持久缓存；临时 active geometry/Gram 不在该静态列中。全进程GPU峰值含完整训练feature cache、shadow head和观察开销。共同的 frozen BERT/cache 文件大小与全部外部文件哈希见 ARTIFACT_MANIFEST。缓存并非零 storage。

## 检查与证据

数值检查：解析 Jacobian 对照随机输入 autograd最大误差6.66e-16；结构化 Gram/apply/reconstruct 对照显式矩阵通过；12个合成带球约束问题对照独立 SLSQP；不可行球例正确报告上限；固定buffer256与functional输出逐字节一致；native-safe无Jacobian；shadow与in-place AdamW逐字节一致。

合成 oracle 初次因 SLSQP 的1e-12目标容差触发 linesearch failure；在读取 smoke 结果之前，将**测试oracle**目标容差改成1e-10后通过。生产solver协议与容差没有因此改变。abs(input)缓存后又完整重跑数值检查，两份结果均保留。

独立 audit 使用单模型 in-place AdamW 重放原 shadow更新链，全部840个 executed heads、moments、counter和controller决策一致，并用独立 functional endpoint/逐类比较复核每步floor。初始K分别为[10,8,9,10,7,7,5]与[10,8,9,9,3,9,9]，任务内固定。210步有4次correct→wrong和4次wrong→correct；630步均为0，代码允许成员替换。

完整的 selected/active IDs、逐轮 reserve/误差/球命中、norm/cosine、head/moment/Gram/RHS/dual哈希在压缩STEPS中。汇总统计、Jacobian calls/pairs、endpoint/recheck/read数在 SUMMARY.json。所有 cap候选检查及4个局部QP的独立求解结果单独保存。

## 复现

Python: `E:/gfg-python/governor-v4-bench/Scripts/python.exe`，PyTorch/CUDA/设备版本见 FREEZE。需同版本外部旧证据 `E:/gfg-evidence/banking77-nonlinear-v1` 与 BERT模型；旧scalar实验源码无需导入。当前旧证据尚是外部依赖，compact package不能离线独立重建数据缓存。

设置 `GFG_JACOBIAN_ROOT` 为新的不存在目录，依次运行 `python -B -m experiments.gfg_banking77_jacobian_hildreth_v1.preflight`、`.numerics`、`.run`、`.audit`、`.diagnose`。所有输出create-only。`.report`会生成新 results/README/manifest，必须在新副本中运行，不覆盖当前封存包。没有启动正式11-task，也没有新的test比较。
