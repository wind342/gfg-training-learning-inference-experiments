# Banking77：仅验证 INNER_LIMIT 的真实 endpoint 裁决

**两档task2独立重跑与独立审计完成，实验止于smoke。** 唯一算法修改是将内层达上限后的立即回退改为重建当前候选并真实FP32检查：安全则接受，不安全则继续原外循环，8轮仍不安全才回退。

已有3e57e59中的相关结果在开展本次前已经知晓。本次是补齐独立审计与严格数据访问隔离的验证，不是新发现，也不是结果驱动的挑选重跑。保留旧v1/v2所有证据；仅新增本目录。

## 决策计数

INNER_LIMIT按局部solve/outer round计数，不是按scheduled step。cap-safe rate分母为全部INNER_LIMIT事件。

| Budget | Unsafe native | 普通PASS且实际安全 | INNER_LIMIT | Cap实际安全接受 | Cap不安全继续 | 最终rollback | Cap-safe rate | Floor violations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 210 | 208 | 125 | 100 | 83 | 17 | 0 | 83.00% | 0 |
| 630 | 618 | 497 | 121 | 121 | 0 | 0 | 100.00% | 0 |

完整事件满足 INNER_LIMIT = cap-safe接受 + cap-unsafe继续 + cap-unsafe外循环上限回退；普通PASS但实际不安全的继续事件另行记录。所有原始决策字符串保存在逐步日志。

## 训练指标与gate

以下为非control训练集macro，不是test。每档从与v1相同的对应task1 checkpoint重新训练，K与schedule hash相同。1x/3x的task1 checkpoint本来不同，不混为同一个checkpoint。

| Budget | Old train % | New train % | All train % | Current CE入口→结束 | 触发步norm ratio中位数 | 非零执行cosine中位数 | Gate |
|---|---:|---:|---:|---|---:|---:|---|
| 210 | 51.4938 | 58.8442 | 55.1690 | 2.686831 → 2.139460 | 0.884034 | 0.882970 | PASS |
| 630 | 66.8580 | 76.5460 | 71.7020 | 3.042068 → 1.483108 | 0.909822 | 0.909885 | PASS |

固定gate：unsafe≥20，floor violations0，总rollback≤10%，unsafe修复≥90%，new train macro≥50%，CE≤入口90%，触发步executed/native norm中位数≥0.1（含rollback零），old CE backward0，所有独立状态审计通过。没有在看到结果后调整阈值。

| Budget | correct→wrong | wrong→correct | Unsafe成功修复 | 独立cap候选检查 | 独立continuation检查 |
|---|---:|---:|---:|---:|---:|
| 210 | 5 | 5 | 100.00% | 100 | 99 |
| 630 | 0 | 0 | 100.00% | 121 | 0 |

## 工作量与成本

| Budget | Exact总调用 | Projected recheck | Commit recheck | Outer rounds | Inner sweeps | Jacobian calls | Jacobian pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| 210 | 728 | 307 | 210 | 307 | 48686 | 307 | 9002 |
| 630 | 1879 | 618 | 630 | 618 | 64285 | 618 | 11698 |

Exact总调用还包含每步native检查与一次入口检查。actual endpoint rechecks=projected+commit，完整拆分见CHECKS。

| Budget | Control wall s | Algorithm wall s | Endpoint GFLOPs | Cutter估计 GFLOPs | Old CE backward examples | Old CE backward FLOPs |
|---|---:|---:|---:|---:|---:|---:|
| 210 | 8.5122 | 9.4129 | 80.6303 | 15.9023 | 0 | 0 |
| 630 | 16.4705 | 19.1857 | 208.1102 | 21.3563 | 0 | 0 |

Algorithm wall含current更新、controller和入口；不含训练指标观察、step日志写入、运行前缓存校验或独立审计。controller内逐轮哈希/决策证据与同步计时开销包含在内。因此不把本次时间当成无插桩速度对比。各细分计时存于CHECKS，子区间不能与父区间重复求和。

Endpoint是dense multiply-add FLOPs；cutter沿用v1的解析导数/Gram/方向作用/coordinate/reconstruction估计，未完整覆盖GELU、逐元素、归约、AdamW等指令，不是精确硬件Total FLOPs。old CE backward=0不等于没有旧数据导数计算，解析margin Jacobian已计入cutter。

## 与冻结v1比较

| Budget | Rollback v1→本次 | Unsafe修复率 v1→本次 | Floor violations v1→本次 |
|---|---|---|---|
| 210 | 66 → 0 | 68.27% → 100.00% | 0 → 0 |
| 630 | 237 → 0 | 61.65% → 100.00% | 0 → 0 |

**这支持task2的高rollback主要来自“solver cap被当作functional failure”。** 接受与继续都是本次训练时的真实决策，没有用v1的post-hoc安全比例替代训练。它不证明局部QP已经严格最优、不证明所有后续任务几何都无问题，也不回答扩类边界或泛化能力；本次不扩展这些问题。

## 独立审计与访问边界

生产使用shadow AdamW；audit.py使用in-place AdamW，并独立实现selection、FP64几何/reserve、Hildreth/Dykstra、候选重建、FP32 endpoint、trust更新与停止状态机。审计不import或调用生产cutter/project/Geometry。逐轮核对coef/dual、current/candidate/endpoint/Gram/RHS哈希；每个cap还用显式dense Jacobian验证重建，显式大矩阵仅存在于独立审计。每个continuation和最终head均检查，moments只由一次native proposal推进。

生产geometry/solver/selection、模型/优化器/缓存/数值设置与冻结v1做AST逐项一致性检查。合成A/B/C分支测试与独立solver/endpoint/geometry验证先于smoke。current batch IDs每步确认只含task2非controls；生产cutter使用NumPy解析导数且处于no_grad，旧controls不进入CE或backward。

所有实验Python进程使用输入白名单，只读训练缓存、对应task1 checkpoint、必要训练元数据、v1两个task2 CHECKS及BERT文件。其他历史evidence、test数据与test结果的open被拒绝；没有递归哈希旧目录。模型/cache允许输入在前后复核hash。访问日志随各阶段保留；这是Python open审计，不冒称覆盖操作系统或CUDA库内部的全部文件IO。此前已知历史test结果的事实无法撤销，但本次进程没有访问test文件。

## 复现

Base: 3e57e593f0d4af4098dd203c72af2191739225b2。Python为 E:/gfg-python/governor-v4-bench/Scripts/python.exe。设置 GFG_CAP_SMOKE_ROOT 为新建输出目录路径；依次运行本包 preflight、numerics、run、audit、report。每个artifact采用create-only；既有目录拒绝覆盖。没有formal/evaluate入口。

完整外部authority与输入hash见ARTIFACT_MANIFEST，compact results只含指标/日志/哈希，不含原始文本、特征或checkpoint。缓存和解析低维因子路径保持v1，不为本次审计引入新的训练算法优化。
