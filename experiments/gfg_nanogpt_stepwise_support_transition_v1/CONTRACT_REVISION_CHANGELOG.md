# 合同修订清单

> 本文件原先记录的V1修订已由V2根本定义修正取代。V2不再把405对称为一步反例或一步反例消解总体；当前权威草案是同目录的 `MINIMAL_ONE_STEP_COUNTEREXAMPLE_RESOLUTION_CONTRACT_DRAFT.md`，状态为 `AWAITING_USER_CONFIRMATION`，执行授权为 `false`。

1. 合同改名为“一步转移理论驱动的405对远期反例消解实验”。
2. 明确 405 对是 phase-start 近状态、h100 真实未来不同的远期正例；一步模型只负责相邻状态，远期通过在线逐步递归裁决。
3. 外层折从 `entry_id` 改为验证过的无序 `run_id` 对；58 个运行对覆盖全部 405 对。
4. 删除 405 正例总体上的 precision、recall 和 F1；本稿不建立匹配负对照。
5. 明确连续预测不是逐浮点完全相同，新增开发运行形成的 MAE、RMSE、NRMSE、逐坐标容差和相对基线最低改善门。
6. 冻结“更新已形成、目标 CSRG 尚未执行”的时间边界，并排除可由原生公式直接计算的参数/Adam 下一状态得分。
7. 将多步统一改称在线逐步递归预测，禁止提前读取未来 \(U\) 或用真实中间状态重置。
8. 新增按运行/运行对聚类的置信区间与显著性规则，并要求同时报告新增严格消解和新增错误方向。
9. 保持 `AWAITING_USER_CONFIRMATION` 与 `execution_authorization=false`；没有授权或启动任何实验。
