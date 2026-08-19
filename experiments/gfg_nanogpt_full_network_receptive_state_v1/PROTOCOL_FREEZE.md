# 全网络目标微观功能接收态诊断协议

## 证据地位

本实验使用已经完成并公开结果的12个nanoGPT运行，因此属于：

`POST_HOC_MECHANISM_DIAGNOSTIC_ONLY`

它检验一个明确假设：当前F1/F3/F5压缩状态相近但未来不同，是否主要因为不同目标对同一次实际参数更新具有不同的全网络方向接收结构。

JVP是额外执行的功能分析，不是原训练时自然形成的输入。它只能用于发现和证伪机制，不能直接登记为正式在线预测特征。

## 时间边界

每个区段只使用：

- 更新前模型参数；
- 已经形成的实际参数更新；
- 当前212个评价输入及目标身份；
- 更新前alpha=0 logits。

任何alpha>0 logits、更新后margin、最终正确性、响应类型或竞争者换手结果不得进入坐标计算。

alpha=-0.125与alpha=0.125只允许在坐标形成以后验证JVP方向，不得作为输入、邻居距离或特征选择材料。

## 接收矩阵

参数划分为九个互斥且完备的组件：

1. embedding/readout；
2. position embedding；
3. block 0 normalization；
4. block 0 attention；
5. block 0 MLP；
6. block 1 normalization；
7. block 1 attention；
8. block 1 MLP；
9. final normalization。

对目标e、错误竞争者c和组件b，计算：

\[
r_{e,c,b}
=
\nabla_{\theta_b}(\ell_y-\ell_c)^\top\Delta\theta_b.
\]

九个组件之和形成全网络一阶方向接收：

\[
r_{e,c}=\sum_b r_{e,c,b}.
\]

竞争者顺序只由alpha=0错误类别logit从高到低确定。类别身份不得作为数值坐标。

## 数值门

- 使用PyTorch math attention后端计算JVP，不修改模型参数或源证据；
- 更新前前向与记录alpha=0 logits的最大绝对差不得超过5e-5；
- 九组件必须覆盖每个命名参数且互不重叠；
- 组件和与完整方向的一致性由线性构造保证，并在首区段额外计算完整JVP核验；
- 全网络JVP与中心有限差分方向的相关系数每个区段不得低于0.98。

## 诊断比较

所有方法均从全部合法训练历史重新检索64个邻居，同一运行不得互为邻居：

- F1/F3/F5；
- F1/F3/F5 + 全网络总JVP；
- F1/F3/F5 + 九组件JVP；
- F1/F3/F5 + 当前竞争者差距 + 全网络总JVP；
- F1/F3/F5 + 当前竞争者差距 + 九组件JVP。

主要诊断方法为最后一种。裁决对象是最终正确/错误结果；曲线拟合不参与裁决。分别报告全部样本、严重冲突和冻结311例，以及历史development、历史confirmation和逐运行结果。
