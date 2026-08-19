# 实际更新目标边界预测实验合同 v1

## 唯一预测对象

本实验只预测一次真实训练更新完成后，每个具体评价目标最终为正确还是错误，以及由更新前后正确性形成的四类转移：

- `MAINTAIN_CORRECT`；
- `CORRECT_TO_WRONG`；
- `MAINTAIN_WRONG`；
- `WRONG_TO_CORRECT`。

不预测完整响应曲线，不预测CSRG，不预测支撑重组，不建立或使用困难样本子集。全部15,264个目标使用同一算法和同一裁决规则。

## 时间边界

预测时允许使用：

- 更新前完整参数状态 `theta_t`；
- 当前训练步骤已经形成的实际参数更新 `delta_theta_t`；
- 当前评价输入与正确目标身份；
- 在更新前模型上由自动微分形成的方向量。

预测建立并密封以后，才允许读取完整更新端点的真实目标结果进行裁决。

禁止将以下内容用于预测：

- alpha大于零的真实响应；
- 更新后参数或隐藏表示；
- 更新后logit、margin、正确性或四类标签；
- CSRG探针结果；
- 历史KNN邻居、run_id、record_id或绝对训练步。

本任务的准确名称是“实际更新已经形成后、目标响应发生前的目标级边界预测”。它不预测尚未形成的梯度或参数更新。

## 冻结算法

对目标 `y` 和每个竞争目标 `c`，在更新前状态建立：

```text
b_c  = w_y - w_c
b'_c = delta_w_y - delta_w_c
h    = hidden(theta_t, x)
h'   = D hidden(theta_t, x)[delta_theta_t]
h''  = D^2 hidden(theta_t, x)[delta_theta_t, delta_theta_t]
```

冻结四个消融，其中主算法预先固定为 `quadratic_complete`：

```text
linear             = b_c h + b'_c h + b_c h'
joint_rotation     = linear + b'_c h'
hidden_curvature   = linear + 0.5 b_c h''
quadratic_complete = linear + b'_c h' + 0.5 b_c h''
```

当且仅当目标相对全部竞争者的预测端点gap均大于零，预测更新后目标正确。四类转移必须从已知更新前正确性和预测更新后正确性机械派生，不得训练独立分类器。

## 数据与评测

使用既有12条真实nanoGPT运行中的15,264个目标更新实例。沿用此前冻结的开发/确认运行划分，仅用于报告跨运行重复性；本轮是既有算法的独立重执行，不得称为新的前瞻确认。

分别报告：

- 样本数、正确数和准确率；
- balanced accuracy；
- final-correct precision、recall和F1；
- 四类转移的混淆矩阵、每类召回率和宏平均召回率；
- `linear`、`joint_rotation`、`hidden_curvature`和`quadratic_complete`的完整消融；
- 每条运行的主算法结果；
- 相对linear的修复数、新增错误数和净修复数。

事实身份、参数版本、实际更新身份、评价输入身份、竞争者顺序、alpha=0重建、一阶JVP重建及所有源文件哈希必须通过严格检查。结果层不设置困难样本豁免，也不删除任何错误。

## 声明边界

本实验可以检验该算法能否预测紧邻的一次目标级学习结果。它不直接检验长期训练轨迹、尚未形成的更新、完整响应曲线数值或显式支撑分配。

