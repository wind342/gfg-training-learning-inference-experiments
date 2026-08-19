# 目标表示—竞争边界共同旋转诊断协议 v1

## 目的

检验全网络一阶JVP遗漏的困难响应，是否主要来自目标隐藏表示与竞争边界在同一次有限参数更新中的共同旋转，以及隐藏表示自身沿该更新方向的二阶曲率。

## 时间边界

每个坐标只能使用：更新前参数状态、当前评价输入和已经形成的实际参数更新。不得读取 alpha>0 的真实响应、更新后的参数、最终margin或最终正确性来建立坐标。

本轮通过更新前模型上的自动微分计算：

- 隐藏表示 `h`；
- 隐藏表示方向变化 `h' = D h[delta_theta]`；
- 隐藏表示方向二阶变化 `h'' = D^2 h[delta_theta, delta_theta]`；
- 输出边界 `b_c = w_y - w_c`；
- 输出边界变化 `b'_c = delta_w_y - delta_w_c`。

对于每个目标与23个竞争者，冻结四种端点近似：

1. `linear = gap + J delta_theta`；
2. `joint_rotation = linear + b'_c dot h'`；
3. `hidden_curvature = linear + 0.5 b_c dot h''`；
4. `quadratic_complete = linear + b'_c dot h' + 0.5 b_c dot h''`。

主方法预先固定为 `quadratic_complete`。只有它相对 `linear` 在开发运行和确认运行的311例上都取得正净修复，才判定共同旋转与隐藏曲率形成可重复改善。其余两种方法仅用于机制消融，不得事后替换主方法。

## 完整性门

- alpha=0隐藏表示重建的logits必须与冻结响应一致；
- 当前竞争边界必须与既有完整竞争边界坐标一致；
- 由 `h`、`h'`、`b`、`b'`重建的一阶gap JVP必须与既有全网络JVP一致；
- 所有坐标必须有限；
- 记录身份、目标身份和竞争者排序必须精确对齐。

本轮为 `POST_HOC_MECHANISM_DIAGNOSTIC_ONLY`。二阶自动微分量用于判断机制，不自动升级为正式在线预测输入。
