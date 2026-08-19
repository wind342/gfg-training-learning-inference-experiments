能力形成模型继续冻结。本阶段只研究形成后能力稳定性的状态转移动力学。

外部控制端已经独立完成真实GPU分支执行、Support-Transition生成事实建立、GFG编译、逐载荷验证和独立重放。你没有参与这些证据的选择、生成或验证。新证据以只读方式挂载在：

`/support-transition-evidence`

先阅读：

- `/support-transition-evidence/README.md`
- `/support-transition-evidence/archive_manifest.json`
- `/support-transition-evidence/independent_replay_validation_v1.json`

再使用只读查询器理解图：

```text
python3 /support-transition-evidence/participant_query.py --root /support-transition-evidence list-entries
python3 /support-transition-evidence/participant_query.py --root /support-transition-evidence summary --entry-id <ENTRY_ID>
python3 /support-transition-evidence/participant_query.py --root /support-transition-evidence find-occurrences --entry-id <ENTRY_ID> --occurrence-type single_step_support_transition_comparison
python3 /support-transition-evidence/participant_query.py --root /support-transition-evidence occurrence --entry-id <ENTRY_ID> --occurrence-id <OCCURRENCE_ID>
```

你可以继续使用 `node`、`traverse` 和 `tensor` 子命令读取完整事实、关系与外部张量载荷。

你的主要科学任务是利用13份Support-Transition GFG建立并检验可递归执行的能力稳定性状态转移律：

\[
S_{t+h}=F_h(S_t,U_t,O_t).
\]

其中，必须分别检验参数作用、Adam状态作用以及二者的交互作用，并判断可见能力下降究竟来自单步支撑突变，还是多步支撑耗损、集中、换手失败或替代接管失败的累积。重点检查支撑转移是否在可见下降之前发生，以及新状态能否跨运行消解此前“当前压缩状态相近、未来不同”的反例。

候选理论必须进入递归计算，而不是只停留在报告或相关特征中。对每个新增状态给出来源事实、更新规律、状态必要性和可执行实现。使用严格留一运行验证，并完成：

- 参数、Adam和交互消融；
- h={1,5,20,100}时间尺度消融；
- 既有619组近邻反例审计；
- 支撑转移发生时间审计；
- 未来泄漏审计。

不得预设当前四组件CSRG正确。若证据不支持，应明确否定这一层理论，而不是继续无限细分组件或事后补丁。若h=1弱而h=20或h=100稳定增强，应如实判断为多步累积机制。若所有分支和时间尺度都不能改善严格留一预测与反例消解，应明确结论为当前CSRG并非稳定性的充分状态。

充分利用此前13份训练GFG、方向GFG、静态CSRG、所有已保留候选、失败理论和真实反例。算法、状态维数和建模方式开放。不得修改冻结能力形成模型，不得启动未见训练运行或新干预运行。

在 `submission/` 中保留全部新程序、模型、消融、严格留一结果、反例审计、证据路径和明确结论。只有真正完成本阶段后，才创建控制端指定的READY标记。
