"""Render audited results without changing sealed result bytes."""
from . import run


def main():
    b=run.b;r=b.read(run.HERE/'RESULTS.json');a=b.read(run.HERE/'AUDIT.json')
    assert a['status']=='PASS' and a['result_sha256']==b.sha(run.HERE/'RESULTS.json')
    lines=['# 固定缩放与目标边界动态控制：严格配对对照','',
        '基线 `'+run.BASE+'`。仅新增本实验目录，原实验、main、正式冻结标签和旧证据不变。','',
        '## 判定','',r['verdict']['label'],'',
        '大步长控制器定位与本轮证据一致：'+str(r['verdict']['aggressive_specific_role_consistent'])+'。',
        '判据预先封存：达标数不减、平均限制步数下降、至少两个历史获益，且逐一移除任一历史仍保持达标数不减和平均步数下降。',
        '这是已有 witness family 上的描述性稳健性要求，不是总体显著性检验。只比较四档常数，不声称胜过所有学习率或调度。','',
        '## 设置','',
        '同12个历史权重、原始有序批次和212个带标签控制/评估目标；fresh fused AdamW、零weight decay、norm1裁剪、FP32、三档cosine基础LR。',
        '固定α=1/.75/.5/.25与原动态控制器，动态每5步刷新；moments保持原生递推，不乘α。固定缩放在精确算术下等价于同比缩小LR，本实验保持native proposal后缩放的确切浮点顺序。',
        '留一只用于固定α选择，不是未见目标泛化；212个目标也被控制器使用。各LR都进行180份搜索中的相应部分及三次精确重放计时，总计324份计时记录。',
        '首次达到99%即停止，失败200步计入；95%为次要指标，初始已达标者分别排除。搜索成本另计。','']
    for lr,g in r['regimes'].items():
        lines+=['## 基础 LR '+lr,'',
            '|阈值|方法|达标/有效历史|平均计入步数|计算秒|含核验秒|',
            '|---|---|---:|---:|---:|---:|']
        for th,t in g['thresholds'].items():
            for arm in run.ARMS:
                s=t[arm];lines.append(f"|{th}|{arm}|{s['reached']}/{s['eligible']}|{s['restricted_steps_mean']:.3f}|{s['compute_seconds_mean']:.6f}|{s['checked_wall_seconds_mean']:.6f}|")
        lines+=['',f"99%动态 vs 留一固定：{g['thresholds']['0.99']['paired']}。",f"预定额外价值判据：{g['extra_value_supported']}。",f"动态缩放介入率：{g['intervention_rate']:.4%}；各α更新计数：{g['alpha_counts']}；刷新决策计数：{g['decision_alpha_counts']}。",f"有效99%历史固定α分布：{g['selected_fixed_eligible99_distribution']}；全部历史：{g['selected_fixed_distribution']}。",'',
            '|历史|阈值|native步数|LOO固定策略/步数|dynamic步数|native/fixed/dynamic含核验秒|','|---|---|---:|---|---:|---|']
        for th,t in g['thresholds'].items():
            for n,f,d in zip(t['native']['entries'],t['loo_fixed']['entries'],t['dynamic']['entries']):
                def h(x):return 'FAIL(200)' if x['hit'] is None else str(x['hit'])
                lines.append(f"|{n['entry']}|{th}|{h(n)}|{f['policy']} / {h(f)}|{h(d)}|{n['checked_wall_seconds']:.6f}/{f['checked_wall_seconds']:.6f}/{d['checked_wall_seconds']:.6f}|")
        lines+=['','全部12历史与五策略的first99（0表示初始已达标，None表示失败）：','', '|历史|fixed1|fixed075|fixed05|fixed025|dynamic|','|---|---:|---:|---:|---:|---:|']
        for e,v in g['all_fixed_first99'].items():lines.append('|'+e+'|'+'|'.join(str(v[p]) for p in run.POLICIES)+'|')
        lines+=['']
    lines+=['## 审计和成本','',f"独立审计PASS：{a['search_records']}份搜索、{a['timing_records']}份计时、{a['total_updates']}条真实更新；72条native/dynamic轨迹与旧基线精确对应。",f"新原始证据：`{run.OUT}`，{a['artifact_bytes']:,}字节。",'所有目标边界、动态决策、LOO选择、计时中位数、逐步哈希链和结论均从原始文件独立复算。参数与优化器的真实性通过实际native精确重放核验，不把哈希本身当作参数重算。',
        '计算时间含构造和同步更新，含核验时间另含评估、参数哈希和选择检查；均不含数据加载、预热和最终写盘。每历史取三次中位数，再对有效历史求均值。没有混入旧工程加速数字，也未测耗电。',
        'RESULTS保持生成时COMPLETE_PENDING_INDEPENDENT_AUDIT状态；后续AUDIT绑定其SHA-256，不回写旧结果。','',
        '搜索额外成本：','', '|策略|更新数|计算秒|含核验秒|','|---|---:|---:|---:|']
    for p,v in a['tuning_costs'].items():lines.append(f"|{p}|{int(v['updates'])}|{v['compute_seconds']:.3f}|{v['checked_wall_seconds']:.3f}|")
    lines+=['','## 运行','', '`python -m experiments.gfg_controller_fixed_shrinkage_control_v1.run`',
        '`python -m experiments.gfg_controller_fixed_shrinkage_control_v1.audit`',
        '输出采用独占创建，拒绝覆盖；重跑需新的输出位置与封存。RUN_SEAL包含本轮源文件及全部基线Python文件的哈希。',
        '独立审计和报告同样不覆盖已有收据。完整α序列、逐历史结果见RESULTS.json；原始logits、坐标和fact链在独立证据目录。']
    with (run.HERE/'README.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')


if __name__=='__main__':main()
