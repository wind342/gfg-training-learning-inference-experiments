"""Package both completed v2 streams, reused baselines, and unchanged v1 evidence."""
from .common import *
from .run import history

def statistics(values):
    values=[float(x) for x in values if x is not None]
    return dict(n=len(values),mean=float(np.mean(values)),median=float(np.median(values)),min=min(values),max=max(values)) if values else None

def main():
    guard();ev=read(ROOT/'EVALUATION.json');audit=read(ROOT/'SEAL_AUDIT.json');assert ev['status']==audit['status']=='PASS'
    methods={};details={};cache=read(OLD/'CACHE.json');ledger=read(OLD/'OPERATOR_COST_LEDGER.json')
    for budget in (210,630):
        for kind in ('native','er'):
            name=f'{kind}_{budget}';rr=read(OLD/'formal'/name/'SEALED.json')['task_reports'];last=rr[-1]
            methods[name]=dict(metrics=ev['reused_baselines'][name][-1],reused=True,
                old_backward_examples=sum(r['old_ce_backward_examples'] for r in rr),old_replay_reads=sum(r['old_replay_reads'] for r in rr),
                old_forward_flops=sum(r['old_forward_dense_flops'] for r in rr),cutter_flops=0,
                total_accounted_flops=sum(r['total_dense_flops'] for r in rr),wall=sum(r['algorithm_wall'] for r in rr),
                storage_bytes=last['model_bytes']+last['optimizer_tensor_bytes']+last['old_bank_bytes'],
                old_backward_dense_flops=(budget*10*7553024 if kind=='er' else 0))
        name=f'jacobian_{budget}';rr=read(ROOT/'formal'/name/'SEALED.json')['tasks'];last=rr[-1];steps=[]
        for task in range(2,12):steps.extend(history(ROOT/('smoke' if task==2 else 'formal')/name/f'task{task}'/'STEPS.jsonl.gz'))
        rounds=[q for r in steps for q in r['decision']['rounds']];profile=defaultdict(float);flops=defaultdict(int)
        for r in rr[1:]:
            for k,v in r['regions_seconds'].items():profile[k]+=v
            for k,v in r['flop_ledger'].items():flops[k]+=v
        ce=rr[0]['train_dense_flops']+sum(r['current_ce_dense_flops'] for r in rr[1:]);cutter_cost=sum(v for k,v in flops.items() if k!='endpoint')
        methods[name]=dict(metrics=ev['arms'][name][-1],reused=False,old_backward_examples=0,old_replay_reads=0,old_backward_dense_flops=0,
            old_forward_flops=flops['endpoint'],cutter_flops=cutter_cost,total_accounted_flops=ce+sum(flops.values()),
            wall=sum(r['algorithm_wall'] for r in rr),control_wall=sum(r['control_wall'] for r in rr),
            storage_bytes=last['model_bytes']+last['optimizer_bytes']+last['bank_gpu_bytes']+last['bank_cpu_bytes'])
        details[name]=dict(tasks=rr,totals=next(a for a in audit['streams'] if a['budget']==budget),profile_seconds=dict(profile),flop_ledger=dict(flops),
            selected_controls_per_trigger=statistics([len(r['decision']['selected_controls']) for r in steps if r['decision']['triggered']]),
            final_pairs_per_trigger=statistics([len(r['decision']['final_active']) for r in steps if r['decision']['triggered']]),
            outer_per_trigger=statistics([r['decision']['outer_iterations'] for r in steps if r['decision']['triggered']]),
            sweeps_per_local=statistics([q['solver']['sweeps'] for q in rounds]),
            norm_ratio_triggered=statistics([r['norm_ratio'] for r in steps if r['decision']['triggered']]),
            cosine_nonzero_triggered=statistics([r['cosine'] for r in steps if r['decision']['triggered']]),
            jacobian_calls=sum(r['derivative_calls'].get('jacobian_batches',0) for r in rr[1:]),
            jacobian_pair_evaluations=sum(r['derivative_calls'].get('jacobian_pairs',0) for r in rr[1:]),
            endpoint_calls=sum(r['endpoint_calls'] for r in rr[1:]),endpoint_rows=sum(r['endpoint_rows'] for r in rr[1:]),
            padded_endpoint_rows=sum(r['padded_endpoint_rows'] for r in rr[1:]),
            correct_to_wrong=sum(r['correct_to_wrong'] for r in rr[1:]),wrong_to_correct=sum(r['wrong_to_correct'] for r in rr[1:]),
            linearization_error=statistics([q.get('linearization_max_error') for q in rounds]),
            actual_over_predicted_repair=statistics([q.get('repair_ratio') for q in rounds]))
    rr=read(OLD/'formal/ranpac/SEALED.json');last=rr['task_reports'][-1]
    methods['ranpac']=dict(metrics=ev['reused_baselines']['ranpac'][-1],reused=True,old_backward_examples=0,old_replay_reads=0,old_backward_dense_flops=0,
        old_forward_flops=sum(r['old_forward_dense_flops'] for r in rr['task_reports']),cutter_flops=0,
        total_accounted_flops=sum(r['total_accounted_flops'] for r in rr['task_reports']),wall=rr['setup_wall']+sum(r['algorithm_wall'] for r in rr['task_reports']),
        storage_bytes=last['sufficient_statistics_bytes']+last['projection_bytes']+last['classifier_bytes'],
        sufficient_statistics_bytes=last['sufficient_statistics_bytes'],rp_dimension=10000,
        solve_wall=sum(r['solve_wall'] for r in rr['task_reports']),statistics_update_wall=sum(r['statistics_update_wall'] for r in rr['task_reports']))
    write(ROOT/'SUMMARY.json',dict(status='PASS',methods=methods,controller_details=details,common_cache_wall=cache['cache_wall'],
        common_bert_cache_dense_flops=ledger['bert_train_cache_dense_flops'],common_pretrained_bytes=ledger['pretrained_parameter_bytes'],
        common_train_cache_tensor_bytes=cache['feature_bytes'],coverage='Covered dense lower bounds plus cutter/classical-solve estimates; not complete hardware FLOPs.',
        independent_task2_audit=read(ROOT/'AUDIT.json'),all_task_seal_audit=audit,expansion_audit=read(ROOT/'EXPANSION_AUDIT.json')))
    table=[]
    for name in ('native_210','er_210','jacobian_210','native_630','er_630','jacobian_630','ranpac'):
        m=methods[name];a=m['metrics'];table.append(f"| {name} | {a['old']*100:.2f} | {a['new']*100:.2f} | {a['all']*100:.2f} | {m['old_backward_examples']:,} | {m['old_forward_flops']/1e9:.2f} | {m['cutter_flops']/1e9:.2f} | {m['total_accounted_flops']/1e9:.2f} | {m['wall']:.2f} | {m['storage_bytes']/1e6:.2f} |")
    smoke=[];control=[];profile=[]
    for budget in (210,630):
        for root,label in ((V1,'v1'),(ROOT,'v2')):
            r=read(root/'smoke'/f'jacobian_{budget}'/'task2/CHECKS.json')
            smoke.append(f"| {label} {budget//210}x | {r['train_metrics']['current']['macro']*100:.2f} | {r['repair_rate']*100:.2f}% | {r['rollback']} | {r['floor_violations']} | {r['algorithm_wall']:.3f} | {r['status']} |")
        d=details[f'jacobian_{budget}'];t=d['totals']['totals'];p=d['profile_seconds']
        control.append(f"| {budget//210}x | {t['scheduled']:,} | {t['native_safe']:,} | {t['rollback']} | 0 | {d['jacobian_calls']:,} | {d['jacobian_pair_evaluations']:,} | {t['inner_sweeps']:,} | {d['endpoint_calls']:,} | {t['approximate_local_commits']:,} |")
        profile.append(f"| {budget//210}x | {p['endpoint']:.3f} | {p['geometry_factors_subregion']:.3f} | {p['gram_subregion']:.3f} | {p['budget_subregion']:.3f} | {p['hildreth_ball']:.3f} | {p['parameter_geometry']:.3f} | {p['reconstruction']:.3f} | {p['materialization']:.3f} | {methods[f'jacobian_{budget}']['control_wall']:.3f} |")
    expansion=[]
    for r in read(ROOT/'EXPANSION_AUDIT.json')['records']:
        expansion.append(f"| {r['budget']//210}x | {r['task']} | {r['correct_before_expansion']} | {r['correct_after_expansion']} | {r['lost_at_expansion']} | {r['zero_K_intents']}/{r['old_intents']} |")
    lines=['# Banking77 nonlinear V2: exact-check capped local candidates',
        '', '**v2 修复了 v1 过早回退：两档 task2 smoke 通过，随后按冻结协议完成两档11-task。** v1全部源码与证据保持原样。新版本允许有限且满足trust ball的内层上限候选进入实际FP32检查；实际floor安全则first-safe提交，否则重新线性化并继续原8轮外循环。没有放宽floor，没有old CE backward，也没有重新推进AdamW moments。',
        '', '**但正式能力验证没有成功达到“接近ER”的目标。** 3x最终All为12.73%，ER为35.75%；1x则为21.75%，ER为9.38%。更多训练预算在这条轨迹上反而更差。主要暴露的问题是任务入口扩类时旧能力先丢失、之后K随之降低；3x task10又在零训练update的扩类边界丢失212个旧正确controls。任务内floor=0 violations并没有覆盖这些损失。',
        '', '这是不同的 actuator 停止规则与新轨迹。未充分收敛的局部候选明确标记为近似QP结果，不能称为精确最优projection或byte-identical工程提速。rho、reserve、200内循环上限、数值容差和数据预算均未改变。',
        '', '## 最终测试结果', '',
        'Old=最终70个旧intent的macro，New=最后7个intent的macro，All=77个intent的macro。test只在两条完整新轨迹封存并审计后评价；此前旧Banking77 test结果已被观察，属于后续实验，不能宣称未见过此benchmark。Native、ER、RanPAC为同输入/同模型协议的已封存旧基线，哈希验证复用，没有重新选择结果。', '',
        '| 方法 | Old % | New % | All % | Old backward examples | Old forward/eval G | Cutter估计 G | 已覆盖总量 G | Algorithm wall s | 方法持久状态 MB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',*table,'',
        'FLOPs以G计。已覆盖总量包含dense multiply-add下界、cutter估计或RanPAC经典求解算术；未完整计入GELU、部分逐元素/归约、优化器或CV eigensolver指令，不是精确硬件Total FLOPs。旧controls的解析margin Jacobian计入cutter，并非“没有旧数据导数计算”。Old forward/eval对controller指actual endpoint；RanPAC是释放旧controls的RP变换。', '',
        f"共同的frozen BERT特征预计算另需 {cache['cache_wall']:.3f}s、{ledger['bert_train_cache_dense_flops']/1e12:.3f}T dense FLOPs；预训练参数 {ledger['pretrained_parameter_bytes']/1e6:.3f}MB、训练feature tensor {cache['feature_bytes']/1e6:.3f}MB。上表方法状态不重复这些公共项。",
        '', '方法持久状态：MLP含head和AdamW；ER加旧feature bank；controller加CPU H/abs(H)/input Gram/IDs/labels与GPU bank buffers；RanPAC含G/Q、RP矩阵与classifier。临时Jacobian因子、Gram/求解工作区、shadow head和峰值分配不在该静态列中，相关测量保存在各task CHECKS。',
        '', '旧基线wall为之前封存运行；新controller为本次带同步分区计时运行。此表不是同轮无插桩速度排名。算法时间包含current训练、controller与任务入口，不含证据hash/序列化、观察性训练指标、独立回放、缓存预计算和最终test。共享task1使用对应Native的封存时长。',
        '', '## 修复前后的task2 smoke', '',
        'New train为排除control bank的当前任务训练macro，不是test。通过门槛保持v1原样，1x/3x同时预声明，均未根据结果放宽。', '',
        '| 版本/预算 | New train % | Unsafe修复率 | Rollback | Floor violation | Algorithm wall s | Gate |',
        '|---|---:|---:|---:|---:|---:|---|',*smoke,'',
        'v1在cap直接回退；独立事后检查发现其51/66与237/237个上限候选实际上floor-safe。v2重新完整训练与审计，没有直接把这些旧候选计成新成绩。修复使更多更新执行，耗时未因此下降。',
        '', '## 扩类边界损失：floor=0 violations不能等同于跨任务不遗忘', '',
        '按原协议，K在seen-class expansion之后计算。共享的第一层改变后，尚未训练的输出行也可能对当前hidden activation产生较大logit；新类别进入竞争时，即使没有任何训练update，旧control也会立刻变错。这些损失发生在任务内floor建立之前。独立EXPANSION_AUDIT固定head逐一对照扩类前后，验证了这条边界。', '',
        '| 预算 | 进入task | 扩类前旧正确数 | 扩类后K总和 | 扩类损失 | K=0 intents / old intents |',
        '|---|---:|---:|---:|---:|---:|',*expansion,'',
        '特别是3x task6，旧control从153个正确降至0，所有35个旧intent的K都变成0，因此该task不触发cutter。此后任务内floor安全不能恢复这次入口损失。该结果保留且未修改扩类/K语义来隐藏；解决它需要另立跨类扩展的保护协议，不能冒充本版cutter工程修复。',
        '', '## 正式controller行为与工程成本', '',
        '以下统计tasks2–11；task1没有旧floor/cutter。Native-safe立即提交且不算Jacobian；correct→wrong允许被同intent的wrong→correct替代。K按每个task入口固定，跨task可重算，不做任务内ratchet。floor只保证有限control bank，不等于旧test能力不下降。', '',
        '| 预算 | Scheduled | Native safe | Rollback | Floor violations | Jacobian batches/outer rounds | Pair evaluations | Inner sweeps | Exact endpoints | 近似局部解安全提交 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',*control,'',
        '| 预算 | Endpoint s | Factors s | Gram s | Reserve s | Hildreth+ball s | 参数CPU几何 s | 重建 s | Materialize s | Control s |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',*profile,'',
        'Factors/Gram/Reserve是jacobian_gram_guard的子区间，不能与父区间重复相加。解析Jacobian采用结构化因子和小Gram，生产路径不展开pair×216653矩阵。输入/abs(input)、labels/IDs、relation rows、input Gram一次缓存；随参数变化的激活/Jacobian仍每轮重算。固定buffer256 input/hidden/output预分配，不逐tile填充复制；GELU仍分配小临时激活。没有transition screening或旧warm-QP替换。',
        '', 'SUMMARY保存norm ratio/cosine、selected/active规模、成员替换、linearization误差、cap检查与剩余QP残差、control bank reads等详细统计；逐step记录完整选择/active pairs、Gram/RHS/dual/head/moment哈希。不要将approximate-local commit解释成严格局部QP已收敛。',
        '', 'RanPAC固定RP=10000，独立累积统计式classifier adaptation，没有3x重复训练或old CE backward。充分统计为806.16MB；其表现不证明可训练MLP也能用相同统计量替代所有训练。ER每步16 current+16 old统一CE；old replay reads与backward examples、拆分估计见SUMMARY。',
        '', '## 验证与复现', '',
        '解析Jacobian/autograd、结构化Gram/apply/reconstruct、12个带球QP oracle、不可行球、fixed256、shadow/in-place AdamW和强制cap-safe提交数值检查全部通过。task2的全部840步用独立in-place AdamW程序回放，head/moments/counter/决策与独立functional endpoint完全一致。正式tasks2–11的全部8400步检查floor、native-safe无Jacobian、提交来源、trust-ball、checkpoint链与最终optimizer counter；这部分为证据链审计，不冒称重复执行全部formal轨迹。',
        '', '所有旧原始authority与v1源码/证据逐文件哈希不变。BERT模型文件hash复核，使用已验证CLS缓存。新数据源、seed、类序和reserved control bank均未改变；旧源码不作为动态训练依赖。',
        '', '创建独立新工作副本，设置 GFG_JACOBIAN_V2_ROOT 为不存在目录；按顺序执行本包 preflight、numerics、run、audit、formal、seal_audit、expansion_audit、evaluate、report（python -B -m experiments.gfg_banking77_jacobian_hildreth_v2.<module>）。输出create-only，不覆盖封存目录。外部数据/cache/checkpoints位置及hash见ARTIFACT_MANIFEST；compact结果不包含原始文本或完整模型。',
        '', '基线提交 aaf66dbfb8f28c890b2645f22992226becc9f273。本目录为v2；相邻v1目录保留失败smoke及其诊断。']
    with (HERE/'README.md').open('x',encoding='utf8',newline='\n') as f:f.write('\n'.join(lines)+'\n')
    destination=HERE/'results';destination.mkdir(exist_ok=False);artifacts={}
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(ROOT);artifacts[str(rel).replace('\\','/')]=dict(bytes=p.stat().st_size,sha256=sha(p))
        if p.suffix in ('.json','.gz'):
            out=destination/rel;out.parent.mkdir(parents=True,exist_ok=True)
            with out.open('xb') as f:f.write(p.read_bytes())
            assert sha(out)==sha(p)
    with (HERE/'.gitattributes').open('x',encoding='utf8',newline='\n') as f:f.write('results/** -text\n*.py text eol=lf\n*.md text eol=lf\n')
    write(HERE/'ARTIFACT_MANIFEST.json',dict(base=BASE,authority=str(ROOT),artifacts=artifacts,
        sources={p.name:sha(p) for p in HERE.iterdir() if p.is_file()},old_evidence_hashes=read(ROOT/'FREEZE.json')['old_evidence_hashes'],
        v1_authority_hashes=read(ROOT/'FREEZE.json')['v1_authority_hashes'],prior_files_unchanged=True))
    print('v2 full report and compact evidence created',flush=True)

if __name__=='__main__':main()
