"""One create-only task10 trajectory with isolated observers and references."""
from .core import *
from .tests import tests
import traceback


def source_hashes():
    paths = subprocess.check_output(['git', 'ls-files', 'experiments'], cwd=REPO, text=True).splitlines()
    files = [REPO/q for q in paths if q.endswith('.py')]
    files += list(HERE.glob('*.py'))+[HERE/'PROTOCOL.md']
    return {q.relative_to(REPO).as_posix(): sha(q) for q in sorted(set(files))}


def guard():
    subprocess.run(['git', 'diff', '--exit-code', BASE, '--', '.', ':(exclude)experiments/'+HERE.name], cwd=REPO, check=True, capture_output=True)
    assert subprocess.check_output(['git','rev-parse','paper-experiments-cross-system-feedback-release^{}'],cwd=REPO,text=True).strip() == '36dab5ce347dbbdac157ef23205f556606d18294'


class Observer:
    def __init__(self, screen, stream):
        self.s, self.stream, self.step = screen, stream, 0
        self.wall = 0.
        self.rows = []

    @torch.no_grad()
    def evaluate(self, state, selected, label):
        result = self.s.evaluate(state, selected)
        sync()
        start = time.perf_counter()
        s = self.s
        z = p.affine(state, s.h, s.seen)
        m = margins(z, s.y)
        actual = m > 0
        false_safe = torch.nonzero(result['certified'] & ~actual[:s.n]).flatten().cpu().tolist()
        sign_mismatch = torch.nonzero(result['correct'] != actual).flatten().cpu().tolist()
        fullC = counts(m[:s.n].cpu().numpy(), s.labels, s.groups)
        exact = result['exact_mask']
        difference = float((result['m'][exact]-m[exact]).abs().max()) if bool(exact.any()) else 0.
        wrong = torch.nonzero(~actual[:s.n]).flatten().cpu().numpy()
        ownm = result['m'][:s.n].cpu().numpy()
        fullm = m[:s.n].cpu().numpy()
        order_mismatches = 0
        for relation in range(s.groups):
            indices = wrong[s.labels[wrong] == relation].tolist()
            if sorted(indices, key=lambda i: (-float(ownm[i]), i)) != sorted(indices, key=lambda i: (-float(fullm[i]), i)):
                order_mismatches += 1
        certificate = result['certified'].cpu().numpy()
        row = dict(step=self.step, point=label, full_rows=s.n,
                   certified=int(certificate.sum()), frontier=int((~certificate).sum()),
                   exact_old_rows=int(exact[:s.n].sum()), selected=len(selected),
                   false_safe=false_safe, sign_mismatch=sign_mismatch,
                   count_match=result['C'].tolist() == fullC.tolist(), counts=fullC.tolist(),
                   max_margin_difference=difference, wrong_order_relation_mismatches=order_mismatches,
                   certified_bits=np.packbits(certificate).tobytes().hex(),
                   candidate_sha256=tensor_hash(state))
        self.stream.write(json.dumps(row, separators=(',',':'))+'\n')
        self.stream.flush()
        self.rows.append(row)
        sync()
        self.wall += time.perf_counter()-start
        assert not false_safe and not sign_mismatch and row['count_match'], 'SCREENING_MISMATCH_PRESERVED'
        return result


def warmup(h, state):
    # Unmeasured library initialization only; no gradients, control tuning or data changes.
    for _ in range(3):
        p.affine(state, h, 80)
        p.affine(state, h[:128], 80)
    sync()


def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip() == BASE
    guard()
    assert shutil.disk_usage(ROOT.parent).free > 5*1024**3
    ROOT.mkdir(exist_ok=False)
    p.setup()
    numerical = tests()
    write(ROOT/'NUMERICAL_PREFLIGHT.json', numerical)
    p.setup()
    assert read(INPUT/'CHECKS.json')['status'] == 'PASS'
    ckpath = INPUT/'v2_210/task9/CHECKPOINT.pt'
    cache = read(INPUT/'CACHE.json')
    paths = [ckpath, INPUT/'v2_210/task9/TRAIN_RESULT.json', INPUT/'v2_210/task10/CONFIG.json',
             INPUT/'v2_210/task10/STEPS.jsonl.gz', INPUT/'v2_210/task10/CHECKPOINT.pt',
             INPUT/'EVALUATION_TASK10.json', INPUT/'FEATURES.pt', INPUT/'TEST_FEATURES.pt',
             INPUT/'CACHE.json', INPUT/'CHECKS.json', DATA/'data.npz']
    inputs = {str(q): sha(q) for q in paths}
    assert inputs[str(ckpath)] == read(INPUT/'v2_210/task9/TRAIN_RESULT.json')['checkpoint_sha256']
    assert inputs[str(INPUT/'FEATURES.pt')] == cache['artifacts']['FEATURES.pt']
    assert inputs[str(INPUT/'TEST_FEATURES.pt')] == cache['artifacts']['TEST_FEATURES.pt']
    assert inputs[str(DATA/'data.npz')] == DATA_SHA
    pre = dict(status='PASS', base=BASE, sources=source_hashes(), inputs=inputs,
               runtime=runtime(), numerical_preflight_sha256=sha(ROOT/'NUMERICAL_PREFLIGHT.json'),
               backbone_sha256=cache['backbone_sha256'], created_unix=time.time())
    write(ROOT/'PREFLIGHT.json', pre)
    d = np.load(DATA/'data.npz')
    features = torch.load(INPUT/'FEATURES.pt', weights_only=False, map_location='cpu')
    F = features['features'].cuda()
    saved = torch.load(ckpath, weights_only=False, map_location='cpu')
    head, shadow, opt = make_state(saved)
    assert old.opt_counter(opt) == 1890 and opt.param_groups[0]['lr'] == 5e-5
    h, y, ids, n = bank(d, F, 10)
    assert n == 10072
    cfg = read(INPUT/'v2_210/task10/CONFIG.json')
    K = np.asarray(cfg['K'])
    initial_head, initial_moments = head_hash(head), moments_hash(opt)
    assert initial_head == cfg['head_start'] and initial_moments == cfg['moments_start']
    with gzip.open(INPUT/'v2_210/task10/STEPS.jsonl.gz','rt',encoding='utf8') as log:
        historical = [json.loads(v) for v in log]
    warmup(h, p.snapshot(head))
    sync(); tick = time.perf_counter()
    screen = Screen(h, y, n, 72, 80)
    screen.begin(p.snapshot(head), 1)
    sync(); setup_wall = time.perf_counter()-tick
    write(ROOT/'CONFIG.json', dict(K=K.tolist(), bank_ids=ids.tolist(), n=n, task=10, updates=210,
                                   head_start=initial_head, moments_start=initial_moments,
                                   screening_setup_wall=setup_wall))
    rows = []
    with gzip.open(ROOT/'CALLS.jsonl.gz','xt',encoding='utf8') as calls, gzip.open(ROOT/'STEPS.jsonl.gz','xt',encoding='utf8') as steps:
        observer = Observer(screen, calls)
        for step, current in enumerate(order(d,10), 1):
            observer.step = step
            sync(); start = time.perf_counter()
            _, loss = ce_grad(head, F[current], torch.from_numpy(d['y'][current]).cuda(), 80)
            norm = nn.utils.clip_grad_norm_(head.parameters(), 1.)
            assert torch.isfinite(norm) and np.isfinite(loss)
            before = p.snapshot(head)
            propose(head, shadow, opt)
            native = p.snapshot(shadow)
            sync(); common_wall = time.perf_counter()-start
            mh = moments_hash(opt)
            sync(); start = time.perf_counter()
            refresh = screen.begin(before, step) if step > 1 else None
            obs_before = observer.wall
            executed, report = screened_projection(before, native, screen, K, ids[:n], observer.evaluate)
            with torch.no_grad():
                for name, value in head.named_parameters():
                    value.copy_(executed[name])
                after = observer.evaluate(p.snapshot(head), set(report['selected_controls']), 'commit')
            assert np.all(after['C'] >= K)
            screen.finish()
            sync(); screened_wall = time.perf_counter()-start-(observer.wall-obs_before)
            assert moments_hash(opt) == mh and old.opt_counter(opt) == 1890+step

            # Pure paired reference: its outputs NEVER feed screen/controller.
            sync(); start = time.perf_counter()
            ref, _, reference = floor_projection(before, native, h, y, n, K, ids[:n], seen=80)
            with torch.no_grad():
                refactual = margins(p.affine(ref, h, 80), y).cpu().tolist()
            assert refactual == reference['executed']
            sync(); ref_wall = time.perf_counter()-start
            compare_keys = ('native_counts','native_deficits','projection_triggered','skip','selections','sweeps',
                            'sweep_count','final_active','final_duals','selected_controls','executed_counts','roundoff_budget_sha256')
            mismatches = [key for key in compare_keys if report[key] != reference[key]]
            same_head = equal(executed, ref)
            max_error = max(float((executed[k]-ref[k]).abs().max()) for k in ref)
            hist = historical[step-1]
            row = dict(step=step, current_ids=current.tolist(), loss=loss, optimizer_step=old.opt_counter(opt),
                       head_sha256=head_hash(head), moments_sha256=mh, report=report, refresh=refresh,
                       common_update_wall=common_wall, screened_control_wall=screened_wall,
                       reference_control_wall=ref_wall, observer_wall=observer.wall-obs_before,
                       historical_head_identical=head_hash(head)==hist['head_sha256'],
                       historical_moments_identical=mh==hist['moments_sha256'],
                       reference_head_identical=same_head, reference_max_head_error=max_error,
                       reference_mismatches=mismatches, geometry=p.geometry(before,native,executed))
            if mismatches or not same_head:
                row['reference_report'] = {key:reference[key] for key in compare_keys}
            steps.write(json.dumps(row, separators=(',',':'),allow_nan=False)+'\n'); steps.flush()
            rows.append(row)
            if step == 1 or step % 30 == 0:
                print('STEP',step,'hist',row['historical_head_identical'],'ref',same_head,
                      'cert',observer.rows[-1]['certified'],'sweeps',report['sweep_count'],flush=True)
    assert len(rows) == 210 and source_hashes() == pre['sources']
    assert all(sha(q)==v for q,v in inputs.items())
    guard()
    save(ROOT/'CHECKPOINT.pt', dict(head=p.cpu_state(p.snapshot(head)), optimizer=opt.state_dict()))
    write(ROOT/'REFRESHES.json', screen.refreshes)
    write(ROOT/'TRAINING_COMPLETE.json',dict(steps=210,head=head_hash(head),moments=moments_hash(opt),finished_unix=time.time()))
    # Test data only loaded after training is sealed.
    test = torch.load(INPUT/'TEST_FEATURES.pt',weights_only=False,map_location='cpu')
    identity = head_hash(head)
    with torch.no_grad():
        pred = head(test['features'].cuda()).argmax(1).cpu().tolist()
    assert head_hash(head) == identity and moments_hash(opt) == rows[-1]['moments_sha256']
    ev = dict(indices=test['indices'],predictions=pred,metrics=output_metrics(d,pred,10),readonly=True)
    original = read(INPUT/'EVALUATION_TASK10.json')['arms']['v2_210']
    ev['historical_predictions_identical'] = pred == original['predictions']
    write(ROOT/'EVALUATION.json',ev)
    cr = observer.rows
    ratios = [r['frontier']/n for r in cr]
    algorithm_wall = setup_wall+sum(r['common_update_wall']+r['screened_control_wall'] for r in rows)
    reference_wall = sum(r['common_update_wall']+r['reference_control_wall'] for r in rows)
    result = dict(status='COMPLETED',steps=210,bank_size=n,old_new_all=ev['metrics'],
                  algorithm_wall=algorithm_wall,paired_reference_algorithm_wall=reference_wall,
                  paired_speedup=reference_wall/algorithm_wall,checker_wall=observer.wall,
                  screening_setup_wall=setup_wall,reference_projection_wall=sum(r['reference_control_wall'] for r in rows),
                  original_full_rows=sum(n*(3+r['report']['sweep_count']) for r in rows),
                  screened_exact_rows=sum(r['exact_old_rows'] for r in cr),
                  anchor_refresh_rows=sum(r['rows'] for r in screen.refreshes),
                  relation_refreshes=sum(len(r['relations']) for r in screen.refreshes),
                  frontier_ratio=dict(median=float(np.median(ratios)),p90=float(np.quantile(ratios,.9)),max=max(ratios)),
                  false_safe=sum(len(r['false_safe']) for r in cr),floor_violations=0,
                  count_mismatch_calls=sum(not r['count_match'] for r in cr),
                  max_evaluated_margin_difference=max(r['max_margin_difference'] for r in cr),
                  wrong_order_relation_mismatches=sum(r['wrong_order_relation_mismatches'] for r in cr),
                  reference_head_identical_steps=sum(r['reference_head_identical'] for r in rows),
                  reference_decision_identical_steps=sum(not r['reference_mismatches'] for r in rows),
                  historical_head_identical_steps=sum(r['historical_head_identical'] for r in rows),
                  historical_moments_identical_steps=sum(r['historical_moments_identical'] for r in rows),
                  historical_predictions_identical=ev['historical_predictions_identical'],
                  total_sweeps=sum(r['report']['sweep_count'] for r in rows),
                  skip_steps=sum(r['report']['skip'] for r in rows))
    result['exact_row_reduction'] = 1-result['screened_exact_rows']/result['original_full_rows']
    result['gate'] = 'PASS' if result['reference_decision_identical_steps']==210 and result['historical_head_identical_steps']==210 else 'IDENTITY_MISMATCH_REQUIRES_REVIEW'
    write(ROOT/'RESULTS.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        if ROOT.exists():
            write(ROOT/f'FAILURE_{time.time_ns()}.json',dict(traceback=traceback.format_exc(),time=time.time()))
        raise
