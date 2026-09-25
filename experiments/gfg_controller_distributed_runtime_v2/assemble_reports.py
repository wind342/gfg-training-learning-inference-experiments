"""Record preserved failures and descriptive engineering status, after outcomes."""
from .evidence import OUT,HERE,read,write,sha


def main():
    previous=OUT.parent/'attempt-1';seal=read(previous/'ARCHITECTURE_SEAL.json')
    assert all(sha(previous/'source_snapshot'/name)==identity for name,identity in seal['sources'].items())
    history={'attempt1':{'status':'PRESERVED_PROCESS_FAILURE','receipt':read(previous/'FAILURE_PROCESS.json'),
        'resolution':'New attempt2: singleton collectives are no-ops; all three FSDP1 oracle gates pass.',
        'artifacts':{str(p.relative_to(previous)):sha(p) for p in previous.rglob('*') if p.is_file() and p.suffix in ('.json','.npz','.py','.md')}},
        'windows_tcp_rendezvous':{'original_receipt_sha256':sha(OUT/'REAL_MULTIGPU_AUDIT.json'),
            'first_retry_sha256':sha(OUT/'LAUNCHER_RETRY_RESULTS.json'),
            'cause':'Installed Torch TCPStore default requests unavailable libuv; environment/config retry did not fix its constructor.',
            'resolution':'FileStore rendezvous plus disabled TCPStore sharing; all three runtime world1 entries return0.',
            'resolution_receipt_sha256':sha(OUT/'LAUNCHER_FILESTORE_RESULTS.json')},
        'supplemental_oracle_harness':{'status':'PRESERVED_INVALID_ORACLE_CONFIGURATION',
            'cause':'The added portable audit passed lr instead of cached alpha to the oracle. Production runtime and presealed gates/scale runs were unaffected.',
            'failed_source_sha256':sha(OUT/'supplemental-failed-source'/'distributed_gate.py'),
            'old_seal_sha256':sha(OUT/'DISTRIBUTED_GATE_SUPPLEMENT_SEAL.json'),
            'corrected_seal_sha256':sha(OUT/'DISTRIBUTED_GATE_CORRECTION_SEAL.json'),
            'resolution':'Correct cached-alpha argument; all three separate world1 reruns are BYTE_EXACT. Original mismatch data remain unchanged.'}}
    for rate in (.001,.003,.01):
        assert read(OUT/'distributed-oracle-gate-v2'/f'world1-lr{rate}'/'rank0'/'RESULT.json')['classification']=='BYTE_EXACT'
    assert all(r['returncode']==0 for r in read(OUT/'LAUNCHER_FILESTORE_RESULTS.json')['calls'])
    write(OUT/'failures'/'PRESERVED_HISTORY.json',history)
    gates=read(OUT/'GATES.json')['gates'];assert len(gates)==12 and all(r['all_exact'] for r in gates)
    synthetic=read(OUT/'SYNTHETIC.json')['results']
    workspaces={str(h):{mode:max(c['metadata']['workspace']['measured_peak_increment'] for r in synthetic if r['h']==h for c in r['cases'] if c['mode']==mode and c['dedup']) for mode in ('exact','scalarized','streaming')} for h in (4096,8192)}
    memory={s:read(OUT/'profiling'/f'{s}.json')['memory_bytes'] for s in ('70','160')}
    write(OUT/'ENGINEERING_STATUS.json',{
        'SCIENTIFIC_REFERENCE':'PASS','CANONICAL_PARAMETER_IDENTITY':'PASS',
        'GLOBAL_DISTRIBUTED_CLIPPING':'EMULATED_ONLY','STAGE_PACKED_FSDP':'WORLD_SIZE1_ONLY',
        'EXACT_DISTRIBUTED_BOUNDARY':'EMULATED_ONLY','SCALARIZED_BOUNDARY':'CANDIDATE',
        'FLASH_BOUNDARY':'CANDIDATE','REAL_MULTIGPU':'NOT_TESTED',
        '70M_TRANSPORT':'MIXED','160M_TRANSPORT':'NOT_ESTABLISHED',
        'SPARSE_CONTROL':'NOT_RUN','ENGINEERING_STATUS':'DISTRIBUTED_RESEARCH_PROTOTYPE',
        'transport_scope':'Both model sizes ran and dynamic reached99 in every tested formal seed. Labels concern the near-boundary advantage, not mere ability to execute.',
        'hardware':read(OUT/'ARCHITECTURE_SEAL.json')['hardware'],
        'largest_model_parameters':163633152,'largest_cuda_world_size':1,'cpu_collective_world_size':2,
        'single_gpu_persistent_bytes':memory,'boundary_peak_increment_bytes_N33_V73':workspaces,
        'communication':{'measured_network_bytes_world1':0,'real_multigpu_wire_bytes':'NOT_MEASURED',
            'delta_data_collectives_per_stage_world_gt1':1,'boundary_data_collectives_world_gt1':4,
            'full_vocab_gather':False,'exact_target_bytes':'4*N+8*U*H','scalarized_target_bytes':'36*N',
            'min_and_valid_payload_bytes':'32*N+4','native_fsdp_traffic':'additional, NOT_MEASURED'},
        'dynamic_vs_fixed':'70 SAFE:3/3 vs0/3; 70 NEAR:3/3 both but dynamic slower in every seed; 160 SAFE:3/3 vs2/3.',
        'time_to_capability':'Dynamic did not beat native/tuned-native in any tested formal regime.',
        'resource_limits':['Only one CUDA GPU; NCCL unavailable','No real CUDA world2 end-to-end gate',
            'No Triton fused kernel; pure PyTorch streaming candidate only','160M NEAR/SEVERE and70M SEVERE not established',
            'No qualifying NEAR benefit, so no Phase3/Pareto/preset recommendation',
            'Directional workspace isolated and multi-GPU wire timing not measured',
            'Only nanoGPT model-specific adapter and pure one-dimensional FSDP ownership tested',
            'Per-stage telemetry lists accumulate; long-run production bounding is not certified'],
        'profile_source_sha256':sha(HERE/'profile_diagnostic.py')})


if __name__=='__main__':main()
