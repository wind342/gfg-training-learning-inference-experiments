"""Existing GFG writer, tensor store and GeneratedOrigin incidence semantics."""
from .common import *
from experiments.gfg_nanogpt_support_redundancy_v1.support_gfg import SupportGFGWriter, GraphRef

class Evidence:
    def __init__(self, root, row, branch, protocol_hash):
        self.root, self.row, self.branch = root, row, branch
        self.store = TensorStore(root / 'tensor-objects')
        self.writer = SupportGFGWriter(root / 'execution.sqlite3', root / 'tensor-objects',
            scope_id=identity({'entry': row['entry_id'], 'branch': branch, 'protocol': protocol_hash}),
            source_bundle_id=row['original_training_gfg']['bundle_manifest_sha256'], contract_sha256=protocol_hash,
            graph_schema=SCHEMA, block_schema=SCHEMA+'-block', manifest_schema=SCHEMA+'-manifest')
        self.protocol_hash = protocol_hash
        self.sequence = 0

    def object(self, key, payload, step, kind='analysis_result'):
        return self.writer.object(semantic_key=f'{self.branch}:{step}:{key}', role=key,
            optimizer_step=step, payload=payload, object_kind=kind)

    def generated(self, ref, step):
        origin = self.object('receiving_generated_origin', {'origin_kind':'GeneratedOrigin',
            'source_graph_schema': SCHEMA, 'source_object_id':ref.object_id,
            'source_content_sha256':ref.content_sha256, 'source_role':ref.role}, step, 'GeneratedOrigin')
        self.writer.relation('generated_origin_dependency', ref.object_id, origin.object_id,
            {'basis':'same_graph_generated_result_continuity'})
        return GraphRef(origin.object_id, origin.content_sha256, origin.role, 'generated_origin')

    def event(self, key, payload, sources, step):
        oid = self.writer.occurrence(occurrence_type=key, optimizer_step=step,
            transform_reference={'operation':key,'protocol_sha256':self.protocol_hash,
                                 'code_manifest_sha256':sha(HERE / 'CODE_MANIFEST.json')},
            payload={'branch':self.branch,'entry_id':self.row['entry_id'],'sequence':self.sequence})
        self.sequence += 1
        result = self.object(key, payload, step)
        self.writer.bind(oid, [(s,s.role) for s in sources], result)
        return result, oid

def load_ref(root, ref):
    path = root / ref['locator']
    require(sha(path) == ref['file_sha256'], 'TENSOR_FILE_HASH')
    a = np.load(path, allow_pickle=False)
    require(list(a.shape) == ref['shape'] and str(a.dtype) == ref['dtype'], 'TENSOR_SHAPE_DTYPE')
    require(hashlib.sha256(a.tobytes()).hexdigest() == ref['raw_tensor_sha256'], 'TENSOR_RAW_HASH')
    return a

def check_references(root, value, seen):
    if isinstance(value, dict):
        if 'locator' in value and 'raw_tensor_sha256' in value:
            key = identity(value)
            if key not in seen:
                load_ref(root, value)
                if 'layout' in value:
                    load_named_array(root, value)
                seen.add(key)
        for v in value.values():
            check_references(root,v,seen)
    elif isinstance(value,list):
        for v in value:
            check_references(root,v,seen)
