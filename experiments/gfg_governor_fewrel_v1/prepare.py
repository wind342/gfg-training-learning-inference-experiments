"""Download public data outside Git; pin sources and retain exact split identities."""
import hashlib
import json
import random
from pathlib import Path
import urllib.request
import numpy as np
from transformers import BertTokenizerFast

ROOT=Path('E:/gfg-evidence/governor-fewrel-v1')
MODEL=Path('E:/gfg-models/bert-base-uncased-v4')
CRL='bcc06732b2ebcc3c8d4adaa5c838b7ce03e5df20'
FEWREL='278a2315d2138810a379cd8d5718914dc56e2582'
SEED=2021


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(name,url):
    path=ROOT/'sources'/name;path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':'http://127.0.0.1:7890'}))
        with opener.open(url,timeout=120) as response, path.open('xb') as out:out.write(response.read())
    return path


def main():
    sources={}
    for repo,rev,names in [('thuiar/CRL',CRL,['README.md','config.py','run_continual.py','dataloaders/sampler.py','methods/backbone.py','datasets/id2rel.json','datasets/data_with_marker.json']),
                           ('thunlp/FewRel',FEWREL,['LICENSE','data/train_wiki.json','data/val_wiki.json'])]:
        for name in names:
            url=f'https://raw.githubusercontent.com/{repo}/{rev}/{name}'
            path=fetch(repo.split('/')[-1]+'-'+name.replace('/','-'),url)
            sources[str(path.name)]={'url':url,'sha256':sha(path),'bytes':path.stat().st_size}
    data=json.loads((ROOT/'sources/CRL-datasets-data_with_marker.json').read_text())
    labels=json.loads((ROOT/'sources/CRL-datasets-id2rel.json').read_text())
    assert len(labels)==80 and set(labels)==set(data)
    assert all(len(v)==700 for v in data.values())
    tokenizer=BertTokenizerFast.from_pretrained(MODEL,local_files_only=True)
    tokenizer.add_special_tokens({'additional_special_tokens':['[E11]','[E12]','[E21]','[E22]']})
    marker_ids=[tokenizer.convert_tokens_to_ids(t) for t in ['[E11]','[E21]']]
    shuffled=list(range(80));random.Random(SEED).shuffle(shuffled)
    order=np.argsort(shuffled).tolist()  # CRL sampler, not arbitrary reordered tasks
    packed=[];targets=[];splits=[];sample_ids=[];missing=[]
    for new_id,old_id in enumerate(order):
        samples=list(enumerate(data[labels[old_id]]));random.Random(SEED).shuffle(samples)
        encoded=tokenizer([' '.join(s['tokens']) for _,s in samples],padding='max_length',truncation=True,max_length=256)['input_ids']
        for j,((original,sample),tokens) in enumerate(zip(samples,encoded)):
            pos=[]
            for m in marker_ids:
                if tokens.count(m)!=1:missing.append([new_id,original,m])
                pos.append(tokens.index(m) if m in tokens else -1)
            packed.append(pos+tokens);targets.append(new_id);splits.append(0 if j<420 else 1 if j<560 else 2)
            sample_ids.append([labels[old_id],original])
    if missing:raise ValueError(f'missing/truncated/duplicate entity markers: {missing[:10]}')
    output=ROOT/'data.npz'
    if output.exists():raise FileExistsError(output)
    with output.open('xb') as f:np.savez_compressed(f,x=np.array(packed,dtype=np.int64),y=np.array(targets),split=np.array(splits))
    manifest=dict(sources=sources,seed=SEED,relation_order=[labels[i] for i in order],sample_ids=sample_ids,
        split='per relation CRL seeded shuffle:420 train/140 validation/140 test',tasks=10,relations_per_task=8,
        probe_rule='first 10 validation examples per seen relation; no test probes',data_sha256=sha(output),
        license_scope='FewRel MIT; CRL has no root LICENSE in inspected tree; no CRL source redistributed or executed',
        model_files={p.name:sha(p) for p in MODEL.iterdir() if p.is_file()},tokenizer_size=len(tokenizer))
    with (ROOT/'DATA_MANIFEST.json').open('x',encoding='utf8') as f:json.dump(manifest,f,indent=2)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('sample_ids','sources','model_files')},indent=2))


if __name__=='__main__':main()
