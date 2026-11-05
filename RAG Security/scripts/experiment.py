"""Checkpoint each attempt and freeze selection before opening held-out tasks."""
import gc
import sys
import time
import torch
import psutil
from rag_security.artifacts import ROOT, read, write, digest, source_identity
from rag_security.models import LocalModel
from rag_security.detectors import Screening
from rag_security.engine import attempt, CONTROLS

stage=sys.argv[1]
assert stage in ('pilot','development','evaluation','positions','ingestion','adaptive')
config=read(ROOT/'configs/study.json')
source=source_identity()
screen=Screening()
base_identity=digest([source,config,read(ROOT/'models.lock.json'),read(ROOT/'sources.lock.json'),read(ROOT/'reports/partitions.json'),read(ROOT/'reports/detector-freeze.json')])
freeze_path=ROOT/'reports/selection-freeze.json'
if stage in ('evaluation','positions','ingestion','adaptive'):
    freeze=read(freeze_path)
    assert freeze['source_identity']==source and freeze['experiment_identity']==base_identity,'Frozen implementation changed.'
split='development' if stage in ('pilot','development','adaptive') else 'evaluation'
tasks=read(ROOT/f'data/tasks/{split}.json')
attacks=read(ROOT/'data/attacks.json')[split]
defenses=list(CONTROLS) if stage=='development' else config['defenses'] if stage=='evaluation' else ['D0','D4'] if stage=='pilot' else ['D0',freeze['selected_defense']]
if stage=='evaluation' and freeze['selected_defense'] not in defenses:
    defenses.append(freeze['selected_defense'])
if stage=='pilot':
    tasks=[tasks[0],tasks[-1]]
if stage in ('positions','ingestion'):
    tasks=tasks[:2]+tasks[-2:]
if stage=='adaptive':
    # Bounded development-only best-of-three placement search. No payload search
    # observes the protected confirmation tasks.
    tasks=tasks[-2:]
results=[]
resources=[]
start=time.perf_counter()
for key in config['models']:
    model=LocalModel(key)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for task in tasks:
        compatible=[a for a in attacks if (task['domain']=='bipia-table') == a['source'].startswith('BIPIA')]
        cases=[None,*compatible] if stage in ('pilot','development','evaluation') else compatible[:2]
        positions=['start','middle','end'] if stage in ('positions','adaptive') else ['end']
        ranks=[0,1] if stage=='positions' else [None]
        for attack in cases:
            for position in positions:
                for rank in ranks:
                    baseline=None
                    for defense in defenses:
                        eid=digest([base_identity,stage,key,task,attack,position,rank,defense])
                        path=ROOT/f'data/traces/{stage}/{eid}.json'
                        if path.exists():
                            row=read(path)
                        else:
                            row=attempt(model,screen,task,attack,defense,base_identity,position,rank,'ingestion' if stage=='ingestion' else 'passage')
                            row.update(experiment_id=base_identity,attempt_id=eid,stage=stage)
                            if defense=='D0':
                                baseline=row
                            # D0 always precedes comparisons for a matched case.
                            if baseline is None:
                                raise RuntimeError('Missing paired reference.')
                            row['valid_attack']=bool(attack and baseline['valid_artifact'] and baseline['payload_visible'])
                            write(path,row)
                        if defense=='D0':
                            baseline=row
                        results.append(row)
        print(stage,key,task['id'],len(results),flush=True)
    resources.append({'model':key,'loading_seconds':model.load_seconds,'peak_gpu_allocated_bytes':int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,'process_rss_bytes':psutil.Process().memory_info().rss})
    del model
    gc.collect()
    torch.cuda.empty_cache()
summary={'stage':stage,'identity':base_identity,'attempts':len(results),'elapsed_seconds':time.perf_counter()-start,'resources':resources,'attempt_ids':[r['attempt_id'] for r in results]}
write(ROOT/f'reports/{stage}-run.json',summary)
if stage=='development':
    from collections import defaultdict
    by=defaultdict(list)
    for r in results:
        by[(r['model'],r['defense'])].append(r)
    candidates=[]
    for (key,defense),rs in by.items():
        if defense=='D0':
            continue
        attacked=[r for r in rs if r['attack_id']]
        clean=[r for r in rs if not r['attack_id']]
        score=sum(r['completion']-r['attack_success'] for r in attacked)/len(attacked)-.5*sum(r['rejected'] for r in clean)/len(clean)
        candidates.append({'model':key,'defense':defense,'score':score,'clean_completion':sum(r['completion'] for r in clean)/len(clean),'attacked_completion':sum(r['completion'] for r in attacked)/len(attacked),'asr':sum(r['attack_success'] for r in attacked)/len(attacked)})
    selected=max(candidates,key=lambda c:(c['score'],-len(CONTROLS[c['defense']]),c['clean_completion']))
    freeze={'experiment_identity':base_identity,'source_identity':source,'selected_model':selected['model'],'selected_defense':selected['defense'],'selection':selected,'candidates':candidates,'development_trace_identity':digest(summary['attempt_ids']),'confirmation_policy':'No tuning after this freeze; positions/ingestion are diagnostic only. Adaptive search uses development tasks and fixed payloads only.'}
    if freeze_path.exists():
        assert freeze==read(freeze_path)
    else:
        write(freeze_path,freeze)
    print('FROZEN',selected,flush=True)
print(summary['attempts'],summary['elapsed_seconds'],flush=True)
