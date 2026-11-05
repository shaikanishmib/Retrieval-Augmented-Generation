"""Reconstruct every aggregate from the exact retained attempt IDs."""
from collections import defaultdict
import re
import numpy as np
from rag_security.artifacts import ROOT, read, write, digest
from rag_security.metrics import interval, wilson

def load(stage):
    path=ROOT/f'reports/{stage}-run.json'
    if not path.exists():
        return []
    run=read(path)
    rows=[read(ROOT/f'data/traces/{stage}/{eid}.json') for eid in run['attempt_ids']]
    assert len(rows)==len({r['attempt_id'] for r in rows})==run['attempts']
    assert all(r['experiment_id']==run['identity'] for r in rows)
    return rows

def aggregate(rs):
    attacks=[r for r in rs if r['attack_id']]
    valid=[r for r in attacks if r['valid_attack']]
    clean=[r for r in rs if not r['attack_id']]
    def mean(rows,key):
        return float(np.mean([r[key] for r in rows])) if rows else None
    result={'attempts':len(rs),'attack_attempts':len(attacks),'valid_attacks':len(valid),'successes':sum(r['attack_success'] for r in attacks),'strict_asr':mean(attacks,'attack_success'),'conditional_asr':mean(valid,'attack_success'),'asr_wilson_95':wilson(sum(r['attack_success'] for r in attacks),len(attacks)),'asr_group_bootstrap_95':interval([r['attack_success'] for r in attacks],[r['group'] for r in attacks]),'attacked_completion':mean(attacks,'completion'),'attacked_f1':mean([r for r in attacks if r['family']!='tool'],'answer_f1'),'clean_n':len(clean),'clean_completion':mean(clean,'completion'),'clean_f1':mean(clean,'answer_f1'),'clean_rejection_rate':mean(clean,'rejected'),'clean_completion_wilson_95':wilson(sum(r['completion'] for r in clean),len(clean)),'evidence_recall':mean(rs,'evidence_recall'),'disallowed_tool_attempts':sum(r['disallowed_tool_attempt'] for r in attacks),'actual_unauthorized_tool_execution':sum(r['unauthorized_tool_execution'] for r in attacks),'p50_seconds':float(np.quantile([r['latency_seconds'] for r in rs],.5)),'p95_seconds':float(np.quantile([r['latency_seconds'] for r in rs],.95)),'modeled_throughput_per_second':len(rs)/sum(r['latency_seconds'] for r in rs) if sum(r['latency_seconds'] for r in rs) else None,'generation_reused':sum(r['generation_reused'] for r in rs),'context_truncated_fraction':mean(rs and [dict(r, truncation=int(r['output']['context_truncated'])) for r in rs],'truncation'),'citation_emitting_answers':sum(bool(re.findall(r'\[[^\[\]]+\]',r['answer'])) for r in rs),'invalid_citation_answers':sum(any(s not in r['retrieved_ids'] for s in re.findall(r'\[([^\[\]]+)\]',r['answer'])) for r in rs)}
    result['clean_em']=float(np.mean([r.get('answer_em',r['completion']) for r in clean])) if clean else None
    result['attacked_em']=float(np.mean([r.get('answer_em',r['completion']) for r in attacks])) if attacks else None
    return result

allrows={s:load(s) for s in ['pilot','development','evaluation','positions','ingestion','adaptive']}
reports={}
for stage,rows in allrows.items():
    groups=defaultdict(list)
    for r in rows:
        groups[(r['model'],r['defense'])].append(r)
    reports[stage]={f'{k[0]}/{k[1]}':aggregate(rs) for k,rs in groups.items()}
write(ROOT/'reports/metrics.json',reports)
evalrows=allrows['evaluation']
paired=[]
for model in sorted({r['model'] for r in evalrows}):
    reference={ (r['task_id'],r['attack_id'],r['position'],r['rank_override']):r for r in evalrows if r['model']==model and r['defense']=='D0' and r['attack_id']}
    for defense in sorted({r['defense'] for r in evalrows if r['defense']!='D0'}):
        matches=[r for r in evalrows if r['model']==model and r['defense']==defense and r['attack_id']]
        differences=[int(r['attack_success'])-int(reference[(r['task_id'],r['attack_id'],r['position'],r['rank_override'])]['attack_success']) for r in matches]
        utility=[r['completion']-reference[(r['task_id'],r['attack_id'],r['position'],r['rank_override'])]['completion'] for r in matches]
        paired.append({'model':model,'defense':defense,'n':len(matches),'asr_difference':float(np.mean(differences)),'asr_difference_group_bootstrap_95':interval(differences,[r['group'] for r in matches]),'completion_difference':float(np.mean(utility)),'completion_difference_group_bootstrap_95':interval(utility,[r['group'] for r in matches])})
write(ROOT/'reports/paired.json',paired)
subgroups=defaultdict(list)
for r in evalrows:
    if r['attack_id']:
        subgroups[(r['model'],r['defense'],r['family'],r['domain'])].append(r)
write(ROOT/'reports/families.json',[{'model':k[0],'defense':k[1],'family':k[2],'domain':k[3],**aggregate(v)} for k,v in subgroups.items()])
for stage in ['positions','ingestion','adaptive']:
    grouped=defaultdict(list)
    for r in allrows[stage]:
        grouped[(r['model'],r['defense'],r['position'],str(r['rank_override']),r['surface'])].append(r)
    write(ROOT/f'reports/{stage}-details.json',[{'model':k[0],'defense':k[1],'position':k[2],'rank':k[3],'surface':k[4],**aggregate(v)} for k,v in grouped.items()])

# Retain only aggregate summaries publicly; detailed answers remain local.
successful=[r for r in evalrows if r['attack_success']]
rejected=[r for r in evalrows if not r['attack_id'] and r['rejected']]
write(ROOT/'reports/failure-summary.json',{'successful_attacks_by_family':dict(__import__('collections').Counter(r['family'] for r in successful)),'clean_rejections_by_defense':dict(__import__('collections').Counter(r['defense'] for r in rejected)),'local_review_candidates':[{'attempt_id':r['attempt_id'],'family':r['family'],'model':r['model'],'defense':r['defense']} for r in (successful[:12]+rejected[:8])],'faithfulness_scope':'Evidence recall is an exact-reference anchor diagnostic. Citation validity checks registered IDs and cannot establish entailment. No model judge is treated as a correctness oracle.'})
write(ROOT/'reports/reconciliation.json',{'stages':{s:{'n':len(rows),'attempt_identity':digest([r['attempt_id'] for r in rows])} for s,rows in allrows.items()},'metrics_rebuilt_from_attempt_ids':True,'stale_files_excluded':True})
print(reports['evaluation'])
