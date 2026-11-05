"""Exploratory NLI replay across all packed passages; no new controller decisions."""
import time
from agentic_rag.artifacts import ROOT,read,write,sha
from agentic_rag.models import Services,elapsed

if __name__=='__main__':
    services=Services()
    raw=[]
    summary={}
    began=time.perf_counter()
    for name in ['hotpot','squad']:
        for strategy in ['fixed','adaptive','plan','agent','verify']:
            label=f'{strategy}-{name}-evaluation'
            manifest=read(ROOT/f'data/manifests/{label}.json')
            rows=[]
            for entry in manifest['rows']:
                path=ROOT/entry['path']
                prediction=read(path)
                claim=prediction['claim']
                if not claim:
                    continue
                pairs=[dict(id=e['id'],document=e['document'],**services.verify(e['text'],claim)) for e in prediction['context']]
                model_conflict=any(v['entailment']>=.5 for v in pairs) and any(v['contradiction']>=.5 for v in pairs)
                rows.append(dict(id=prediction['id'],trace_sha256=sha(path),accepted=bool(prediction['answer']),model_conflict=model_conflict,pairs=pairs))
            raw.extend(dict(configuration=label,**r) for r in rows)
            summary[label]=dict(claim_proposals=len(rows),model_conflict_flags=sum(r['model_conflict'] for r in rows),
                                accepted_with_conflict_flag=sum(r['accepted'] and r['model_conflict'] for r in rows),
                                truncated_pairs=sum(v['truncated'] for r in rows for v in r['pairs']))
    write(ROOT/'data/conflict-replay.json',raw)
    write(ROOT/'reports/conflict-diagnostics.json',dict(scope='Exploratory post-run NLI replay of retained final claims against every packed passage. A flag requires entailment >=0.5 in one window and contradiction >=0.5 in another. These are model signals, not independently annotated contradictions; they never alter frozen predictions.',
             results=summary,replay_ms=elapsed(began),nli_model='nli',model_load_ms=services.nli_ms))
    print(summary)
