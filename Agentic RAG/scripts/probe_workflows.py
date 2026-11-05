"""Small uncached workflow and repeatability probes, separate from real benchmarks."""
from agentic_rag.artifacts import ROOT,read,write,digest
from agentic_rag.models import Generator,Services
from agentic_rag.controller import Controller
from agentic_rag.data import Task

if __name__=='__main__':
    freeze=read(ROOT/'reports/selection-freeze.json')
    generator=Generator('flan-base')
    planner=generator if freeze['planner']=='flan-base' else Generator(freeze['planner'])
    services=Services()
    engine=Controller(generator,planner,services,freeze['config'])
    semantic=[]
    for premise,claim,expected in [('Dogs are animals.','Dogs are animals.','entailment'),('The capital of France is Paris.','The capital of France is Berlin.','contradiction'),('A person is outdoors.','A person is outside.','entailment')]:
        semantic.append(dict(premise=premise,claim=claim,expected=expected,**services.verify(premise,claim)))
    documents={'Ada':'Ada is based in London.','London':'London is in England.'}
    task=Task('synthetic-workflow','In which country is Ada based?',documents,{k:[v] for k,v in documents.items()})
    results=[]
    for strategy in ['fixed','adaptive','plan','agent','verify']:
        passes=[engine.run(task,strategy,retrieval=freeze['retrieval']) for _ in range(2)]
        assert all(r['state']['status']=='finished' and r['state']['calls']<=freeze['config']['max_calls'] and r['state']['steps']<=freeze['config']['max_steps'] and r['state']['tokens']<=freeze['config']['max_tokens'] for r in passes)
        signatures=[digest(dict(answer=r['answer'],citations=r['citations'],actions=[t['action'] for t in r['state']['trace'] if 'action' in t])) for r in passes]
        results.append(dict(strategy=strategy,answer=passes[0]['answer'],repeat_signature_equal=signatures[0]==signatures[1],signatures=signatures,
                            reason=passes[0]['reason'],calls=passes[0]['state']['calls'],tokens=passes[0]['state']['tokens']))
        write(ROOT/f'data/probes/{strategy}.json',passes)
    write(ROOT/'reports/workflow-probes.json',dict(scope='One synthetic two-source question, two uncached greedy passes per strategy, and three simple NLI sentence-pair diagnostics. These are operational/semantic probes, not benchmark or comprehensive security evidence.',results=results,nli_labels=services.labels,nli_semantic_probes=semantic))
    print(results)
