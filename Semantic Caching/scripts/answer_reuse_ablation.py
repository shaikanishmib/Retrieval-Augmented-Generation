"""Measure permissive answer reuse on genuine nearby QA and synthetic wrappers."""
import numpy as np
from semantic_cache.artifacts import ROOT,read,write,identity
from semantic_cache.data import load_qa
from semantic_cache.models import Encoder,Generator
from semantic_cache.security import Corpus,Document,Principal
from semantic_cache.rag import Pipeline
from semantic_cache.cache import Cache
from semantic_cache.metrics import qa_score

cfg = read(ROOT/'configs/study.json')
freeze = read(ROOT/'reports/selection-freeze.json')
assert freeze['identity']==identity()
encoder = Encoder(freeze['encoder'])
generator = Generator(freeze['generator'])
encoder.encode(['warm-up'])
generator.generate('What is two plus two?',['Two plus two is four.'])
all_rows = load_qa('squad')
selected = read(ROOT/'data/partitions/squad.json')['evaluation'][:16]
traces = []
for source in selected:
    # Genuine different question; different annotated answer identifies unsafe interchangeability.
    candidates = [r for r in all_rows if r['group']==source['group'] and r['id']!=source['id'] and r['question']!=source['question'] and set(r['answers']).isdisjoint(source['answers'])]
    assert candidates
    vectors,_ = encoder.encode([source['question']]+[r['question'] for r in candidates])
    target = candidates[int(np.argmax(vectors[1:]@vectors[0]))]
    corpus = Corpus([Document(k,v) for k,v in source['documents'].items()])
    principal = Principal('public','ablation',frozenset(corpus.documents))
    pipeline = Pipeline(corpus,generator,encoder,cfg)
    for variant,question,answers,family in [('nearby_authentic',target['question'],target['answers'],target['id']),('synthetic_wrapper','Please answer this question: '+source['question'],source['answers'],source['id'])]:
        baseline = pipeline.request(question,principal)
        baseline_em,baseline_f1 = qa_score(baseline['answer'],answers)
        for threshold in [.8,.95,freeze['threshold']]:
            for strategy in ['semantic','evidence']:
                cache = Cache(threshold=threshold)
                seed = pipeline.request(source['question'],principal,strategy,cache,source['id'])
                result = pipeline.request(question,principal,strategy,cache,family)
                em,f1 = qa_score(result['answer'],answers)
                traces.append(dict(source_id=source['id'],target_id=family,variant=variant,threshold=threshold,strategy=strategy,source_answer=seed['answer'],baseline_answer=baseline['answer'],baseline_f1=baseline_f1,baseline_ms=baseline['total_ms'],em=em,f1=f1,false_hit=result['hit']!='miss' and variant=='nearby_authentic',**result))
write(ROOT/'data/ablations/answer-reuse.json',traces)
aggregates = []
for variant in ['nearby_authentic','synthetic_wrapper']:
    for threshold in [.8,.95,freeze['threshold']]:
        for strategy in ['semantic','evidence']:
            rows = [r for r in traces if r['variant']==variant and r['threshold']==threshold and r['strategy']==strategy]
            hits = sum(r['hit']!='miss' for r in rows)
            aggregates.append(dict(variant=variant,threshold=threshold,strategy=strategy,requests=len(rows),hits=hits,false_reuse=sum(r['false_hit'] for r in rows),f1=float(np.mean([r['f1'] for r in rows])),baseline_f1=float(np.mean([r['baseline_f1'] for r in rows])),p50_ms=float(np.median([r['total_ms'] for r in rows])),baseline_p50_ms=float(np.median([r['baseline_ms'] for r in rows])),generation_calls=sum(r['generation_calls'] for r in rows)))
write(ROOT/'reports/answer-reuse-ablation.json',aggregates)
print(aggregates,flush=True)
