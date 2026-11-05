"""Use adjacent matched requests to diagnose drift separately from traffic-level trials."""
import numpy as np
from semantic_cache.artifacts import ROOT,read,write,identity
from semantic_cache.cache import Cache
from semantic_cache.models import Encoder,Generator
from semantic_cache.rag import Pipeline
from semantic_cache.security import Corpus,Document,Principal
from semantic_cache.metrics import qa_score

cfg = read(ROOT/'configs/study.json')
freeze = read(ROOT/'reports/selection-freeze.json')
assert freeze['identity']==identity()
encoder = Encoder(freeze['encoder'])
generator = Generator(freeze['generator'])
encoder.encode(['discarded warm-up'])
generator.generate('What is two plus two?',['Two plus two is four.'])
traces = []
for dataset in ['hotpot','squad']:
    for index,row in enumerate(read(ROOT/f'data/partitions/{dataset}.json')['evaluation'][:8]):
        corpus = Corpus([Document(k,v) for k,v in row['documents'].items()])
        p = Principal('public','adjacent-diagnostic',frozenset(corpus.documents))
        pipeline = Pipeline(corpus,generator,encoder,cfg)
        cache = Cache(threshold=freeze['threshold'])
        seed = pipeline.request(row['question'],p,'exact',cache,row['id'])
        # Alternate order to reduce a systematic first/second request effect.
        result = {}
        for policy in (['none','exact'] if index%2==0 else ['exact','none']):
            result[policy] = pipeline.request(row['question'],p,policy,cache,row['id'])
        assert result['exact']['hit']=='exact' and result['exact']['generation_calls']==0
        assert result['none']['generation_calls']==1
        traces.append(dict(dataset=dataset,id=row['id'],question=row['question'],answers=row['answers'],seed=seed,baseline=result['none'],warm_hit=result['exact'],same_answer=result['none']['answer']==result['exact']['answer']))
write(ROOT/'data/probes/adjacent-latency.json',dict(identity=identity(),rows=traces))
report = dict(pairs=len(traces),warm_model=True,traffic_estimate=False,order='Alternating baseline/hit order after one real seed request per family',cold_cache_seed_p50_ms=float(np.median([r['seed']['total_ms'] for r in traces])),no_cache_p50_ms=float(np.median([r['baseline']['total_ms'] for r in traces])),warm_exact_hit_p50_ms=float(np.median([r['warm_hit']['total_ms'] for r in traces])),same_answer=sum(r['same_answer'] for r in traces),generation_calls_avoided=len(traces))
write(ROOT/'reports/adjacent-latency.json',report)
print(report,flush=True)
