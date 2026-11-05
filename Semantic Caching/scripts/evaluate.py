"""Run real sequential inference trials; completed trial files are resumable checkpoints."""
import time
import numpy as np
import psutil
import torch
from semantic_cache.artifacts import ROOT,read,write,identity,digest,sha
from semantic_cache.cache import Cache
from semantic_cache.data import workload
from semantic_cache.metrics import qa_score
from semantic_cache.models import Encoder,Generator
from semantic_cache.rag import Pipeline
from semantic_cache.security import Corpus,Document,Principal

cfg = read(ROOT/'configs/study.json')
freeze = read(ROOT/'reports/selection-freeze.json')
assert freeze['identity']==identity(),'Study sources changed after selection freeze.'
encoder = Encoder(freeze['encoder'])
generator = Generator(freeze['generator'])
encoder.encode(['discarded warm-up'])
generator.generate('What is two plus two?',['Two plus two is four.'])
resources = dict(encoder_load_ms=encoder.load_ms,generator_load_ms=generator.load_ms,model_warmup='one discarded encoder and generator request',rss_loaded=psutil.Process().memory_info().rss,cuda_allocated=torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,trials=cfg['trials'],cache_state='fresh per strategy/trial/workload/dataset')
ledger = []
for dataset in ['hotpot','squad']:
    rows = read(ROOT/f'data/partitions/{dataset}.json')['evaluation']
    for mode in ['mixed','unique-heavy','unique']:
        requests = workload(rows,mode)
        for trial in range(cfg['trials']):
            # Every trial reorders source families identically across strategies.
            order = np.random.default_rng(cfg['seed']+trial).permutation(len(rows)).tolist()
            ordered_rows = [rows[i] for i in order]
            requests = workload(ordered_rows,mode)
            for strategy in cfg['strategies']:
                label = f'{dataset}-{mode}-{strategy}-{trial}'
                path = ROOT/f'data/traces/{label}.json'
                run_id = digest([identity(),freeze,requests,label])
                if path.exists():
                    saved = read(path)
                    assert saved['run_id']==run_id and len(saved['rows'])==len(requests)
                    ledger.append(dict(label=label,requests=len(requests),sha256=sha(path),resumed=True))
                    continue
                cache = Cache(capacity=cfg['capacity'],ttl=cfg['ttl_seconds'],threshold=freeze['threshold']) if strategy!='none' else None
                pipelines = {}
                trace = []
                # Candidate-pool scope is per original QA task; avoids leaking between differing corpora.
                # Different authentic questions within a shared article still compete when corpus hashes match.
                for index,request in enumerate(requests):
                    row = request['row']
                    if row['id'] not in pipelines:
                        corpus = Corpus([Document(k,v) for k,v in row['documents'].items()])
                        pipelines[row['id']] = Pipeline(corpus,generator,encoder,cfg)
                    pipeline = pipelines[row['id']]
                    p = Principal('public','evaluation-user',frozenset(pipeline.corpus.documents))
                    result = pipeline.request(request['question'],p,strategy,cache,request['family'])
                    em,f1 = qa_score(result['answer'],row['answers'])
                    result.update(request_id=index,trial=trial,id=row['id'],group=row['group'],kind=request['kind'],question=request['question'],answers=row['answers'],em=em,f1=f1,reuse_equivalent=result['cached_family'] in {None,request['family']},index_ms=pipeline.index_ms if request['kind']=='unique' else 0.)
                    # Include one-time task indexing in the first request, across all strategies.
                    result['total_ms'] += result['index_ms']
                    trace.append(result)
                write(path,dict(run_id=run_id,identity=identity(),label=label,rows=trace,cache_entries=len(cache.entries) if cache else 0,evictions=cache.evictions if cache else 0))
                ledger.append(dict(label=label,requests=len(trace),sha256=sha(path),resumed=False))
                write(ROOT/'reports/execution-ledger.json',dict(primary_complete=False,completed=ledger))
                print(label,len(trace),round(sum(r['total_ms'] for r in trace)/1000,2),'seconds',flush=True)
resources.update(rss_final=psutil.Process().memory_info().rss,cuda_peak_allocated=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)
write(ROOT/'reports/resources.json',resources)
write(ROOT/'reports/execution-ledger.json',dict(primary_complete=True,completed=ledger,request_runs=sum(r['requests'] for r in ledger)))
