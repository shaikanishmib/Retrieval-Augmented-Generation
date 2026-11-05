"""Recompute every primary aggregate from saved requests and render source-safe figures."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from semantic_cache.artifacts import ROOT,read,write,sha
from semantic_cache.metrics import aggregate,paired_interval,qa_score

records,raw = [],{}
trial_metrics = []
ledger = read(ROOT/'reports/execution-ledger.json')
assert ledger['primary_complete']
for dataset in ['hotpot','squad']:
    for mode in ['mixed','unique-heavy','unique']:
        for strategy in read(ROOT/'configs/study.json')['strategies']:
            label = f'{dataset}-{mode}-{strategy}'
            rows = []
            hashes = []
            for trial in range(3):
                path = ROOT/f'data/traces/{label}-{trial}.json'
                saved = read(path)
                assert all(tuple(qa_score(r['answer'],r['answers']))==(r['em'],r['f1']) for r in saved['rows'])
                rows.extend(saved['rows'])
                trial_metrics.append(dict(label=f'{label}-{trial}',dataset=dataset,workload=mode,strategy=strategy,trial=trial,**aggregate(saved['rows'])))
                hashes.append(sha(path))
            raw[label] = rows
            result = dict(dataset=dataset,workload=mode,strategy=strategy,**aggregate(rows),trace_sha256=hashes)
            records.append(result)
for row in records:
    baseline = next(r for r in records if r['dataset']==row['dataset'] and r['workload']==row['workload'] and r['strategy']=='none')
    row.update(p50_speedup=baseline['p50_ms']/row['p50_ms'],p95_speedup=baseline['p95_ms']/row['p95_ms'],generation_calls_avoided=baseline['generation_calls']-row['generation_calls'],gpu_generation_ms_avoided=baseline['gpu_generation_ms']-row['gpu_generation_ms'],f1_change=row['f1']-baseline['f1'])
write(ROOT/'reports/metrics.json',records)
write(ROOT/'reports/trial-metrics.json',trial_metrics)
phases = []
for label,rows in raw.items():
    for kind in ['unique','exact_repeat','synthetic_wrapper']:
        subset = [r for r in rows if r['kind']==kind]
        if subset:
            phases.append(dict(configuration=label,phase=kind,**aggregate(subset)))
write(ROOT/'reports/cold-warm-phases.json',phases)
pd.DataFrame([{k:v for k,v in r.items() if k!='trace_sha256'} for r in records]).to_csv(ROOT/'reports/metrics.csv',index=False)
intervals = {}
for dataset in ['hotpot','squad']:
    for mode in ['mixed','unique-heavy','unique']:
        for strategy in ['exact','semantic','evidence','hybrid']:
            label = f'{dataset}-{mode}-{strategy}'
            baseline = raw[f'{dataset}-{mode}-none']
            intervals[label] = {field:paired_interval(baseline,raw[label],field) for field in ['f1','total_ms']}
write(ROOT/'reports/paired-intervals.json',intervals)
figures = ROOT/'reports/figures'
figures.mkdir(exist_ok=True)
fig,axes = plt.subplots(1,2,figsize=(11,4))
for ax,dataset in zip(axes,['hotpot','squad']):
    subset = [r for r in records if r['dataset']==dataset and r['workload']=='mixed']
    ax.bar([r['strategy'] for r in subset],[r['p50_ms'] for r in subset])
    ax.set(title=dataset,ylabel='End-to-end median milliseconds')
fig.tight_layout()
fig.savefig(figures/'latency.png',dpi=150)
plt.close(fig)
fig,axes = plt.subplots(1,2,figsize=(11,4))
for ax,alias in zip(axes,['minilm','bge-small']):
    rows = read(ROOT/f'data/evaluation/{alias}-pairs.json')
    for label in [0,1]:
        ax.hist([r['score'] for r in rows if r['label']==label],bins=30,alpha=.5,label='different meaning' if label==0 else 'paraphrase')
    ax.set(title=alias,xlabel='Cosine similarity',ylabel='PAWS pairs')
    ax.legend()
fig.tight_layout()
fig.savefig(figures/'similarity.png',dpi=150)
plt.close(fig)
fig,ax = plt.subplots(figsize=(7,4))
for strategy in ['none','exact','semantic','evidence','hybrid']:
    values = sorted(r['total_ms'] for r in raw['hotpot-mixed-'+strategy])
    ax.plot(values,np.arange(1,len(values)+1)/len(values),label=strategy)
ax.set(xlabel='End-to-end milliseconds',ylabel='Cumulative fraction',xscale='log')
ax.legend()
fig.tight_layout()
fig.savefig(figures/'latency-distribution.png',dpi=150)
plt.close(fig)
summary = ['# Measured benchmark','', 'All values below are reconstructed from actual timed inference traces. Model loading and one discarded warm-up are separate. Three sequential trials use a fresh cache for each strategy, workload and dataset. Candidate-pool indexing is charged to each first question.','', '| Dataset | Workload | Strategy | F1 | Exact hit | Semantic hit | p50 ms | p95 ms | Calls avoided |','| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for r in records:
    summary.append(f"| {r['dataset']} | {r['workload']} | {r['strategy']} | {r['f1']:.3f} | {r['exact_hit_rate']:.3f} | {r['semantic_hit_rate']:.3f} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | {r['generation_calls_avoided']} |")
summary += ['', 'Reuse correctness is based on request-family equivalence, separate from baseline answer correctness. Synthetic wrappers preserve the question and do not represent independent human paraphrases. Different corpus scopes prevent semantic cross-task hits on HotpotQA. PAWS supplies the harder annotated equivalence diagnostic; it is not an end-to-end QA result.', '', 'The hybrid policy tries exact answers, strict semantic answers, then evidence reuse at an experimental threshold of 0.95 with fresh generation. Evidence caching regenerates answers after both exact and semantic evidence hits. Backend microbenchmarks use synthetic normalized vectors and exclude embedding time.', '', 'Confidence intervals resample HotpotQA question families and SQuAD articles. HotpotQA evaluation families can share source pages, so their intervals may understate document-level dependence. Repeated timings share one host, and three trials do not establish production tail latency. Avoided CUDA elapsed time is neither energy nor monetary savings.']
(ROOT/'reports/benchmark.md').write_text('\n'.join(summary)+'\n',encoding='utf-8')
write(ROOT/'reports/metric-audit.json',dict(verified_trace_files=len(ledger['completed']),verified_requests=sum(r['requests'] for r in ledger['completed']),qa_score_mismatches=0,aggregate_rows=len(records)))
print(pd.DataFrame(records)[['dataset','workload','strategy','f1','p50_ms','p95_ms']].to_string(index=False))
