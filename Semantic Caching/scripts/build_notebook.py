"""Create the complete research notebook without replacing an executed notebook."""
import nbformat as nbf
from semantic_cache.artifacts import ROOT,read

path = ROOT/'notebooks/01_complete_semantic_caching_research.ipynb'
if path.exists():
    existing = nbf.read(path,as_version=4)
    if any(c.get('execution_count') is not None for c in existing.cells):
        raise RuntimeError('Preserve the executed local notebook before creating a separate run.')
cells = []
def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))
def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md('''# Semantic caching for RAG

Can a cache reduce end-to-end latency and model computation without reusing answers across different meanings, stale evidence or access boundaries? This study compares real local generation with exact answer reuse, semantic answer reuse and evidence reuse. Exact caching is the first comparator.

This is a small local study, not an official dataset leaderboard or a production security assessment. PAWS is an annotated sentence-equivalence diagnostic. QA traces use authentic HotpotQA and SQuAD records plus explicitly synthetic request wrappers. Raw text and per-request outputs remain local.''')
code('''from pathlib import Path
import sys, json
ROOT = Path.cwd()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
assert (ROOT / 'configs/study.json').is_file()
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import pandas as pd
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from semantic_cache.artifacts import read, digest, sha, identity
from semantic_cache.metrics import aggregate, qa_score, paired_interval
cfg = read(ROOT/'configs/study.json')
freeze = read(ROOT/'reports/selection-freeze.json')
assert freeze['identity'] == identity()
display(pd.Series(cfg))''')
md('''## Hypotheses and staged selection

Repeated identical requests should benefit most from exact caching. Semantic reuse should add value only if useful paraphrases survive a gate with tolerable false accepts. High cosine similarity may hide role reversals, changed entities, negation or dates. Evidence caching avoids retrieval while retaining generation, so it should save fewer model calls.

The development rule maximizes true pair acceptances subject to zero observed false acceptances at prespecified thresholds. A threshold above one disables semantic reuse if every useful candidate makes errors. The generator maximizes development QA F1 and stays fixed. Held-out outcomes never change these choices.''')
code('''display(pd.Series(freeze))
display(pd.DataFrame(read(ROOT/'reports/generator-development.json')).T)
development = read(ROOT/'reports/embedding-development.json')
development_sweeps = pd.DataFrame([dict(encoder=k, **row) for k,v in development.items() for row in v['sweep']])
display(development_sweeps)
eligible = development_sweeps[(development_sweeps.guard) & (development_sweeps.fp == 0) & development_sweeps.encoder.isin(['minilm','bge-small'])]
assert eligible.tp.max() == max(r['tp'] for k in ['minilm','bge-small'] for r in development[k]['sweep'] if r['guard'] and r['fp']==0)
display(Markdown(f"Frozen answer-cache threshold: **{freeze['threshold']}**; encoder: `{freeze['encoder']}`; generator: `{freeze['generator']}`."))''')
md('''## Host and artifact audit

Existing model files are reused read-only. The project imports no implementation modules from Projects 1â€“3. This workspace uses their installed Python package overlays; a standalone installation can use the dependency inventory. Model downloads, paid APIs and hosted services are unnecessary. Model loading, request inference and backend restart costs are distinct measurements.''')
code('''environment = read(ROOT/'reports/environment.json')
display(pd.Series({k:v for k,v in environment.items() if k not in ['versions','prior_artifact_sizes']}))
display(pd.DataFrame(environment['prior_artifact_sizes']).T)
display(pd.Series(environment['versions']))
display(read(ROOT/'reports/cleanup-manifest.json'))''')
md('''## Dataset provenance and licenses

HotpotQA distractor validation and SQuAD 2.0 public training records are reused from checksum-verified local files. SQuAD's development articles were already used in earlier projects. Prior question IDs are excluded; SQuAD also excludes prior articles. Development and evaluation source IDs are disjoint within this study.

PAWS-Wiki Labeled Final provides generated pairs with human fluency and equivalence labels. These are not human-authored question paraphrases, and their annotations do not establish answer correctness. The official archive returned HTTP 403 during acquisition; pinned mirror Parquet files provide reproducible bytes. Google supplies the PAWS license linked below. QQP was considered and excluded because its reconstruction conditions do not improve this study's QA annotation boundary.

References: [PAWS](https://github.com/google-research-datasets/paws), [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/), [HotpotQA](https://hotpotqa.github.io/). Raw releases remain outside Git.''')
code('''sources = read(ROOT/'sources.lock.json')
display(pd.DataFrame(sources).drop(columns=['splits'], errors='ignore'))
validation = read(ROOT/'reports/data-validation.json')
counts = []
for name, record in validation['datasets'].items():
    if 'partitions' in record:
        for split, details in record['partitions'].items():
            counts.append(dict(dataset=name, split=split, **{k:v for k,v in details.items() if k!='ids'}))
    else:
        counts.append(dict(dataset=name, **{k:v for k,v in record.items() if k!='ids'}))
display(pd.DataFrame(counts))''')
md('''## Partition and annotation validation

Question IDs and source documents separate QA development from evaluation. HotpotQA uses disjoint selected source pools rather than claiming a new official split. PAWS normalized sentence overlap is removed; source-page mapping is absent, so document-level separation cannot be established for PAWS. Repeated QA requests and wrappers remain in the original question family. The split assertions below inspect the actual local records.''')
code('''partitions = {name:read(ROOT/f'data/partitions/{name}.json') for name in ['hotpot','squad','paws']}
for name in ['hotpot','squad']:
    a,b = (partitions[name][s] for s in ['development','evaluation'])
    assert not {r['id'] for r in a} & {r['id'] for r in b}
    assert not set().union(*(set(r['documents']) for r in a)) & set().union(*(set(r['documents']) for r in b))
pa,pb = (partitions['paws'][s] for s in ['development','evaluation'])
assert not {r[k].strip().casefold() for r in pa for k in ['sentence1','sentence2']} & {r[k].strip().casefold() for r in pb for k in ['sentence1','sentence2']}
print('QA source and family separation, PAWS sentence separation: passed')
display(pd.DataFrame([dict(dataset=n, kind=r['kind']) for n in ['hotpot','squad'] for r in partitions[n]['evaluation']]).value_counts().rename('questions').reset_index())''')
md('''## Workload composition and order

Mixed workloads contain one first request, one exact repeat and one synthetic request-wrapper paraphrase per family. Unique-heavy workloads repeat and wrap one eighth of families. Unique workloads contain no deliberate repeats. These are controlled sequences rather than traffic sampled from a production deployment. Three deterministic family orders run with fresh cache state per strategy and trial. The first family pass precedes all repeat requests, making capacity effects observable.''')
code('''from semantic_cache.data import workload
composition = []
for dataset in ['hotpot','squad']:
    for mode in ['mixed','unique-heavy','unique']:
        requests = workload(partitions[dataset]['evaluation'],mode)
        composition.extend(dict(dataset=dataset,workload=mode,kind=k,requests=v) for k,v in pd.Series([r['kind'] for r in requests]).value_counts().items())
display(pd.DataFrame(composition))
print('Trial seeds:', [cfg['seed']+i for i in range(cfg['trials'])])''')
md('''## Models and representation choices

MiniLM and BGE produce normalized 384-dimensional vectors. Both use a maximum sequence length of 256 and symmetric, unprefixed question encoding. This differs from BGE's asymmetric retrieval query instruction. TF-IDF unigrams/bigrams provide a feasible third embedding method; it is not a pretrained neural model. Its held-out vocabulary and IDF come only from development sentences.

FLAN-small and FLAN-base are compared using local greedy generation. BM25 retrieves overlapping word windows from each task's candidate documents. SQuAD uses all paragraphs in the article, while HotpotQA uses its distractor candidate pool. These are bounded-corpus tasks, not global open-domain search.''')
code('''models = read(ROOT/'models.lock.json')
display(pd.DataFrame([{k:v for k,v in dict(alias=a,**r).items() if k!='files'} for a,r in models.items()]))
display(pd.DataFrame([{k:v for k,v in dict(encoder=a,**r).items() if k!='sweep'} for a,r in development.items()]))
for a,record in models.items():
    assert all(sha(ROOT.parent/record['local_path']/file)==info['sha256'] for file,info in record['files'].items())
print('Exact model snapshot hashes verified')''')
md('''## Similarity distributions and threshold sweeps

Cosine similarity measures representational proximity; it is not a calibrated probability that two answers are interchangeable. The held-out PAWS distribution below retains both beneficial and harmful matches. Number and negation guards can reject some mistakes while missing changes to entities and argument order.''')
code('''equivalence = read(ROOT/'reports/equivalence-evaluation.json')
sweeps = pd.DataFrame([dict(encoder=k, **row) for k,v in equivalence.items() for row in v['sweep']])
display(sweeps)
fig, axes = plt.subplots(1,2,figsize=(12,4))
for ax,alias in zip(axes,['minilm','bge-small']):
    values = read(ROOT/f'data/evaluation/{alias}-pairs.json')
    for label in [0,1]:
        ax.hist([r['score'] for r in values if r['label']==label],bins=25,alpha=.5,label=str(label))
    ax.set(title=alias,xlabel='Cosine similarity',ylabel='Pairs')
    ax.legend(title='Equivalence label')
plt.tight_layout()
plt.show()''')
md('''## Annotated false matches

These local examples pair authentic PAWS sentences with their original labels. A false acceptance means a non-equivalent pair passed the gate, whether or not a hypothetical QA answer happens to be the same. No QA quality is inferred from this diagnostic.''')
code('''scores = read(ROOT/f"data/evaluation/{freeze['encoder']}-pairs.json")
by_id = {r['id']:r for r in pb}
failures = sorted([r for r in scores if r['label']==0],key=lambda r:-r['score'])[:8]
display(pd.DataFrame([dict(**s, sentence1=by_id[s['id']]['sentence1'],sentence2=by_id[s['id']]['sentence2']) for s in failures]))
display(pd.DataFrame(read(ROOT/'reports/synthetic-stress.json')))''')
md('''## Cache architecture and eligibility

The cache keeps normalized question text, scope, answer or passages, source IDs and hashes, normalized vectors, creation/access/expiry times and a family label used only for evaluation. Family labels never enter the hit decision. Exact normalization collapses whitespace and applies Unicode NFC; it preserves case, punctuation, dates, numbers and negation.

Scope includes tenant, user, allowed sources, permission epoch, full corpus digest, exact generator/embedding revisions and prompt version. The frozen model lock verifies their snapshot bytes. Current source existence, visibility, tenant and hash checks run before content is returned. Full corpus invalidation also covers newly added competing documents. Unknown permissions fail closed.

In-memory and FAISS caches use a lock for lookup, eviction and insertion. FAISS inserts and removes vectors incrementally. SQLite signs entries with an external key, uses WAL and commits updates transactionally. Startup rebuilds the memory index from verified stored entries. Trusted writers and keys remain part of the threat boundary; concurrent misses can duplicate generation.''')
code('''from semantic_cache.security import Corpus, Document, Principal, scope, normalize, lexical_guard
from semantic_cache.cache import Cache
corpus = Corpus([Document('demo','Paris is the capital of France.',tenant='demo')])
principal = Principal('demo','user',frozenset({'demo'}))
sc = scope(principal,corpus,'fixed-model','fixed-prompt')
cache = Cache()
cache.put('What is the capital of France?',sc,'Paris',['demo'],corpus)
assert cache.lookup('What is the capital of France?',sc,principal,corpus)['answer']=='Paris'
assert normalize('2020')!=normalize('2021')
print('Exact answer hit with current authorization: passed')''')
md('''## Baseline and cold-model loading

Every no-cache request retrieves and generates a real answer. Misses in cache policies use the same generator, prompt and retrieval. Model loading and one discarded warm-up are outside steady-state latency. First-use task indexing is included in the first question's end-to-end time. Corpus scope hashing and cache insertion are included in total request time. Model and cache warmth are independent conditions.

A baseline audit found whole-prompt truncation could remove the question after long evidence. The final prompt reserves instructions, the complete question and answer suffix before packing evidence. All affected trials and model-selection outputs were preserved and rerun. Only corrected prompt-version results enter this notebook.''')
code('''resources = read(ROOT/'reports/resources.json')
display(pd.Series(resources))
metrics = read(ROOT/'reports/metrics.json')
frame = pd.DataFrame(metrics).drop(columns=['trace_sha256'])
display(frame[frame.strategy=='none'])''')
md('''## Exact and strict semantic cache results

Exact answer hits avoid generation. The strict semantic policy is exact-first and performs an embedding lookup after an exact miss; when its selected threshold disables semantic hits, it should add overhead without adding answer reuse. Hybrid tries exact answers, strict semantic answers, then evidence at a prespecified experimental threshold of 0.95 with fresh generation. Evidence hits perform fresh generation.''')
code('''columns = ['dataset','workload','strategy','f1','exact_hit_rate','semantic_hit_rate','false_hits','correct_hit_precision','p50_ms','p95_ms','generation_calls_avoided']
display(frame[columns])
display(pd.DataFrame(read(ROOT/'reports/cold-warm-phases.json')))
fig,axes = plt.subplots(1,2,figsize=(12,4))
for ax,dataset in zip(axes,['hotpot','squad']):
    subset = frame[(frame.dataset==dataset)&(frame.workload=='mixed')]
    ax.bar(subset.strategy,subset.p50_ms)
    ax.set(title=dataset,ylabel='p50 milliseconds')
plt.tight_layout()
plt.show()''')
md('''## Adjacent latency diagnostic

Substantial timing variation during the primary sequential trials motivated this additional diagnostic after the freeze. Sixteen existing held-out questions each seed a cache with real generation. A fresh no-cache request and a warm exact hit then run adjacently, alternating their order across families. This isolates the observed warm-hit mechanism more closely than long sequential policy runs. It is not a production traffic estimate, and it does not change the selected cache settings. Seeding and model loading are reported separately.''')
code('''adjacent = read(ROOT/'reports/adjacent-latency.json')
display(pd.Series(adjacent))
adjacent_rows = read(ROOT/'data/probes/adjacent-latency.json')['rows']
assert len(adjacent_rows)==16
assert all(r['warm_hit']['generation_calls']==0 and r['baseline']['generation_calls']==1 for r in adjacent_rows)
display(pd.DataFrame([dict(dataset=r['dataset'],id=r['id'],seed_ms=r['seed']['total_ms'],no_cache_ms=r['baseline']['total_ms'],warm_exact_ms=r['warm_hit']['total_ms'],same_answer=r['same_answer']) for r in adjacent_rows]))''')
md('''## Latency distributions and paired comparisons

Cache-hit medians can hide expensive misses in p95. The curves include all three trials. Paired confidence intervals resample HotpotQA question families and SQuAD articles, keeping repeated requests together. HotpotQA evaluation families can share source pages, which can make these intervals optimistic. They are pointwise descriptive intervals from one host, not guarantees about deployment tail latency.

Substantial host drift occurred between sequential policy runs. The strict answer gate makes no semantic accepts, so any observed speed advantage over exact caching cannot be attributed to semantic answer reuse. The two policies avoid the same exact-repeat generation calls. Per-trial outcomes retain the drift rather than selecting only faster trials.''')
code('''all_traces = {}
for dataset in ['hotpot','squad']:
    for mode in ['mixed','unique-heavy','unique']:
        for strategy in cfg['strategies']:
            label = f'{dataset}-{mode}-{strategy}'
            all_traces[label] = [r for trial in range(cfg['trials']) for r in read(ROOT/f'data/traces/{label}-{trial}.json')['rows']]
fig,ax = plt.subplots(figsize=(8,4))
for strategy in cfg['strategies']:
    values = sorted(r['total_ms'] for r in all_traces['hotpot-mixed-'+strategy])
    ax.plot(values,np.arange(1,len(values)+1)/len(values),label=strategy)
ax.set(xlabel='Milliseconds',ylabel='Cumulative fraction',xscale='log')
ax.legend()
plt.show()
intervals = read(ROOT/'reports/paired-intervals.json')
display(pd.DataFrame(read(ROOT/'reports/trial-metrics.json'))[['dataset','workload','strategy','trial','p50_ms','p95_ms','requests_per_second']])
display(pd.DataFrame([dict(configuration=k,metric=f,**v) for k,fields in intervals.items() for f,v in fields.items()]))''')
md('''## Independent aggregate reconstruction

The following assertions recompute aggregates and QA scores directly from local saved requests. Cached answers are scored against the current request's annotations. Answer reuse precision remains separate from EM/F1: a correct reuse decision can repeat a wrong baseline answer.''')
code('''checked = 0
for record in metrics:
    label = f"{record['dataset']}-{record['workload']}-{record['strategy']}"
    rows = all_traces[label]
    recalculated = aggregate(rows)
    for key,value in recalculated.items():
        assert value == record[key], (label,key)
    for row in rows:
        assert qa_score(row['answer'],row['answers']) == (row['em'],row['f1'])
        if row['generation_calls']:
            assert ('Question: '+row['question']+chr(10)+'Evidence:'+chr(10)) in row['prompt']
            assert row['input_tokens'] <= cfg['max_input_tokens']
        checked += 1
print(checked, 'request outcomes verified against aggregate reports')''')
md('''## Permissive answer reuse and evidence reuse

This ablation warms each cache with a source question, then asks either a synthetic wrapper or a genuine different SQuAD question from the same article with a disjoint annotated answer. The different question is selected by high embedding similarity within the article. This is an adversarial diagnostic, not a random traffic estimate. It uses held-out records after the freeze and never tunes the final threshold.

An evidence false match means retrieval was reused across non-equivalent questions; it need not produce the wrong answer. Compare fresh-generation QA against the no-cache answer for that request.''')
code('''ablation = pd.DataFrame(read(ROOT/'reports/answer-reuse-ablation.json'))
display(ablation)
ablation_rows = read(ROOT/'data/ablations/answer-reuse.json')
bad = [r for r in ablation_rows if r['false_hit'] and r['strategy']=='semantic'][:8]
display(pd.DataFrame([{k:r[k] for k in ['source_id','target_id','threshold','source_answer','answer','baseline_answer','f1','baseline_f1','similarity']} for r in bad]))''')
md('''## Freshness and invalidation

Controlled document changes exercise updates, deletion, visibility revocation, and additions to the corpus. The old scope becomes incompatible, and direct old-scope probes also check current source hashes. Conservative full-corpus invalidation trades reuse for compatibility. It does not discover changes in an upstream source that the application has not ingested.''')
code('''security = read(ROOT/'reports/security-freshness.json')
display(pd.DataFrame(security['probes']))
assert security['failures']==0
from dataclasses import replace
corpus.documents['demo'] = replace(corpus.documents['demo'],text='Changed evidence',version='2')
assert cache.lookup('What is the capital of France?',sc,principal,corpus) is None
print('Direct old-scope source-change rejection: passed')''')
md('''## Authorization and poisoning boundaries

Controlled tests use synthetic users and tenants. User changes, permission epochs and unknown permissions reject old entries. Persisted entries require a valid HMAC; modified payloads and wrong keys fail restoration. This protects storage integrity under the tested key boundary. The incorrect-trusted-insertion test deliberately shows a retained failure: a validly inserted wrong answer is reused. Integrity does not establish answer truth. The mechanism cannot stop a compromised trusted writer, a stolen signing key, malicious source ingestion or prompt injection in generation. No production penetration test or distributed-cache security assessment was performed.''')
code('''import xml.etree.ElementTree as ET
test_report = ET.parse(ROOT/'reports/tests.xml').getroot()
suites = list(test_report.iter('testsuite'))
display(pd.DataFrame([s.attrib for s in suites]))
assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ['failures','errors','skipped'])
print('Recorded controlled security checks:', security['tested_requests'])''')
md('''## Backend comparison and recovery

The microbenchmark inserts 512 normalized synthetic vectors incrementally, then measures exact and semantic lookup. SQLite includes signed payload reads/writes through the cache and recovery after closing the connection. Qdrant uses persistent local mode with scope filtering. These paths differ in Python overhead and payload handling; the results do not isolate the underlying engines. Redis was absent and service installation was not justified for this local study.''')
code('''backends = read(ROOT/'reports/backend-comparison.json')
display(pd.DataFrame(backends).T)
assert 'restart_ms' in backends['sqlite-faiss']
assert backends['qdrant-local'].get('entries')==512
print('SQLite and Qdrant restart checks passed')''')
md('''## Capacity, eviction and changing traffic

The synthetic sequence cycles 64 keys twice, introduces 64 new keys, then returns to the first set. LRU is compared at four capacities; TTL advances a controlled clock by one second per request. This separates deterministic eviction behavior from noisy inference timings. It does not estimate the best capacity for an unknown production distribution.''')
code('''eviction = pd.DataFrame(read(ROOT/'reports/eviction.json'))
display(eviction)
fig,ax = plt.subplots(figsize=(7,4))
for policy,group in eviction.groupby('policy'):
    ax.plot(group.capacity,group.hits/group.requests,marker='o',label=policy)
ax.set(xlabel='Entry capacity',ylabel='Hit fraction',xscale='log')
ax.legend()
plt.show()''')
md('''## Resource consumption and computation avoided

CUDA event elapsed time covers generation operations and associated stream work. Avoided calls count real model invocations skipped by answer reuse. Neither metric establishes energy savings or dollar savings. Cache entry JSON bytes, FAISS index bytes and persistent file bytes describe different storage representations; process RSS deltas are affected by allocator reuse.''')
code('''display(frame[['dataset','workload','strategy','generation_calls','generation_calls_avoided','gpu_generation_ms','gpu_generation_ms_avoided','mean_embedding_ms','mean_lookup_ms','mean_retrieval_ms','mean_generation_ms']])
display(pd.Series(resources))
display(pd.DataFrame(backends).T[['serialized_entry_bytes','index_bytes','disk_bytes','rss_delta']])''')
md('''## Retained QA failures and answerability

The baseline is a functioning bounded RAG system with measured errors. FLAN may answer unsupported questions or miss multihop answers. A cache cannot repair those failures by repeating the answer. The examples below retain actual generated outputs and their dataset annotations locally.''')
code('''failures = [r for r in all_traces['hotpot-mixed-none']+all_traces['squad-mixed-none'] if r['trial']==0 and r['kind']=='unique' and r['f1']==0][:10]
display(pd.DataFrame([{k:r[k] for k in ['id','question','answer','answers','sources','f1','total_ms']} for r in failures]))
unanswerable_ids = {r['id'] for r in partitions['squad']['evaluation'] if r['answers']==['']}
answerability = []
for strategy in cfg['strategies']:
    rows = [r for r in all_traces['squad-mixed-'+strategy] if r['id'] in unanswerable_ids]
    answerability.append(dict(strategy=strategy,requests=len(rows),abstention_fraction=np.mean([not r['answer'] for r in rows]) if rows else None))
display(pd.DataFrame(answerability))''')
quantitative_conclusion = ''
if (ROOT/'reports/metrics.json').exists() and (ROOT/'reports/equivalence-evaluation.json').exists():
    measured = read(ROOT/'reports/metrics.json')
    paragraphs = []
    for dataset in ['hotpot','squad']:
        baseline = next(r for r in measured if r['dataset']==dataset and r['workload']=='mixed' and r['strategy']=='none')
        exact = next(r for r in measured if r['dataset']==dataset and r['workload']=='mixed' and r['strategy']=='exact')
        paragraphs.append(f"On {dataset} mixed traffic, exact caching recorded p50 {exact['p50_ms']:.2f} ms and p95 {exact['p95_ms']:.2f} ms, compared with {baseline['p50_ms']:.2f} and {baseline['p95_ms']:.2f} ms without caching. F1 changed by {exact['f1']-baseline['f1']:+.3f}; {exact['generation_calls_avoided']} generation calls were avoided across three trials.")
    eq = read(ROOT/'reports/equivalence-evaluation.json')
    for alias in ['minilm','bge-small']:
        row = next(r for r in eq[alias]['sweep'] if r['threshold']==.95 and r['guard'])
        paragraphs.append(f"At the diagnostic 0.95 threshold with the lexical guard, {alias} accepted {row['accepted']} held-out PAWS pairs, including {row['fp']} annotated non-equivalent pairs. This gate was not selected as a safe final-answer reuse rule.")
    quantitative_conclusion = '\n\n'+'\n\n'.join(paragraphs)
md('''## Conclusions and limits

Read the exact-cache results alongside the selected gate and the permissive ablations. Latency reduction is useful only when the hit is compatible with meaning, evidence and access. Zero observed stale or unauthorized fixture returns does not establish a zero deployment failure rate.

The study is limited by small QA samples, one GPU host, bounded per-task candidate pools, synthetic request wrappers and no independently annotated QA paraphrases. PAWS sentences test a different domain from QA. Similarity-selected adversarial QA neighbors intentionally overrepresent difficult reuse cases. The hybrid evidence threshold is experimental and does not establish a safe evidence-equivalence gate. No third neural encoder, large-generator upgrade, distributed service, energy measurement or production traffic trace is included.

The source-only public notebook preserves every markdown and code cell in this local notebook. Local outputs contain real tables, plots and generated answers. Reproduction scripts verify source and partition identities before resuming trials.'''+quantitative_conclusion)
code('''display(read(ROOT/'reports/metric-audit.json'))
ledger = read(ROOT/'reports/execution-ledger.json')
assert ledger['primary_complete']
print('Completed trial files:',len(ledger['completed']))
print('Timed requests:',ledger['request_runs'])
print('Notebook code cells:', 'counted by publication verifier after execution')''')
notebook = nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python','version':'3.12.10'}})
path.parent.mkdir(parents=True,exist_ok=True)
nbf.write(notebook,path)
print(len(cells),'cells created')
