"""Build the research notebook from measured results and original data."""

import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build():
    cells = []
    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text.strip()))
    def code(text):
        cells.append(nbf.v4.new_code_cell(text.strip()))
    md("""# RAG foundations and retrieval optimization

## Research questions

How do lexical and pretrained semantic retrieval differ between scientific claims and financial questions? Does segmentation help when evaluation requires complete document identifiers? How much accuracy does fusion or cross-encoder reranking buy for its added latency? Which query transformations lose relevant material?

This study uses the original BEIR SciFact and FiQA-2018 archives. It evaluates retrieval against their relevance judgments. It does not evaluate generated answers, factuality, or a deployed RAG application. All measurements below come from local execution. The final test labels are reserved for reporting; chunking and fusion choices use development labels or a training-derived validation set.""")
    code("""from pathlib import Path
import sys, json, math, hashlib, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
get_ipython().run_line_magic('matplotlib', 'inline')
ROOT = Path.cwd()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
assert (ROOT / 'src/rag_data').exists()
sys.path.insert(0, str(ROOT / 'src'))
from rag_data.validation import load_records, validate, digest
from rag_data.retrieval import tokenize, evaluate
from rag_data.research import qrels, run_dataset, REQUIRED_RUNS
from rag_data.chunking import chunks
plt.style.use('default')
plt.rcParams.update({'figure.figsize': (10, 4), 'axes.spines.top': False, 'axes.spines.right': False})
FIGURES = ROOT / 'data/figures'
FIGURES.mkdir(parents=True, exist_ok=True)
def read(path):
    return json.loads(path.read_text(encoding='utf-8'))
def figure(name):
    plt.tight_layout()
    plt.savefig(FIGURES / (name + '.png'), dpi=140, bbox_inches='tight')
    plt.show()
print('Project:', ROOT)
print('Kernel Python:', sys.executable, sys.version.split()[0])
""")
    md("""## Provenance, rights, and benchmark identity

BEIR supplies corpus JSONL, query JSONL, and split-specific TSV judgments. Archive SHA-256 values pin the exact downloaded bytes because the download paths do not include release identifiers. The original IDs and positive relevance grades remain unchanged. Raw files, source texts, embeddings, indexes, and model weights stay local.

SciFact's authors distinguish CC BY 4.0 claims and annotations from the ODC-By 1.0 abstract database. That database license does not resolve independent publisher rights in every abstract. FiQA's underlying source rights also require care; access through BEIR grants no blanket redistribution permission. See [dataset provenance](../docs/DATA.md), [BEIR](https://github.com/beir-cellar/beir), [SciFact](https://github.com/allenai/scifact), and the [FiQA challenge](https://sites.google.com/view/fiqa/home).""")
    code("""sources = read(ROOT / 'sources.lock.json')
for item in sources:
    archive = ROOT / 'data/raw' / item['filename']
    assert archive.stat().st_size == item['bytes']
    assert digest(archive) == item['sha256']
display(pd.DataFrame(sources)[['name', 'bytes', 'sha256', 'url']])
""")
    code("""datasets = {}
for name in ('scifact', 'fiqa'):
    raw = ROOT / 'data/raw' / name
    corpus, _ = load_records(raw / 'corpus.jsonl', corpus=True, allow_empty=True)
    queries, _ = load_records(raw / 'queries.jsonl', corpus=False)
    labels = {p.stem: qrels(p) for p in sorted((raw / 'qrels').glob('*.tsv'))}
    datasets[name] = {'corpus': corpus, 'queries': queries, 'qrels': labels,
                      'audit': validate(raw, allow_empty_corpus=True)}
display(pd.DataFrame([{'dataset': n, 'documents': len(d['corpus']), 'queries': len(d['queries']),
                       'splits': ', '.join(d['qrels'])} for n,d in datasets.items()]))
""")
    md("""## Corpus quality

Empty fields, duplicate texts, and missing reference IDs measure different problems. A duplicate text with a distinct original ID is retained because the benchmark judges IDs. FiQA's empty documents remain empty; no title or body is inferred. Relevance scores of zero, if present, would be retained as judged negatives.""")
    code("""quality = []
for name,d in datasets.items():
    for component in ('corpus','queries'):
        audit = d['audit'][component]
        quality.append({'dataset':name,'component':component,'records':audit['count'],
                        'empty_texts':audit['empty_fields'].get('text',0),
                        'empty_titles':audit['empty_fields'].get('title',0),
                        'duplicate_text_excess':audit['duplicate_text_excess'],
                        'duplicate_ids':audit['duplicate_ids'],
                        'missing_required_fields':audit['missing_required_fields']})
display(pd.DataFrame(quality))
display(pd.DataFrame([{'dataset':name,'query_document_namespace_id_overlap':d['audit']['query_corpus_id_overlap']} for name,d in datasets.items()]))
assert all(d['audit']['valid'] for d in datasets.values())
""")
    code("""from collections import defaultdict
duplicate_quality=[]
for name,d in datasets.items():
    by_body=defaultdict(list)
    for did,row in d['corpus'].items(): by_body[row['text']].append(did)
    groups=[ids for body,ids in by_body.items() if body.strip() and len(ids)>1]
    test_positives={did for labels in d['qrels']['test'].values() for did,grade in labels.items() if grade>0}
    empty_ids={did for did,row in d['corpus'].items() if not row['text'].strip()}
    duplicate_quality.append({'dataset':name,'nonempty_duplicate_body_groups':len(groups),
                              'duplicate_groups_with_test_positives':sum(bool(set(ids)&test_positives) for ids in groups),
                              'empty_documents_with_test_positives':len(empty_ids & test_positives)})
display(pd.DataFrame(duplicate_quality))
""")
    code("""split_rows=[]
for name,d in datasets.items():
    for split, audit in d['audit']['splits'].items():
        split_rows.append({'dataset':name,'split':split,**{k:v for k,v in audit.items() if k != 'labels'},
                           'relevance_grades':str(audit['labels'])})
display(pd.DataFrame(split_rows))
display(pd.DataFrame({n:d['audit']['split_query_overlap'] for n,d in datasets.items()}).fillna(0))
""")
    md("""## Length distributions and vocabulary

Exploratory lengths use Unicode word matches (`\\w+`), while chunk sizes use whitespace tokens. Embedding models use their own subword tokenizers and a 256-token input limit. These units are reported separately: a 128-word chunk can exceed 128 subword tokens.

Length plots use logarithmic axes so long financial posts do not obscure short records. Zero-length records are reported separately and omitted from logarithmic histograms. The study retains the complete corpora, including their tails.""")
    code("""lengths={}
for name,d in datasets.items():
    lengths[name]={
        'documents':np.array([len(tokenize((r['title']+' '+r['text']).strip())) for r in d['corpus'].values()]),
        'queries':np.array([len(tokenize(r['text'])) for r in d['queries'].values()])}
display(pd.DataFrame([{'dataset':n,'kind':kind,'count':len(x),'mean':x.mean(),
                        'median':np.median(x),'p95':np.percentile(x,95),'max':x.max(),'zero':int((x==0).sum())}
                       for n,groups in lengths.items() for kind,x in groups.items()]))
""")
    code("""fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,kind in zip(axes,('documents','queries')):
    bins=np.geomspace(1,max(max(lengths[n][kind]) for n in datasets)+1,45)
    for name in datasets:
        x=lengths[name][kind]
        ax.hist(x[x>0],bins=bins,alpha=.55,label=name)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_title(kind.capitalize())
    ax.set_xlabel('Unicode word tokens'); ax.set_ylabel('Records'); ax.legend()
figure('length-distributions')
""")
    md("""SciFact documents have a median of 213 Unicode word tokens, compared with 95 for FiQA. Their 95th percentiles are 376.9 and 380, respectively. FiQA's maximum is 3,056 tokens, compared with 1,575 in SciFact. The financial corpus therefore combines many short posts with a long upper tail. Query medians are closer: 12 tokens in SciFact and 11 in FiQA. These word counts do not determine the exact amount of subword truncation; the tokenizer audit below measures that separately.""")
    code("""fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,kind in zip(axes,('documents','queries')):
    for name in datasets:
        x=np.sort(lengths[name][kind])
        ax.plot(x,np.arange(1,len(x)+1)/len(x),label=name)
    ax.set_xscale('symlog',linthresh=1); ax.set_xlabel('Unicode word tokens')
    ax.set_ylabel('Cumulative fraction'); ax.set_title(kind.capitalize()); ax.legend()
figure('length-ecdf')
""")
    code("""from collections import Counter
from rag_data.transforms import STOP
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
term_stop=STOP | set(ENGLISH_STOP_WORDS)
terms={}
for name,d in datasets.items():
    terms[name]=Counter(t for r in d['corpus'].values() for t in tokenize(r['text']) if len(t)>2 and t.isalpha() and t not in term_stop)
fig,axes=plt.subplots(1,2,figsize=(12,5))
for ax,(name,counts) in zip(axes,terms.items()):
    top=counts.most_common(15)[::-1]
    ax.barh([t for t,_ in top],[c for _,c in top]); ax.set_title(name); ax.set_xlabel('Corpus occurrences')
figure('corpus-terminology')
""")
    md("""Term counts describe vocabulary frequency, not topic labels. Scientific and financial corpora differ in document length, task, and annotation density as well as terminology. A difference in aggregate retrieval score cannot isolate domain alone.""")
    md("""### Distinctive scientific and financial vocabulary

The frequent-term plot excludes a standard English stop list. The next comparison uses log-base-two frequency ratios with additive smoothing of 0.5 over the shared vocabulary. A term must occur at least 50 times in the focal corpus. This reduces single-occurrence artifacts and shows words concentrated in one domain. The ratios are descriptive, without significance tests or inferred topic labels.""")
    code("""vocabulary=set(terms['scifact'])|set(terms['fiqa'])
distinctive={}
for name,counts in terms.items():
    other=terms['fiqa' if name=='scifact' else 'scifact']
    total=sum(counts.values())+.5*len(vocabulary)
    other_total=sum(other.values())+.5*len(vocabulary)
    rows=[{'term':term,'count':count,'log2_frequency_ratio':float(np.log2(((count+.5)/total)/((other[term]+.5)/other_total)))}
          for term,count in counts.items() if count>=50]
    distinctive[name]=pd.DataFrame(rows).sort_values('log2_frequency_ratio',ascending=False).head(15)
fig,axes=plt.subplots(1,2,figsize=(12,5))
for ax,(name,frame) in zip(axes,distinctive.items()):
    frame=frame.sort_values('log2_frequency_ratio')
    ax.barh(frame['term'],frame['log2_frequency_ratio']); ax.set_title(name)
    ax.set_xlabel('Log2 frequency ratio against the other corpus')
figure('distinctive-terminology')
display(pd.concat(distinctive))
""")
    md("""## Query-document relationships

Each query can have several judged relevant documents, and a document can support several queries. Unjudged documents are treated as nonrelevant for benchmark scoring, although incomplete annotation can penalize useful retrieval. The plots below describe positive judgments without copying source text.""")
    code("""relationship=[]
for name,d in datasets.items():
    for split,labels in d['qrels'].items():
        counts=[sum(v>0 for v in row.values()) for row in labels.values()]
        relationship.append({'dataset':name,'split':split,'judged_queries':len(labels),
                              'mean_relevant':np.mean(counts),'max_relevant':max(counts),
                              'recall_at1_ceiling':np.mean([1/n for n in counts]),
                              'unique_relevant_documents':len({did for row in labels.values() for did,v in row.items() if v>0})})
display(pd.DataFrame(relationship))
""")
    md("""The test judgments contain 1.13 relevant documents per SciFact query on average and 2.63 per FiQA query. A query with several positives cannot achieve Recall@1 of one: its ceiling is `1 / number of positives`. MRR can still reach one when the first result is relevant. Annotation density is another reason to interpret each dataset's comparisons independently.""")
    code("""fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,(name,d) in zip(axes,datasets.items()):
    counts=[sum(v>0 for v in row.values()) for row in d['qrels']['test'].values()]
    ax.hist(counts,bins=np.arange(0,max(counts)+2)-.5)
    ax.set_title(name+' test'); ax.set_xlabel('Relevant documents per query'); ax.set_ylabel('Queries')
figure('judgment-density')
""")
    code("""lexical_relationships={}
for name,d in datasets.items():
    rows=[]
    for qid,labels in d['qrels']['test'].items():
        qtokens=set(tokenize(d['queries'][qid]['text']))
        for did,grade in labels.items():
            if grade<=0: continue
            record=d['corpus'][did]
            dtokens=set(tokenize(record['title']+' '+record['text']))
            rows.append({'query_id':qid,'query_tokens':len(qtokens),'document_tokens':len(dtokens),
                         'query_term_coverage':len(qtokens & dtokens)/max(1,len(qtokens))})
    lexical_relationships[name]=pd.DataFrame(rows)
display(pd.concat({n:f[['query_tokens','document_tokens','query_term_coverage']].describe() for n,f in lexical_relationships.items()}))
""")
    code("""fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,(name,frame) in zip(axes,lexical_relationships.items()):
    ax.scatter(frame['query_tokens'],frame['query_term_coverage'],s=10,alpha=.25)
    ax.set_title(name+' judged pairs'); ax.set_xlabel('Unique query tokens'); ax.set_ylabel('Fraction found in relevant document')
figure('relevant-pair-lexical-coverage')
""")
    md("""## Experimental design

The staged matrix avoids a full Cartesian product. Chunking compares whole documents, fixed windows (128 words with zero or 24-word overlap; 256 words with 32-word overlap), paragraph/sentence boundary splitting, and sentence boundary splitting. Boundary methods cap segment length and retain overlap. They are deterministic heuristics; neither is semantic segmentation. Only the BM25-selected segmentation advances to dense retrieval. When whole documents win validation, no additional dense chunk configurations are encoded.

At most 150 validation queries are selected by SHA-256 order. FiQA uses its development split. SciFact uses its training split because BEIR provides no development qrels. IDs and normalized text duplicates shared with the test queries are excluded from validation selection. The validation choice is fixed before test scoring. Final test comparisons are descriptive and are not used to choose a deployment configuration.

BM25 uses `k1=0.9`, `b=0.4`, lowercase Unicode word tokenization, and deterministic ID tie breaking. Chunk candidates are mapped to unique document IDs before scoring. A 100-chunk pool may contain fewer than 20 unique documents; this coverage cost is part of the chunking comparison.""")
    code("""results={}
for name in datasets:
    path=ROOT/'reports/research'/f'{name}.json'
    results[name]=read(path) if path.exists() else run_dataset(name)
    assert REQUIRED_RUNS <= set(results[name]['runs']), f'Incomplete experiment matrix: {name}'
    assert all(row['query_count']==len(datasets[name]['qrels']['test']) for row in results[name]['runs'].values())
display(pd.DataFrame({n:r['selection'] for n,r in results.items()}).T)
display(read(ROOT/'reports/research/environment.json'))
""")
    code("""chunk_table=pd.concat([pd.DataFrame(r['chunking']).assign(dataset=n) for n,r in results.items()],ignore_index=True)
display(chunk_table[['dataset','method','size','overlap','chunks','covered_documents','build_seconds','ndcg@10','recall@20','p50_ms','p95_ms']])
display(pd.DataFrame({'selected_configuration':{n:str(r['selected_chunking']) for n,r in results.items()}}))
""")
    code("""fig,axes=plt.subplots(1,2,figsize=(13,4))
for ax,(name,frame) in zip(axes,chunk_table.groupby('dataset',sort=False)):
    labels=[f'{r.method} {r.size}/{r.overlap}' for r in frame.itertuples()]
    ax.barh(labels,frame['ndcg@10']); ax.set_title(name+' validation'); ax.set_xlabel('nDCG@10')
figure('chunking-validation')
""")
    code("""coverage=[]
for name,d in datasets.items():
    docs={did:(r['title']+' '+r['text']).strip() for did,r in d['corpus'].items()}
    for method,size,overlap in [('whole',128,24),('fixed',128,24),('recursive',128,24),('sentence',128,24)]:
        rows=chunks(docs,method,size,overlap)
        coverage.append({'dataset':name,'method':method,'chunks':len(rows),'mean_chunk_words':np.mean([c.tokens for c in rows]),
                         'p95_chunk_words':np.percentile([c.tokens for c in rows],95),
                         'word_replication_ratio':sum(c.tokens for c in rows)/max(1,sum(len(t.split()) for t in docs.values()))})
display(pd.DataFrame(coverage))
""")
    md("""## Pretrained embeddings and truncation

The study uses `all-MiniLM-L6-v2` and `multi-qa-MiniLM-L6-cos-v1`. The primary dense model is fixed to all-MiniLM before test evaluation; multi-qa is a model-comparison baseline. Exact model commits and file hashes are recorded in `models.lock.json`. Inputs combine title and body with one space, use the original query text, and truncate at 256 subword tokens. Vectors are L2 normalized, so inner product measures cosine similarity. Empty original records receive the model's empty-input representation and retain their IDs.

Encoding time is measured once when each matrix is first built. Loading a cached matrix preserves that original measurement. Search timings exclude cached query encoding; they are not end-to-end application latency. The tokenizer truncation audit quantifies an important whole-document limitation.""")
    code("""model_specs=read(ROOT/'models.lock.json')
display(pd.DataFrame([{k:v for k,v in s.items() if k!='files'} for s in model_specs]))
encoding_rows=[]
for name,r in results.items():
    for model,info in r['encoding'].items():
        for kind in ('corpus','queries'):
            encoding_rows.append({'dataset':name,'model':model,'kind':kind,'seconds':info[kind]['seconds'],
                                  'rows':info[kind]['shape'][0],'dimensions':info[kind]['shape'][1],
                                  'matrix_MiB':info[kind]['bytes']/2**20})
encoding=pd.DataFrame(encoding_rows)
display(encoding)
""")
    code("""embedding_integrity=[]
for name,r in results.items():
    for model,info in r['encoding'].items():
        path=ROOT/'data/experiments'/name/f'{model}-corpus.npy'
        vectors=np.load(path,mmap_mode='r')
        assert list(vectors.shape)==info['corpus']['shape']
        assert vectors.shape[0]==len(datasets[name]['corpus'])
        assert np.isfinite(vectors).all()
        error=float(np.max(np.abs(np.linalg.norm(vectors,axis=1)-1)))
        assert error<1e-5
        embedding_integrity.append({'dataset':name,'model':model,'rows':vectors.shape[0],
                                    'dimensions':vectors.shape[1],'maximum_norm_error':error,'file_sha256':digest(path)})
        del vectors
display(pd.DataFrame(embedding_integrity))
_=(ROOT/'reports/research/embedding-integrity.json').write_text(json.dumps(embedding_integrity,indent=2),encoding='utf-8')
""")
    code("""from transformers import AutoTokenizer
truncation=[]
for spec in [s for s in model_specs if s['role']=='embedding']:
    model_name=spec['id'].split('/')[-1]
    tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'data/models'/model_name),local_files_only=True)
    for name,d in datasets.items():
        text=[(r['title']+' '+r['text']).strip() for r in d['corpus'].values()]
        lens=[]
        for start in range(0,len(text),512):
            lens.extend(len(x) for x in tokenizer(text[start:start+512],truncation=False,verbose=False)['input_ids'])
        truncation.append({'dataset':name,'model':model_name,'documents_above_256':sum(x>256 for x in lens),
                           'fraction_above_256':np.mean(np.array(lens)>256),'p95_subwords':np.percentile(lens,95)})
display(pd.DataFrame(truncation))
""")
    code("""fig,axes=plt.subplots(1,2,figsize=(12,4))
frame=encoding[encoding['kind']=='corpus']
labels=frame['dataset']+' / '+frame['model'].str.replace('MiniLM-','',regex=False)
axes[0].barh(labels,frame['seconds']); axes[0].set_xlabel('Corpus encoding seconds')
axes[1].barh(labels,frame['matrix_MiB']); axes[1].set_xlabel('Float32 matrix MiB')
figure('encoding-costs')
""")
    md("""## Retrieval metrics

nDCG@10 uses linear relevance gain and log-base-two discount, matching the binary grades in these two snapshots. MRR@10 is the reciprocal rank of the first positive in the top ten. MAP@10 divides accumulated precision at relevant ranks by `min(10, number of positives)`. Recall@k divides retrieved positives by every judged positive for the query. Each metric is macro-averaged across the full test query set, including misses.

Chunk results are mapped to unique original document IDs before evaluation. A correct document match does not establish that the returned chunk contains an evidence sentence. SciFact's BEIR judgments identify cited articles, including citations without annotated evidence; these scores do not measure claim verification or generated-answer factuality.

Exact NumPy similarity and FAISS `IndexFlatIP` perform exhaustive search over normalized vectors. Local Qdrant uses the Python client's embedded persistent database with cosine similarity. It is a database integration comparison, not a benchmark of a networked Qdrant server or an approximate HNSW configuration.""")
    code("""metrics=pd.concat([pd.DataFrame(r['runs']).T.assign(dataset=n,run=lambda f:f.index) for n,r in results.items()],ignore_index=True)
numeric=['ndcg@10','mrr@10','map@10','recall@1','recall@3','recall@5','recall@10','recall@20','p50_ms','p95_ms']
metrics[numeric]=metrics[numeric].astype(float)
display(metrics[['dataset','run']+numeric].round(5))
""")
    code("""baseline_names=['bm25','bm25_chunked','dense_0','dense_1','faiss','qdrant','hybrid','rerank_20','rerank_50']
fig,axes=plt.subplots(1,2,figsize=(12,5))
for ax,name in zip(axes,datasets):
    frame=metrics[(metrics.dataset==name)&metrics.run.isin(baseline_names)].set_index('run')
    frame['ndcg@10'].sort_values().plot.barh(ax=ax)
    ax.set_title(name+' test'); ax.set_xlabel('nDCG@10')
figure('test-retrieval-comparison')
""")
    code("""fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,name in zip(axes,datasets):
    for label in ['bm25','dense_0','dense_1','hybrid','rerank_50']:
        row=metrics[(metrics.dataset==name)&(metrics.run==label)].iloc[0]
        ax.plot([1,3,5,10,20],[row[f'recall@{k}'] for k in [1,3,5,10,20]],marker='o',label=label)
    ax.set_title(name); ax.set_xlabel('k'); ax.set_ylabel('Recall'); ax.legend(fontsize=8)
figure('recall-curves')
""")
    code("""display(pd.DataFrame([{'dataset':name,'backend':backend,**values} for name,r in results.items() for backend,values in r['backend_checks'].items()]))
display(metrics[metrics.run.isin(['dense_0','faiss','qdrant'])][['dataset','run','ndcg@10','p50_ms','p95_ms','index_seconds']])
""")
    md("""The backend overlap check uses the first 30 test IDs in sorted order. It checks integration consistency, not statistical accuracy. Tied scores may produce different top-k boundary IDs across engines. Exhaustive index construction does not imply that every API wrapper has the same overhead or tie behavior.""")
    code("""from qdrant_client import QdrantClient
stored_identifiers=[]
for name,d in datasets.items():
    client=QdrantClient(path=str(ROOT/'data/experiments'/name/'qdrant'))
    try:
        count=client.count('documents',exact=True).count
        identifiers=[]; offset=None
        while True:
            points,offset=client.scroll('documents',limit=1024,offset=offset,with_payload=True,with_vectors=False)
            identifiers.extend(p.payload['id'] for p in points)
            if offset is None: break
        assert count==len(identifiers)==len(set(identifiers))==len(d['corpus'])
        assert set(identifiers)==set(d['corpus'])
        stored_identifiers.append({'dataset':name,'qdrant_points':count,'unique_original_document_ids':len(set(identifiers)),'all_ids_match':True})
    finally:
        client.close()
display(pd.DataFrame(stored_identifiers))
""")
    md("""### Index construction and storage

The following cell separately measures BM25 construction from the original documents and serializes an exhaustive FAISS index. FAISS construction times come from the experiment runner. Qdrant's constructor timer includes opening persistent storage and replacing an existing collection when present, so it is not a clean-slate-only build comparison. Qdrant storage is the size of its closed local database files. These sizes exclude model weights and Python dependencies. Matrix bytes and index files describe storage requirements; they are not peak process memory estimates.""")
    code("""import faiss
from rag_data.retrieval import BM25
from rag_data.dense import normalize
index_costs=[]
for name,d in datasets.items():
    docs={did:(r['title']+' '+r['text']).strip() for did,r in d['corpus'].items()}
    start=time.perf_counter(); temporary_index=BM25(docs); sparse_seconds=time.perf_counter()-start
    del temporary_index
    vectors=np.load(ROOT/'data/experiments'/name/'all-MiniLM-L6-v2-corpus.npy')
    index=faiss.IndexFlatIP(vectors.shape[1]); index.add(normalize(vectors))
    index_path=ROOT/'data/experiments'/name/'primary-flat.faiss'
    faiss.write_index(index,str(index_path))
    qdrant_path=ROOT/'data/experiments'/name/'qdrant'
    index_costs.append({'dataset':name,'bm25_build_seconds':sparse_seconds,
                       'faiss_build_seconds':results[name]['runs']['faiss']['index_seconds'],
                       'qdrant_build_seconds':results[name]['runs']['qdrant']['index_seconds'],
                       'dense_matrix_bytes':vectors.nbytes,'faiss_file_bytes':index_path.stat().st_size,
                       'qdrant_file_bytes':sum(p.stat().st_size for p in qdrant_path.rglob('*') if p.is_file())})
    del index,vectors
display(pd.DataFrame(index_costs))
_=(ROOT/'reports/research/index-costs.json').write_text(json.dumps(index_costs,indent=2),encoding='utf-8')
""")
    code("""memory_path=ROOT/'reports/research/memory-observation.json'
if memory_path.exists():
    display(read(memory_path))
else:
    print('The optional process-memory observer was not run. Matrix bytes are not process peak memory.')
""")
    md("""## Fusion and candidate pools

Reciprocal rank fusion combines BM25 and the first dense model using `1 / (constant + rank)` for each list. Constants 10 and 60 and pool depths 20 and 100 are compared on validation queries. Equal weights are fixed in advance. Fusion does not calibrate sparse and dense score magnitudes.""")
    code("""fusion=pd.concat([pd.DataFrame(r['fusion_validation']).assign(dataset=n) for n,r in results.items()],ignore_index=True)
display(fusion[['dataset','constant','depth','ndcg@10','recall@20','p50_ms','p95_ms']])
delta=[]
for name,r in results.items():
    for baseline in ['bm25','dense_0','dense_1']:
        delta.append({'dataset':name,'baseline':baseline,'hybrid_minus_baseline_ndcg':r['runs']['hybrid']['ndcg@10']-r['runs'][baseline]['ndcg@10']})
display(pd.DataFrame(delta))
""")
    md("""## Thresholds, filtering, and latency

Dense thresholds 0.2, 0.4, and 0.6 are sensitivity analyses, with no threshold chosen on test labels. The length filter permits documents with at least 20 Unicode word tokens. It measures a quality-filter trade-off; it does not claim that shorter documents are irrelevant. Both ablations can discard judged positives.

p50 and p95 use one sequential pass over queries on this machine, with one query submitted at a time. Timings exclude network round trips and do not estimate cold-cache behavior or confidence intervals over repeated trials. Dense queries are encoded in advance. Cross-encoder timings include candidate retrieval and model scoring.""")
    code("""display(metrics[metrics.run.str.startswith('threshold_')|metrics.run.isin(['dense_0','bm25','length_filter'])][['dataset','run','ndcg@10','recall@20','p50_ms','p95_ms']])
fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,name in zip(axes,datasets):
    frame=metrics[(metrics.dataset==name)&metrics.run.isin(baseline_names)]
    ax.scatter(frame['p50_ms'],frame['ndcg@10'])
    for _,row in frame.iterrows(): ax.annotate(row['run'],(row['p50_ms'],row['ndcg@10']),fontsize=7)
    ax.set_xscale('log'); ax.set_xlabel('p50 search / reranking ms'); ax.set_ylabel('nDCG@10'); ax.set_title(name)
figure('accuracy-latency')
""")
    md("""### Timing variation between completed passes

A cached verification pass repeats the retrieval comparisons with the same source hashes and stored embeddings. The following table retains both completed passes. Background host load was not controlled, so changes in p50 or p95 do not isolate an algorithmic effect. These two observations per method do not support a latency confidence interval. The main comparison tables use the verified pass rather than selecting the faster measurement.""")
    code("""repeat_path=ROOT/'reports/research/latency-repeatability.csv'
if repeat_path.exists():
    repeated=pd.read_csv(repeat_path)
    accuracy=repeated.pivot(index=['dataset','method'],columns='pass',values='ndcg@10')
    display(pd.DataFrame({'maximum_absolute_ndcg_difference':[(accuracy['initial']-accuracy['verified']).abs().max()]}))
    selected=repeated[repeated['method'].isin(['bm25','dense_0','faiss','qdrant','hybrid','rerank_50'])]
    display(selected.pivot(index=['dataset','method'],columns='pass',values=['p50_ms','p95_ms']))
else:
    print('Only one completed retrieval pass is recorded.')
""")
    md("""## Cross-encoder reranking

`ms-marco-TinyBERT-L2-v2` scores query-document pairs from the selected hybrid candidate generator. Candidate limits 20 and 50 use identical model parameters and the same 256-token pair limit. These are maximum limits: when validation selects a 20-result pool per retriever, their union contains at most 40 documents, often fewer because of overlap. The table records the actual candidate counts. The cross-encoder was pretrained elsewhere; no fine tuning uses these test judgments. These test comparisons describe a cost curve and do not select a final serving depth.""")
    code("""candidate_counts=[]
for name,r in results.items():
    depth=r['runs']['hybrid']['selected_depth']
    sparse=read(ROOT/'data/experiments'/name/'bm25-rankings.json')
    dense=read(ROOT/'data/experiments'/name/'dense_0-rankings.json')
    for limit in (20,50):
        counts=[min(limit,len({did for did,_ in sparse[qid]}|{did for did,_ in dense[qid]})) if depth==20 else limit for qid in datasets[name]['qrels']['test']]
        candidate_counts.append({'dataset':name,'configured_limit':limit,'fusion_pool_per_retriever':depth,
                                 'actual_min':min(counts),'actual_mean':np.mean(counts),'actual_max':max(counts)})
display(pd.DataFrame(candidate_counts))
_=(ROOT/'reports/research/candidate-counts.json').write_text(json.dumps(candidate_counts,indent=2),encoding='utf-8')
""")
    code("""display(metrics[metrics.run.isin(['hybrid','rerank_20','rerank_50'])][['dataset','run','ndcg@10','mrr@10','map@10','recall@20','p50_ms','p95_ms']])
for name,r in results.items():
    a,b=r['runs']['hybrid'],r['runs']['rerank_50']
    display(Markdown(f"On {name}, reranking with a maximum of 50 candidates changes nDCG@10 by {b['ndcg@10']-a['ndcg@10']:+.4f}; median measured retrieval/reranking time changes from {a['p50_ms']:.2f} to {b['p50_ms']:.2f} ms."))
""")
    md("""## Query transformations and semantic drift

Expansion uses the three most frequent eligible terms in the top three BM25 documents. It is unsupervised pseudo-relevance feedback and can reinforce an early mistake. Rewriting removes a fixed list of function words. Multi-query retrieval fuses the original, rewritten, and expanded BM25 lists. These methods execute locally without a language-generation service. They should not be interpreted as an LLM rewriting or HyDE experiment.

Semantic drift is operationalized as a lower query-level nDCG@10 than the unmodified BM25 query. This detects retrieval harm, not a direct semantic-distance measurement.""")
    code("""display(metrics[metrics.run.isin(['bm25','expansion','rewrite','multiquery'])][['dataset','run','ndcg@10','mrr@10','recall@20','p50_ms','p95_ms']])
per_query={}
for name,r in results.items():
    per_query[name]={label:pd.DataFrame(read(ROOT/'data/experiments'/name/f'{label}-per-query.json')).T for label in r['runs']}
drift=[]
for name,groups in per_query.items():
    base=groups['bm25']['ndcg@10']
    for method in ['expansion','rewrite','multiquery']:
        change=groups[method]['ndcg@10']-base
        drift.append({'dataset':name,'method':method,'improved':int((change>1e-12).sum()),'harmed':int((change < -1e-12).sum()),'tied':int((change.abs()<=1e-12).sum()),'mean_change':change.mean()})
display(pd.DataFrame(drift))
""")
    code("""polarity=[]
for name,d in datasets.items():
    groups=per_query[name]
    affected=[qid for qid in d['qrels']['test'] if 'not' in tokenize(d['queries'][qid]['text'])]
    delta=(groups['rewrite']['ndcg@10']-groups['bm25']['ndcg@10']).reindex(affected)
    polarity.append({'dataset':name,'test_queries_with_not':len(affected),
                     'rewrite_removes_not':len(affected),
                     'harmed_after_rewrite':int((delta < -1e-12).sum()),
                     'mean_ndcg_change':delta.mean() if len(delta) else np.nan})
display(pd.DataFrame(polarity))
""")
    md("""The fixed rewriting stop list removes `not`. That can change a claim's meaning even when retrieval metrics barely change, because the same article may be relevant to both a claim and its negation. The polarity table records this mechanism separately from the aggregate retrieval-harm measure. This rewriting heuristic requires revision before use in an application that must preserve question meaning.""")
    code("""fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,(name,groups) in zip(axes,per_query.items()):
    changes=[groups[m]['ndcg@10']-groups['bm25']['ndcg@10'] for m in ['expansion','rewrite','multiquery']]
    ax.boxplot(changes,tick_labels=['expansion','rewrite','multiquery'],showfliers=False)
    ax.axhline(0,color='grey',linewidth=.8); ax.set_title(name); ax.set_ylabel('Query nDCG change from BM25')
figure('transformation-drift')
""")
    md("""## Error analysis

Aggregate means can hide different failure populations. This analysis separates complete misses at 20, late relevant results, and first-page ranking errors. Query IDs and the original query text in the local failure tables remain local. The public notebook contains the same analysis code with every saved output cleared; no source documents or queries are copied into its cell sources.""")
    code("""error_rows=[]
for name,groups in per_query.items():
    for method in ['bm25','dense_0','hybrid','rerank_50']:
        frame=groups[method]
        error_rows.append({'dataset':name,'method':method,'complete_misses_at20':int((frame['recall@20']==0).sum()),
                           'positive_only_after10':int(((frame['recall@20']>0)&(frame['recall@10']==0)).sum()),
                           'positive_after_rank1':int(((frame['recall@1']==0)&(frame['recall@10']>0)).sum())})
display(pd.DataFrame(error_rows))
""")
    code("""error_features={}
for name,d in datasets.items():
    base=per_query[name]['bm25'].copy()
    base['query_words']=[len(tokenize(d['queries'][qid]['text'])) for qid in base.index]
    base['relevant_documents']=[sum(v>0 for v in d['qrels']['test'][qid].values()) for qid in base.index]
    base['length_bin']=pd.qcut(base['query_words'].rank(method='first'),4,labels=['Q1','Q2','Q3','Q4'])
    error_features[name]=base
    display(Markdown(f'### {name}: query-length strata'))
    display(base.groupby('length_bin',observed=True)[['query_words','ndcg@10','recall@20','relevant_documents']].agg(['mean','count']))
""")
    code("""for name,groups in per_query.items():
    frame=error_features[name][['query_words','relevant_documents']].copy()
    frame['bm25_ndcg']=groups['bm25']['ndcg@10']
    frame['dense_ndcg']=groups['dense_0']['ndcg@10']
    frame['rerank_ndcg']=groups['rerank_50']['ndcg@10']
    frame['rerank_minus_bm25']=frame['rerank_ndcg']-frame['bm25_ndcg']
    display(Markdown(f'### {name}: largest reranking regressions'))
    display(frame.sort_values('rerank_minus_bm25').head(8))
    display(Markdown(f'### {name}: largest reranking gains'))
    display(frame.sort_values('rerank_minus_bm25',ascending=False).head(8))
""")
    code("""for name,d in datasets.items():
    groups=per_query[name]
    change=(groups['expansion']['ndcg@10']-groups['bm25']['ndcg@10']).sort_values()
    examples=pd.DataFrame({'query':[d['queries'][qid]['text'] for qid in change.head(5).index],
                            'bm25_ndcg':groups['bm25'].loc[change.head(5).index,'ndcg@10'],
                            'expanded_ndcg':groups['expansion'].loc[change.head(5).index,'ndcg@10'],
                            'change':change.head(5)})
    display(Markdown(f'### {name}: queries where the original beats expansion'))
    display(examples)
""")
    md("""## Paired uncertainty and ablations

The paired bootstrap resamples test queries with replacement 2,000 times using seed 42. Intervals describe the mean query-level nDCG difference for the sampled benchmark queries. Resampling treats query records as independent; repeated or related claims can make intervals too narrow. The intervals do not account for corpus selection, annotation incompleteness, model pretraining overlap, or multiple comparisons. The tests are exploratory; no family-wise significance claim follows from an interval excluding zero.""")
    code("""rng=np.random.default_rng(42)
uncertainty=[]
for name,groups in per_query.items():
    for baseline,method in [('bm25','dense_0'),('bm25','hybrid'),('dense_0','hybrid'),('hybrid','rerank_20'),('hybrid','rerank_50'),('bm25','expansion')]:
        diff=(groups[method]['ndcg@10']-groups[baseline]['ndcg@10']).to_numpy()
        draws=rng.choice(diff,size=(2000,len(diff)),replace=True).mean(axis=1)
        uncertainty.append({'dataset':name,'comparison':method+' minus '+baseline,'mean_change':diff.mean(),
                            'ci_low':np.percentile(draws,2.5),'ci_high':np.percentile(draws,97.5),
                            'queries':len(diff)})
uncertainty=pd.DataFrame(uncertainty)
display(uncertainty)
uncertainty.to_csv(ROOT/'reports/research/paired-bootstrap.csv',index=False)
""")
    code("""fig,axes=plt.subplots(1,2,figsize=(13,5))
for ax,(name,frame) in zip(axes,uncertainty.groupby('dataset',sort=False)):
    x=frame['mean_change'].to_numpy(); lo=frame['ci_low'].to_numpy(); hi=frame['ci_high'].to_numpy()
    ax.errorbar(x,range(len(frame)),xerr=[x-lo,hi-x],fmt='o',capsize=3)
    ax.set_yticks(range(len(frame)),frame['comparison']); ax.axvline(0,color='grey',linewidth=.8)
    ax.set_title(name); ax.set_xlabel('Paired mean nDCG@10 change, 95% bootstrap interval')
figure('paired-uncertainty')
""")
    md("""## Recomputing the benchmark metrics

The saved ranking lists contain original document IDs and scores. Recomputing each reported aggregate from these lists checks that labels, deduplication, and the stored results agree. The local lists are inspection artifacts and are excluded from publication.""")
    code("""recomputed=[]
for name,r in results.items():
    for method,reported in r['runs'].items():
        scored=read(ROOT/'data/experiments'/name/f'{method}-rankings.json')
        rankings={qid:[did for did,score in rows] for qid,rows in scored.items()}
        assert set(rankings)==set(datasets[name]['qrels']['test'])
        assert all(len(row)==len(set(row)) for row in rankings.values())
        measured,_=evaluate(datasets[name]['qrels']['test'],rankings,set(datasets[name]['corpus']))
        for metric,value in measured.items():
            assert np.isclose(value,reported[metric],atol=1e-12)
        recomputed.append({'dataset':name,'run':method,'queries':len(rankings),'all_metrics_match':True})
display(pd.DataFrame(recomputed))
""")
    md("""## Sparse and dense complementarity

Two systems can have similar means while succeeding on different queries. The table compares complete misses at rank 20 and mean top-20 document-set overlap. A union oracle indicates candidate coverage only; it is not a deployable ranker or an additional benchmark score.""")
    code("""complementarity=[]
for name,groups in per_query.items():
    sparse=read(ROOT/'data/experiments'/name/'bm25-rankings.json')
    dense=read(ROOT/'data/experiments'/name/'dense_0-rankings.json')
    overlap=[]; union_recall=[]
    for qid,labels in datasets[name]['qrels']['test'].items():
        a={did for did,_ in sparse[qid][:20]}; b={did for did,_ in dense[qid][:20]}
        positives={did for did,grade in labels.items() if grade>0}
        overlap.append(len(a & b)/max(1,len(a | b)))
        union_recall.append(len((a | b) & positives)/len(positives))
    smiss=groups['bm25']['recall@20']==0
    dmiss=groups['dense_0']['recall@20']==0
    complementarity.append({'dataset':name,'both_miss':int((smiss & dmiss).sum()),
                             'sparse_only_finds_positive':int((~smiss & dmiss).sum()),
                             'dense_only_finds_positive':int((smiss & ~dmiss).sum()),
                             'mean_top20_jaccard':np.mean(overlap),'union_candidate_recall':np.mean(union_recall)})
display(pd.DataFrame(complementarity))
""")
    md("""## Conclusions from the executed comparisons""")
    code("""for name,r in results.items():
    drift_frame=pd.DataFrame(drift)
    harmed=int(drift_frame[(drift_frame.dataset==name)&(drift_frame.method=='expansion')]['harmed'].iloc[0])
    display(Markdown(f"Pseudo-relevance expansion lowers query-level nDCG for **{harmed}** of {r['selection']['test_queries']} test queries. Average gains therefore need to be assessed alongside regressions."))
""")
    md("""## Limitations and next experiments

The matrix uses two small pretrained encoders and one small reranker on CPU. It does not establish performance for larger models or production traffic. The models may have encountered related material during pretraining; this study does not audit their training corpora. Sentence and recursive segmentation are heuristic boundary methods. Embedding-based semantic chunking and generative HyDE are not executed. No answer generator is evaluated.

Validation uses at most 150 queries, so configuration selection can be noisy. Choosing segmentation with BM25 can miss configurations that would suit a dense encoder; unselected dense chunk configurations are not measured. Text deduplication protects exact normalized query matches but cannot remove paraphrase leakage. Test-set error strata and transformation comparisons are exploratory, with no subsequent tuning on those labels. Larger candidate pools can improve recall while increasing reranker cost. Score thresholds are model-specific and should be chosen on separate validation data before serving.

Encoding matrices, stored index files, and experiment directories measure artifact sizes. The optional process observer records sampled RSS and the Windows lifetime peak-working-set counter seen during observation; it does not isolate each index's memory cost. Timings come from one machine and one sequential pass. FiQA exceeds the Qdrant client's 20,000-point recommendation for embedded mode. These local results do not predict server throughput. Raw text rights remain separate from archive availability; this repository distributes acquisition instructions and aggregate results only.

## Reproducibility and engineering

The maintained Python modules separate validation, chunking, dense indexing, lexical retrieval, fusion, transformations, and experiments. The tests cover numerical examples, backend consistency, original-ID aggregation, deterministic selection, invalid inputs, and real dataset integrity. Use the commands in [README](../README.md) to acquire pinned archives and models, run experiments, execute this notebook in a clean kernel, and verify the output-free public copy. The full local notebook retains every output; publication changes only execution state and metadata in the public copy.""")
    code("""metrics[['dataset','run']+numeric].to_csv(ROOT/'reports/research/metrics.csv',index=False)
print('Saved local figures:', len(list(FIGURES.glob('*.png'))))
print('Complete test queries:', {n:r['selection']['test_queries'] for n,r in results.items()})
print('Aggregate metrics:', ROOT/'reports/research/metrics.csv')
""")
    result_paths = {n: ROOT / 'reports/research' / f'{n}.json' for n in ('scifact', 'fiqa')}
    required = {'bm25', 'dense_0', 'dense_1', 'hybrid', 'rerank_20', 'rerank_50', 'multiquery'}
    measured = {n: json.loads(p.read_text()) for n,p in result_paths.items() if p.exists()}
    if len(measured) == 2 and all(required <= set(r['runs']) for r in measured.values()):
        findings = []
        for name,r in measured.items():
            runs = r['runs']
            base, d0, d1, hybrid, cross = [runs[k] for k in ('bm25','dense_0','dense_1','hybrid','rerank_50')]
            chosen = r['selected_chunking']
            text = f"### {name}: measured findings\n\n"
            text += f"BM25 reaches test nDCG@10 {base['ndcg@10']:.4f}, compared with {d0['ndcg@10']:.4f} for all-MiniLM and {d1['ndcg@10']:.4f} for multi-qa-MiniLM. Recall@20 is {base['recall@20']:.4f}, {d0['recall@20']:.4f}, and {d1['recall@20']:.4f}, respectively.\n\n"
            text += f"Validation selects `{chosen[0]}` segmentation (size {chosen[1]}, overlap {chosen[2]}; size is ignored for whole documents). Hybrid retrieval changes nDCG@10 by {hybrid['ndcg@10']-base['ndcg@10']:+.4f} relative to BM25. Reranking with a maximum of 50 candidates changes it by {cross['ndcg@10']-hybrid['ndcg@10']:+.4f} relative to hybrid retrieval.\n\n"
            boundary = [row for row in r['chunking'] if row['method'] in ('recursive','sentence')]
            if len(boundary) == 2 and boundary[0]['chunks'] == boundary[1]['chunks'] and boundary[0]['ndcg@10'] == boundary[1]['ndcg@10']:
                text += "Recursive and sentence boundary variants produce the same chunk count and validation nDCG in this matrix. They provide no separate evidence of a segmentation gain.\n\n"
            text += f"The measured median for hybrid retrieval is {hybrid['p50_ms']:.2f} ms. Reranking with a maximum of 50 candidates has median {cross['p50_ms']:.2f} ms and p95 {cross['p95_ms']:.2f} ms. These values exclude dense query encoding and describe the recorded local environment. Expansion changes nDCG@10 by {runs['expansion']['ndcg@10']-base['ndcg@10']:+.4f}; multi-query fusion changes it by {runs['multiquery']['ndcg@10']-base['ndcg@10']:+.4f}."
            findings.append(nbf.v4.new_markdown_cell(text))
        position = next(i for i,c in enumerate(cells) if c.cell_type == 'markdown' and c.source.startswith('## Conclusions from'))
        cells[position + 1:position + 1] = findings
    notebook=nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec={"display_name":"Python 3", "language":"python", "name":"python3"}
    destination=ROOT/'notebooks/01_complete_retrieval_research.ipynb'
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        prior = nbf.read(destination,as_version=4)
        if any(c.get('outputs') or c.get('execution_count') is not None for c in prior.cells):
            raise FileExistsError('The local notebook contains execution state; edit its sources directly.')
    nbf.write(notebook,destination)
    print(f'Wrote {len(cells)} cells ({sum(c.cell_type=="code" for c in cells)} code cells).')


if __name__=='__main__':
    build()
