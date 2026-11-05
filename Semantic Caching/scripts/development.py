"""Select models and thresholds on development records, then freeze the study."""
import time
import numpy as np
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer
from semantic_cache.artifacts import ROOT,read,write,identity,digest
from semantic_cache.data import stress_pairs
from semantic_cache.metrics import qa_score
from semantic_cache.models import Encoder,Generator
from semantic_cache.rag import Pipeline
from semantic_cache.security import Corpus,Document,Principal,lexical_guard

cfg = read(ROOT/'configs/study.json')
freeze_path = ROOT/'reports/selection-freeze.json'
if freeze_path.exists():
    assert read(freeze_path)['identity']==identity()
    print('Development selection already frozen; preserving it.')
    raise SystemExit(0)
paws = read(ROOT/'data/partitions/paws.json')['development']
texts = [r['sentence1'] for r in paws]+[r['sentence2'] for r in paws]
labels = np.asarray([r['label'] for r in paws])
model_reports = {}
for alias in ['minilm','bge-small','tfidf']:
    start = time.perf_counter()
    rss = psutil.Process().memory_info().rss
    if alias=='tfidf':
        encoder = TfidfVectorizer(ngram_range=(1,2),max_features=20000)
        vectors = encoder.fit_transform(texts)
        scores = np.asarray(vectors[:len(paws)].multiply(vectors[len(paws):]).sum(axis=1)).ravel()
        load_ms,encode_ms,dimensions = 0.,(time.perf_counter()-start)*1000,vectors.shape[1]
    else:
        encoder = Encoder(alias)
        encoder.encode(['discarded warm-up'])
        vectors,encode_ms = encoder.encode(texts)
        scores = np.sum(vectors[:len(paws)]*vectors[len(paws):],axis=1)
        load_ms,dimensions = encoder.load_ms,encoder.dimensions
    sweep = []
    guard = np.asarray([lexical_guard(r['sentence1'],r['sentence2']) for r in paws])
    for threshold in cfg['thresholds']:
        for guarded in [False,True]:
            accepted = (scores>=threshold)&(guard if guarded else True)
            tp,fp = int(np.sum(accepted&(labels==1))),int(np.sum(accepted&(labels==0)))
            sweep.append(dict(threshold=threshold,guard=guarded,tp=tp,fp=fp,accepted=int(accepted.sum()),precision=tp/(tp+fp) if tp+fp else None,recall=tp/int(labels.sum()),false_positive_rate=fp/int(np.sum(labels==0))))
    model_reports[alias] = dict(load_ms=load_ms,encode_ms=encode_ms,dimensions=dimensions,rss_delta=psutil.Process().memory_info().rss-rss,sweep=sweep)
    write(ROOT/f'data/development/{alias}-scores.json',[dict(id=r['id'],label=int(r['label']),score=float(score),guard=bool(g)) for r,score,g in zip(paws,scores,guard)])
    if alias!='tfidf':
        encoder.close()
    print('Encoder development',alias,flush=True)
write(ROOT/'reports/embedding-development.json',model_reports)
choices = [(r['tp'], -model_reports[a]['encode_ms'],a,r['threshold']) for a in ['minilm','bge-small'] for r in model_reports[a]['sweep'] if r['guard'] and r['fp']==0]
_,_,selected_encoder,threshold = max(choices)
generator_reports = {}
for alias in ['flan-small','flan-base']:
    generator = Generator(alias)
    generator.generate('What is two plus two?',['Two plus two is four.'])
    traces = []
    for dataset in ['hotpot','squad']:
        for row in read(ROOT/f'data/partitions/{dataset}.json')['development']:
            corpus = Corpus([Document(k,v) for k,v in row['documents'].items()])
            p = Principal('public','development',frozenset(corpus.documents))
            pipeline = Pipeline(corpus,generator,None,cfg)
            result = pipeline.request(row['question'],p)
            em,f1 = qa_score(result['answer'],row['answers'])
            traces.append(dict(dataset=dataset,id=row['id'],em=em,f1=f1,**result))
    write(ROOT/f'data/development/{alias}-qa.json',traces)
    generator_reports[alias] = dict(load_ms=generator.load_ms,mean_f1=float(np.mean([r['f1'] for r in traces])),mean_em=float(np.mean([r['em'] for r in traces])),p50_ms=float(np.median([r['total_ms'] for r in traces])),questions=len(traces))
    print('Generator development',alias,generator_reports[alias],flush=True)
    generator.close()
selected_generator = max(generator_reports,key=lambda a:generator_reports[a]['mean_f1'])
write(ROOT/'reports/generator-development.json',generator_reports)
freeze = dict(identity=identity(),encoder=selected_encoder,threshold=threshold,generator=selected_generator,created_unix=time.time(),selection_rule='Maximize development true accepts with zero observed false accepts and guard; ties use measured batch encoding time. Generator maximizes development QA F1.',partitions={p.name:digest(read(p)) for p in (ROOT/'data/partitions').glob('*.json')})
write(freeze_path,freeze)
print(freeze,flush=True)
