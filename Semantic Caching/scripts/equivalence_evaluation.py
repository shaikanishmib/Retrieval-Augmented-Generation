"""Held-out sentence equivalence, synthetic stress pairs and permissive QA reuse."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from semantic_cache.artifacts import ROOT,read,write,identity
from semantic_cache.data import stress_pairs
from semantic_cache.models import Encoder
from semantic_cache.security import lexical_guard

freeze = read(ROOT/'reports/selection-freeze.json')
assert freeze['identity']==identity()
partitions = read(ROOT/'data/partitions/paws.json')
rows = partitions['evaluation']
texts = [r['sentence1'] for r in rows]+[r['sentence2'] for r in rows]
labels = np.asarray([r['label'] for r in rows])
guards = np.asarray([lexical_guard(r['sentence1'],r['sentence2']) for r in rows])
reports = {}
stress_reports = []
for alias in ['minilm','bge-small','tfidf']:
    if alias=='tfidf':
        import time
        encoder = TfidfVectorizer(ngram_range=(1,2),max_features=20000)
        # Fit only on development text; held-out text never sets IDF or vocabulary.
        encoder.fit([r[k] for r in partitions['development'] for k in ['sentence1','sentence2']])
        tick = time.perf_counter()
        v = encoder.transform(texts)
        ms = (time.perf_counter()-tick)*1000
        scores = np.asarray(v[:len(rows)].multiply(v[len(rows):]).sum(axis=1)).ravel()
    else:
        encoder = Encoder(alias)
        encoder.encode(['discarded warm-up'])
        v,ms = encoder.encode(texts)
        scores = np.sum(v[:len(rows)]*v[len(rows):],axis=1)
        stress = stress_pairs()
        sv,_ = encoder.encode([p[0] for p in stress]+[p[1] for p in stress])
        for i,(a,b,kind) in enumerate(stress):
            stress_reports.append(dict(encoder=alias,kind=kind,similarity=float(sv[i]@sv[i+len(stress)]),guard=lexical_guard(a,b),synthetic=True,expected_equivalent=False))
    sweep = []
    for threshold in read(ROOT/'configs/study.json')['thresholds']:
        for guarded in [False,True]:
            accepted = (scores>=threshold)&(guards if guarded else True)
            tp,fp = int(np.sum(accepted&(labels==1))),int(np.sum(accepted&(labels==0)))
            sweep.append(dict(threshold=threshold,guard=guarded,tp=tp,fp=fp,accepted=int(accepted.sum()),precision=tp/(tp+fp) if tp+fp else None,recall=tp/int(labels.sum()),false_positive_rate=fp/int(np.sum(labels==0))))
    reports[alias] = dict(pairs=len(rows),encode_ms=ms,sweep=sweep,score_quantiles={str(label):np.quantile(scores[labels==label],[0,.25,.5,.75,1]).tolist() for label in [0,1]})
    write(ROOT/f'data/evaluation/{alias}-pairs.json',[dict(id=r['id'],score=float(s),label=int(r['label']),guard=bool(g)) for r,s,g in zip(rows,scores,guards)])
    if alias!='tfidf':
        encoder.close()
write(ROOT/'reports/equivalence-evaluation.json',reports)
write(ROOT/'reports/synthetic-stress.json',stress_reports)
print(reports[freeze['encoder']],flush=True)
