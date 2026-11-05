"""QA scores, cache denominators and paired source-group uncertainty."""
from collections import Counter
import re
import string
import numpy as np

def normalized_answer(text):
    text = ''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',text).split())

def qa_score(prediction, answers):
    p = normalized_answer(prediction).split()
    em,f1 = 0.,0.
    for answer in answers:
        a = normalized_answer(answer).split()
        em = max(em,float(p==a))
        common = sum((Counter(p)&Counter(a)).values())
        score = float(p==a) if not p or not a else 2*common/(len(p)+len(a))
        f1 = max(f1,score)
    return em,f1

def aggregate(rows):
    n = len(rows)
    hits = [r for r in rows if r['hit']!='miss']
    false = sum(not r['reuse_equivalent'] for r in hits)
    ms = np.asarray([r['total_ms'] for r in rows])
    return dict(requests=n,exact_hit_rate=sum(r['hit']=='exact' for r in rows)/n,semantic_hit_rate=sum(r['hit']=='semantic' for r in rows)/n,miss_rate=1-len(hits)/n,false_hits=false,false_hit_rate_per_request=false/n,false_hit_rate_per_hit=false/len(hits) if hits else None,correct_hit_precision=1-false/len(hits) if hits else None,em=float(np.mean([r['em'] for r in rows])),f1=float(np.mean([r['f1'] for r in rows])),p50_ms=float(np.percentile(ms,50)),p95_ms=float(np.percentile(ms,95)),requests_per_second=1000*n/float(ms.sum()),generation_calls=sum(r['generation_calls'] for r in rows),gpu_generation_ms=sum(r['gpu_generation_ms'] for r in rows),mean_lookup_ms=float(np.mean([r['lookup_ms'] for r in rows])),mean_embedding_ms=float(np.mean([r['embedding_ms'] for r in rows])),mean_retrieval_ms=float(np.mean([r['retrieval_ms'] for r in rows])),mean_generation_ms=float(np.mean([r['generation_ms'] for r in rows])))

def paired_interval(left,right,field,seed=20261102,repetitions=2000):
    assert [(r['trial'],r['request_id']) for r in left]==[(r['trial'],r['request_id']) for r in right]
    groups = sorted({r['group'] for r in left})
    differences = {g:[b[field]-a[field] for a,b in zip(left,right) if a['group']==g] for g in groups}
    rng = np.random.default_rng(seed)
    samples = [np.mean([v for g in rng.choice(groups,len(groups),replace=True) for v in differences[g]]) for _ in range(repetitions)]
    return dict(mean_difference=float(np.mean([b[field]-a[field] for a,b in zip(left,right)])),low=float(np.percentile(samples,2.5)),high=float(np.percentile(samples,97.5)),unit='source group',replicates=repetitions)
