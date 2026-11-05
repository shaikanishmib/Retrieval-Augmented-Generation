"""Task-group uncertainty for frozen detector comparisons."""
import numpy as np
from sklearn.metrics import average_precision_score
from rag_security.artifacts import ROOT, read, write

data=read(ROOT/'data/detectors/predictions.json')
n=len(read(ROOT/'data/tasks/development.json'))
# Each task contributes a different number of compatible attacks. Use IDs rather
# than an assumed fixed examples-per-task count to identify the held-out rows.
heldout={r['id'] for r in read(ROOT/'data/tasks/evaluation.json')}
indices=[i for i,r in enumerate(data['rows']) if r['group'] in heldout]
rows=[data['rows'][i] for i in indices]
y=np.array([r['label'] for r in rows])
groups=sorted({r['group'] for r in rows})
members={g:np.array([i for i,r in enumerate(rows) if r['group']==g]) for g in groups}
rng=np.random.default_rng(606)
samples=[np.concatenate([members[g] for g in rng.choice(groups,len(groups),replace=True)]) for _ in range(2000)]
summary={}
f1draws={}
for method,all_scores in data['scores'].items():
    scores=np.array([all_scores[i] for i in indices])
    pred=scores>=data['thresholds'][method]
    draws=[]
    aps=[]
    for ids in samples:
        a,b=y[ids],pred[ids]
        tp=int(sum((a==1)&b));fp=int(sum((a==0)&b));fn=int(sum((a==1)&~b))
        draws.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
        aps.append(float(average_precision_score(a,scores[ids])))
    f1draws[method]=np.array(draws)
    summary[method]={'f1_task_group_bootstrap_95':np.quantile(draws,[.025,.975]).tolist(),'average_precision_task_group_bootstrap_95':np.quantile(aps,[.025,.975]).tolist()}
paired={method:np.quantile(f1draws[method]-f1draws['tfidf'],[.025,.975]).tolist() for method in f1draws if method!='tfidf'}
write(ROOT/'reports/detector-uncertainty.json',{'groups':len(groups),'examples':len(rows),'replicates':2000,'methods':summary,'paired_f1_difference_from_tfidf_95':paired,'limitations':'Task-group resampling preserves repeated family outcomes within tasks, but does not create independent payload families.'})
print(summary)
