"""Describe frozen errors by original annotations; no configurations are changed."""
import numpy as np
from llm_judge.artifacts import ROOT, read, write
from llm_judge.metrics import evaluate
from llm_judge.claims import tokens

# The adapter records original-versus-bounded evidence lengths for Qwen only.
# Missing instrumentation must not appear as a measured zero-truncation rate.
for split in ['development','validation','test']:
    path = ROOT/f'data/scores/{split}.json'
    artifact = read(path)
    for name,record in artifact['runtime'].items():
        if not name.startswith('qwen') and 'truncated_context_fraction' in record:
            record['truncated_context_fraction'] = None
            record['truncation_scope'] = 'Pair tokenization is capped at 512; original pair lengths were not instrumented. Fraction unmeasured.'
    write(path,artifact)
heldout_path = ROOT/'reports/heldout.json'
heldout = read(heldout_path)
heldout['runtime'] = read(ROOT/'data/scores/test.json')['runtime']
write(heldout_path,heldout)

rows = read(ROOT/'data/partitions/test.json')
freeze = read(ROOT/'reports/selection-freeze.json')
p = np.array(read(ROOT/'data/scores/test.json')['scores'][freeze['selected']])
y = np.array([r['y'] for r in rows])
pred = p >= freeze['thresholds'][freeze['selected']]
categories = {
    'any implicit_true': [any(s.get('implicit_true') for s in r['labels']) for r in rows],
    'any conflict': [any('Conflict' in s['label_type'] for s in r['labels']) for r in rows],
    'any baseless information': [any('Baseless' in s['label_type'] for s in r['labels']) for r in rows],
    'any subtle error': [any('Subtle' in s['label_type'] for s in r['labels']) for r in rows],
}
breakdown = {}
for category,mask in categories.items():
    ix = np.array(mask,dtype=bool)
    breakdown[category] = {'responses':int(ix.sum()),'missed_hallucination_fraction':float(np.mean(~pred[ix])) if ix.any() else None}
quality = {key:{'responses':sum(r['quality']==key for r in rows),'predicted_hallucinations':sum(bool(v) for r,v in zip(rows,pred) if r['quality']==key)} for key in sorted({r['quality'] for r in rows})}
by_generator = {key:evaluate(y[[r['model']==key for r in rows]],p[[r['model']==key for r in rows]],freeze['thresholds'][freeze['selected']]) for key in sorted({r['model'] for r in rows})}
groups = sorted({r['source_id'] for r in rows})
indexes = {g:[i for i,r in enumerate(rows) if r['source_id']==g] for g in groups}
rng = np.random.default_rng(58021)
intervals = {key:[] for key in ['accuracy','precision','recall','f1']}
for _ in range(2000):
    ix = np.concatenate([indexes[g] for g in rng.choice(groups,len(groups),replace=True)])
    actual,guess = y[ix],pred[ix]
    tp = np.sum((actual==1)&guess); fp = np.sum((actual==0)&guess); fn = np.sum((actual==1)&~guess)
    values = {'accuracy':np.mean(actual==guess),'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'f1':2*tp/max(1,2*tp+fp+fn)}
    for key,value in values.items(): intervals[key].append(value)
write(ROOT/'reports/error-analysis.json',{'configuration':freeze['selected'],'annotation_subsets':breakdown,'subset_policy':'Annotation subsets overlap. Miss rates measure detection of human evidence-support labels, including true-but-unestablished assertions; they do not establish factual correctness.','quality_flags':quality,'by_generator':by_generator,'selected_source_bootstrap_intervals_95':{key:np.quantile(values,[.025,.975]).tolist() for key,values in intervals.items()},'bootstrap_replicates':2000,'resampling_unit':'source_id','causal_limit':'Generator-specific error rates describe the sampled responses and are not evidence of self-preference or causal identity bias.'})
print('Described annotation-type errors and selected-system confidence intervals.')
