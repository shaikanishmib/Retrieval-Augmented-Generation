"""Measure perturbation sensitivity of the frozen selected system itself."""
import gc
import subprocess
import sys
import numpy as np
import torch
from llm_judge.artifacts import ROOT, read, write
from llm_judge.bias import perturb
from llm_judge.claims import lexical_score, token_f1
from llm_judge.classifier import predict
from llm_judge.models import Adapter
from llm_judge.tracking import infer

freeze = read(ROOT/'reports/selection-freeze.json')
rows = read(ROOT/'data/partitions/test.json')
subset = []
for task in ['QA','Summary','Data2txt']:
    unique = {}
    for row in rows:
        if row['task']==task: unique.setdefault(row['source_id'],row)
    subset.extend(list(unique.values())[:6])
kinds = ['baseline','verbosity','identity','injection','irrelevant','truncate','reverse-evidence']
combined = [r if kind=='baseline' else perturb(r,kind) for kind in kinds for r in subset]
required = set()
def collect(name):
    if name in freeze['calibration']:
        for feature in freeze['calibration'][name]['features']: collect(feature)
    else:
        required.add(name)
collect(freeze['selected'])
scores = {}
for name in required:
    if name=='lexical': scores[name] = [lexical_score(r) for r in combined]
    elif name=='supervised-tfidf': scores[name] = predict(combined)[0]
    elif name=='exact-substring': scores[name] = [float(r['response'] not in r['context']) for r in combined]
    elif name=='token-f1-proxy': scores[name] = [1-token_f1(r['response'],r['context']) for r in combined]
    elif name in ['majority','prevalence']: scores[name] = [float(freeze[name])]*len(combined)
keys = sorted({name.split('/')[0] for name in required if '/' in name})
for key in keys:
    adapter = Adapter(key)
    for name in sorted(required):
        if not name.startswith(key+'/'): continue
        variant = name.split('/')[1]
        actual_variant = 'claims' if variant=='claims-mean' else variant
        traces = infer(combined,adapter,actual_variant)
        scores[name] = [t['mean_score'] if variant=='claims-mean' else t['score'] for t in traces]
    del adapter
    gc.collect()
    torch.cuda.empty_cache()
def assemble(name):
    if name in scores: return np.array(scores[name])
    cal = freeze['calibration'][name]
    x = np.array([assemble(feature) for feature in cal['features']]).T
    return 1/(1+np.exp(-(x@np.array(cal['coefficient'])+cal['intercept'])))
values = assemble(freeze['selected']).reshape(len(kinds),len(subset))
threshold = freeze['thresholds'][freeze['selected']]
results = {kind:{'n':len(subset),'decision_flip_fraction':float(np.mean((values[0]>=threshold)!=(values[i]>=threshold))),'mean_score_change':float(np.mean(values[i]-values[0]))} for i,kind in enumerate(kinds) if kind!='baseline'}
for i,kind in enumerate(kinds):
    if kind=='baseline': continue
    changed = np.array([perturb(r,kind)!=r for r in subset])
    results[kind]['changed_inputs'] = int(changed.sum())
    results[kind]['flip_fraction_among_changed_inputs'] = float(np.mean((values[0,changed]>=threshold)!=(values[i,changed]>=threshold))) if changed.any() else None
write(ROOT/'data/selected-bias-scores.json',{'kinds':kinds,'scores':values.tolist(),'ids':[r['id'] for r in subset]})
write(ROOT/'reports/selected-bias.json',{'configuration':freeze['selected'],'sampling':'One response from each of the first six source groups per task: 18 source groups. Additional post-freeze descriptive diagnostic chosen by source IDs, without inspecting test errors.','reference_policy':'Sensitivity only. No transformed human labels or perturbation accuracy claims.','results':results})
print('Measured frozen selected-system perturbations.')
subprocess.run([sys.executable,str(ROOT/'scripts/run.py'),'audit_perturbations.py'],cwd=ROOT,check=True)
