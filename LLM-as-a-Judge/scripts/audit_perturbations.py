"""Distinguish actual evidence-order changes from unchanged single-line inputs."""
from llm_judge.artifacts import ROOT, read, write

path = ROOT/'reports/bias-robustness.json'
report = read(path)
rows = read(ROOT/'data/partitions/test.json')
subset = [r for task in ['Data2txt','QA','Summary'] for r in [x for x in rows if x['task']==task][:10]]
for kind in ['verbosity','identity','injection','irrelevant','truncate','reverse-evidence']:
    changed = sum('\n'.join(reversed(r['context'].splitlines()))!=r['context'] for r in subset) if kind=='reverse-evidence' else len(subset)
    report[kind]['changed_inputs'] = changed
    report[kind]['unchanged_inputs'] = len(subset)-changed
    report[kind]['flip_fraction_among_changed_inputs'] = round(report[kind]['decision_flip_fraction']*len(subset))/changed if changed else None
report['perturbation_scope'] = 'Evidence-line reversal can leave single-line JSON unchanged. Rates among changed inputs are reported separately; unchanged inputs reuse identical cached predictions.'
write(path,report)
selected_path = ROOT/'reports/selected-bias.json'
if selected_path.exists():
    selected = read(selected_path)
    retained = read(ROOT/'data/selected-bias-scores.json')
    lookup = {r['id']:r for r in rows}
    selected_rows = [lookup[identifier] for identifier in retained['ids']]
    threshold = read(ROOT/'reports/selection-freeze.json')['thresholds'][selected['configuration']]
    baseline = retained['scores'][0]
    for i,kind in enumerate(retained['kinds']):
        if kind=='baseline': continue
        changed = ['\n'.join(reversed(r['context'].splitlines()))!=r['context'] if kind=='reverse-evidence' else True for r in selected_rows]
        flips = [(a>=threshold)!=(b>=threshold) for a,b in zip(baseline,retained['scores'][i])]
        assert abs(sum(flips)/len(flips)-selected['results'][kind]['decision_flip_fraction'])<1e-12
        selected['results'][kind]['changed_inputs'] = sum(changed)
        selected['results'][kind]['flip_fraction_among_changed_inputs'] = sum(f for f,c in zip(flips,changed) if c)/sum(changed) if any(changed) else None
    write(selected_path,selected)
print('Audited actual perturbation counts.')
