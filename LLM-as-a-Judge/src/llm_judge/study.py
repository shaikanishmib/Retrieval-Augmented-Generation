"""Bounded development selection followed by a single frozen held-out study."""
import gc
import sys
import time
import numpy as np
import torch
from llm_judge.artifacts import ROOT, digest, read, write, source_identity
from llm_judge.claims import lexical_score, token_f1
from llm_judge.metrics import evaluate, choose_threshold, paired_interval, selective
from llm_judge.models import Adapter
from llm_judge.tracking import infer
from llm_judge.calibration import fit_calibrator
from llm_judge.classifier import predict as classifier_predict

VARIANTS = [('nli', ['whole', 'claims', 'one-window']), ('nli-base', ['whole', 'claims']), ('qwen06', ['direct', 'rubric', 'affirmative']), ('qwen17', ['direct', 'rubric', 'affirmative']), ('bge-small', ['embedding'])]

def get_scores(partition):
    rows = read(ROOT / f'data/partitions/{partition}.json')
    scores = {'lexical': [lexical_score(r) for r in rows], 'exact-substring': [float(r['response'].strip() not in r['context']) for r in rows], 'token-f1-proxy': [1 - token_f1(r['response'], r['context']) for r in rows]}
    runtime = {}
    scores['supervised-tfidf'], classifier_seconds = classifier_predict(rows)
    runtime['supervised-tfidf'] = {'total_inference_seconds': classifier_seconds, 'evaluations_per_second': len(rows) / classifier_seconds, 'scope': 'Batch transform and prediction; no individual latency quantiles measured.'}
    for key, variants in VARIANTS:
        adapter = Adapter(key)
        for variant in variants:
            name = key + '/' + variant
            traces = infer(rows, adapter, variant)
            scores[name] = [r['score'] for r in traces]
            if key.startswith('nli') and variant == 'claims':
                scores[key + '/claims-mean'] = [r['mean_score'] for r in traces]
            seconds = [r['seconds'] for r in traces]
            runtime[name] = {'load_seconds': adapter.load_seconds, 'total_inference_seconds': sum(seconds), 'p50_seconds': float(np.median(seconds)), 'p95_seconds': float(np.quantile(seconds, .95)), 'evaluations_per_second': len(seconds) / sum(seconds), 'peak_process_rss': max(r['process_rss'] for r in traces), 'peak_gpu_allocated': max(r['gpu_peak_allocated'] for r in traces), 'mean_input_tokens': float(np.mean([r['input_tokens'] for r in traces])), 'truncated_context_fraction': float(np.mean([r.get('context_truncated', False) for r in traces]))}
            print(partition, name, 'completed', len(traces), flush=True)
        del adapter
        gc.collect()
        torch.cuda.empty_cache()
    result = {'partition_identity': digest(rows), 'source_identity': source_identity(), 'scores': scores, 'runtime': runtime}
    write(ROOT / f'data/scores/{partition}.json', result)
    return rows, result

def development():
    if (ROOT / 'reports/selection-freeze.json').exists():
        raise RuntimeError('Study already frozen; development cannot overwrite selection')
    rows, result = get_scores('development')
    valrows, validation = get_scores('validation')
    scores = result['scores']
    y = [r['y'] for r in rows]
    vy = [r['y'] for r in valrows]
    # Complete source groups are split once for calibration and selection.
    groups = sorted({r['source_id'] for r in rows}, key=lambda g: digest(g))
    fitgroups, selectgroups = groups[:24], groups[24:]
    fitix = [i for i, r in enumerate(rows) if r['source_id'] in fitgroups]
    selectix = [i for i, r in enumerate(rows) if r['source_id'] in selectgroups]
    majority = int(np.mean(np.array(y)[fitix]) >= .5)
    scores['majority'] = [float(majority)] * len(rows)
    validation['scores']['majority'] = [float(majority)] * len(valrows)
    prevalence = float(np.mean(np.array(y)[fitix]))
    scores['prevalence'] = [prevalence] * len(rows)
    validation['scores']['prevalence'] = [prevalence] * len(valrows)
    base_features = ['nli/claims', 'nli-base/claims-mean', 'nli-base/whole', 'qwen17/rubric', 'qwen17/affirmative', 'supervised-tfidf', 'lexical', 'bge-small/embedding']
    calibration = {}
    for name, features in [('calibrated-nli', ['nli/claims']), ('ensemble', base_features)]:
        x = np.array([scores[f] for f in features]).T
        model = fit_calibrator(x[fitix], np.array(y)[fitix], fitgroups, selectgroups + [r['source_id'] for r in valrows])
        scores[name] = model.predict_proba(x)[:, 1].tolist()
        validation['scores'][name] = model.predict_proba(np.array([validation['scores'][f] for f in features]).T)[:, 1].tolist()
        calibration[name] = {'features': features, 'coefficient': model.coef_[0].tolist(), 'intercept': float(model.intercept_[0]), 'fit_groups': fitgroups}
    config = read(ROOT / 'configs/study.json')
    thresholds, comparison = {}, {}
    for name, p in scores.items():
        candidates = sorted(set(config['thresholds'] + np.quantile(np.array(p)[fitix], np.linspace(0, 1, 21)).tolist()))
        threshold = .5 if name == 'majority' else choose_threshold(np.array(y)[fitix], np.array(p)[fitix], candidates)
        thresholds[name] = threshold
        comparison[name] = {'selection': evaluate(np.array(y)[selectix], np.array(p)[selectix], threshold), 'validation': evaluate(vy, validation['scores'][name], threshold)}
    singles = [name for name in scores if name not in ('ensemble', 'calibrated-nli')]
    best_single = max(singles, key=lambda n: comparison[n]['selection']['macro_f1'])
    eligible = [best_single]
    for name in ['calibrated-nli', 'ensemble']:
        if comparison[name]['selection']['macro_f1'] > comparison[best_single]['selection']['macro_f1'] + .01:
            eligible.append(name)
    selected = max(eligible, key=lambda n: comparison[n]['selection']['macro_f1'])
    freeze = {'selected': selected, 'best_single': best_single, 'thresholds': thresholds, 'calibration': calibration, 'majority': majority, 'prevalence': prevalence, 'fit_groups': fitgroups, 'selection_groups': selectgroups, 'selection_policy': 'Fit thresholds/calibration on 24 development source groups; choose on 12 other development source groups. Validation is diagnostic only. Ensemble requires >0.01 selection macro-F1 gain.', 'source_identity': source_identity(), 'config': config, 'models_identity': digest(read(ROOT / 'models.lock.json')), 'development_identity': digest(rows), 'validation_identity': digest(valrows), 'test_partition_identity': digest(read(ROOT / 'data/partitions/test.json')), 'frozen_unix_time': time.time()}
    write(ROOT / 'reports/development.json', comparison)
    write(ROOT / 'reports/selection-freeze.json', freeze)
    write(ROOT / 'data/scores/development.json', result)
    write(ROOT / 'data/scores/validation.json', validation)
    print('Frozen selection:', selected, flush=True)

def heldout():
    freeze = read(ROOT / 'reports/selection-freeze.json')
    assert freeze['source_identity'] == source_identity(), 'Frozen source changed'
    assert freeze['models_identity'] == digest(read(ROOT / 'models.lock.json'))
    rows = read(ROOT / 'data/partitions/test.json')
    assert freeze['test_partition_identity'] == digest(rows)
    rows, result = get_scores('test')
    scores = result['scores']
    scores['majority'] = [float(freeze['majority'])] * len(rows)
    scores['prevalence'] = [freeze['prevalence']] * len(rows)
    for name, cal in freeze['calibration'].items():
        x = np.array([scores[f] for f in cal['features']]).T
        z = x @ np.array(cal['coefficient']) + cal['intercept']
        scores[name] = (1 / (1 + np.exp(-z))).tolist()
    y = [r['y'] for r in rows]
    results = {name: evaluate(y, p, freeze['thresholds'][name]) for name, p in scores.items()}
    selected = freeze['selected']
    intervals = {name: paired_interval(rows, scores[selected], p, freeze['thresholds'][selected], freeze['thresholds'][name]) for name, p in scores.items() if name != selected}
    task_results = {}
    for task in sorted({r['task'] for r in rows}):
        ix = [i for i, r in enumerate(rows) if r['task'] == task]
        task_results[task] = {name: evaluate(np.array(y)[ix], np.array(p)[ix], freeze['thresholds'][name]) for name, p in scores.items()}
    write(ROOT / 'data/scores/test.json', result)
    write(ROOT / 'reports/heldout.json', {'selected': selected, 'results': results, 'paired_comparisons': intervals, 'by_task': task_results, 'selective': selective(y, scores[selected], freeze['thresholds'][selected]), 'runtime': result['runtime'], 'positive_class': '1=at least one original human-annotated hallucination span', 'test_scope': '540 responses from 90 source groups sampled without labels from the official test partition; not full official-test benchmark performance.'})
    print('Held-out study complete:', results[selected], flush=True)
