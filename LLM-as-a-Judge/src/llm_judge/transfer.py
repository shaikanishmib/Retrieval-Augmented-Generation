"""Frozen cross-dataset transfer, retaining the different annotation origin."""
import gc
import numpy as np
import pandas as pd
import torch
from llm_judge.artifacts import ROOT, read, write, digest
from llm_judge.study import get_scores
from llm_judge.metrics import evaluate

def run():
    files = sorted((ROOT / 'data/raw/ragbench/techqa').glob('*test*.parquet'))
    frame = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    frame = frame.sort_values('id', key=lambda s: s.astype(str)).head(120)
    rows = []
    for _, r in frame.iterrows():
        context = '\n\n'.join(r['documents'])
        rows.append({'id': 'ragbench-' + str(r['id']), 'source_id': digest(context), 'context': context, 'question': r['question'], 'response': r['response'], 'y': int(not r['adherence_score']), 'task': 'TechQA', 'evidence_hash': digest(context)})
    write(ROOT / 'data/partitions/transfer.json', rows)
    _, scores = get_scores('transfer')
    freeze = read(ROOT / 'reports/selection-freeze.json')
    p = scores['scores']
    p['majority'] = [float(freeze['majority'])] * len(rows)
    p['prevalence'] = [freeze['prevalence']] * len(rows)
    for name, cal in freeze['calibration'].items():
        x = np.array([p[f] for f in cal['features']]).T
        p[name] = (1 / (1 + np.exp(-(x @ np.array(cal['coefficient']) + cal['intercept'])))).tolist()
    write(ROOT / 'data/scores/transfer.json', scores)
    write(ROOT / 'reports/transfer.json', {'n': len(rows), 'positive_labels': sum(r['y'] for r in rows), 'annotation_models': sorted(set(frame['annotating_model_name'])), 'positive_class': '1 = not adherence_score; model-generated sentence-support aggregation, not RAGTruth human span labels', 'selection': 'First 120 IDs sorted lexically from official TechQA test; no thresholds or weights retuned', 'results': {name: evaluate([r['y'] for r in rows], values, freeze['thresholds'][name]) for name, values in p.items()}})
