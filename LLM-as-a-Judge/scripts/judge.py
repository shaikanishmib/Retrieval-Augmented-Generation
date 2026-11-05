"""Evaluate a local answer against supplied evidence with the frozen configuration."""
import argparse
import gc
import json
import math
import time
import psutil
from pathlib import Path
import torch
from llm_judge.artifacts import ROOT, read, digest, source_identity
from llm_judge.claims import lexical_score, token_f1
from llm_judge.models import Adapter
from llm_judge.classifier import predict

parser = argparse.ArgumentParser(description='Score evidence support using the frozen local study. This does not establish factual correctness.')
parser.add_argument('--evidence', type=Path, required=True, help='UTF-8 evidence text file')
parser.add_argument('--answer', type=Path, required=True, help='UTF-8 answer text file')
parser.add_argument('--question', default='')
args = parser.parse_args()
freeze = read(ROOT/'reports/selection-freeze.json')
assert freeze['source_identity'] == source_identity(), 'Study source differs from the frozen configuration'
row = {'id':'local','source_id':'local','context':args.evidence.read_text(encoding='utf-8'),'response':args.answer.read_text(encoding='utf-8'),'question':args.question}
if not row['context'].strip() or not row['response'].strip():
    raise ValueError('Evidence and answer must both contain text')
scores = {}
load_seconds = []
started = time.perf_counter()
required = set()
def require(name):
    required.add(name)
    if name in freeze['calibration']:
        for feature in freeze['calibration'][name]['features']:
            require(feature)
require(freeze['selected'])
def score(name):
    if name in scores:
        return scores[name]
    if name in freeze['calibration']:
        config = freeze['calibration'][name]
        z = config['intercept'] + sum(w * score(feature) for w,feature in zip(config['coefficient'],config['features']))
        value = 1 / (1 + math.exp(-z))
    elif name == 'supervised-tfidf':
        value = predict([row])[0][0]
    elif name == 'lexical':
        value = lexical_score(row)
    elif name == 'exact-substring':
        value = float(row['response'] not in row['context'])
    elif name == 'token-f1-proxy':
        value = 1 - token_f1(row['response'],row['context'])
    elif name in ('majority','prevalence'):
        value = freeze[name]
    else:
        key,variant = name.split('/')
        adapter = Adapter(key)
        load_seconds.append(adapter.load_seconds)
        for feature in sorted(required):
            if not feature.startswith(key+'/'):
                continue
            feature_variant = feature.split('/')[1]
            if key.startswith('nli'):
                output = adapter.nli(row,decompose=feature_variant!='whole',count=1 if feature_variant=='one-window' else 3)
                scores[feature] = output['mean_score'] if feature_variant=='claims-mean' else output['score']
            elif key == 'bge-small':
                scores[feature] = adapter.embedding(row)['score']
            else:
                scores[feature] = adapter.qwen(row,feature_variant)['score']
        value = scores[name]
        del adapter
        gc.collect()
        torch.cuda.empty_cache()
    scores[name] = float(value)
    return float(value)
name = freeze['selected']
value = score(name)
threshold = freeze['thresholds'][name]
memory = psutil.Process().memory_info()
print(json.dumps({'configuration':name,'unsupported_score':value,'threshold':threshold,'predicted_hallucination':value>=threshold,'distance_from_threshold':abs(value-threshold),'input_identity':digest(row),'evaluation_seconds_including_loads':time.perf_counter()-started,'model_loading_seconds':sum(load_seconds),'peak_process_working_set':getattr(memory,'peak_wset',memory.rss),'peak_gpu_allocated':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,'interpretation':'Evidence-support prediction; not an independent factual-correctness judgment.'},indent=2))
