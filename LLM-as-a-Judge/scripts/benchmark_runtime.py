"""Measure bounded tokenization, batching and CPU/GPU NLI behavior."""
import time
import subprocess
import sys
import numpy as np
import psutil
import torch
from llm_judge.artifacts import ROOT, read, write
from llm_judge.models import Adapter, sync
from llm_judge.claims import select_windows

rows = read(ROOT / 'data/partitions/development.json')[:6]
adapter = Adapter('nli-base')
pairs = [(select_windows(r['context'], r['response'], 1)[0]['text'], r['response']) for r in rows]
start = time.perf_counter()
batch = adapter.tokenizer(pairs, padding=True, truncation=True, max_length=512, return_tensors='pt')
tokenization = time.perf_counter() - start
results = {'n_pairs': len(pairs), 'tokenization_seconds': tokenization, 'model_load_seconds': adapter.load_seconds, 'model_bytes': adapter.lock['bytes'], 'scope': 'Six real development pairs; includes warm-up. Batching changes numeric rounding and is checked for decision consistency, not assumed equivalent.'}
outputs = {}
for device in ['cuda', 'cpu']:
    adapter.model.to(device)
    for size in [1, 6]:
        sync()
        start = time.perf_counter()
        values = []
        with torch.inference_mode():
            for i in range(0, len(pairs), size):
                item = {k: v[i:i + size].to(device) for k, v in batch.items()}
                values.extend(adapter.model(**item).logits.float().softmax(-1).cpu().tolist())
        sync()
        key = f'{device}-batch{size}'
        outputs[key] = np.array(values)
        results[key] = {'seconds': time.perf_counter() - start, 'process_rss': psutil.Process().memory_info().rss}
results['maximum_probability_difference'] = float(max(np.max(abs(value - outputs['cuda-batch1'])) for value in outputs.values()))
results['argmax_decision_agreement'] = {key: float(np.mean(value.argmax(-1) == outputs['cuda-batch1'].argmax(-1))) for key, value in outputs.items()}
write(ROOT / 'reports/runtime-profile.json', results)
print(results)
del adapter
torch.cuda.empty_cache()
subprocess.run([sys.executable,str(ROOT/'scripts/run.py'),'benchmark_selected.py'],cwd=ROOT,check=True)
