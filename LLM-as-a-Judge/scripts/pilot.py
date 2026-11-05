"""Confirm real inference on a bounded development-only sample."""
import gc
import torch
from llm_judge.artifacts import ROOT, read, write
from llm_judge.models import Adapter
from llm_judge.tracking import infer
from llm_judge.metrics import evaluate

rows = read(ROOT / 'data/partitions/development.json')
# Interleave tasks rather than selecting a source-file prefix.
tasks = sorted({r['task'] for r in rows})
rows = [r for pair in zip(*[[r for r in rows if r['task'] == t][:6] for t in tasks]) for r in pair]
results = {}
for key, variants in [('nli', ['whole', 'claims']), ('nli-base', ['whole', 'claims']), ('qwen06', ['direct', 'rubric']), ('qwen17', ['direct', 'rubric']), ('bge-small', ['embedding'])]:
    adapter = Adapter(key)
    for variant in variants:
        traces = infer(rows, adapter, variant)
        results[f'{key}/{variant}'] = {**evaluate([r['y'] for r in rows], [r['score'] for r in traces]), 'seconds': sum(r['seconds'] for r in traces), 'load_seconds': adapter.load_seconds, 'labels': adapter.labels}
        print(key, variant, results[f'{key}/{variant}'], flush=True)
    del adapter
    gc.collect()
    torch.cuda.empty_cache()
write(ROOT / 'reports/pilot.json', results)
