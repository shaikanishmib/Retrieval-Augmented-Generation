"""Content-addressed resumable traces; failed inference never becomes a score."""
import time
import psutil
import torch
from llm_judge.artifacts import ROOT, digest, read, write, source_identity
from llm_judge.models import sync

def infer(rows, adapter, variant):
    identity = digest({'source': source_identity(), 'model': adapter.lock, 'variant': variant})
    records = []
    for row in rows:
        path = ROOT / 'data/traces' / identity / (row['id'] + '-' + digest(row)[:16] + '.json')
        if path.exists():
            record = read(path)
        else:
            sync()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            try:
                if adapter.key.startswith('nli'):
                    result = adapter.nli(row, decompose=variant != 'whole', count=1 if variant == 'one-window' else 3)
                elif adapter.key == 'bge-small':
                    result = adapter.embedding(row)
                else:
                    result = adapter.qwen(row, variant)
                sync()
            except Exception as exc:
                write(path.with_suffix('.failure.json'), {'error_type': type(exc).__name__, 'message': str(exc), 'row_hash': digest(row), 'identity': identity})
                raise
            record = {'id': row['id'], 'source_id': row['source_id'], 'row_hash': digest(row), 'identity': identity, 'model': adapter.key, 'variant': variant, 'seconds': time.perf_counter() - start, 'process_rss': psutil.Process().memory_info().rss, 'gpu_peak_allocated': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0, **result}
            write(path, record)
        records.append(record)
    return records
