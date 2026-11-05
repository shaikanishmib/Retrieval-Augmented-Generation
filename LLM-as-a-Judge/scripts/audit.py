"""Record current resources, prior tracked bytes, and reusable model provenance."""
import importlib.metadata
import platform
import subprocess
import sys
import psutil
import torch
from llm_judge.artifacts import ROOT, read, sha, write

repo = ROOT.parent
git = ['git', '-c', f'safe.directory={repo.as_posix()}']
tracked = subprocess.check_output(git + ['ls-files', '-z'], cwd=repo).decode().split('\0')
previous = {p: sha(repo / p) for p in tracked if p and p.split('/')[0] in ['Retrieval Optimization', 'Advanced RAG', 'Agentic RAG', 'Semantic Caching']}
write(ROOT / 'data/prior-projects.json', previous)
write(ROOT / 'reports/environment.json', {'python': sys.version, 'platform': platform.platform(), 'cpu': platform.processor(), 'physical_cores': psutil.cpu_count(logical=False), 'logical_cores': psutil.cpu_count(), 'ram': psutil.virtual_memory()._asdict(), 'disk': psutil.disk_usage(str(ROOT))._asdict(), 'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,driver_version', '--format=csv']).decode().strip(), 'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}, 'prior_tracked_files_hashed': len(previous), 'target_existed_before_work': False, 'cleanup': [], 'cache_policy': 'Reuse explicitly pinned project model paths read-only; no global cache deletion.'})
if (ROOT / 'models.lock.json').exists():
    selected = read(ROOT / 'models.lock.json')
else:
    models = read(repo / 'Agentic RAG/models.lock.json')
    selected = {key: models[key] for key in ['qwen06', 'qwen17', 'nli', 'bge-small']}
for key, model in selected.items():
    path = repo / model['local_path']
    for name, info in model['files'].items():
        if sha(path / name) != info['sha256']:
            raise RuntimeError(f'Model hash mismatch: {key}/{name}')
write(ROOT / 'models.lock.json', selected)
print('Verified cached model files and recorded resources.')
