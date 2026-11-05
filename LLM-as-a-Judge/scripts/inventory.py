"""Inventory environment and cache locations without reading credentials or changing files."""
import os
import winreg
from pathlib import Path
from llm_judge.artifacts import ROOT, write

cache_roots = [Path('C:/Users/shaik/.cache/huggingface'), Path('C:/Users/shaik/.cache/torch')]
for key in ['HF_HOME','HUGGINGFACE_HUB_CACHE','TRANSFORMERS_CACHE']:
    if os.environ.get(key):
        cache_roots.append(Path(os.environ[key]))
caches = {}
for path in cache_roots:
    try:
        caches[str(path)] = {'exists':path.exists(),'children':[p.name for p in path.iterdir()] if path.exists() else []}
    except PermissionError:
        caches[str(path)] = {'accessible':False}
project_dirs = {}
with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
    cpu_name = winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
for name in ['Retrieval Optimization','Advanced RAG','Agentic RAG','Semantic Caching']:
    base = ROOT.parent / name
    data = base / 'data'
    project_dirs[name] = {'data_directories':[p.name for p in data.iterdir() if p.is_dir()] if data.exists() else [], 'python_environments':[str(p.relative_to(base)) for p in base.glob('data/*/pyvenv.cfg')], 'instructions':[str(p.relative_to(base)) for p in base.rglob('AGENTS.md')]}
write(ROOT/'reports/cache-inventory.json',{'cpu_registry_name':cpu_name,'global_cache_locations':caches,'previous_project_locations':project_dirs,'inference_libraries':['Transformers','PyTorch CUDA'],'reuse':'Read-only model paths from the published model lock; prior package overlays when outside a virtual environment.','cleanup':'None. The target did not exist before this build; no prior unique work was removed.'})
print('Recorded cache and environment locations without changing them.')
