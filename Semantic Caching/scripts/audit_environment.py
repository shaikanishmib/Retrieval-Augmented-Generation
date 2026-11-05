"""Inspect resources and preserve previous project file hashes without changing them."""
import importlib.metadata
import importlib.util
import platform
import subprocess
import psutil
from semantic_cache.artifacts import ROOT, sha, write

preservation = {}
sizes = {}
for name in ['Retrieval Optimization', 'Advanced RAG', 'Agentic RAG']:
    base = ROOT.parent/name
    files = [p for p in base.rglob('*') if p.is_file()]
    sizes[name] = {'files':len(files), 'bytes':sum(p.stat().st_size for p in files)}
    for p in files:
        if not set(p.relative_to(base).parts) & {'packages','gpu-packages','__pycache__','hf','matplotlib','ipython'}:
            preservation[p.relative_to(ROOT.parent).as_posix()] = sha(p)
write(ROOT/'data/preservation.json', preservation)
write(ROOT/'reports/cleanup-manifest.json', {'target_existed_at_start':False,'planned_deletions':[], 'deleted':[], 'policy':'Preserve previous projects, shared packages and model weights. No cleanup planned.'})
versions = {}
for name in ['torch','transformers','sentence-transformers','numpy','psutil','nbformat','nbclient','ipykernel','matplotlib','pandas','pytest','pyarrow','faiss-cpu','qdrant-client','redis']:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
report = {'python':platform.python_version(),'platform':platform.platform(),'cpu':platform.processor(),'logical_cpus':psutil.cpu_count(),'physical_cpus':psutil.cpu_count(logical=False),'ram':psutil.virtual_memory()._asdict(),'disk':psutil.disk_usage(str(ROOT))._asdict(),'process_rss':psutil.Process().memory_info().rss,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,driver_version','--format=csv']).decode().strip(),'versions':versions,'prior_artifact_sizes':sizes,'prior_files_hashed':len(preservation)}
write(ROOT/'reports/environment.json',report)
print(report)
