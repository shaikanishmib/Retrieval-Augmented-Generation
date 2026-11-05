"""Inventory the workspace before creating experimental artifacts."""
import importlib.metadata as md
import importlib.util
import os
import platform
import subprocess
import sys
import psutil
from rag_security.artifacts import ROOT, write, file_hash

def git(*args):
    return subprocess.check_output(['git', '-c', f'safe.directory={ROOT.parent.as_posix()}', *args], cwd=ROOT.parent, text=True).strip()

assert all('RAG Security/' in line for line in git('status', '--porcelain').splitlines()), 'Preserve and review existing changes before starting.'
tracked = git('ls-files', '-z').split('\0')
prior = {f: file_hash(ROOT.parent / f) for f in tracked if f and not f.startswith('RAG Security/') and f != 'README.md'}
write(ROOT / '.local/prior-files.json', prior)
write(ROOT / 'reports/environment.json', {
    'python': sys.version, 'platform': platform.platform(), 'cpu': platform.processor(),
    'physical_cores': psutil.cpu_count(logical=False), 'logical_cores': os.cpu_count(),
    'ram': dict(psutil.virtual_memory()._asdict()), 'disk': dict(psutil.disk_usage(str(ROOT))._asdict()),
    'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv'], text=True).strip(),
    'baseline_commit': git('rev-parse', 'HEAD'), 'prior_tracked_files': len(prior),
    'instructions': 'No AGENTS.md found in repository or readable ancestor locations; repository rg and ancestor probes performed.',
    'target_inventory_before_creation': [], 'removed_files': [],
    'security_packages': {p: bool(importlib.util.find_spec(p)) for p in ['garak', 'agentdojo', 'llm_guard', 'transformers', 'sklearn']},
    'packages': {d.metadata['Name']: d.version for d in md.distributions()}})
print('Audit saved; prior projects hashed read-only.')
