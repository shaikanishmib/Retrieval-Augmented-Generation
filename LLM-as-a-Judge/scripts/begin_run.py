"""Preserve published references before an independent run in a fresh checkout."""
import shutil
from llm_judge.artifacts import ROOT

if (ROOT / 'data/traces').exists() or (ROOT / 'data/classifier.joblib').exists():
    raise RuntimeError('This workspace has inference artifacts; resume it or use a separate checkout')
archive = ROOT / 'data/published-reference'
if archive.exists():
    raise RuntimeError('Published references already preserved; do not reset this run')
archive.mkdir(parents=True)
for name in ['reports','README.md','models.lock.json','sources.lock.json','ragbench.lock.json']:
    path = ROOT / name
    if path.is_dir():
        shutil.copytree(path, archive / name)
    elif path.exists():
        shutil.copy2(path, archive / name)
freeze = ROOT / 'reports/selection-freeze.json'
if freeze.exists():
    assert (archive / 'reports/selection-freeze.json').read_bytes() == freeze.read_bytes()
    freeze.unlink()
print('Published references preserved locally. This fresh checkout can run independent selection.')
