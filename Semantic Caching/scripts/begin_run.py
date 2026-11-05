"""Preserve published reports and local execution output before an independent reproduction."""
import shutil
import time
from semantic_cache.artifacts import ROOT,read
if (ROOT/'data/traces').exists():
    raise RuntimeError('Existing local trials must be resumed or preserved explicitly before a new run.')
archive = (ROOT/'data/reproduction-history'/str(time.time_ns())).resolve()
assert archive.is_relative_to(ROOT.resolve())
archive.mkdir(parents=True)
if (ROOT/'reports').exists():
    shutil.copytree(ROOT/'reports',archive/'reports')
# These are run-specific selection/runtime artifacts; partition and provenance locks remain.
for name in ['selection-freeze.json','execution-ledger.json','resources.json']:
    path = ROOT/'reports'/name
    if path.exists():
        shutil.move(str(path),str(archive/name))
full = ROOT/'notebooks/01_complete_semantic_caching_research.ipynb'
if full.exists():
    shutil.move(str(full),str(archive/full.name))
print('Previous publication retained at',archive)
