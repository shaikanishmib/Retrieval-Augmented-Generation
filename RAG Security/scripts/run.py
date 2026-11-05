"""Run a stage with an isolated environment or existing read-only packages."""
import os
from pathlib import Path
import runpy
import site
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
paths = [ROOT / 'src']
if sys.prefix == sys.base_prefix:
    paths += [ROOT.parent / 'Advanced RAG/data/gpu-packages', ROOT.parent / 'Advanced RAG/data/packages', ROOT.parent / 'Semantic Caching/data/packages']
sys.path[:0] = list(map(str, paths))
for p in paths:
    if (p / 'pywin32.pth').exists():
        site.addsitedir(str(p))
os.environ['PYTHONPATH'] = os.pathsep.join(map(str, paths))
for key, folder in [('HF_HOME', 'hf'), ('MPLCONFIGDIR', 'matplotlib'), ('IPYTHONDIR', 'ipython'), ('JUPYTER_RUNTIME_DIR', 'jupyter')]:
    os.environ[key] = str(ROOT / 'data' / folder)
os.environ.update(TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1', HF_HUB_DISABLE_XET='1')
sys.stdout.reconfigure(encoding='utf-8')
stage = sys.argv.pop(1)
if stage == 'test':
    import pytest
    raise SystemExit(pytest.main([str(ROOT / 'tests'), *sys.argv[1:]]))
runpy.run_path(str(ROOT / 'scripts' / stage), run_name='__main__')
