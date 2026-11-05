"""Run research stages with local packages or a standalone environment."""
import os
from pathlib import Path
import runpy
import site
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
paths = [ROOT / 'src', ROOT / 'data/packages']
if sys.prefix == sys.base_prefix:
    paths += [ROOT.parent / 'Advanced RAG/data/gpu-packages', ROOT.parent / 'Advanced RAG/data/packages']
sys.path[:0] = list(map(str, paths))
if (ROOT/'data/packages/pywin32.pth').exists():
    site.addsitedir(str(ROOT/'data/packages'))
os.environ['PYTHONPATH'] = os.pathsep.join(map(str, paths))
for key, folder in [('HF_HOME','hf'),('MPLCONFIGDIR','matplotlib'),('IPYTHONDIR','ipython')]:
    os.environ[key] = str(ROOT / 'data' / folder)
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True
stage = sys.argv.pop(1)
if stage == 'test':
    import pytest
    raise SystemExit(pytest.main([str(ROOT/'tests'), '-p', 'no:cacheprovider', *sys.argv[1:]]))
runpy.run_path(str(ROOT/'scripts'/stage), run_name='__main__')
