"""Execute a fresh kernel and retain real outputs, including partial failures."""
import os
import sys
import nbformat
from nbclient import NotebookClient
from llm_judge.artifacts import ROOT, write

path = ROOT / 'notebooks/01_complete_llm_judge_research.ipynb'
notebook = nbformat.read(path, as_version=4)
if any(c.get('execution_count') is not None for c in notebook.cells):
    raise RuntimeError('Preserve existing executed outputs before another notebook run')
kernel_root = ROOT / 'data/jupyter'
startup = ''
if sys.prefix == sys.base_prefix:
    package_path = ROOT.parent / 'Semantic Caching/data/packages'
    startup = f'import site; site.addsitedir({str(package_path)!r}); '
startup += 'from ipykernel.kernelapp import IPKernelApp; IPKernelApp.launch_instance()'
write(kernel_root / 'kernels/python3/kernel.json', {'argv':[sys.executable,'-c',startup,'-f','{connection_file}'],'display_name':'Python 3','language':'python'})
os.environ['JUPYTER_PATH'] = str(kernel_root)
try:
    NotebookClient(notebook, timeout=1200, kernel_name='python3', resources={'metadata':{'path':str(ROOT)}}).execute()
finally:
    nbformat.write(notebook, path)
print('Executed every code cell in a clean kernel.')
