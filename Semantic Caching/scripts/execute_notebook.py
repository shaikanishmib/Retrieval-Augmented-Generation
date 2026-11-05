"""Execute the complete notebook in a fresh kernel and preserve partial output on failure."""
import nbformat
import os
import sys
from nbclient import NotebookClient
from semantic_cache.artifacts import ROOT,write
path = ROOT/'notebooks/01_complete_semantic_caching_research.ipynb'
nb = nbformat.read(path,as_version=4)
if any(c.get('execution_count') is not None for c in nb.cells):
    raise RuntimeError('The local notebook already has execution output; preserve it before a separate run.')
kernel_root = ROOT/'data/jupyter'
write(kernel_root/'kernels/python3/kernel.json',dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],display_name='Python 3',language='python'))
os.environ['JUPYTER_PATH'] = str(kernel_root)+(os.pathsep+os.environ['JUPYTER_PATH'] if os.environ.get('JUPYTER_PATH') else '')
(ROOT/'data/runtime').mkdir(parents=True,exist_ok=True)
os.environ['JUPYTER_RUNTIME_DIR'] = str(ROOT/'data/runtime')
try:
    NotebookClient(nb,timeout=1200,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
finally:
    nbformat.write(nb,path)
print('All notebook cells executed in a clean kernel.')
