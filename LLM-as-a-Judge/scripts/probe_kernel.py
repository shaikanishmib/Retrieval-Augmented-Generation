"""Check a fresh notebook kernel before executing the research notebook."""
import os
import sys
import nbformat
from nbclient import NotebookClient
from llm_judge.artifacts import ROOT, write

startup = ''
if sys.prefix == sys.base_prefix:
    startup = f'import site; site.addsitedir({str(ROOT.parent / "Semantic Caching/data/packages")!r}); '
startup += 'from ipykernel.kernelapp import IPKernelApp; IPKernelApp.launch_instance()'
kernel_root = ROOT / 'data/jupyter'
write(kernel_root / 'kernels/judge-probe/kernel.json', {'argv':[sys.executable,'-c',startup,'-f','{connection_file}'],'display_name':'Judge probe','language':'python'})
os.environ['JUPYTER_PATH'] = str(kernel_root)
nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("import nbformat, numpy; print('kernel_ready')")])
NotebookClient(nb,timeout=120,kernel_name='judge-probe',resources={'metadata':{'path':str(ROOT)}}).execute()
nbformat.write(nb,ROOT/'data/kernel-probe.ipynb')
print(nb.cells[0].outputs)
