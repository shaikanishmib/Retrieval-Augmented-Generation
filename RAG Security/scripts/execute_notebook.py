"""Execute all cells in a clean kernel and preserve failures locally."""
import nbformat
from nbclient import NotebookClient
from rag_security.artifacts import ROOT

path=ROOT/'notebooks/01_complete_rag_security_research.ipynb'
nb=nbformat.read(path,as_version=4)
assert not any(c.get('outputs') or c.get('execution_count') for c in nb.cells if c.cell_type=='code'),'Refusing to overwrite an executed notebook.'
client=NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}})
try:
    client.execute()
except Exception:
    nbformat.write(nb,ROOT/'.local/failed-notebook.ipynb')
    raise
nbformat.write(nb,path)
print('Executed',sum(c.cell_type=='code' for c in nb.cells),'code cells')
