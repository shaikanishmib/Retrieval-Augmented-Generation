"""Publish identical cell sources while retaining the complete executed notebook."""
import copy
import nbformat
from semantic_cache.artifacts import ROOT,sha,write
full = ROOT/'notebooks/01_complete_semantic_caching_research.ipynb'
public = ROOT/'notebooks/01_public_semantic_caching_research.ipynb'
nb = nbformat.read(full,as_version=4)
code = [c for c in nb.cells if c.cell_type=='code']
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in code)
assert sum('image/png' in o.get('data',{}) for c in code for o in c.outputs)>=4
clean = copy.deepcopy(nb)
for cell in clean.cells:
    if cell.cell_type=='code':
        cell.outputs=[]
        cell.execution_count=None
        cell.metadata.pop('execution',None)
nbformat.write(clean,public)
saved = nbformat.read(public,as_version=4)
assert [(c.cell_type,c.source) for c in nb.cells]==[(c.cell_type,c.source) for c in saved.cells]
assert all(not c.get('outputs') and c.get('execution_count') is None for c in saved.cells)
report = dict(cells=len(nb.cells),code_cells=len(code),executed_code_cells=len(code),outputs=sum(len(c.outputs) for c in code),figures=sum('image/png' in o.get('data',{}) for c in code for o in c.outputs),source_equivalent=True,public_outputs=0,full_sha256=sha(full),public_sha256=sha(public))
write(ROOT/'reports/notebook-verification.json',report)
print(report)
