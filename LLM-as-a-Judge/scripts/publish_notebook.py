"""Publish identical source cells while preserving the executed local notebook."""
import copy
import nbformat
from llm_judge.artifacts import ROOT, write, sha

local = ROOT / 'notebooks/01_complete_llm_judge_research.ipynb'
public = ROOT / 'notebooks/01_public_llm_judge_research.ipynb'
a = nbformat.read(local, as_version=4)
assert all(c.execution_count is not None for c in a.cells if c.cell_type == 'code')
assert not any(o.output_type == 'error' for c in a.cells if c.cell_type == 'code' for o in c.outputs)
b = copy.deepcopy(a)
for cell in b.cells:
    if cell.cell_type == 'code':
        cell.outputs = []
        cell.execution_count = None
        cell.metadata = {}
nbformat.write(b, public)
c = nbformat.read(public, as_version=4)
assert [(x.cell_type,x.source) for x in a.cells] == [(x.cell_type,x.source) for x in c.cells]
write(ROOT / 'reports/notebook-audit.json', {'cells':len(a.cells),'code_cells':sum(x.cell_type=='code' for x in a.cells),'local_outputs':sum(len(x.outputs) for x in a.cells if x.cell_type=='code'),'public_outputs':0,'public_execution_counts':'all null','source_cell_identity':True,'local_sha256':sha(local),'public_sha256':sha(public),'figures':len(list((ROOT/'reports/figures').glob('*.png')))})
print('Verified identical sources and an output-free public copy.')
