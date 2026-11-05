"""Create the public copy only after the executed notebook passes review."""
import copy
import nbformat
from rag_security.artifacts import ROOT, write, digest

local=nbformat.read(ROOT/'notebooks/01_complete_rag_security_research.ipynb',as_version=4)
codes=[c for c in local.cells if c.cell_type=='code']
assert all(isinstance(c.execution_count,int) for c in codes)
assert all(o.output_type!='error' for c in codes for o in c.outputs)
public=copy.deepcopy(local)
for cell in public.cells:
    if cell.cell_type=='code':
        cell.outputs=[]
        cell.execution_count=None
nbformat.write(public,ROOT/'notebooks/01_public_rag_security_research.ipynb')
assert [(c.cell_type,c.source) for c in local.cells]==[(c.cell_type,c.source) for c in public.cells]
figures=sum('image/png' in o.get('data',{}) for c in codes for o in c.outputs)
write(ROOT/'reports/notebook-audit.json',{'cells':len(local.cells),'executed_code_cells':len(codes),'local_outputs':sum(len(c.outputs) for c in codes),'embedded_figures':figures,'public_outputs':0,'public_execution_counts':0,'identical_sources_types_order':True,'source_identity':digest([(c.cell_type,c.source) for c in public.cells])})
print('Public notebook source identity verified; local outputs retained.')
