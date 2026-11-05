"""Validate notebook structure and compile every source cell before expensive execution."""
import ast
import nbformat
from semantic_cache.artifacts import ROOT
path = ROOT/'notebooks/01_complete_semantic_caching_research.ipynb'
nb = nbformat.read(path,as_version=4)
nbformat.validate(nb)
for index,cell in enumerate(nb.cells):
    if cell.cell_type=='code':
        ast.parse(cell.source,filename=f'cell-{index}')
print(len(nb.cells),'notebook cells validated and code sources compiled.')
