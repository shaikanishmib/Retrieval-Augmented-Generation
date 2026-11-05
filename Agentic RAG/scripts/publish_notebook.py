"""Create an output-free public copy without changing the executed original."""
import copy
import nbformat
from agentic_rag.artifacts import ROOT,sha,write,digest

if __name__ == "__main__":
    full=ROOT / "notebooks/01_complete_agentic_rag_research.ipynb"
    public=ROOT / "notebooks/01_public_agentic_rag_research.ipynb"
    before=sha(full)
    local=nbformat.read(full,as_version=4)
    code=[c for c in local.cells if c.cell_type=="code"]
    assert all(c.execution_count is not None for c in code)
    assert not any(o.output_type=="error" for c in code for o in c.outputs)
    stripped=copy.deepcopy(local)
    for cell in stripped.cells:
        if cell.cell_type=="code":
            cell.outputs=[]
            cell.execution_count=None
        cell.metadata={}
    public.write_text(nbformat.writes(stripped),encoding='utf-8',newline='\n')
    checked=nbformat.read(public,as_version=4)
    assert [(c.cell_type,c.source) for c in local.cells]==[(c.cell_type,c.source) for c in checked.cells]
    assert all(not c.get("outputs") and c.get("execution_count") is None for c in checked.cells)
    assert sha(full)==before
    outputs=[o for c in code for o in c.outputs]
    write(ROOT / "reports/notebook-verification.json",dict(cells=len(local.cells),code_cells=len(code),markdown_cells=len(local.cells)-len(code),
            executed_code_cells=len(code),outputs=len(outputs),plots=sum("image/png" in o.get("data",{}) for o in outputs),
            full_sha256=before,public_sha256=sha(public),source_identity=digest([(c.cell_type,c.source) for c in local.cells]),
            exact_source_identity=True,public_outputs=0,public_execution_counts=0,local_unchanged=True))
    print("Verified",len(local.cells),"source-identical cells; local outputs preserved")
