from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_public_sources_and_execution_state():
    public=ROOT/"notebooks/01_public_retrieval_research.ipynb"
    local=ROOT/"notebooks/01_complete_retrieval_research.ipynb"
    if not public.exists():
        pytest.skip("Generate the reviewed public notebook after executing the local report")
    p=nbformat.read(public,as_version=4)
    assert len(p.cells)>=40
    assert all(c.execution_count is None and c.outputs==[] for c in p.cells if c.cell_type=="code")
    assert not any(c.get("attachments") for c in p.cells)
    if local.exists():
        full=nbformat.read(local,as_version=4)
        assert [(c.cell_type,c.source) for c in full.cells]==[(c.cell_type,c.source) for c in p.cells]
        code=[c for c in full.cells if c.cell_type=="code"]
        assert [c.execution_count for c in code]==list(range(1,len(code)+1))
        assert any(c.outputs for c in code)
        assert not any(o.output_type=="error" for c in code for o in c.outputs)
