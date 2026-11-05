"""Publish a source-identical copy while retaining the local executed bytes."""
import copy
import hashlib
import json
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]

def main():
    local_path = ROOT / "notebooks/01_complete_advanced_rag_research.ipynb"
    public_path = ROOT / "notebooks/01_public_advanced_rag_research.ipynb"
    before = local_path.read_bytes()
    local = nbformat.reads(before.decode("utf-8"), as_version=4)
    code = [c for c in local.cells if c.cell_type == "code"]
    assert [c.execution_count for c in code] == list(range(1, len(code) + 1))
    assert not any(o.output_type == "error" for c in code for o in c.outputs)
    assert any(c.outputs for c in code)
    public = copy.deepcopy(local)
    public.metadata.pop("widgets", None)
    for cell in public.cells:
        assert not cell.get("attachments")
        cell.metadata.pop("execution", None)
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
    nbformat.write(public, public_path)
    reread = nbformat.read(public_path, as_version=4)
    assert [(c.cell_type, c.source) for c in local.cells] == [(c.cell_type, c.source) for c in reread.cells]
    assert all(not c.outputs and c.execution_count is None for c in reread.cells if c.cell_type == "code")
    assert local_path.read_bytes() == before
    outputs = [o for c in code for o in c.outputs]
    result = {"cells": len(local.cells), "executed_code_cells": len(code), "local_output_objects": len(outputs),
              "local_png_outputs": sum("image/png" in o.get("data", {}) for o in outputs), "identical_sources": True,
              "public_outputs_empty": True, "local_unchanged": True, "local_sha256": hashlib.sha256(before).hexdigest(),
              "public_sha256": hashlib.sha256(public_path.read_bytes()).hexdigest()}
    (ROOT / "reports/notebook-verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
