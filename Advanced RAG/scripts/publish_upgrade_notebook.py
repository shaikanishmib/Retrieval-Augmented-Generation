"""Create an output-free upgrade copy and verify both executed notebooks remain intact."""
import copy
import hashlib
import nbformat
from advanced_rag.upgrade.artifacts import ROOT, REPORTS, save, preserved

preserved()
local = ROOT / "notebooks/02_complete_grounding_upgrade.ipynb"
public = ROOT / "notebooks/02_public_grounding_upgrade.ipynb"
before = local.read_bytes()
notebook = nbformat.reads(before.decode("utf-8"), as_version=4)
code = [c for c in notebook.cells if c.cell_type == "code"]
assert [c.execution_count for c in code] == list(range(1, len(code) + 1))
assert all(o.output_type != "error" for c in code for o in c.outputs)
assert any(c.outputs for c in code)
copy_notebook = copy.deepcopy(notebook)
copy_notebook.metadata.pop("widgets", None)
for cell in copy_notebook.cells:
    assert not cell.get("attachments")
    cell.metadata.pop("execution", None)
    if cell.cell_type == "code":
        cell.execution_count = None
        cell.outputs = []
nbformat.write(copy_notebook, public)
public.write_bytes(public.read_bytes().replace(b"\r\n", b"\n"))
loaded = nbformat.read(public, as_version=4)
assert [(c.cell_type, c.source) for c in notebook.cells] == [(c.cell_type, c.source) for c in loaded.cells]
assert all(not c.outputs and c.execution_count is None for c in loaded.cells if c.cell_type == "code")
assert local.read_bytes() == before
outputs = [o for c in code for o in c.outputs]
save(REPORTS / "notebook-verification.json", {"cells": len(notebook.cells), "executed_code_cells": len(code),
     "local_outputs": len(outputs), "png_outputs": sum("image/png" in o.get("data", {}) for o in outputs),
     "sources_identical": True, "public_outputs_empty": True, "executed_local_unchanged": True, "baseline_preserved": True,
     "local_sha256": hashlib.sha256(before).hexdigest(), "public_sha256": hashlib.sha256(public.read_bytes()).hexdigest()})
preserved()
print(len(notebook.cells), "source-identical cells; local output bytes preserved")
