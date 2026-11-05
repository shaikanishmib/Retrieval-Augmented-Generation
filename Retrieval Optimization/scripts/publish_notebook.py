"""Create and verify an output-free copy without modifying the local notebook."""

import copy
import hashlib
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "notebooks/01_complete_retrieval_research.ipynb"
PUBLIC = ROOT / "notebooks/01_public_retrieval_research.ipynb"


def main():
    before = LOCAL.read_bytes()
    local = nbformat.reads(before.decode("utf-8"), as_version=4)
    code = [c for c in local.cells if c.cell_type == "code"]
    if not code or any(c.execution_count is None for c in code):
        raise ValueError("Every local code cell must be executed before publication")
    if [c.execution_count for c in code] != list(range(1, len(code) + 1)):
        raise ValueError("Local execution counts must follow one clean sequential run")
    if any(o.output_type == "error" for c in code for o in c.outputs):
        raise ValueError("The local notebook contains an execution error")
    if not any(c.outputs for c in code):
        raise ValueError("The local notebook has no saved outputs")
    public = copy.deepcopy(local)
    public.metadata.pop("widgets", None)
    for cell in public.cells:
        cell.metadata.pop("execution", None)
        cell.metadata.pop("ExecuteTime", None)
        if cell.get("attachments"):
            raise ValueError("Embedded Markdown attachments require review")
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    nbformat.write(public, PUBLIC)
    loaded = nbformat.read(PUBLIC, as_version=4)
    assert len(local.cells) == len(loaded.cells)
    assert all(a.cell_type == b.cell_type and a.source == b.source for a,b in zip(local.cells, loaded.cells))
    assert all(c.outputs == [] and c.execution_count is None for c in loaded.cells if c.cell_type == "code")
    assert LOCAL.read_bytes() == before
    outputs = [o for c in code for o in c.outputs]
    report = {"cells": len(local.cells), "executed_code_cells": len(code), "local_output_objects": len(outputs),
              "local_png_outputs": sum("image/png" in o.get("data", {}) for o in outputs),
              "identical_cell_sources": True, "public_outputs_empty": True, "local_unchanged": True,
              "local_sha256": hashlib.sha256(before).hexdigest(), "public_sha256": hashlib.sha256(PUBLIC.read_bytes()).hexdigest()}
    path = ROOT / "reports/notebook-verification.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
