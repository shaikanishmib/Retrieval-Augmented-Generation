"""Execute an unexecuted notebook in the project's dependency environment."""
import os
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "notebooks/01_complete_advanced_rag_research.ipynb"
notebook = nbformat.read(path, as_version=4)
if any(c.get("outputs") or c.get("execution_count") is not None for c in notebook.cells):
    raise FileExistsError("Preserve the executed notebook; execution requires an unexecuted source notebook")
os.environ["IPYTHONDIR"] = str(ROOT / "data/ipython")
os.environ["JUPYTER_RUNTIME_DIR"] = str(ROOT / "data/runtime")
os.environ["MPLCONFIGDIR"] = str(ROOT / "data/matplotlib")
manager = KernelManager(kernel_name="python3")
manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
try:
    NotebookClient(notebook, km=manager, timeout=600, resources={"metadata": {"path": str(ROOT)}}).execute()
except BaseException:
    nbformat.write(notebook, ROOT / "data/failed-execution.ipynb")
    raise
else:
    nbformat.write(notebook, path)
    print("Executed", len(notebook.cells), "cells")
finally:
    if manager.has_kernel:
        manager.shutdown_kernel(now=True)
        manager.cleanup_resources()
