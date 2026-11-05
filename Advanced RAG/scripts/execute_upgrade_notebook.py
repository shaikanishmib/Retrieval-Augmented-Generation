"""Execute the new notebook while retaining both original executed notebooks."""
import os
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from advanced_rag.upgrade.artifacts import ROOT, LOCAL, preserved

preserved()
path = ROOT / "notebooks/02_complete_grounding_upgrade.ipynb"
notebook = nbformat.read(path, as_version=4)
if any(c.get("outputs") or c.get("execution_count") is not None for c in notebook.cells):
    raise FileExistsError("The upgrade notebook already contains execution state; preserve it before a new run")
os.environ["IPYTHONDIR"] = str(LOCAL / "ipython")
os.environ["JUPYTER_RUNTIME_DIR"] = str(LOCAL / "runtime")
os.environ["MPLCONFIGDIR"] = str(LOCAL / "matplotlib")
manager = KernelManager(kernel_name="python3")
manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
try:
    NotebookClient(notebook, km=manager, timeout=600, resources={"metadata": {"path": str(ROOT)}}).execute()
except BaseException:
    from datetime import datetime
    failed = LOCAL / ("failed-notebook-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".ipynb")
    nbformat.write(notebook, failed)
    raise
else:
    nbformat.write(notebook, path)
    print("Executed", len(notebook.cells), "cells")
finally:
    if manager.has_kernel:
        manager.shutdown_kernel(now=True)
        manager.cleanup_resources()
preserved()
