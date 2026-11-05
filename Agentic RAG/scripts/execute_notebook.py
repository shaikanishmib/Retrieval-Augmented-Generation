"""Execute a clean local notebook and retain any failed attempt separately."""
import os
import sys
import time
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from agentic_rag.artifacts import ROOT

if __name__ == "__main__":
    path=ROOT / "notebooks/01_complete_agentic_rag_research.ipynb"
    nb=nbformat.read(path,as_version=4)
    if any(c.get("outputs") or c.get("execution_count") is not None for c in nb.cells):
        raise FileExistsError("Executed local notebook is protected")
    os.environ["JUPYTER_RUNTIME_DIR"]=str(ROOT / "data/runtime")
    manager=KernelManager(kernel_name="python3")
    manager.kernel_spec.argv=[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"]
    try:
        NotebookClient(nb,km=manager,timeout=600,resources={"metadata":{"path":str(ROOT)}}).execute()
    except BaseException:
        failed=ROOT / f"data/failed-notebook-{time.time_ns()}.ipynb"
        failed.parent.mkdir(parents=True,exist_ok=True)
        nbformat.write(nb,failed)
        raise
    else:
        nbformat.write(nb,path)
        print("Executed",len(nb.cells),"cells in a clean kernel")
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
            manager.cleanup_resources()
