"""Execute every research cell in a fresh project-local Python kernel."""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eda-only", action="store_true", help="Validate EDA cells in a separate local notebook.")
    args = parser.parse_args()
    os.environ["IPYTHONDIR"] = str(ROOT / "data/ipython")
    os.environ["JUPYTER_RUNTIME_DIR"] = str(ROOT / "data/jupyter-runtime")
    os.environ["MPLCONFIGDIR"] = str(ROOT / "data/matplotlib")
    path = ROOT / "notebooks/01_complete_retrieval_research.ipynb"
    notebook = nbformat.read(path, as_version=4)
    if args.eda_only:
        stop = next(i for i,c in enumerate(notebook.cells) if c.cell_type == "markdown" and c.source.startswith("## Experimental design"))
        notebook.cells = notebook.cells[:stop]
        path = ROOT / "data/eda-validation.ipynb"
    elif any(c.get("outputs") for c in notebook.cells):
        prior = path.read_bytes()
        backup = ROOT / ".local/notebook-execution-backups" / (hashlib.sha256(prior).hexdigest() + ".ipynb")
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            backup.write_bytes(prior)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(notebook, km=manager, timeout=3600, resources={"metadata":{"path":str(ROOT)}})
    try:
        client.execute()
    except BaseException:
        failed = ROOT / "data" / ("failed-eda-validation.ipynb" if args.eda_only else "failed-notebook-execution.ipynb")
        nbformat.write(notebook, failed)
        raise
    else:
        nbformat.write(notebook, path)
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
            manager.cleanup_resources()
    label = "EDA validation" if args.eda_only else "complete research"
    print(f"Executed and saved the {label} notebook.")


if __name__ == "__main__":
    main()
