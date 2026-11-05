"""Inventory the reference without changing its files."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
reference = ROOT.parent / "Retrieval Optimization"
inventory = {}
for path in sorted(reference.rglob("*")):
    if path.is_file():
        content = path.read_bytes()
        inventory[path.relative_to(reference).as_posix()] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        if path.suffix in {".py", ".md", ".toml", ".json", ".csv"}:
            content.decode("utf-8-sig")
notebook = json.loads((reference / "notebooks/01_public_retrieval_research.ipynb").read_text())
report = {"files": inventory, "notebook_cells": len(notebook["cells"]), "local_data_present": (reference / "data").exists(), "local_executed_notebook_present": (reference / "notebooks/01_complete_retrieval_research.ipynb").exists()}
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports/reference-audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print({k: v for k, v in report.items() if k != "files"})
