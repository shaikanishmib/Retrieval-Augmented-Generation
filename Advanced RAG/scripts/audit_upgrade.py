"""Record baseline preservation and local inference resources."""
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/upgrade"

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def main():
    OUT.mkdir(exist_ok=True)
    baseline = OUT / "preservation.json"
    if not baseline.exists():
        paths = subprocess.check_output(["git", "-c", f"safe.directory={ROOT.parent.as_posix()}", "-c", "core.quotepath=false", "ls-tree", "-r", "--name-only", "92a85e1", "--", "Advanced RAG", "Retrieval Optimization"], cwd=ROOT.parent, text=True).splitlines()
        files = {name: digest(ROOT.parent / name) for name in paths}
        local = ROOT / "notebooks/01_complete_advanced_rag_research.ipynb"
        files[local.relative_to(ROOT.parent).as_posix()] = digest(local)
        for path in (ROOT / "data/experiments").glob("*.json"):
            files[path.relative_to(ROOT.parent).as_posix()] = digest(path)
        (ROOT / ".local/upgrade-baseline").mkdir(parents=True, exist_ok=True)
        backup = ROOT / ".local/upgrade-baseline/01_complete_advanced_rag_research.ipynb"
        if not backup.exists():
            shutil.copyfile(local, backup)
        baseline.write_text(json.dumps({"commit": "92a85e1485be22bc762422b52bb554c09b8590f8", "files": files, "notebook_backup_sha256": digest(backup)}, indent=2), encoding="utf-8")
    for name, expected in json.loads(baseline.read_text())["files"].items():
        if digest(ROOT.parent / name) != expected:
            raise ValueError(f"Preserved baseline file changed: {name}")
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(ROOT)
    import torch
    report = {"python": platform.python_version(), "cpu": platform.processor(), "physical_cores": psutil.cpu_count(logical=False),
              "logical_cores": os.cpu_count(), "ram_total_bytes": memory.total, "ram_available_bytes": memory.available,
              "disk_total_bytes": disk.total, "disk_free_bytes": disk.free, "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
              "backends": {name: shutil.which(name) for name in ("ollama", "llama-server", "llama-cli", "uv")},
              "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}
    try:
        report["nvidia_smi"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv"], text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        report["nvidia_error"] = str(exc)
    (OUT / "hardware-initial.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    from advanced_rag.pipeline import Models, Pipeline
    from advanced_rag.data import load
    started = time.perf_counter()
    models = Models.load(ROOT)
    loading = time.perf_counter() - started
    rows = []
    for name in ("hotpot", "squad"):
        for row in load(name)[0][:4]:
            started = time.perf_counter()
            pipeline = Pipeline(models, row["docs"])
            build = time.perf_counter() - started
            result = pipeline.run(row["question"], "hybrid")
            rows.append({"dataset": name, "id": row["id"], "build_ms": build * 1000, **{k: result[k] for k in ("total_ms", "input_tokens", "output_tokens")}})
    probe = {"scope": "Eight baseline development questions, before acquiring larger models", "device": models.device, "loading_seconds": loading,
             "max_observed_rss_bytes": psutil.Process().memory_info().rss, "workload": rows}
    (OUT / "baseline-feasibility.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "packages"}, indent=2))
    print(json.dumps(probe, indent=2))

if __name__ == "__main__":
    main()
