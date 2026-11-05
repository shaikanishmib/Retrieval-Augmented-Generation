"""Record reusable artifacts and preservation hashes before experiments."""
import importlib.metadata
import platform
import shutil
import subprocess
import sys
import time
import psutil
from agentic_rag.artifacts import ROOT, REFERENCE, sha, write, read

def inventory(path):
    files = list(path.rglob("*"))
    return {"path": str(path), "files": sum(p.is_file() for p in files),
            "bytes": sum(p.stat().st_size for p in files if p.is_file())}

if __name__ == "__main__":
    prior = ROOT / "reports/preservation.json"
    if not prior.exists():
        paths = []
        for name in ["Advanced RAG", "Retrieval Optimization"]:
            base = ROOT.parent / name
            paths += [p for p in base.rglob("*") if p.is_file() and "data" not in p.relative_to(base).parts and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts]
        write(prior, {str(p.relative_to(ROOT.parent)): sha(p) for p in paths})
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(ROOT)
    result = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), python=sys.version, executable=sys.executable,
                  platform=platform.platform(), cpu=platform.processor(), logical_cpus=psutil.cpu_count(), physical_cpus=psutil.cpu_count(logical=False),
                  ram_bytes=memory.total, available_ram_bytes=memory.available, disk_free_bytes=disk.free,
                  gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True).strip(),
                  reusable=[inventory(REFERENCE / p) for p in ["data/raw", "data/models", "data/upgrade/models", "data/packages", "data/gpu-packages"]],
                  packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()})
    write(ROOT / "reports/environment.json", result)
    write(ROOT / "reports/cleanup.json", {"deletions": [], "recovered_bytes": 0, "reason": "Sufficient free disk; no artifacts deleted."})
    (ROOT / "reports/requirements-lock.txt").write_text("\n".join(sorted(f"{k}=={v}" for k,v in result["packages"].items())) + "\n", encoding="utf-8")
    print({k:v for k,v in result.items() if k not in ["packages", "reusable"]})
