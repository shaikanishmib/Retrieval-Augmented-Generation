"""Record the selected package overlay and model runtime without changing baseline reports."""
import importlib.metadata
import os
import platform
import subprocess
import sys
import torch
from advanced_rag.upgrade.artifacts import ROOT, REPORTS, save, preserved

preserved()
packages = {}
for distribution in importlib.metadata.distributions():
    name = distribution.metadata["Name"]
    if name and name.lower() not in packages:
        packages[name.lower()] = (name, distribution.version)
(REPORTS / "requirements-lock.txt").write_text("\n".join(f"{name}=={version}" for name, version in sorted(packages.values(), key=lambda x:x[0].lower())) + "\n", encoding="utf-8")
save(REPORTS / "environment.json", {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
     "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "threads": 4,
     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
     "paid_api_calls": 0, "api_charges_usd": 0, "local_cost": "Electricity and hardware depreciation unmeasured",
     "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM"), "package_overlay": [str(p) for p in sys.path[:3]]})
print(torch.__version__, torch.cuda.is_available(), len(packages))
