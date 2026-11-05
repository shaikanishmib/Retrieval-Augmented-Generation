"""Record installed package versions selected by the research interpreter."""
import importlib.metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
packages = {}
for distribution in importlib.metadata.distributions():
    name = distribution.metadata["Name"]
    if name and name.lower() not in packages:
        packages[name.lower()] = (name, distribution.version)
text = "\n".join(f"{name}=={version}" for name, version in sorted(packages.values(), key=lambda x: x[0].lower())) + "\n"
(ROOT / "reports/requirements-lock.txt").write_text(text, encoding="utf-8")
print(len(packages), "package versions recorded")
