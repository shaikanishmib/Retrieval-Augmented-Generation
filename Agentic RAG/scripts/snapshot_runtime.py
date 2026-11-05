"""Record importable package versions with local overlay precedence."""
import importlib.metadata
import platform
import sys
from agentic_rag.artifacts import ROOT,write

if __name__ == "__main__":
    packages={}
    for item in importlib.metadata.distributions():
        name=item.metadata.get("Name")
        if name:
            packages.setdefault(name,item.version)
    import torch,transformers,sentence_transformers,numpy,psutil
    actual={m.__name__:m.__version__ for m in [torch,transformers,sentence_transformers,numpy,psutil]}
    write(ROOT / "reports/runtime-environment.json",dict(python=sys.version,platform=platform.platform(),actual_imports=actual,distributions=packages))
    (ROOT / "reports/requirements-lock.txt").write_text("\n".join(sorted(f"{k}=={v}" for k,v in packages.items()))+"\n",encoding="utf-8")
    print(actual)
