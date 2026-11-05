"""Run a stage using an installed environment or read-only local package overlays."""
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
os.environ["PYTHONIOENCODING"] = "utf-8"
reference = ROOT.parent / "Advanced RAG"
paths = [ROOT / "src"]
if sys.prefix == sys.base_prefix:
    paths += [reference / "data/gpu-packages", reference / "data/packages"]
sys.path[:0] = list(map(str, paths))
os.environ["PYTHONPATH"] = os.pathsep.join(map(str, paths))
for key, value in {"HF_HOME": "data/hf", "MPLCONFIGDIR": "data/matplotlib", "IPYTHONDIR": "data/ipython"}.items():
    os.environ[key] = str(ROOT / value)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
stage = sys.argv.pop(1)
if stage == "test":
    import pytest
    raise SystemExit(pytest.main([str(ROOT / "tests"), "-p", "no:cacheprovider", *sys.argv[1:]]))
runpy.run_path(str(ROOT / "scripts" / stage), run_name="__main__")
