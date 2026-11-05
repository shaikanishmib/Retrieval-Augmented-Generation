"""Run upgrade stages with an isolated CUDA package overlay."""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
paths = ([ROOT / "src"] if sys.prefix != sys.base_prefix else
         [ROOT / "data/gpu-packages", ROOT / "data/packages", ROOT / "src"])
sys.path[:0] = [str(p) for p in paths]
os.environ["PYTHONPATH"] = os.pathsep.join(map(str, paths))
os.environ["HF_HOME"] = str(ROOT / "data/upgrade/hf")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
script = sys.argv.pop(1)
runpy.run_path(str(ROOT / "scripts" / script), run_name="__main__")
