"""Run project scripts with dependencies stored inside the ignored data directory."""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data/packages"))
sys.path.insert(0, str(ROOT / "src"))
os.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT / "data/packages"), str(ROOT / "src")])
script = sys.argv.pop(1)
runpy.run_path(str(ROOT / "scripts" / script), run_name="__main__")
