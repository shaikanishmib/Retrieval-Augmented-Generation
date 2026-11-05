"""Run syntax, unit, and artifact checks using project-local dependencies."""
import compileall
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
assert compileall.compile_dir(str(ROOT / "src"), quiet=1)
assert compileall.compile_dir(str(ROOT / "scripts"), quiet=1)
assert compileall.compile_dir(str(ROOT / "tests"), quiet=1)
subprocess.run([sys.executable, "-m", "pytest", "--junitxml=reports/tests.xml", "-q"], cwd=ROOT, check=True)
