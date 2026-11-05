"""Check the preserved baseline and upgrade without replacing baseline test reports."""
import compileall
import argparse
import subprocess
import sys
from advanced_rag.upgrade.artifacts import ROOT, preserved

preserved()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--unit-only", action="store_true")
args = parser.parse_args()
for directory in ("src", "scripts", "tests_upgrade"):
    assert compileall.compile_dir(str(ROOT / directory), quiet=1)
targets = ["tests_upgrade/" + name for name in ("test_grounding.py", "test_evaluation.py", "test_adapters.py", "test_reasoning.py", "test_evidence_spans.py", "test_question_support.py")] if args.unit_only else ["tests", "tests_upgrade"]
report = "reports/upgrade/unit-tests.xml" if args.unit_only else "reports/upgrade/tests.xml"
subprocess.run([sys.executable, "-m", "pytest", *targets, "-q", "--junitxml=" + report], cwd=ROOT, check=True)
preserved()
