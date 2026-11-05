"""Resume the frozen generation study and its prespecified diagnostics."""
import subprocess
import sys
from rag_security.artifacts import ROOT

for stage in ['development','evaluation','positions','ingestion','adaptive']:
    subprocess.run([sys.executable,str(ROOT/'scripts/run.py'),'experiment.py',stage],cwd=ROOT,check=True)
for stage in ['boundaries.py','smoke.py','analyze.py']:
    subprocess.run([sys.executable,str(ROOT/'scripts/run.py'),stage],cwd=ROOT,check=True)
