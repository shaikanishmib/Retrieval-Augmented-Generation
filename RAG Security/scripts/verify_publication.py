"""Publish the reviewed notebook copy and run final local checks."""
import subprocess
import sys
from rag_security.artifacts import ROOT

for stage in ['publish_notebook.py','artifact_audit.py','build_report.py','checks.py','publication_audit.py']:
    print('Running',stage,flush=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/run.py'),stage],cwd=ROOT,check=True)
