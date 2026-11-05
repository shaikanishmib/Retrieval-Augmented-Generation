"""Run supplementary controls and execute the reviewed notebook sources."""
import subprocess
import sys
from rag_security.artifacts import ROOT

for stage in ['additional_probes.py','tool_controls.py','cold_profile.py','runtime.py','figures.py','build_notebook.py','execute_notebook.py']:
    print('Running',stage,flush=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/run.py'),stage],cwd=ROOT,check=True)
