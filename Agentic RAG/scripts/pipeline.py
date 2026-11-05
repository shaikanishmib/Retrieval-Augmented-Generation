"""Run remaining stages sequentially so GPU models do not compete for memory."""
import subprocess
import sys
from agentic_rag.artifacts import ROOT

if __name__ == "__main__":
    stages=[['snapshot_runtime.py'],['experiment.py','pilot'],['probe_plan.py'],['generator_comparison.py'],['experiment.py','development'],['experiment.py','evaluation'],['experiment.py','ablations'],['analyze.py'],['audit_metrics.py'],['analyze_conflicts.py'],['probe_workflows.py']]
    for arguments in stages:
        print('Stage:',*arguments,flush=True)
        subprocess.run([sys.executable,str(ROOT/'scripts/run.py'),*arguments],check=True,cwd=ROOT.parent)
