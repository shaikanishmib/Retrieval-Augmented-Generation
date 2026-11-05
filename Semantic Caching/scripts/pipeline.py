"""Execute frozen evaluation, diagnostics and analysis sequentially without shared GPU trials."""
import subprocess
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
for stage in ['development.py','evaluate.py','equivalence_evaluation.py','answer_reuse_ablation.py','analyze.py']:
    subprocess.run([sys.executable,str(root/'scripts/run.py'),stage],check=True)
