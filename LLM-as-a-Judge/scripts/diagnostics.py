from llm_judge.bias import diagnostics
import subprocess
import sys
from llm_judge.artifacts import ROOT, read, write
diagnostics()
report_path = ROOT/'reports/bias-robustness.json'
report = read(report_path)
report['configuration'] = 'qwen17/rubric for perturbations and uncached repeatability; Qwen3-1.7B A/B prompt for pairwise; nli-base/claims for sentence units'
report['model_revisions'] = {key:value['revision'] for key,value in read(ROOT/'models.lock.json').items() if key in ['qwen17','nli-base']}
write(report_path,report)
subprocess.run([sys.executable,str(ROOT/'scripts/run.py'),'audit_perturbations.py'],cwd=ROOT,check=True)
subprocess.run([sys.executable,str(ROOT/'scripts/run.py'),'analyze_errors.py'],cwd=ROOT,check=True)
subprocess.run([sys.executable,str(ROOT/'scripts/run.py'),'selected_bias.py'],cwd=ROOT,check=True)
