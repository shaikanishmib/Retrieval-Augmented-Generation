"""Run frozen study stages sequentially, retaining stage logs and failures."""
import subprocess
import sys
import time
from llm_judge.artifacts import ROOT, write

if '--wait-for-freeze' in sys.argv:
    deadline = time.monotonic() + 2400
    while not (ROOT/'reports/selection-freeze.json').exists():
        if time.monotonic() >= deadline:
            raise TimeoutError('Development did not produce a selection freeze')
        time.sleep(5)

stages = [
    ['experiment.py','heldout'],
    ['diagnostics.py'],
    ['transfer.py'],
    ['synthetic_probes.py'],
    ['benchmark_runtime.py'],
    ['runtime_versions.py'],
    ['audit_annotations.py'],
    ['test','-q','--junitxml=reports/tests.xml'],
    ['build_notebook.py'],
    ['execute_notebook.py'],
    ['publish_notebook.py'],
    ['build_report.py'],
    ['test','-q','--junitxml=reports/tests.xml'],
]
for index, stage in enumerate(stages):
    print('Running', *stage, flush=True)
    result = subprocess.run([sys.executable,str(ROOT/'scripts/run.py'),*stage],cwd=ROOT)
    write(ROOT/'data/pipeline-status.json', {'stage':stage,'index':index,'returncode':result.returncode})
    if result.returncode:
        raise SystemExit(result.returncode)
print('Experiments, notebook and tests completed.',flush=True)
