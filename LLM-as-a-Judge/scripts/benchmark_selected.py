"""Measure the frozen CLI on three real task-balanced development examples."""
import json
import subprocess
import sys
import time
import numpy as np
from llm_judge.artifacts import ROOT, read, write

rows = read(ROOT/'data/partitions/development.json')
subset = [next(r for r in rows if r['task']==task) for task in ['QA','Summary','Data2txt']]
freeze = read(ROOT/'reports/selection-freeze.json')
reference = read(ROOT/'data/scores/development.json')['scores'][freeze['selected']]
records = []
folder = ROOT/'data/runtime-cli'
folder.mkdir(parents=True,exist_ok=True)
for row in subset:
    evidence,answer = folder/'evidence.txt',folder/'answer.txt'
    evidence.write_text(row['context'],encoding='utf-8')
    answer.write_text(row['response'],encoding='utf-8')
    start = time.perf_counter()
    run = subprocess.run([sys.executable,str(ROOT/'scripts/run.py'),'judge.py','--evidence',str(evidence),'--answer',str(answer),'--question',row['question']],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    seconds = time.perf_counter()-start
    (folder/(row['id']+'.stderr.txt')).write_text(run.stderr,encoding='utf-8')
    if run.returncode:
        raise RuntimeError(run.stderr)
    result = json.loads(run.stdout)
    expected = reference[next(i for i,r in enumerate(rows) if r['id']==row['id'])]
    assert abs(result['unsupported_score']-expected)<1e-6
    records.append({'id':row['id'],'task':row['task'],'cold_process_wall_seconds':seconds,'score_matches_retained_prediction':True,**result})
write(folder/'records.json',records)
write(ROOT/'reports/selected-runtime.json',{'configuration':freeze['selected'],'n':len(records),'scope':'Three task-balanced development inputs, separate cold CLI processes. Includes imports, component model loading, tokenization and inference. Each required model loads once per request and supplies its related features; not a persistent-serving benchmark. p95 is descriptive for this small sample.','p50_cold_process_seconds':float(np.median([r['cold_process_wall_seconds'] for r in records])),'p95_cold_process_seconds':float(np.quantile([r['cold_process_wall_seconds'] for r in records],.95)),'evaluations_per_second':len(records)/sum(r['cold_process_wall_seconds'] for r in records),'peak_process_working_set':max(r['peak_process_working_set'] for r in records),'peak_gpu_allocated':max(r['peak_gpu_allocated'] for r in records),'mean_loading_seconds':float(np.mean([r['model_loading_seconds'] for r in records])),'retained_prediction_agreement':True})
print('Frozen CLI agrees with retained scores; cold-process runtime recorded.')
