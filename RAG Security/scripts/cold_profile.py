"""Separate fresh-process CLI latency from warmed model inference."""
import gc
import json
import subprocess
import sys
import time
import numpy as np
import psutil
import torch
from rag_security.artifacts import ROOT, write, read
from rag_security.models import LocalModel
from rag_security.policy import Document

fixture=ROOT/'data/cold-profile/evidence.txt'
fixture.parent.mkdir(parents=True,exist_ok=True)
fixture.write_text('The station identification color is amber.',encoding='utf-8')
profiles=[]
for key in read(ROOT/'configs/study.json')['models']:
    cold=[]
    peaks=[]
    answers=[]
    for _ in range(2):
        command=[sys.executable,str(ROOT/'scripts/run.py'),'answer.py','--question','What identification color does the station use?','--document',str(fixture),'--model',key,'--defense','D6']
        start=time.perf_counter()
        proc=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        peak=0
        while proc.poll() is None:
            if time.perf_counter()-start>60:
                proc.kill()
                raise TimeoutError('Local CLI exceeded the 60-second profile budget.')
            try:
                peak=max(peak,psutil.Process(proc.pid).memory_info().rss)
            except psutil.Error:
                pass
            time.sleep(.05)
        out,err=proc.communicate()
        assert proc.returncode==0,err
        response=json.loads(out)
        answers.append(response['answer'])
        cold.append(time.perf_counter()-start)
        peaks.append(peak)
    model=LocalModel(key)
    warm=[]
    for _ in range(3):
        result=model.generate('What identification color does the station use?',[Document('source-0',fixture.read_text(encoding='utf-8'))])
        warm.append(result['inference_seconds'])
    profiles.append({'model':key,'cold_process_seconds':cold,'cold_process_p50_seconds':float(np.median(cold)),'cold_answers':answers,'cold_clean_completion':sum('amber' in a.casefold() for a in answers)/len(answers),'warm_inference_seconds':warm,'warm_inference_p50_seconds':float(np.median(warm)),'sampled_peak_child_rss_bytes':max(peaks),'scope':'Cold process uses D6 and includes imports, screening, tokenization, weight loading and any generation; failed answers are retained. OS file caches are warm; no cache clearing. Warm values time unfiltered model generation, not the complete D6 policy. Two cold and three warm repetitions are descriptive, with different policy scopes.'})
    del model
    gc.collect()
    torch.cuda.empty_cache()
write(ROOT/'reports/cold-profile.json',profiles)
print(profiles)
