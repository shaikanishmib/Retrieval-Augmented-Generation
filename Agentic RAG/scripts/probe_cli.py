"""Exercise the actual CLI on synthetic documents and retain its complete output locally."""
import json
import subprocess
import sys
import time
from agentic_rag.artifacts import ROOT,write

if __name__=='__main__':
    documents=ROOT/'data/probes/cli-documents.json'
    write(documents,{'Ada':'Ada is based in London.','London':'London is in England.'})
    started=time.perf_counter()
    command=[sys.executable,str(ROOT/'scripts/run.py'),'answer.py','--documents',str(documents),'--question','In which country is Ada based?','--strategy','fixed']
    process=subprocess.run(command,cwd=ROOT.parent,capture_output=True,text=True,encoding='utf-8')
    (ROOT/'data/probes/cli-stderr.txt').write_text(process.stderr,encoding='utf-8')
    process.check_returncode()
    result=json.loads(process.stdout)
    assert result['state']['status']=='finished' and result['state']['calls']==1
    assert not result['answer'] or result['citation_valid']
    write(ROOT/'data/probes/cli-output.json',result)
    report=dict(scope='One synthetic two-source CLI smoke test; not benchmark or production-reliability evidence.',
                exit_code=process.returncode,answer=result['answer'],citation_valid=result['citation_valid'],calls=result['state']['calls'],
                reason=result['reason'],process_ms=(time.perf_counter()-started)*1000)
    write(ROOT/'reports/cli-smoke.json',report)
    print(report)
