"""Run required tests and retain a compact public test record."""
import subprocess
import sys
from rag_security.artifacts import ROOT, write

result=subprocess.run([sys.executable,str(ROOT/'scripts/run.py'),'test','-q'],cwd=ROOT,text=True,capture_output=True)
print(result.stdout)
print(result.stderr)
write(ROOT/'reports/tests.json',{'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'scope':'Unit and recorded-artifact regression tests; real local model inference is separately verified by smoke.py and the executed notebook.'})
raise SystemExit(result.returncode)
