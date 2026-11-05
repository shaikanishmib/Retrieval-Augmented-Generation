"""Record imported versions and a portable package inventory."""
import importlib
import importlib.metadata as md
import subprocess
import sys
from rag_security.artifacts import ROOT, write

modules=['torch','transformers','numpy','scipy','sklearn','pandas','matplotlib','nbformat','nbclient','psutil','joblib']
versions={name:{'version':getattr(importlib.import_module(name),'__version__','unknown')} for name in modules}
import torch
versions['torch']['cuda_available']=torch.cuda.is_available()
versions['torch']['cuda_runtime']=torch.version.cuda
write(ROOT/'reports/runtime-environment.json',{'python':sys.version,'modules':versions})
packages={d.metadata.get('Name'):d.version for d in md.distributions() if d.metadata.get('Name')}
(ROOT/'reports/requirements-lock.txt').write_text('\n'.join(f'{k}=={v}' for k,v in sorted(packages.items(),key=lambda p:p[0].lower()))+'\n',encoding='utf-8')
print(versions)
