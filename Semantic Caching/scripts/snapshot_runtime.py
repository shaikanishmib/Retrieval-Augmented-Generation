"""Record the actual import environment and exact installed distribution inventory."""
import importlib.metadata
import platform
import sys
from semantic_cache.artifacts import ROOT,write
versions = {}
for dist in importlib.metadata.distributions():
    name = dist.metadata.get('Name')
    if name and name.lower() not in {k.lower() for k in versions}:
        versions[name] = dist.version
write(ROOT/'reports/runtime-environment.json',dict(python=platform.python_version(),executable=sys.executable,versions=versions))
(ROOT/'reports/requirements-lock.txt').write_text('\n'.join(f'{k}=={v}' for k,v in sorted(versions.items(),key=lambda pair:pair[0].lower()))+'\n',encoding='utf-8')
print(len(versions),'distributions recorded')
