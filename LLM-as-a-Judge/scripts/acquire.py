"""Acquire a pinned RAGTruth release; raw source text remains local."""
import urllib.request
from llm_judge.artifacts import ROOT, read, sha, write

REV = 'c103204b9ce28d6bbad859304bf30de72b8ed8fe'
files = {}
for name in ['dataset/response.jsonl', 'dataset/source_info.jsonl', 'README.md', 'LICENSE']:
    url = f'https://raw.githubusercontent.com/ParticleMedia/RAGTruth/{REV}/{name}'
    path = ROOT / 'data/raw/ragtruth' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url, timeout=120) as response:
            content = response.read()
        path.write_bytes(content)
    files[name] = {'url': url, 'sha256': sha(path), 'bytes': path.stat().st_size}
lock = {'repository': 'https://github.com/ParticleMedia/RAGTruth', 'revision': REV, 'license': 'MIT repository license; upstream content rights remain with original sources', 'files': files}
target = ROOT / 'sources.lock.json'
if target.exists() and read(target) != lock:
    raise RuntimeError('Downloaded files differ from the existing source lock')
write(target, lock)
print({name: value['bytes'] for name, value in files.items()})
