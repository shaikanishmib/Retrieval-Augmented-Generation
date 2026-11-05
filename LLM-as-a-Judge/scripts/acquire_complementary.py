"""Download only the RAGBench TechQA test subset and original task documentation."""
import json
import urllib.request
from llm_judge.artifacts import ROOT, read, sha, write

repo = 'galileo-ai/ragbench'
lock_path = ROOT/'ragbench.lock.json'
previous = read(lock_path) if lock_path.exists() else None
if previous:
    revision = previous['revision']
    names = list(previous['files'])
else:
    with urllib.request.urlopen(f'https://huggingface.co/api/datasets/{repo}', timeout=60) as stream:
        info = json.load(stream)
    revision = info['sha']
    names = [s['rfilename'] for s in info['siblings'] if s['rfilename'] == 'README.md' or ('techqa/' in s['rfilename'] and 'test-' in s['rfilename'] and s['rfilename'].endswith('.parquet'))]
assert any(x.endswith('.parquet') for x in names), names
files = {}
for name in names:
    path = ROOT / 'data/raw/ragbench' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}'
    if not path.exists():
        with urllib.request.urlopen(url, timeout=180) as response:
            path.write_bytes(response.read())
    files[name] = {'url': url, 'sha256': sha(path), 'bytes': path.stat().st_size}
    if previous:
        assert files[name] == previous['files'][name], name
write(ROOT / 'ragbench.lock.json', {'id': repo, 'revision': revision, 'license': 'cc-by-4.0', 'files': files, 'role': 'Complementary transfer agreement with model-generated labels; not independent human ground truth.'})
print(files)
