"""Download only pinned research files; inference never uses the network."""
import urllib.request
from rag_security.artifacts import ROOT, read, write, file_hash

def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'local-rag-security-research'})
    return urllib.request.urlopen(req, timeout=60).read()

lock_path = ROOT / 'sources.lock.json'
if lock_path.exists():
    lock = read(lock_path)
else:
    import json
    lock = {}
    for name, repo in [('bipia', 'microsoft/BIPIA'), ('agentdojo', 'ethz-spylab/agentdojo')]:
        rev = json.loads(fetch(f'https://api.github.com/repos/{repo}/commits/main'))['sha']
        lock[name] = {'repo': repo, 'revision': rev, 'license': 'MIT', 'files': {}}
    write(lock_path, lock)

for name, paths in {
    'bipia': ['LICENSE', 'README.md', 'benchmark/README.md', 'benchmark/email/train.jsonl', 'benchmark/email/test.jsonl', 'benchmark/table/train.jsonl', 'benchmark/table/test.jsonl', 'benchmark/text_attack_train.json', 'benchmark/text_attack_test.json', 'bipia/data/email.py', 'bipia/data/table.py', 'pyproject.toml'],
    'agentdojo': ['LICENSE', 'README.md', 'pyproject.toml'],
}.items():
    entry = lock[name]
    for path in paths:
        local = ROOT / 'data/raw' / name / path
        if not local.exists():
            url = f"https://raw.githubusercontent.com/{entry['repo']}/{entry['revision']}/{path}"
            data = fetch(url)
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(data)
        measured = {'sha256': file_hash(local), 'bytes': local.stat().st_size}
        if path in entry['files']:
            assert measured == entry['files'][path], path
        entry['files'][path] = measured
        write(lock_path, lock)
    print(name, entry['revision'], len(entry['files']))
