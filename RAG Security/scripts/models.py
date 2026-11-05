"""Verify cached models and acquire one compact injection classifier."""
import json
import urllib.request
from rag_security.artifacts import ROOT, read, write, file_hash

def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'rag-security-research'}), timeout=120).read()

target = ROOT / 'models.lock.json'
if target.exists():
    locks = read(target)
else:
    old = read(ROOT.parent / 'Agentic RAG/models.lock.json')
    locks = {k: old[k] for k in ['flan-base', 'qwen06', 'qwen17', 'bge-small']}
    write(target, locks)
for key, lock in list(locks.items()):
    if key == 'injection':
        continue
    folder=ROOT.parent / lock['local_path']
    if not folder.exists():
        folder=ROOT / 'data/models' / key
    for name, meta in lock['files'].items():
        path = folder / name
        if not path.exists():
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(fetch(f"https://huggingface.co/{lock['id']}/resolve/{lock['revision']}/{name}"))
        assert file_hash(path) == meta['sha256'], path
    print(key, 'verified')

if 'injection' not in locks:
    info = json.loads(fetch('https://huggingface.co/api/models/protectai/deberta-v3-base-prompt-injection-v2'))
    locks['injection'] = {'id': info['id'], 'revision': info['sha'], 'license': 'apache-2.0', 'local_path': 'RAG Security/data/models/injection', 'files': {}, 'training_overlap': 'Upstream training membership not fully disclosed; independence from BIPIA cannot be established.'}
    write(target, locks)
lock = locks['injection']
for name in ['README.md', 'config.json', 'tokenizer_config.json', 'special_tokens_map.json', 'spm.model', 'tokenizer.json', 'model.safetensors']:
    path = ROOT.parent / lock['local_path'] / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fetch(f"https://huggingface.co/{lock['id']}/resolve/{lock['revision']}/{name}"))
    meta = {'sha256': file_hash(path), 'bytes': path.stat().st_size}
    if name in lock['files']:
        assert meta == lock['files'][name]
    lock['files'][name] = meta
    write(target, locks)
print('injection verified')
