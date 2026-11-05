"""Acquire one stronger specialized NLI candidate after the hardware audit."""
import json
import sys
import urllib.request
from llm_judge.artifacts import ROOT, read, sha, write

repo = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
existing_lock = read(ROOT/'models.lock.json')
if 'nli-base' in existing_lock:
    model = existing_lock['nli-base']
    for name,info in model['files'].items():
        assert sha(ROOT.parent/model['local_path']/name)==info['sha256'], name
    print('The stronger NLI checkpoint is already pinned and verified.')
    sys.exit(0)
with urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}', timeout=60) as response:
    info = json.load(response)
revision = info['sha']
target = ROOT / 'data/models/nli-base'
target.mkdir(parents=True, exist_ok=True)
names = ['config.json', 'tokenizer_config.json', 'special_tokens_map.json', 'spm.model', 'README.md']
available = {s['rfilename'] for s in info['siblings']}
names += ['model.safetensors' if 'model.safetensors' in available else 'pytorch_model.bin']
files = {}
for name in names:
    if name not in available:
        continue
    url = f'https://huggingface.co/{repo}/resolve/{revision}/{name}'
    path = target / name
    if not path.exists():
        tmp = path.with_suffix(path.suffix + '.download')
        with urllib.request.urlopen(url, timeout=180) as response, tmp.open('wb') as out:
            while block := response.read(8 * 1024 * 1024):
                out.write(block)
        tmp.replace(path)
    files[name] = {'sha256': sha(path), 'bytes': path.stat().st_size}
lock = read(ROOT / 'models.lock.json')
lock['nli-base'] = {'id': repo, 'revision': revision, 'license': info.get('cardData', {}).get('license'), 'model_card': f'https://huggingface.co/{repo}/blob/{revision}/README.md', 'local_path': 'LLM-as-a-Judge/data/models/nli-base', 'files': files, 'bytes': sum(f['bytes'] for f in files.values())}
write(ROOT / 'models.lock.json', lock)
print(lock['nli-base'])
