"""Verify reusable weights or download exact locked files into this project."""
import urllib.request
from llm_judge.artifacts import ROOT, read, sha, write

lock = read(ROOT / 'models.lock.json')
for key, model in lock.items():
    existing = ROOT.parent / model['local_path']
    reusable = existing.exists() and all((existing/name).exists() and sha(existing/name)==info['sha256'] for name,info in model['files'].items())
    destination = existing if reusable else ROOT / 'data/models' / key
    destination.mkdir(parents=True, exist_ok=True)
    for name, info in model['files'].items():
        path = destination / name
        if not path.exists():
            url = f"https://huggingface.co/{model['id']}/resolve/{model['revision']}/{name}"
            temp = path.with_suffix(path.suffix + '.download')
            with urllib.request.urlopen(url, timeout=180) as stream, temp.open('wb') as output:
                while block := stream.read(8 * 1024 * 1024):
                    output.write(block)
            assert sha(temp) == info['sha256'], name
            temp.replace(path)
        assert sha(path) == info['sha256'], name
    if destination != existing:
        model['local_path'] = destination.relative_to(ROOT.parent).as_posix()
write(ROOT / 'models.lock.json', lock)
print('Verified all locked files. Missing weights were downloaded inside this project.')
