"""Stable identities and atomic local artifacts."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temp.replace(path)

def source_identity():
    paths=[p for folder in ('src', 'configs') for p in sorted((ROOT / folder).rglob('*')) if p.is_file() and '__pycache__' not in str(p)]
    paths += [ROOT / 'scripts' / name for name in ('run.py','models.py','acquire.py','prepare.py','detectors.py','experiment.py')]
    return digest({str(p.relative_to(ROOT)): file_hash(p) for p in paths})
