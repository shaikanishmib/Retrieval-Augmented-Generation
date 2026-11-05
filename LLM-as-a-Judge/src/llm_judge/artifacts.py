"""Canonical identities and atomic local artifacts."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def source_identity():
    return digest({p.relative_to(ROOT).as_posix(): sha(p) for folder in ['src', 'configs'] for p in sorted((ROOT / folder).rglob('*')) if p.is_file() and '__pycache__' not in p.parts})
