"""Stable experiment identities and atomic JSON artifacts."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[2]

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def identity():
    files = sorted([*ROOT.glob('src/**/*.py'), *ROOT.glob('configs/*.json'), ROOT/'models.lock.json', ROOT/'sources.lock.json'])
    files += [ROOT/'scripts'/name for name in ['development.py','evaluate.py','equivalence_evaluation.py','answer_reuse_ablation.py']]
    return digest({p.relative_to(ROOT).as_posix(): sha(p) for p in files if p.exists()})
