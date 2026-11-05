"""Artifact storage with explicit hashes and atomic checkpoint writes."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT.parent / "Advanced RAG"

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)

def fingerprint():
    paths = sorted((ROOT / "src").rglob("*.py")) + [ROOT / "configs/study.json", ROOT / "models.lock.json", ROOT / "sources.lock.json"]
    return digest({str(p.relative_to(ROOT)): sha(p) for p in paths})
