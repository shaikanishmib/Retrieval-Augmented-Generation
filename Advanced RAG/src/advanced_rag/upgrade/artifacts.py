"""Store resumable results under content fingerprints."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "reports/upgrade"
LOCAL = ROOT / "data/upgrade"

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = path.read_bytes()
        current = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode()
        if previous != current:
            archive = LOCAL / "artifact-history" / (hashlib.sha256(previous).hexdigest() + ".json")
            archive.parent.mkdir(parents=True, exist_ok=True)
            if not archive.exists():
                archive.write_bytes(previous)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(partial, path)

def sources():
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "src/advanced_rag/upgrade").glob("*.py")}

def preserved():
    for name, expected in read(REPORTS / "preservation.json")["files"].items():
        with (ROOT.parent / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Preserved baseline file changed: {name}")
