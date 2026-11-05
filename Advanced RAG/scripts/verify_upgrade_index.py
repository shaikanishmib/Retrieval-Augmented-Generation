"""Check staged publication bytes against working files and recorded hashes."""
import hashlib
import json
import subprocess
from advanced_rag.upgrade.artifacts import ROOT, REPORTS, read, preserved

def git(*args):
    return subprocess.check_output(["git", "-c", f"safe.directory={ROOT.parent.as_posix()}", *args], cwd=ROOT.parent)

def staged(path):
    return git("show", ":" + path.relative_to(ROOT.parent).as_posix())

preserved()
paths = git("diff", "--cached", "--name-only", "-z").decode().split("\0")
paths = [p for p in paths if p]
assert paths and all(p.startswith("Advanced RAG/") for p in paths)
for name in paths:
    path = ROOT.parent / name
    assert staged(path) == path.read_bytes(), f"Staged bytes differ: {name}"
review = json.loads(staged(REPORTS / "publication-review.json"))
for name, expected in review["provenance_sha256"].items():
    assert hashlib.sha256(staged(ROOT / name)).hexdigest() == expected, name
notebook = json.loads(staged(REPORTS / "notebook-verification.json"))
assert hashlib.sha256(staged(ROOT / "notebooks/02_public_grounding_upgrade.ipynb")).hexdigest() == notebook["public_sha256"]
freeze = read(REPORTS / "selection-freeze.json")
assert hashlib.sha256(staged(REPORTS / "selection.json")).hexdigest() == freeze["selection_sha256"]
print("Verified staged bytes and publication hashes for", len(paths), "files")
