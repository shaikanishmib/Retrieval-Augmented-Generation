"""Audit new publication files and verify preservation of the complete baseline."""
import hashlib
import re
import subprocess
import nbformat
from advanced_rag.upgrade.artifacts import ROOT, REPORTS, read, save, preserved

def git(*args):
    return subprocess.check_output(["git", "-c", f"safe.directory={ROOT.parent.as_posix()}", "-c", "core.quotepath=false", *args], cwd=ROOT.parent, text=True)

preserved()
paths = git("ls-files", "--cached", "--others", "--exclude-standard", "--", ROOT.name).splitlines()
assert all(p.startswith(ROOT.name + "/") for p in git("diff", "--cached", "--name-only").splitlines())
for name in paths:
    path = ROOT.parent / name
    assert path.resolve().is_relative_to(ROOT.resolve()) and not path.is_symlink()
    assert "data" not in path.relative_to(ROOT).parts
    assert path.suffix not in {".bin", ".safetensors", ".npy", ".parquet", ".jsonl", ".part", ".zip"}
    assert path.stat().st_size < 1_000_000
    text = path.read_text(encoding="utf-8-sig")
    assert not re.search(r"AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", text)
local = ROOT / "notebooks/02_complete_grounding_upgrade.ipynb"
public = ROOT / "notebooks/02_public_grounding_upgrade.ipynb"
a, b = [nbformat.read(p, as_version=4) for p in (local, public)]
assert [(c.cell_type, c.source) for c in a.cells] == [(c.cell_type, c.source) for c in b.cells]
assert all(c.execution_count is None and not c.outputs for c in b.cells if c.cell_type == "code")
for path in (local, ROOT / "data/upgrade/models/qwen17/model.safetensors", ROOT / "data/upgrade/predictions"):
    relative = path.relative_to(ROOT.parent).as_posix()
    assert relative not in paths
    assert git("check-ignore", relative).strip() == relative
identity = read(REPORTS / "experiment-identity.json")
freeze = read(REPORTS / "selection-freeze.json")
assert hashlib.sha256((REPORTS / "selection.json").read_bytes()).hexdigest() == freeze["selection_sha256"]
question_selection = read(REPORTS / "answer-support-selection.json")
assert hashlib.sha256((ROOT / "src/advanced_rag/answer_support.py").read_bytes()).hexdigest() == question_selection["source_sha256"]
assert "fresh" not in read(REPORTS / "development-pilot/metrics.json")
save(REPORTS / "publication-audit.json", {"candidate_files": len(paths), "baseline_preserved": True, "source_equivalence": True,
     "public_outputs_empty": True, "local_artifacts_ignored": True, "credential_pattern_scan": "No matches",
     "development_pilot_had_no_fresh_evaluation": True, "study_fingerprint": identity["fingerprint"], "scope": "Advanced RAG only"})
print("Publication audit passed", len(paths), "files")
