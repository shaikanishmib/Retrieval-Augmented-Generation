"""Verify publication scope, notebook state, hashes, and excluded artifacts."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent

def git(*args):
    return subprocess.check_output(["git", "-c", f"safe.directory={REPO.as_posix()}", *args], cwd=REPO, text=True)

def main():
    paths = git("-c", "core.quotepath=false", "ls-files", "--cached", "--others", "--exclude-standard", "--", ROOT.name).splitlines()
    assert paths
    staged = git("diff", "--cached", "--name-only").splitlines()
    assert all(p.startswith(ROOT.name + "/") for p in staged), "Unrelated staged files require separate handling"
    forbidden = {".jsonl", ".parquet", ".npy", ".safetensors", ".bin", ".zip", ".gz", ".part"}
    patterns = [r"AKIA[0-9A-Z]{16}", r"gh[pousr]_[A-Za-z0-9]{30,}", r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"]
    for name in paths:
        path = REPO / name
        assert path.resolve().is_relative_to(ROOT.resolve()) and not path.is_symlink()
        assert path.suffix not in forbidden and "data" not in path.relative_to(ROOT).parts
        assert path.stat().st_size < 1_000_000
        text = path.read_text(encoding="utf-8-sig")
        assert not any(re.search(pattern, text) for pattern in patterns), name
    local = ROOT / "notebooks/01_complete_advanced_rag_research.ipynb"
    public = ROOT / "notebooks/01_public_advanced_rag_research.ipynb"
    a, b = [nbformat.read(p, as_version=4) for p in (local, public)]
    assert [(c.cell_type, c.source) for c in a.cells] == [(c.cell_type, c.source) for c in b.cells]
    assert all(c.execution_count is None and not c.outputs for c in b.cells if c.cell_type == "code")
    for excluded in (local, ROOT / "data/raw/hotpot.json", ROOT / "data/models/generator/model.safetensors"):
        relative = excluded.relative_to(REPO).as_posix()
        assert relative not in paths
        assert git("check-ignore", relative).strip() == relative
    baseline = json.loads((ROOT / "reports/reference-audit.json").read_text())
    for name, info in baseline["files"].items():
        assert hashlib.sha256((REPO / "Retrieval Optimization" / name).read_bytes()).hexdigest() == info["sha256"]
    report = {"candidate_files": len(paths), "source_identity": True, "public_outputs_empty": True,
              "excluded_artifacts": True, "reference_unchanged": True, "credential_pattern_scan": "No matches", "scope": "Advanced RAG only"}
    (ROOT / "reports/publication-audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
