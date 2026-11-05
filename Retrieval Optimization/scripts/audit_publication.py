"""Check publication candidates for excluded artifacts and notebook execution state."""

import json
import re
import subprocess
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
PREFIX = ROOT.name + "/"


def git(*args):
    return subprocess.check_output(["git", "-c", f"safe.directory={REPO.as_posix()}", *args], cwd=REPO, text=True)


def main():
    paths = sorted(set(git("ls-files", "--cached", "--others", "--exclude-standard", "--", ROOT.name).splitlines()))
    if not paths:
        raise ValueError("No publication candidates")
    staged = git("diff", "--cached", "--name-only").splitlines()
    if any(not p.startswith(PREFIX) for p in staged):
        raise ValueError("Unrelated files are staged; they must remain outside this contribution")
    rejected = {".jsonl", ".npy", ".faiss", ".safetensors", ".bin", ".zip", ".tar", ".gz", ".tsv"}
    secrets = [re.compile(r"AKIA[0-9A-Z]{16}"), re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
               re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")]
    output = ROOT / "reports/publication-audit.json"
    checked = []
    for name in paths:
        path = REPO / name
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError(f"Publication candidate leaves the project: {name}")
        if path.suffix in rejected or "data" in path.relative_to(ROOT).parts or ".local" in path.parts:
            raise ValueError(f"Local artifact is a publication candidate: {name}")
        if path.stat().st_size > 1_000_000:
            raise ValueError(f"Large publication candidate requires review: {name}")
        text = path.read_text(encoding="utf-8-sig")
        if any(pattern.search(text) for pattern in secrets):
            raise ValueError(f"Possible credential material: {name}")
        if path != output:
            checked.append({"path": name, "bytes": path.stat().st_size})
    local = ROOT / "notebooks/01_complete_retrieval_research.ipynb"
    public = ROOT / "notebooks/01_public_retrieval_research.ipynb"
    a,b = [nbformat.read(p, as_version=4) for p in (local,public)]
    assert [(c.cell_type,c.source) for c in a.cells] == [(c.cell_type,c.source) for c in b.cells]
    assert all(c.outputs == [] and c.execution_count is None for c in b.cells if c.cell_type == "code")
    assert all(c.execution_count is not None for c in a.cells if c.cell_type == "code")
    for excluded in [local, ROOT/"data/raw/scifact/corpus.jsonl", ROOT/"data/raw/fiqa/corpus.jsonl", ROOT/"data/models/all-MiniLM-L6-v2/model.safetensors", ROOT/".local/legacy-work.zip"]:
        relative = excluded.relative_to(REPO).as_posix()
        assert relative not in paths
        assert git("check-ignore", relative).strip() == relative
    report = {"candidate_files_excluding_audit": checked, "candidate_bytes_excluding_audit": sum(p["bytes"] for p in checked),
              "excluded_artifact_checks": True, "cell_source_identity": True, "public_outputs_empty": True,
              "credential_pattern_scan": "No matching patterns in publication candidates",
              "scope": "Only rag-retrieval-optimization; source-text and prose review are separate manual checks"}
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "candidate_files_excluding_audit"}, indent=2))


if __name__ == "__main__":
    main()
