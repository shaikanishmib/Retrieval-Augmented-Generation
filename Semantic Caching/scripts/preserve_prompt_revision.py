"""Preserve all affected trial artifacts before correcting prompt truncation."""
import shutil
from semantic_cache.artifacts import ROOT,write,sha

target = (ROOT/'data/superseded-prompt-v1').resolve()
report_name = 'prompt-revision.json'
reason = 'Whole-prompt truncation could remove the question after long evidence. Reserve instructions/question/suffix, truncate evidence only, assert final token limit, and reselect models before all final trials.'
if target.exists():
    target = (ROOT/'data/superseded-scope-v1').resolve()
    report_name = 'scope-revision.json'
    reason = 'Bind cache keys to exact generator and embedding model revisions, rather than aliases alone. Preserve early trials and reselect before final evaluation.'
assert target.is_relative_to(ROOT.resolve())
target.mkdir(parents=True,exist_ok=True)
paths = [ROOT/'reports/selection-freeze.json',ROOT/'reports/execution-ledger.json',ROOT/'reports/generator-development.json',ROOT/'reports/embedding-development.json',ROOT/'data/traces',ROOT/'data/development']
manifest = {p.relative_to(ROOT).as_posix():{f.relative_to(p).as_posix():sha(f) for f in p.rglob('*.json')} if p.is_dir() else sha(p) for p in paths if p.exists()}
write(ROOT/'reports'/report_name,dict(reason=reason,preserved=manifest,deleted=[]))
for path in paths:
    if path.exists():
        destination = target/path.name
        assert not destination.exists()
        shutil.move(str(path),str(destination))
