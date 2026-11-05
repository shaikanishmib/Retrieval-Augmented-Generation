"""Preserve superseded selection and partial trials before the hybrid correction."""
import shutil
from semantic_cache.artifacts import ROOT,write,sha

target = (ROOT/'data/superseded-hybrid-v1').resolve()
assert target.is_relative_to(ROOT.resolve())
target.mkdir(parents=True,exist_ok=True)
paths = [ROOT/'reports/selection-freeze.json',ROOT/'reports/execution-ledger.json',ROOT/'reports/generator-development.json',ROOT/'reports/embedding-development.json',ROOT/'data/traces']
manifest = {p.relative_to(ROOT).as_posix():{f.relative_to(p).as_posix():sha(f) for f in p.rglob('*.json')} if p.is_dir() else sha(p) for p in paths if p.exists()}
write(ROOT/'reports/development-revision.json',dict(reason='Hybrid initially duplicated answer-cache logic. Add evidence fallback and fresh generation; preserve superseded partial trials, then reselect before final evaluation.',preserved=manifest,deleted=[]))
for path in paths:
    if path.exists():
        destination = target/path.name
        assert not destination.exists()
        shutil.move(str(path),str(destination))
