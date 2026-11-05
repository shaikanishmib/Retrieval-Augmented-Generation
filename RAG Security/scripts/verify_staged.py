"""Verify the staged Git blobs against reviewed normalization hashes."""
import ast
import json
import subprocess
from rag_security.artifacts import ROOT, read, write, file_hash

def git(*args):
    return subprocess.check_output(['git','-c',f'safe.directory={ROOT.parent.as_posix()}',*args],cwd=ROOT.parent,text=True).strip()

expected=read(ROOT/'reports/source-provenance.json')['git_normalized_blobs']
checked={}
for path,blob in expected.items():
    actual=git('rev-parse',':'+path)
    assert actual==blob,(path,actual,blob)
    checked[path]=actual
paths=[p for p in git('diff','--cached','--name-only','-z').split('\0') if p]
assert paths and all(p=='README.md' or p.startswith('RAG Security/') for p in paths)
assert all('/data/' not in p and '/.local/' not in p and '01_complete_' not in p for p in paths)
for path in paths:
    p=ROOT.parent/path
    if p.suffix=='.py':
        ast.parse(p.read_text(encoding='utf-8'))
    elif p.suffix in ('.json','.ipynb'):
        json.loads(p.read_text(encoding='utf-8'))
prior=read(ROOT/'.local/prior-files.json')
assert all(file_hash(ROOT.parent/p)==h for p,h in prior.items())
git('diff','--cached','--check')
write(ROOT/'.local/index-audit.json',{'verified_normalized_blobs':checked,'staged_files':paths,'prior_files_verified':len(prior)})
audit=read(ROOT/'reports/publication-audit.json')
audit['git_index_blobs_verified']=True
audit['staged_scope_verified']=True
write(ROOT/'reports/publication-audit.json',audit)
print('Verified',len(checked),'Git-normalized staged blobs and',len(paths),'scoped changes.')
