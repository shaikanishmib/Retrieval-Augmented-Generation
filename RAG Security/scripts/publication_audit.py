"""Audit publication scope, preserved projects and Git-normalized file bytes."""
import re
import subprocess
from rag_security.artifacts import ROOT, read, write, file_hash, source_identity

def git(*args):
    return subprocess.check_output(['git','-c',f'safe.directory={ROOT.parent.as_posix()}',*args],cwd=ROOT.parent,text=True).strip()

assert source_identity()==read(ROOT/'reports/selection-freeze.json')['source_identity']
prior=read(ROOT/'.local/prior-files.json')
changed=[f for f,h in prior.items() if not (ROOT.parent/f).exists() or file_hash(ROOT.parent/f)!=h]
assert not changed,changed
assert all(read(ROOT/f'reports/{s}-run.json')['attempts']>0 for s in ('development','evaluation','positions','ingestion','adaptive'))
assert read(ROOT/'reports/reconciliation.json')['metrics_rebuilt_from_attempt_ids']
assert read(ROOT/'reports/notebook-audit.json')['identical_sources_types_order']
files=git('ls-files','--cached','--others','--exclude-standard','-z','--','RAG Security','README.md').split('\0')
files=sorted(set(f for f in files if f))
for f in files:
    if f.startswith('RAG Security/'):
        assert not any('/'+bad in f for bad in ('data/','.local/','__pycache__/'))
        assert '01_complete_' not in f
        assert (ROOT.parent/f).stat().st_size<5_000_000
    path=ROOT.parent/f
    if path.suffix.lower() in ('.py','.md','.json','.toml','.txt','.ipynb'):
        content=path.read_text(encoding='utf-8')
        assert not re.search(r'(?<![A-Z0-9])AKIA[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{30,}|sk-proj-[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----',content),f
hashes={f:git('hash-object','--path='+f,f) for f in files if f not in ('RAG Security/reports/source-provenance.json','RAG Security/reports/publication-audit.json')}
write(ROOT/'reports/source-provenance.json',{'git_normalized_blobs':hashes,'scope':'Reviewed public files excluding these self-referential audit reports. Hashes use Git clean filters for each path.'})
write(ROOT/'reports/publication-audit.json',{'prior_projects_unchanged':True,'prior_tracked_files_verified':len(prior),'public_files_reviewed':len(files),'raw_data_weights_traces_excluded':True,'secret_pattern_scan_passed':True,'notebook_sources_identical':True,'frozen_sources_match':True,'metrics_reconciled':True,'licenses_reviewed':'docs/provenance.md','scope':'Regex secret checks supplement explicit path exclusions and manual review; they cannot prove absence of unknown sensitive content.'})
print('Verified',len(prior),'prior files and',len(files),'public files')
