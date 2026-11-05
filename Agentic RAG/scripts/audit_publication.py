"""Validate local research artifacts and optionally inspect exact staged Git bytes."""
import argparse
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
import nbformat
from agentic_rag.artifacts import ROOT,read,sha,write,fingerprint,digest

def git(*args,input=None):
    return subprocess.check_output(['git','-c',f'safe.directory={ROOT.parent.as_posix()}',*args],cwd=ROOT.parent,input=input)

def audit(staged=False):
    failures=[]
    required=['metrics.json','paired-intervals.json','experiment-provenance.json','selection-freeze.json','data-validation.json','notebook-verification.json','tests.xml','runtime-environment.json','execution-ledger.json','support-review.json','metric-convention-audit.json','conflict-diagnostics.json','workflow-probes.json','source-provenance.json','cli-smoke.json']
    assert all((ROOT/'reports'/p).is_file() for p in required)
    for name,expected in read(ROOT/'reports/preservation.json').items():
        assert sha(ROOT.parent/name)==expected,name
    freeze=read(ROOT/'reports/selection-freeze.json')
    assert freeze['source_identity']==fingerprint()
    assert read(ROOT/'reports/experiment-provenance.json')['source_identity']==fingerprint()
    source_map=read(ROOT/'reports/source-provenance.json')
    assert source_map['executed_source_identity']==fingerprint()
    assert all(sha(ROOT/name)==r['workspace_sha256'] for name,r in source_map['files'].items())
    assert read(ROOT/'reports/execution-ledger.json')['primary_complete']
    metric_audit=read(ROOT/'reports/metric-convention-audit.json')
    assert metric_audit['mismatches']==0
    assert metric_audit['question_results_checked']==read(ROOT/'reports/execution-ledger.json')['completed_question_runs']
    assert read(ROOT/'reports/cli-smoke.json')['exit_code']==0
    assert all(r['answer_without_valid_citation']==0 and r['termination_failures']==0 for label,r in read(ROOT/'reports/reliability.json').items() if not label.startswith('none-'))
    tests=ET.parse(ROOT/'reports/tests.xml').getroot()
    suites=list(tests.iter('testsuite'))
    assert suites and all(int(s.attrib.get(k,0))==0 for s in suites for k in ['failures','errors','skipped'])
    sources=read(ROOT/'sources.lock.json')
    for record in sources:
        path=ROOT/('data/raw/squad-train.json' if record['name']=='squad' else 'data/raw/hotpot.json')
        assert sha(path)==record['sha256']
    for alias,record in read(ROOT/'models.lock.json').items():
        location=ROOT.parent/record['local_path']
        for name,info in record['files'].items():
            assert sha(location/name)==info['sha256'],(alias,name)
    full=ROOT/'notebooks/01_complete_agentic_rag_research.ipynb'
    public=ROOT/'notebooks/01_public_agentic_rag_research.ipynb'
    local=nbformat.read(full,as_version=4)
    stripped=nbformat.read(public,as_version=4)
    assert [(c.cell_type,c.source) for c in local.cells]==[(c.cell_type,c.source) for c in stripped.cells]
    assert all(not c.get('outputs') and c.get('execution_count') is None for c in stripped.cells)
    assert all(c.execution_count is not None for c in local.cells if c.cell_type=='code')
    assert not any(o.output_type=='error' for c in local.cells if c.cell_type=='code' for o in c.outputs)
    assert len(list((ROOT/'notebooks').glob('*.ipynb')))==2
    verification=read(ROOT/'reports/notebook-verification.json')
    assert sha(full)==verification['full_sha256'] and sha(public)==verification['public_sha256']
    provenance=read(ROOT/'reports/experiment-provenance.json')
    for label,record in provenance['artifacts'].items():
        path=ROOT/f'data/manifests/{label}.json'
        manifest=read(path)
        assert sha(path)==record['manifest_sha256']
        assert digest({r['id']:sha(ROOT/r['path']) for r in manifest['rows']})==record['trace_hash']
        if label.endswith('-evaluation'):
            assert all((ROOT/r['path']).stat().st_mtime>=(ROOT/'reports/selection-freeze.json').stat().st_mtime for r in manifest['rows'])
    staged_files={}
    candidates=[p for p in ROOT.rglob('*') if p.is_file() and not set(p.relative_to(ROOT).parts)&{'data','.local','__pycache__','.pytest_cache'} and p!=full]
    secrets=re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{32,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)')
    for p in candidates:
        assert p.stat().st_size<5*2**20,(p,'Oversized publication artifact')
        assert not secrets.search(p.read_bytes()),p
    if staged:
        names=git('diff','--cached','--name-only','-z').decode().split('\0')
        names=[n for n in names if n]
        assert names,'No files staged'
        assert all(n=='README.md' or n.startswith('Agentic RAG/') for n in names)
        for name in names:
            assert not set(Path(name).parts)&{'data','.local','__pycache__','.pytest_cache'},name
            assert '01_complete_' not in name
            blob=git('show',':'+name)
            actual=git('rev-parse',':'+name).decode().strip()
            expected=git('hash-object','--path='+name,'--stdin',input=(ROOT.parent/name).read_bytes()).decode().strip()
            assert actual==expected,name
            assert len(blob)<5*2**20 and not secrets.search(blob),name
            import hashlib
            staged_files[name]=dict(bytes=len(blob),git_blob=actual,sha256=hashlib.sha256(blob).hexdigest())
            if name.startswith('Agentic RAG/'):
                relative=str(Path(name).relative_to('Agentic RAG'))
                if relative in source_map['files']:
                    assert staged_files[name]['sha256']==source_map['files'][relative]['git_lf_sha256'],name
        git('diff','--cached','--check')
    result=dict(passed=True,staged=staged,source_identity=fingerprint(),preserved_files=len(read(ROOT/'reports/preservation.json')),
                notebook=verification,tests=sum(int(s.attrib['tests']) for s in suites),checked_models=len(read(ROOT/'models.lock.json')),
                checked_datasets=len(sources),reviewed_publication_files=len(candidates),staged_files=staged_files,
                third_party_text='No raw source passages or model text in public outputs. Dataset/model URLs, identifiers and attribution retained.',
                secret_scan='Token/private-key patterns plus publication-scope inspection; not a guarantee against all secrets.')
    # Keep the staged byte manifest local so it does not hash itself.
    write(ROOT/('data/staged-byte-audit.json' if staged else 'reports/publication-audit.json'),result)
    print('Publication audit passed:',len(candidates),'publication candidates;',len(staged_files),'staged files')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--staged',action='store_true')
    audit(parser.parse_args().staged)
