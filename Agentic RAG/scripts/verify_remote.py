"""Verify published Git objects and download the public notebook at the exact commit."""
import hashlib
import json
import subprocess
import urllib.request
from agentic_rag.artifacts import ROOT,read,write

if __name__=='__main__':
    git=['git','-c',f'safe.directory={ROOT.parent.as_posix()}']
    commit=subprocess.check_output([*git,'rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip()
    remote=subprocess.check_output([*git,'ls-remote','origin','refs/heads/main'],cwd=ROOT.parent,text=True).split()[0]
    assert remote==commit
    endpoint=f'https://api.github.com/repos/shaikanishmib/Retrieval-Augmented-Generation/git/trees/{commit}?recursive=1'
    request=urllib.request.Request(endpoint,headers={'User-Agent':'agentic-rag-publication-audit'})
    with urllib.request.urlopen(request,timeout=120) as response:
        tree=json.load(response)
    remote_files={v['path']:v['sha'] for v in tree['tree'] if v['type']=='blob'}
    expected=read(ROOT/'data/staged-byte-audit.json')['staged_files']
    assert all(remote_files[p]==r['git_blob'] for p,r in expected.items())
    notebook='Agentic RAG/notebooks/01_public_agentic_rag_research.ipynb'
    url=f'https://raw.githubusercontent.com/shaikanishmib/Retrieval-Augmented-Generation/{commit}/'+notebook.replace(' ','%20')
    with urllib.request.urlopen(url,timeout=120) as response:
        payload=response.read()
    assert hashlib.sha256(payload).hexdigest()==expected[notebook]['sha256']
    record=dict(commit=commit,remote_main=remote,verified_git_blobs=len(expected),public_notebook_sha256=hashlib.sha256(payload).hexdigest(),url=url)
    write(ROOT/'data/remote-publication-verification.json',record)
    print(record)
