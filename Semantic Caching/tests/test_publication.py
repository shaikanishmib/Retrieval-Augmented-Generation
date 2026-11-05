import nbformat
from semantic_cache.artifacts import ROOT,read,sha,identity

def test_notebook_sources_and_execution():
    full = ROOT/'notebooks/01_complete_semantic_caching_research.ipynb'
    public = ROOT/'notebooks/01_public_semantic_caching_research.ipynb'
    local = nbformat.read(full,as_version=4)
    clean = nbformat.read(public,as_version=4)
    assert [(c.cell_type,c.source) for c in local.cells]==[(c.cell_type,c.source) for c in clean.cells]
    assert all(not c.get('outputs') and c.get('execution_count') is None for c in clean.cells)
    assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in local.cells if c.cell_type=='code')
    record = read(ROOT/'reports/notebook-verification.json')
    assert sha(full)==record['full_sha256'] and sha(public)==record['public_sha256']

def test_final_trace_identity():
    freeze = read(ROOT/'reports/selection-freeze.json')
    assert freeze['identity']==identity()
    ledger = read(ROOT/'reports/execution-ledger.json')
    assert ledger['primary_complete']
    assert len(ledger['completed'])==90
    for row in ledger['completed']:
        path = ROOT/f"data/traces/{row['label']}.json"
        assert sha(path)==row['sha256']
        trace = read(path)
        assert trace['identity']==freeze['identity']
        assert len(trace['rows'])==row['requests']
