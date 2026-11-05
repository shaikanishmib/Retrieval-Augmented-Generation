"""Check the experiment/publication boundaries after recorded stages complete."""
import json
from pathlib import Path
import pytest
from rag_security.artifacts import ROOT, read, source_identity, digest

def test_no_network_in_attack_runtime():
    for path in (ROOT/'src/rag_security').glob('*.py'):
        source=path.read_text(encoding='utf-8')
        assert 'import requests' not in source
        assert 'import socket' not in source
        assert 'import urllib' not in source
    models=(ROOT/'src/rag_security/models.py').read_text(encoding='utf-8')
    assert 'local_files_only=True' in models

def test_model_and_source_locks():
    models=read(ROOT/'models.lock.json')
    sources=read(ROOT/'sources.lock.json')
    for lock in [*models.values(),*sources.values()]:
        assert len(lock['revision'])==40
        assert lock['license']
        assert lock['files']
        assert all(len(meta['sha256'])==64 and meta['bytes']>0 for meta in lock['files'].values())

def test_frozen_sources():
    path=ROOT/'reports/selection-freeze.json'
    if not path.exists():
        pytest.skip('Development has not completed.')
    assert read(path)['source_identity']==source_identity()

def test_notebook_identity():
    import nbformat
    path=ROOT/'notebooks/01_public_rag_security_research.ipynb'
    if not path.exists():
        pytest.skip('Notebook publication has not completed.')
    public=nbformat.read(path,as_version=4)
    for cell in public.cells:
        if cell.cell_type=='code':
            assert cell.outputs==[] and cell.execution_count is None
    local_path=ROOT/'notebooks/01_complete_rag_security_research.ipynb'
    if local_path.exists():
        local=nbformat.read(local_path,as_version=4)
        assert [(c.cell_type,c.source) for c in local.cells]==[(c.cell_type,c.source) for c in public.cells]
        assert all(c.execution_count for c in local.cells if c.cell_type=='code')

def test_actual_inference_smoke():
    path=ROOT/'reports/smoke.json'
    if not path.exists():
        pytest.skip('Inference smoke stage has not completed.')
    report=read(path)
    assert report['real_inference']
    assert len({r['model'] for r in report['rows']})==2
    assert all(r['input_tokens']>0 and r['inference_seconds']>0 and not r['private_retrieved'] and not r['canary_exposed'] for r in report['rows'])

def test_attempt_manifests():
    for stage in ('development','evaluation','positions','ingestion','adaptive'):
        path=ROOT/f'reports/{stage}-run.json'
        if not path.exists():
            pytest.skip('All stages have not completed.')
        report=read(path)
        assert len(set(report['attempt_ids']))==report['attempts']
        local=ROOT/f'data/traces/{stage}'
        if local.exists():
            for eid in report['attempt_ids']:
                row=read(local/(eid+'.json'))
                assert row['attempt_id']==eid and row['experiment_id']==report['identity']

def test_validity_is_paired():
    run_path=ROOT/'reports/evaluation-run.json'
    if not run_path.exists():
        pytest.skip('Evaluation has not completed.')
    run=read(run_path)
    local=ROOT/'data/traces/evaluation'
    if not local.exists():
        pytest.skip('Detailed local traces are excluded from publication.')
    cases={}
    for eid in run['attempt_ids']:
        row=read(local/(eid+'.json'))
        key=(row['model'],row['task_id'],row['attack_id'])
        cases.setdefault(key,set()).add(row['valid_attack'])
    assert all(len(values)==1 for values in cases.values())

def test_recorded_outcomes_recompute():
    path=ROOT/'reports/evaluation-run.json'
    if not path.exists() or not (ROOT/'data/traces/evaluation').exists():
        pytest.skip('Detailed evaluation traces remain local.')
    from rag_security.attacks import success
    attacks={a['id']:a for a in read(ROOT/'data/attacks.json')['evaluation']}
    for eid in read(path)['attempt_ids']:
        row=read(ROOT/f'data/traces/evaluation/{eid}.json')
        attack=attacks.get(row['attack_id'])
        assert row['attack_success']==success(row['answer'],attack,row['reference'])
        assert not row['unauthorized_tool_execution']

def test_boundary_results():
    report=read(ROOT/'reports/boundaries.json')
    assert report['families']['tenant-miss']['protected_retrieval']['count']==0
    assert report['families']['tenant-cache-hit']['protected_exposure']['count']==0
    assert report['families']['tenant-cache-hit']['authorized_hit']['rate']==1
    assert report['families']['index-metadata']['forged_authorized']['count']==0
    assert report['families']['index-id-substitution']['id_only_exposure']['rate']==1
    assert report['families']['index-id-substitution']['integrity_checked_exposure']['count']==0
    assert report['families']['tool-policy']['executed']['count']==0
    assert report['families']['resource-budget']['blocked']['rate']==1

def test_tfidf_training_membership():
    parts=read(ROOT/'reports/partitions.json')
    assert len(parts['tasks']['development'])==16
    assert len(parts['tasks']['evaluation'])==32
    report=read(ROOT/'reports/detector-freeze.json')
    assert report['training_examples']==240
    assert len(report['training_identity'])==64

def test_public_exclusions():
    ignored=(ROOT/'.gitignore').read_text(encoding='utf-8')
    for pattern in ['data/','.local/','notebooks/01_complete_rag_security_research.ipynb']:
        assert pattern in ignored
