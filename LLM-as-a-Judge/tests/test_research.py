"""Check annotation semantics, leakage boundaries and experiment accounting."""
import copy
import numpy as np
import pytest
from llm_judge.artifacts import ROOT, digest, read, source_identity
from llm_judge.data import hallucination_label, assert_disjoint
from llm_judge.claims import claims, citation_ids, citation_validity, select_windows, token_f1
from llm_judge.metrics import evaluate, paired_interval, choose_threshold, selective
from llm_judge.calibration import fit_calibrator
from llm_judge.bias import perturb

def test_label_mapping_keeps_implicit_truth():
    assert hallucination_label([]) == 0
    assert hallucination_label([{'implicit_true': True, 'label_type': 'Evident Baseless Info'}]) == 1
    assert hallucination_label([{'label_type': 'Subtle Conflict'}]) == 1

def test_metrics_known_confusion():
    result = evaluate([0, 0, 1, 1], [.1, .6, .2, .9])
    assert result['accuracy'] == .5
    assert result['precision'] == .5
    assert result['recall'] == .5
    assert result['confusion'] == [[1, 1], [1, 1]]
    assert result['brier'] == pytest.approx(.255)

def test_metrics_reject_nonfinite():
    with pytest.raises(AssertionError):
        evaluate([0], [float('nan')])

def test_group_and_evidence_overlap_rejected():
    a = {'source_id': 'a', 'evidence_hash': 'same'}
    b = {'source_id': 'b', 'evidence_hash': 'same'}
    with pytest.raises(AssertionError):
        assert_disjoint({'a': [a], 'b': [b]})
    with pytest.raises(AssertionError):
        assert_disjoint({'a': [a], 'b': [a]})

def test_actual_partitions_disjoint():
    assert_disjoint({k: read(ROOT / f'data/partitions/{k}.json') for k in ['development', 'validation', 'test']})

def test_claim_offsets_and_evidence_retained():
    text = '  Paris is in France. Water freezes. '
    units = claims(text)
    assert len(units) == 2
    for unit in units:
        assert text[unit['start']:unit['end']] == unit['text']
    context = ' '.join(['alpha'] * 200 + ['Paris France'] * 20)
    for window in select_windows(context, 'Paris France'):
        assert context[window['start']:window['end']] == window['text']

def test_citations_missing_and_fabricated():
    assert citation_ids('Fact [1]. Other [99].') == ['1', '99']
    assert citation_validity('Fact [1]. Other [99].', ['1']) == .5
    assert citation_validity('No citations', ['1']) is None

def test_token_f1_known_overlap():
    assert token_f1('a a b', 'a b b') == pytest.approx(2 / 3)

def test_calibration_leakage_rejected():
    with pytest.raises(ValueError):
        fit_calibrator([[0], [1]], [0, 1], ['same'], ['same'])

def test_identity_canonical_and_change_sensitive():
    assert digest({'b': 2, 'a': 1}) == digest({'a': 1, 'b': 2})
    assert digest({'a': 1}) != digest({'a': 2})

def test_bias_perturbations_preserve_original():
    row = {'response': 'Paris is in France.', 'context': 'Paris is in France.', 'question': 'Where?'}
    before = copy.deepcopy(row)
    repeated = perturb(row, 'verbosity')
    assert repeated['response'].count(row['response']) == 2
    assert row == before
    assert perturb(row, 'identity')['context'] == row['context']
    assert 'Ignore' in perturb(row, 'injection')['response']
    with pytest.raises(ValueError):
        perturb(row, 'invalid')

def test_selective_coverage_monotonic():
    curve = selective([0, 1, 1], [.1, .6, .9])
    assert all(a['coverage'] >= b['coverage'] for a, b in zip(curve, curve[1:]))

def test_frozen_identity_and_calibration_groups():
    path = ROOT / 'reports/selection-freeze.json'
    if not path.exists():
        pytest.skip('Development has not frozen yet')
    freeze = read(path)
    assert freeze['source_identity'] == source_identity()
    assert not set(freeze['fit_groups']) & set(freeze['selection_groups'])

@pytest.mark.integration
def test_actual_nli_inference_and_failure():
    from llm_judge.models import Adapter
    adapter = Adapter('nli-base')
    row = {'response': 'Paris is in France.', 'context': 'Paris is the capital of France.'}
    result = adapter.nli(row)
    assert 0 <= result['score'] <= 1
    assert result['claims'][0]['selected_evidence']
    contradicted = adapter.nli({'response':'Paris is in Germany.','context':'Paris is in France, not Germany.'})
    assert result['score'] < contradicted['score']
    multi = adapter.nli({'response':'Paris is in France. Water is wet.','context':'Paris is in France. Water is wet.'})
    assert multi['score'] == max(c['unsupported'] for c in multi['claims'])
    assert multi['mean_score'] == pytest.approx(np.mean([c['unsupported'] for c in multi['claims']]))
    with pytest.raises(ValueError):
        adapter.nli({'response': '', 'context': ''})
    from llm_judge.tracking import infer
    failed = {'id':'integration-empty','source_id':'integration','response':'','context':''}
    with pytest.raises(ValueError):
        infer([failed],adapter,'claims')
    failures = list((ROOT/'data/traces').glob('*/integration-empty-*.failure.json'))
    assert failures and read(failures[-1])['error_type']=='ValueError'
    assert not any(p for p in (ROOT/'data/traces').glob('*/integration-empty-*.json') if not p.name.endswith('.failure.json'))

def test_publication_notebook_identity():
    import nbformat
    complete = ROOT / 'notebooks/01_complete_llm_judge_research.ipynb'
    public = ROOT / 'notebooks/01_public_llm_judge_research.ipynb'
    if not public.exists():
        pytest.skip('Notebook not published yet')
    a, b = [nbformat.read(p, as_version=4) for p in [complete, public]]
    assert [(c.cell_type, c.source) for c in a.cells] == [(c.cell_type, c.source) for c in b.cells]
    assert all(c.execution_count is None and not c.outputs for c in b.cells if c.cell_type == 'code')
    assert all(c.execution_count is not None for c in a.cells if c.cell_type == 'code')
    assert not any(o.output_type == 'error' for c in a.cells if c.cell_type == 'code' for o in c.outputs)
