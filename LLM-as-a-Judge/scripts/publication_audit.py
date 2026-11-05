"""Reconcile scores, notebooks, provenance and the publication staging area."""
import hashlib
import re
import subprocess
import sys
import nbformat
from llm_judge.artifacts import ROOT, read, write, sha, digest, source_identity
from llm_judge.metrics import evaluate

repo = ROOT.parent
git = ['git', '-c', f'safe.directory={repo.as_posix()}']
def command(*args):
    return subprocess.check_output(git + list(args), cwd=repo)

freeze = read(ROOT / 'reports/selection-freeze.json')
assert freeze['source_identity'] == source_identity()
assert freeze['models_identity'] == digest(read(ROOT / 'models.lock.json'))
assert sha(ROOT/'data/classifier.joblib') == read(ROOT/'trained-classifier.lock.json')['sha256']
for relative, expected in read(ROOT / 'data/prior-projects.json').items():
    assert sha(repo / relative) == expected, relative
for name, info in read(ROOT / 'sources.lock.json')['files'].items():
    assert sha(ROOT / 'data/raw/ragtruth' / name) == info['sha256']
for name, info in read(ROOT / 'ragbench.lock.json')['files'].items():
    assert sha(ROOT / 'data/raw/ragbench' / name) == info['sha256']
for model in read(ROOT / 'models.lock.json').values():
    for name, info in model['files'].items():
        assert sha(repo / model['local_path'] / name) == info['sha256']
rows = read(ROOT / 'data/partitions/test.json')
scores = read(ROOT / 'data/scores/test.json')['scores']
results = read(ROOT / 'reports/heldout.json')['results']
for name, values in scores.items():
    actual = evaluate([r['y'] for r in rows], values, freeze['thresholds'][name])
    assert actual == results[name], name
local, public = [nbformat.read(ROOT / f'notebooks/01_{tag}_llm_judge_research.ipynb', as_version=4) for tag in ['complete','public']]
assert [(c.cell_type,c.source) for c in local.cells] == [(c.cell_type,c.source) for c in public.cells]
assert all(c.execution_count is not None for c in local.cells if c.cell_type == 'code')
assert all(c.execution_count is None and not c.outputs for c in public.cells if c.cell_type == 'code')
assert not any(o.output_type == 'error' for c in local.cells if c.cell_type == 'code' for o in c.outputs)
staged = [x for x in command('diff','--cached','--name-only','-z').decode().split('\0') if x]
normalized = {}
for path in staged:
    assert path == 'README.md' or path.startswith('LLM-as-a-Judge/'), path
    assert not any(part in {'data','.local','__pycache__','.aws'} for part in path.split('/')), path
    assert '01_complete_' not in path, path
    content = command('show', ':' + path)
    working = (repo / path).read_bytes()
    # Git applies configured text normalization; compare repository object identities.
    index_id = command('rev-parse', ':' + path).decode().strip()
    working_id = command('hash-object', '--path=' + path, path).decode().strip()
    assert index_id == working_id, path
    if path.endswith(('.py','.md','.json','.toml','.ipynb','.txt')):
        text = content.decode('utf-8')
        assert not re.search(r'(?:sk-[A-Za-z0-9]{24,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)', text), path
    if path != 'LLM-as-a-Judge/reports/publication-audit.json':
        normalized[path] = {'git_blob': index_id, 'published_sha256': hashlib.sha256(content).hexdigest(), 'working_sha256': hashlib.sha256(working).hexdigest()}
write(ROOT / 'reports/publication-audit.json', {'metrics_reconciled':True,'previous_tracked_files_unchanged':len(read(ROOT/'data/prior-projects.json')),'dataset_and_model_hashes_verified':True,'notebook_source_identity':True,'executed_notebook_preserved':True,'raw_text_and_weights_excluded':True,'staged_files_checked':len(staged),'git_normalized_publication_hashes':normalized,'hash_manifest_policy':'This audit file is excluded from its own hash manifest to avoid self-reference.'})
print('Publication checks passed:', len(staged), 'staged files.')
