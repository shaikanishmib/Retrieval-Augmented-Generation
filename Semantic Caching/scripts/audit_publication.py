"""Verify local research, preserved projects, source-safe publication and normalized Git bytes."""
import argparse
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
import nbformat
import numpy as np
from semantic_cache.artifacts import ROOT,read,write,sha,identity,digest
from semantic_cache.metrics import aggregate,qa_score
from semantic_cache.data import load_qa
from semantic_cache.security import lexical_guard

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--staged',action='store_true')
args = parser.parse_args()
def git(*arguments,input=None):
    return subprocess.check_output(['git','-c',f'safe.directory={ROOT.parent.as_posix()}',*arguments],cwd=ROOT.parent,input=input)

freeze = read(ROOT/'reports/selection-freeze.json')
assert identity()==freeze['identity']
for relative,expected in read(ROOT/'data/preservation.json').items():
    assert sha(ROOT.parent/relative)==expected,relative
ledger = read(ROOT/'reports/execution-ledger.json')
assert ledger['primary_complete'] and len(ledger['completed'])==90
for record in ledger['completed']:
    path = ROOT/f"data/traces/{record['label']}.json"
    assert sha(path)==record['sha256']
    trace = read(path)
    assert trace['identity']==freeze['identity']
    assert len(trace['rows'])==record['requests']
    assert path.stat().st_mtime>=freeze['created_unix']
metrics = read(ROOT/'reports/metrics.json')
trial_metrics = read(ROOT/'reports/trial-metrics.json')
assert len(trial_metrics)==90
for record in trial_metrics:
    rows = read(ROOT/f"data/traces/{record['label']}.json")['rows']
    assert all(record[k]==v for k,v in aggregate(rows).items())
for record in metrics:
    label = f"{record['dataset']}-{record['workload']}-{record['strategy']}"
    rows = [r for t in range(3) for r in read(ROOT/f'data/traces/{label}-{t}.json')['rows']]
    for key,value in aggregate(rows).items():
        assert value==record[key],(label,key)
    assert all(qa_score(r['answer'],r['answers'])==(r['em'],r['f1']) for r in rows)
    assert all(r['input_tokens']<=480 and ('Question: '+r['question']+'\nEvidence:\n') in r['prompt'] for r in rows if r['generation_calls'])
ablation_rows = read(ROOT/'data/ablations/answer-reuse.json')
annotations = {r['id']:r['answers'] for r in load_qa('squad')}
assert all(qa_score(r['answer'],annotations[r['target_id']])==(r['em'],r['f1']) for r in ablation_rows)
assert all(qa_score(r['baseline_answer'],annotations[r['target_id']])[1]==r['baseline_f1'] for r in ablation_rows)
for record in read(ROOT/'reports/answer-reuse-ablation.json'):
    rows = [r for r in ablation_rows if all(r[k]==record[k] for k in ['variant','threshold','strategy'])]
    assert len(rows)==record['requests']
    assert sum(r['hit']!='miss' for r in rows)==record['hits']
    assert sum(r['false_hit'] for r in rows)==record['false_reuse']
    assert float(np.mean([r['f1'] for r in rows]))==record['f1']
    assert float(np.mean([r['baseline_f1'] for r in rows]))==record['baseline_f1']
    assert float(np.median([r['total_ms'] for r in rows]))==record['p50_ms']
for alias,record in read(ROOT/'reports/equivalence-evaluation.json').items():
    pairs = read(ROOT/f'data/evaluation/{alias}-pairs.json')
    annotated = {r['id']:r for r in read(ROOT/'data/partitions/paws.json')['evaluation']}
    assert all(r['label']==annotated[r['id']]['label'] and r['guard']==lexical_guard(annotated[r['id']]['sentence1'],annotated[r['id']]['sentence2']) for r in pairs)
    for sweep in record['sweep']:
        accepted = [r for r in pairs if r['score']>=sweep['threshold'] and (r['guard'] or not sweep['guard'])]
        assert sum(r['label']==1 for r in accepted)==sweep['tp']
        assert sum(r['label']==0 for r in accepted)==sweep['fp']
for backend,operations in read(ROOT/'data/backend/operation-traces.json').items():
    record = read(ROOT/'reports/backend-comparison.json')[backend]
    assert float(np.median(operations['insert_ms']))==record['incremental_insert_p50_ms']
    assert float(np.median(operations['semantic_lookup_ms']))==record['semantic_p50_ms']
    if operations['exact_lookup_ms']:
        assert float(np.median(operations['exact_lookup_ms']))==record['exact_p50_ms']
adjacent_rows = read(ROOT/'data/probes/adjacent-latency.json')['rows']
adjacent = read(ROOT/'reports/adjacent-latency.json')
assert len(adjacent_rows)==adjacent['pairs']==16
assert float(np.median([r['baseline']['total_ms'] for r in adjacent_rows]))==adjacent['no_cache_p50_ms']
assert float(np.median([r['warm_hit']['total_ms'] for r in adjacent_rows]))==adjacent['warm_exact_hit_p50_ms']
assert all(r['warm_hit']['generation_calls']==0 and r['warm_hit']['hit']=='exact' and r['baseline']['generation_calls']==1 for r in adjacent_rows)
for dataset in ['hotpot','squad']:
    manifest = read(ROOT/'reports/data-validation.json')['datasets'][dataset]['partitions']
    partitions = read(ROOT/f'data/partitions/{dataset}.json')
    assert all(digest(partitions[k])==v['sha256'] for k,v in manifest.items())
for record in read(ROOT/'sources.lock.json'):
    if record['name']=='paws':
        for split,info in record['splits'].items():
            assert sha(ROOT/f'data/raw/paws-{split}.parquet')==info['sha256']
    else:
        assert sha(ROOT/record['local_file'])==record['sha256']
for alias,record in read(ROOT/'models.lock.json').items():
    for name,info in record['files'].items():
        assert sha(ROOT.parent/record['local_path']/name)==info['sha256'],(alias,name)
local_path = ROOT/'notebooks/01_complete_semantic_caching_research.ipynb'
public_path = ROOT/'notebooks/01_public_semantic_caching_research.ipynb'
local,public = [nbformat.read(path,as_version=4) for path in [local_path,public_path]]
assert [(c.cell_type,c.source) for c in local.cells]==[(c.cell_type,c.source) for c in public.cells]
assert all(not c.get('outputs') and c.get('execution_count') is None for c in public.cells)
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in local.cells if c.cell_type=='code')
verification = read(ROOT/'reports/notebook-verification.json')
assert sha(local_path)==verification['full_sha256'] and sha(public_path)==verification['public_sha256']
suites = list(ET.parse(ROOT/'reports/tests.xml').getroot().iter('testsuite'))
assert suites and all(int(s.attrib.get(k,0))==0 for s in suites for k in ['failures','errors','skipped'])
assert read(ROOT/'reports/security-freshness.json')['failures']==0
assert read(ROOT/'reports/backend-comparison.json')['qdrant-local']['entries']==512
assert read(ROOT/'reports/metric-audit.json')['verified_requests']==ledger['request_runs']
secrets = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{32,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)')
candidates = [p for p in ROOT.rglob('*') if p.is_file() and not set(p.relative_to(ROOT).parts)&{'data','.local','__pycache__','.pytest_cache'} and p!=local_path]
for path in candidates:
    assert path.stat().st_size<5*2**20,path
    assert not secrets.search(path.read_bytes()),path
source_map = {}
for path in candidates:
    if path.suffix in {'.py','.toml'} or path.parent.name=='configs' or path.name.endswith('.lock.json'):
        payload = path.read_bytes()
        normalized = payload.replace(b'\r\n',b'\n')
        import hashlib
        source_map[path.relative_to(ROOT).as_posix()] = dict(workspace_sha256=sha(path),lf_sha256=hashlib.sha256(normalized).hexdigest())
write(ROOT/'reports/source-provenance.json',dict(experiment_identity=identity(),files=source_map))
assert 'Project 4' in (ROOT.parent/'README.md').read_text(encoding='utf-8')
assert 'Measured end-to-end results' in (ROOT/'README.md').read_text(encoding='utf-8')
staged = {}
if args.staged:
    names = [n for n in git('diff','--cached','--name-only','-z').decode().split('\0') if n]
    assert names
    for name in names:
        assert name=='README.md' or name.startswith('Semantic Caching/'),name
        assert not set(Path(name).parts)&{'data','.local','__pycache__','.pytest_cache'} and '01_complete_' not in name
        blob = git('show',':'+name)
        assert len(blob)<5*2**20 and not secrets.search(blob)
        actual = git('rev-parse',':'+name).decode().strip()
        expected = git('hash-object','--path='+name,'--stdin',input=(ROOT.parent/name).read_bytes()).decode().strip()
        assert actual==expected,name
        if name!='Semantic Caching/reports/publication-audit.json':
            staged[name] = actual
    assert not any(name.startswith(('Advanced RAG/','Agentic RAG/','Retrieval Optimization/')) for name in names)
write(ROOT/'reports/publication-audit.json',dict(passed=True,preserved_previous_files=len(read(ROOT/'data/preservation.json')),trace_files=len(ledger['completed']),requests=ledger['request_runs'],public_candidates=len(candidates),largest_public_file=max(p.stat().st_size for p in candidates),secret_matches=0,third_party_review='No raw datasets, model cards, weights, caches or per-request outputs in publication candidates. Aggregate figures and ID/hash manifests only.',staged_git_hashes=staged))
print('Publication audit passed;',len(staged),'normalized staged files checked.')
