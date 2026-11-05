"""Map original spans to response labels and partition complete source groups."""
from collections import Counter, defaultdict
import json
import random
from llm_judge.artifacts import ROOT, digest, read, write

def hallucination_label(spans):
    # implicit_true remains positive: factual truth does not establish context support.
    return int(bool(spans))

def evidence(source):
    value = source['source_info']
    if isinstance(value, str):
        return value, ''
    if source['task_type'] == 'QA':
        return value['passages'], value['question']
    return json.dumps(value, ensure_ascii=False, sort_keys=True), ''

def load_raw():
    base = ROOT / 'data/raw/ragtruth/dataset'
    responses = [json.loads(x) for x in (base / 'response.jsonl').read_text(encoding='utf-8').splitlines()]
    sources = [json.loads(x) for x in (base / 'source_info.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len({x['id'] for x in responses}) == len(responses)
    assert len({x['source_id'] for x in sources}) == len(sources)
    lookup = {x['source_id']: x for x in sources}
    rows = []
    for r in responses:
        s = lookup[r['source_id']]
        context, question = evidence(s)
        assert context and r['response'] and r['split'] in ('train', 'test')
        for span in r['labels']:
            assert 0 <= span['start'] < span['end'] <= len(r['response'])
        rows.append({**r, 'context': context, 'question': question, 'task': s['task_type'], 'y': hallucination_label(r['labels']), 'evidence_hash': digest(context)})
    return rows

def partition(rows, config):
    group_splits = defaultdict(set)
    hashes = defaultdict(set)
    for r in rows:
        group_splits[r['source_id']].add(r['split'])
        hashes[r['evidence_hash']].add(r['source_id'])
    assert all(len(v) == 1 for v in group_splits.values())
    # Exclude all exact duplicate evidence groups before sampling, including across official splits.
    duplicates = {sid for ids in hashes.values() if len(ids) > 1 for sid in ids}
    rng = random.Random(config['seed'])
    selected = {key: set() for key in ['development', 'validation', 'test']}
    for task in sorted({r['task'] for r in rows}):
        train = sorted({r['source_id'] for r in rows if r['task'] == task and r['split'] == 'train'} - duplicates)
        test = sorted({r['source_id'] for r in rows if r['task'] == task and r['split'] == 'test'} - duplicates)
        rng.shuffle(train)
        rng.shuffle(test)
        nd, nv, nt = [config[f'{key}_groups_per_task'] for key in ['development', 'validation', 'test']]
        selected['development'].update(train[:nd])
        selected['validation'].update(train[nd:nd + nv])
        selected['test'].update(test[:nt])
    result = {key: [r for r in rows if r['source_id'] in ids] for key, ids in selected.items()}
    assert_disjoint(result)
    return result, duplicates

def assert_disjoint(partitions):
    seen_ids, seen_evidence = set(), set()
    for rows in partitions.values():
        ids = {r['source_id'] for r in rows}
        hashes = {r['evidence_hash'] for r in rows}
        assert not seen_ids & ids
        assert not seen_evidence & hashes
        seen_ids.update(ids)
        seen_evidence.update(hashes)

def prepare():
    rows = load_raw()
    parts, duplicates = partition(rows, read(ROOT / 'configs/study.json'))
    for name, values in parts.items():
        write(ROOT / f'data/partitions/{name}.json', values)
    audit = {'raw_responses': len(rows), 'raw_sources': len({r['source_id'] for r in rows}), 'official_splits': dict(Counter(r['split'] for r in rows)), 'original_label_types': dict(Counter(s['label_type'] for r in rows for s in r['labels'])), 'implicit_true_spans': sum(bool(s.get('implicit_true')) for r in rows for s in r['labels']), 'duplicate_evidence_groups_excluded': len(duplicates), 'missing_context_or_response': 0, 'duplicate_response_texts': len(rows) - len({r['response'] for r in rows}), 'partitions': {k: {'responses': len(v), 'sources': len({r['source_id'] for r in v}), 'positive_labels': sum(r['y'] for r in v), 'tasks': dict(Counter(r['task'] for r in v)), 'identity': digest(v)} for k, v in parts.items()}, 'positive_class': 'Any original annotated hallucination span, including implicit_true; 1=hallucination', 'scope': 'Response detection; sentence overlap diagnostics are not span localization or independently annotated atomic claims.'}
    write(ROOT / 'reports/data-validation.json', audit)
    print(audit)
