"""Authentic QA records and explicitly labeled synthetic request variants."""
import re
import pyarrow.parquet as pq
from .artifacts import ROOT,read,write,digest

def previous_ids():
    ids = set()
    def visit(obj):
        if isinstance(obj,dict):
            for k,v in obj.items():
                if k in {'id','question_id','_id'} and isinstance(v,str):
                    ids.add(v)
                elif k=='ids' and isinstance(v,list):
                    ids.update(x for x in v if isinstance(x,str))
                else:
                    visit(v)
        elif isinstance(obj,list):
            for item in obj:
                visit(item)
    for project,patterns in [('Agentic RAG',['data/partitions/*.json']),('Advanced RAG',['data/upgrade/manifests/*.json','data/experiments/*traces.json'])]:
        for pattern in patterns:
            for path in (ROOT.parent/project).glob(pattern):
                visit(read(path))
    return ids

def load_qa(name):
    raw = read(ROOT/f'data/raw/{"hotpot.json" if name=="hotpot" else "squad-train.json"}')
    rows = []
    if name=='hotpot':
        for r in raw:
            docs = {title:''.join(sentences) for title,sentences in r['context']}
            if any(title not in docs or not 0<=index<len(dict(r['context'])[title]) for title,index in r['supporting_facts']):
                continue
            rows.append(dict(id=r['_id'],group=r['_id'],question=r['question'],documents=docs,answers=[r['answer']],kind='multihop'))
    else:
        for article in raw['data']:
            docs = {f"{article['title']}:{i}":p['context'] for i,p in enumerate(article['paragraphs'])}
            for p in article['paragraphs']:
                for q in p['qas']:
                    assert all(p['context'][a['answer_start']:a['answer_start']+len(a['text'])]==a['text'] for a in q['answers'])
                    rows.append(dict(id=q['id'],group=article['title'],question=q['question'],documents=docs,answers=[a['text'] for a in q['answers']] or [''],kind='unanswerable' if q['is_impossible'] else 'answerable'))
    return rows

def prepare():
    cfg = read(ROOT/'configs/study.json')
    excluded = previous_ids()
    report = {'prior_ids':len(excluded),'datasets':{}}
    for name in ['hotpot','squad']:
        rows = load_qa(name)
        used_groups = {r['group'] for r in rows if r['id'] in excluded}
        available = sorted([r for r in rows if r['id'] not in excluded and (name!='squad' or r['group'] not in used_groups)],key=lambda r:digest(['semantic-v1',r['id']]))
        selected, used_docs,used_questions = {},set(),set()
        for split in ['development','evaluation']:
            target = cfg[split+'_per_dataset']
            chosen = []
            # Select disjoint source pools. Shared-document rows within a split remain valid.
            split_docs = set()
            for row in available:
                if row['question'].strip().casefold() in used_questions or set(row['documents']) & used_docs:
                    continue
                chosen.append(row)
                split_docs.update(row['documents'])
                used_questions.add(row['question'].strip().casefold())
                if len(chosen)==target:
                    break
            assert len(chosen)==target,(name,split,len(chosen))
            selected[split] = chosen
            used_docs.update(split_docs)
        assert not set().union(*(set(r['documents']) for r in selected['development'])) & set().union(*(set(r['documents']) for r in selected['evaluation']))
        write(ROOT/f'data/partitions/{name}.json', selected)
        report['datasets'][name] = dict(valid_records=len(rows),prior_question_exclusions=sum(r['id'] in excluded for r in rows),prior_article_exclusions=len(used_groups) if name=='squad' else None,partitions={s:dict(count=len(rs),ids=[r['id'] for r in rs],groups=len({r['group'] for r in rs}),unanswerable=sum(r['answers']==[''] for r in rs),sha256=digest(rs)) for s,rs in selected.items()})
    paws = {}
    prior_sentences = set()
    for split,raw_split in [('development','dev'),('evaluation','test')]:
        rows = pq.read_table(ROOT/f'data/raw/paws-{raw_split}.parquet').to_pylist()
        chosen = []
        for row in sorted(rows,key=lambda r:digest(['semantic-paws-v1',r['id']])):
            texts = {row['sentence1'].strip().casefold(),row['sentence2'].strip().casefold()}
            if texts & prior_sentences:
                continue
            chosen.append(row)
            if len(chosen)==cfg['paws_'+split]:
                break
        assert len(chosen)==cfg['paws_'+split]
        paws[split] = chosen
        prior_sentences.update(t.strip().casefold() for r in rows for t in [r['sentence1'],r['sentence2']])
        report['datasets']['paws-'+split] = dict(count=len(chosen),positives=sum(r['label'] for r in chosen),ids=[r['id'] for r in chosen],sha256=digest(chosen),raw_split=raw_split)
    write(ROOT/'data/partitions/paws.json',paws)
    write(ROOT/'reports/data-validation.json',report)
    return report

def workload(rows, mode='mixed'):
    requests = []
    for row in rows:
        requests.append(dict(row=row,question=row['question'],family=row['id'],kind='unique',equivalent=True))
    if mode=='unique':
        return requests
    # First all misses, then repeats: expose capacity effects instead of adjacent-pair-only timing.
    for i,row in enumerate(rows):
        if mode=='unique-heavy' and i%8:
            continue
        requests.append(dict(row=row,question=row['question'],family=row['id'],kind='exact_repeat',equivalent=True))
        requests.append(dict(row=row,question='Please answer this question: '+row['question'],family=row['id'],kind='synthetic_wrapper',equivalent=True))
    return requests

def stress_pairs():
    return [
        ('What is the capital of France?','What is the capital of Germany?','entity'),
        ('Who won the race in 2020?','Who won the race in 2021?','date'),
        ('Which drug is approved?','Which drug is not approved?','negation'),
        ('How many residents are over 18?','How many residents are over 80?','number'),
        ('Did Alice defeat Bob?','Did Bob defeat Alice?','role_reversal'),
        ('Who is the current director?','Who was the previous director?','temporal')]
