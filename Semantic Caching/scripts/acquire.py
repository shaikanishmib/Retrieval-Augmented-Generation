"""Verify reused assets and acquire the small PAWS release without publishing raw text."""
import argparse
import shutil
import urllib.request
from semantic_cache.artifacts import ROOT, read, write, sha

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--verify-only', action='store_true')
args = parser.parse_args()
raw = ROOT/'data/raw'
raw.mkdir(parents=True, exist_ok=True)
sources = []
prior_sources = read(ROOT/'sources.lock.json') if (ROOT/'sources.lock.json').exists() else read(ROOT.parent/'Agentic RAG/sources.lock.json')
for name, filename in [('hotpot','hotpot.json'), ('squad','squad-train.json')]:
    record = next(r.copy() for r in prior_sources if r['name']==name)
    target = raw/filename
    if not target.exists():
        if args.verify_only:
            raise FileNotFoundError(target)
        existing = ROOT.parent/'Agentic RAG/data/raw'/filename
        if existing.exists() and sha(existing)==record['sha256']:
            shutil.copyfile(existing,target)
        elif record['url'].endswith('.parquet'):
            import json
            import pyarrow.parquet as pq
            parquet = target.with_suffix('.parquet')
            urllib.request.urlretrieve(record['url'],parquet)
            assert sha(parquet)==record['download_sha256']
            converted = []
            for row in pq.read_table(parquet).to_pylist():
                row['_id'] = row.pop('id')
                row['context'] = list(zip(row['context']['title'],row['context']['sentences']))
                row['supporting_facts'] = list(zip(row['supporting_facts']['title'],row['supporting_facts']['sent_id']))
                converted.append(row)
            target.write_text(json.dumps(converted,ensure_ascii=False),encoding='utf-8')
        else:
            urllib.request.urlretrieve(record['url'],target)
    assert sha(target)==record['sha256'], target
    record.update(local_file=f'data/raw/{filename}', source_page='https://hotpotqa.github.io/' if name=='hotpot' else 'https://rajpurkar.github.io/SQuAD-explorer/')
    sources.append(record)
revision = '161ece9501cf0a11f3e48bd356eaa82de46d6a09'
sources.append(dict(name='paws',revision=revision,version='PAWS-Wiki Labeled Final 2019',source_page='https://github.com/google-research-datasets/paws',license='Google PAWS dataset license: freely usable for any purpose; supplied AS IS',license_url='https://github.com/google-research-datasets/paws/blob/master/LICENSE',checksum_origin='Measured mirror bytes, not an independent upstream checksum',splits={}))
import pyarrow.parquet as pq
for split,remote in [('dev','validation'),('test','test')]:
    url = f'https://huggingface.co/datasets/google-research-datasets/paws/resolve/{revision}/labeled_final/{remote}-00000-of-00001.parquet'
    target = raw/f'paws-{split}.parquet'
    if not target.exists():
        if args.verify_only:
            raise FileNotFoundError(target)
        urllib.request.urlretrieve(url,target)
    sources[-1]['splits'][split] = dict(url=url,sha256=sha(target),bytes=target.stat().st_size,rows=pq.read_metadata(target).num_rows)
sources[-1]['bytes'] = sum(v['bytes'] for v in sources[-1]['splits'].values())
models = read(ROOT/'models.lock.json')
for alias, record in models.items():
    path = ROOT.parent/record['local_path']
    if not all((path/name).is_file() for name in record['files']):
        path = ROOT/'data/models'/alias
        if not args.verify_only:
            path.mkdir(parents=True,exist_ok=True)
        for name,info in record['files'].items():
            target = path/name
            if not target.exists():
                if args.verify_only:
                    raise FileNotFoundError(target)
                target.parent.mkdir(parents=True,exist_ok=True)
                urllib.request.urlretrieve(f"https://huggingface.co/{record['id']}/resolve/{record['revision']}/{name}",target)
        record['local_path'] = path.relative_to(ROOT.parent).as_posix()
    for file, info in record['files'].items():
        assert sha(path/file)==info['sha256'], (alias,file)
    record['bytes'] = sum(v['bytes'] for v in record['files'].values())
if args.verify_only:
    assert read(ROOT/'sources.lock.json')==sources
    assert read(ROOT/'models.lock.json')==models
else:
    write(ROOT/'sources.lock.json', sources)
    write(ROOT/'models.lock.json', models)
print({r['name']:r['bytes'] for r in sources})
