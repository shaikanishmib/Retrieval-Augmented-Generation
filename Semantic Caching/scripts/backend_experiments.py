"""Measure local backend operations, restart, eviction and controlled safety outcomes."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import os
import time
import sqlite3
import numpy as np
import psutil
from semantic_cache.artifacts import ROOT,write,read
from semantic_cache.cache import Cache
from semantic_cache.security import Corpus,Document,Principal,scope
from semantic_cache.storage import SQLiteStore

folder = ROOT/'data/backend'
folder.mkdir(parents=True,exist_ok=True)
rng = np.random.default_rng(20261102)
vectors = rng.normal(size=(512,384)).astype('float32')
vectors /= np.linalg.norm(vectors,axis=1,keepdims=True)
corpus = Corpus([Document('d','Controlled evidence.',tenant='a')])
p = Principal('a','u',frozenset({'d'}))
sc = scope(p,corpus,'fixture','v1')
results = {}
operation_traces = {}
key_path = folder/'signing.key'
if not key_path.exists():
    key_path.write_bytes(os.urandom(32))
key = key_path.read_bytes()
stamp = str(time.time_ns())
for backend in ['numpy','faiss','sqlite-numpy','sqlite-faiss']:
    persistent = backend.startswith('sqlite')
    path = folder/f'{backend}-{stamp}.db'
    initialization_start = time.perf_counter()
    store = SQLiteStore(path,key) if persistent else None
    cache = Cache(capacity=512,threshold=.9,backend='faiss' if 'faiss' in backend else 'numpy',store=store)
    initialization_ms = (time.perf_counter()-initialization_start)*1000
    rss = psutil.Process().memory_info().rss
    insert,search,exact = [],[],[]
    for i,vector in enumerate(vectors):
        tick = time.perf_counter()
        cache.put(str(i),sc,'fixture-answer',['d'],corpus,vector,str(i))
        insert.append((time.perf_counter()-tick)*1000)
    for i in range(100):
        tick = time.perf_counter()
        hit = cache.lookup(str(i),sc,p,corpus)
        exact.append((time.perf_counter()-tick)*1000)
        assert hit['answer']=='fixture-answer'
        tick = time.perf_counter()
        hit = cache.lookup('variant '+str(i),sc,p,corpus,vectors[i],True)
        search.append((time.perf_counter()-tick)*1000)
        assert hit['family']==str(i)
    record = dict(entries=512,dimensions=384,incremental_insert_p50_ms=float(np.median(insert)),incremental_insert_p95_ms=float(np.percentile(insert,95)),exact_p50_ms=float(np.median(exact)),semantic_p50_ms=float(np.median(search)),semantic_p95_ms=float(np.percentile(search,95)),rss_delta=psutil.Process().memory_info().rss-rss,serialized_entry_bytes=sum(len(__import__('json').dumps(e).encode()) for e in cache.entries.values()),index_bytes=len(__import__('faiss').serialize_index(cache.index)) if cache.index else 0)
    if store:
        record['sqlite_version'] = sqlite3.sqlite_version
        store.close()
        tick = time.perf_counter()
        store = SQLiteStore(path,key)
        restored = Cache(store=store,capacity=512,backend='faiss' if 'faiss' in backend else 'numpy')
        record['restart_ms'] = (time.perf_counter()-tick)*1000
        assert restored.lookup('0',sc,p,corpus)['answer']=='fixture-answer'
        store.close()
        record['disk_bytes'] = path.stat().st_size
    results[backend] = record
    record.update(initialization_ms=initialization_ms,first_insert_ms=insert[0],total_insert_ms=sum(insert),index_initialization='FAISS is lazy on first insertion, including first library import when not already loaded.')
    operation_traces[backend] = dict(insert_ms=insert,semantic_lookup_ms=search,exact_lookup_ms=exact)
    print(backend,record,flush=True)

# Local Qdrant persists actual incremental points; compare to the same synthetic vectors.
try:
    from qdrant_client import QdrantClient,models
    path = folder/f'qdrant-{stamp}'
    initialization_start = time.perf_counter()
    client = QdrantClient(path=str(path))
    initialization_ms = (time.perf_counter()-initialization_start)*1000
    tick = time.perf_counter()
    client.create_collection('cache',vectors_config=models.VectorParams(size=384,distance=models.Distance.COSINE))
    construction = (time.perf_counter()-tick)*1000
    timings = []
    for i,v in enumerate(vectors):
        tick = time.perf_counter()
        client.upsert('cache',[models.PointStruct(id=i,vector=v.tolist(),payload={'scope':sc})],wait=True)
        timings.append((time.perf_counter()-tick)*1000)
    search = []
    for i in range(100):
        tick = time.perf_counter()
        hits = client.query_points('cache',query=vectors[i].tolist(),limit=1,query_filter=models.Filter(must=[models.FieldCondition(key='scope',match=models.MatchValue(value=sc))])).points
        search.append((time.perf_counter()-tick)*1000)
        assert hits[0].id==i
    client.close()
    tick = time.perf_counter()
    client = QdrantClient(path=str(path))
    assert client.count('cache').count==512
    restart = (time.perf_counter()-tick)*1000
    client.close()
    results['qdrant-local'] = dict(entries=512,dimensions=384,construction_ms=construction,incremental_insert_p50_ms=float(np.median(timings)),incremental_insert_p95_ms=float(np.percentile(timings,95)),semantic_p50_ms=float(np.median(search)),semantic_p95_ms=float(np.percentile(search,95)),restart_ms=restart,disk_bytes=sum(f.stat().st_size for f in path.rglob('*') if f.is_file()))
    results['qdrant-local'].update(initialization_ms=initialization_ms,first_insert_ms=timings[0],total_insert_ms=sum(timings))
    operation_traces['qdrant-local'] = dict(insert_ms=timings,semantic_lookup_ms=search,exact_lookup_ms=[])
except Exception as error:
    results['qdrant-local'] = dict(status='failed',error=repr(error))
write(ROOT/'reports/backend-comparison.json',results)
write(ROOT/'data/backend/operation-traces.json',operation_traces)

eviction = []
for capacity in [8,32,128,512]:
    for policy in ['lru','ttl']:
        clock = [0.]
        cache = Cache(capacity=capacity,ttl=16 if policy=='ttl' else 10000,clock=lambda:clock[0])
        hits = 0
        sequence = list(range(64))*2+list(range(64,128))+list(range(64))
        for i in sequence:
            hit = cache.lookup(str(i),sc,p,corpus)
            hits += bool(hit)
            if not hit:
                cache.put(str(i),sc,'fixture',['d'],corpus)
            clock[0] += 1
        eviction.append(dict(capacity=capacity,policy=policy,requests=len(sequence),hits=hits,evictions=cache.evictions,expirations=cache.invalidations))
write(ROOT/'reports/eviction.json',eviction)

probes = []
for scenario in ['update','delete','revocation','permission_epoch','cross_tenant','cross_user','model','prompt','unknown_permission','expired','new_document']:
    current = Corpus([Document('d','Secret fixture.',tenant='a')])
    principal = Principal('a','u',frozenset({'d'}))
    original_scope = scope(principal,current,'m','p')
    now = [0.]
    cache = Cache(ttl=10,clock=lambda:now[0])
    cache.put('q',original_scope,'secret',['d'],current)
    model,prompt = 'm','p'
    if scenario=='update':
        current.documents['d'] = replace(current.documents['d'],text='Updated secret',version='2')
    elif scenario=='delete':
        del current.documents['d']
    elif scenario=='revocation':
        current.documents['d'] = replace(current.documents['d'],visible=False)
    elif scenario=='permission_epoch':
        principal = replace(principal,epoch=1)
    elif scenario=='cross_tenant':
        principal = replace(principal,tenant='b')
    elif scenario=='cross_user':
        principal = replace(principal,user='other')
    elif scenario=='model':
        model = 'm2'
    elif scenario=='prompt':
        prompt = 'p2'
    elif scenario=='unknown_permission':
        principal = replace(principal,verified=False)
    elif scenario=='expired':
        now[0] = 11.
    elif scenario=='new_document':
        current.documents['new'] = Document('new','New competing evidence')
    try:
        hit = cache.lookup('q',scope(principal,current,model,prompt),principal,current)
        returned = hit is not None
    except PermissionError:
        returned = False
    assert not returned,scenario
    probes.append(dict(scenario=scenario,stale_or_unauthorized_return=returned,correctly_rejected=True))
write(ROOT/'reports/security-freshness.json',dict(probes=probes,tested_requests=len(probes),failures=0,scope='Controlled synthetic identities and documents; no production penetration test.'))
