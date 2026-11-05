from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import sqlite3
import numpy as np
import pytest
from semantic_cache.cache import Cache
from semantic_cache.security import Corpus,Document,Principal,scope,normalize,lexical_guard
from semantic_cache.storage import SQLiteStore
from semantic_cache.metrics import qa_score,aggregate,paired_interval
from semantic_cache.artifacts import digest
from semantic_cache.rag import Pipeline

@pytest.fixture
def fixture():
    corpus = Corpus([Document('d','Paris is the capital of France.',tenant='a')])
    principal = Principal('a','u',frozenset({'d'}))
    return corpus,principal,scope(principal,corpus,'m','p')

def insert(cache, fixture, question='q',vector=None):
    corpus,principal,key = fixture
    cache.put(question,key,'Paris',['d'],corpus,vector,'family')
    return corpus,principal,key

def test_exact(fixture):
    cache = Cache()
    corpus,p,key = insert(cache,fixture)
    assert cache.lookup(' q  ',key,p,corpus)['answer']=='Paris'
    assert cache.lookup('Q',key,p,corpus) is None

@pytest.mark.parametrize('a,b',[('2020','2021'),('is approved','is not approved'),('France','Germany'),('May 1','May 2')])
def test_normalization(a,b):
    assert normalize(a)!=normalize(b)

@pytest.mark.parametrize('backend',['numpy','faiss'])
def test_threshold(fixture,backend):
    cache = Cache(threshold=.95,backend=backend)
    corpus,p,key = insert(cache,fixture,vector=[1.,0.])
    assert cache.lookup('other',key,p,corpus,np.array([0.,1.]),True) is None
    assert cache.lookup('other',key,p,corpus,np.array([1.,0.]),True)['hit_kind']=='semantic'

def test_negation_and_number_guard(fixture):
    cache = Cache(threshold=.5)
    corpus,p,key = insert(cache,fixture,question='won in 2020',vector=[1.,0.])
    assert cache.lookup('won in 2021',key,p,corpus,np.array([1.,0.]),True) is None
    assert not lexical_guard('is valid','is not valid')
    assert lexical_guard('Alice beat Bob','Bob beat Alice')  # Retain this known weakness.

@pytest.mark.parametrize('change',['tenant','user','allowed','epoch','verified'])
def test_scope_boundaries(fixture,change):
    corpus,p,key = fixture
    values = {'tenant':'b','user':'v','allowed':frozenset(),'epoch':1,'verified':False}
    changed = replace(p,**{change:values[change]})
    cache = Cache()
    insert(cache,fixture)
    if change=='verified':
        with pytest.raises(PermissionError):
            scope(changed,corpus,'m','p')
    else:
        assert cache.lookup('q',scope(changed,corpus,'m','p'),changed,corpus) is None

@pytest.mark.parametrize('change',['delete','revoke','update','content','tenant'])
def test_current_source_check_even_with_old_scope(fixture,change):
    cache = Cache()
    corpus,p,key = insert(cache,fixture)
    if change=='delete':
        del corpus.documents['d']
    else:
        corpus.documents['d'] = replace(corpus.documents['d'],**{'revoke':{'visible':False},'update':{'version':'2'},'content':{'text':'Lyon'},'tenant':{'tenant':'b'}}[change])
    assert cache.lookup('q',key,p,corpus) is None

def test_model_prompt_and_new_evidence(fixture):
    corpus,p,key = fixture
    assert scope(p,corpus,'m2','p')!=key
    assert scope(p,corpus,'m','p2')!=key
    corpus.documents['new'] = Document('new','Competing evidence')
    assert scope(p,corpus,'m','p')!=key

def test_expiration(fixture):
    now = [10.]
    cache = Cache(ttl=1,clock=lambda:now[0])
    corpus,p,key = insert(cache,fixture)
    now[0] = 11.
    assert cache.lookup('q',key,p,corpus) is None

def test_lru(fixture):
    cache = Cache(capacity=2)
    corpus,p,key = insert(cache,fixture,'one')
    insert(cache,fixture,'two')
    assert cache.lookup('one',key,p,corpus)
    insert(cache,fixture,'three')
    assert cache.lookup('two',key,p,corpus) is None
    assert cache.evictions==1

@pytest.mark.parametrize('backend',['numpy','faiss'])
def test_concurrent_incremental_eviction(fixture,backend):
    cache = Cache(capacity=16,backend=backend)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i:insert(cache,fixture,str(i),[1.,0.]),range(100)))
    assert len(cache.entries)==16
    assert cache.evictions==84
    if backend=='faiss':
        assert cache.index.ntotal==16

def test_persistence_integrity(fixture,tmp_path):
    key = b'x'*32
    path = tmp_path/'cache.db'
    store = SQLiteStore(path,key)
    cache = Cache(store=store)
    corpus,p,sc = insert(cache,fixture,vector=[1.,0.])
    store.close()
    store = SQLiteStore(path,key)
    restored = Cache(store=store)
    assert restored.lookup('q',sc,p,corpus)['answer']=='Paris'
    store.close()
    with sqlite3.connect(path) as db:
        db.execute("UPDATE entries SET payload=replace(payload,'Paris','poison')")
    store = SQLiteStore(path,key)
    with pytest.raises(ValueError,match='integrity'):
        Cache(store=store)
    store.close()

def test_wrong_signing_key(fixture,tmp_path):
    path = tmp_path/'cache.db'
    store = SQLiteStore(path,b'x'*32)
    insert(Cache(store=store),fixture)
    store.close()
    store = SQLiteStore(path,b'y'*32)
    with pytest.raises(ValueError):
        store.load()
    store.close()

def test_trusted_writer_poisoning_is_outside_integrity_boundary(fixture):
    corpus,p,key = fixture
    cache = Cache()
    cache.put('q',key,'incorrect trusted answer',['d'],corpus)
    assert cache.lookup('q',key,p,corpus)['answer']=='incorrect trusted answer'

@pytest.mark.parametrize('vector',[[0.,0.],[float('nan'),1.],[[1.,0.]]])
def test_invalid_vectors(fixture,vector):
    with pytest.raises(ValueError):
        insert(Cache(),fixture,vector=vector)

def test_metrics_and_identity():
    assert qa_score('The Paris',['Paris'])==(1.,1.)
    assert qa_score('',[''])==(1.,1.)
    assert qa_score('Paris France',['Paris'])==(0.,2/3)
    assert digest({'a':1,'b':2})==digest({'b':2,'a':1})
    assert digest([1,2])!=digest([2,1])

class FakeGenerator:
    alias = 'fixture'
    def generate(self,*args):
        return dict(answer='Paris',generation_ms=.1,gpu_generation_ms=0.,input_tokens=1,output_tokens=1,prompt='fixture')

class FakeEncoder:
    def encode(self,texts):
        return np.array([[1.,0.] for _ in texts]),.01

@pytest.mark.parametrize('strategy',['none','exact','semantic','evidence','hybrid'])
def test_pipeline(fixture,strategy):
    corpus,p,_ = fixture
    cfg = dict(prompt_version='p',retrieval_k=1,max_input_tokens=480,max_new_tokens=48)
    pipeline = Pipeline(corpus,FakeGenerator(),FakeEncoder(),cfg)
    cache = Cache(threshold=.8)
    a = pipeline.request('capital?',p,strategy,cache,'f')
    b = pipeline.request('capital?',p,strategy,cache,'f')
    assert a['generation_calls']==1 and a['answer']=='Paris'
    assert b['generation_calls']==int(strategy in {'none','evidence'})
    assert b['answer']=='Paris'
    if strategy in {'semantic','evidence','hybrid'}:
        c = pipeline.request('Please answer: capital?',p,strategy,cache,'f')
        assert c['hit']=='semantic'
        assert c['answer']=='Paris'

def test_tiered_evidence_fallback(fixture):
    corpus,p,_ = fixture
    cfg = dict(prompt_version='p',retrieval_k=1,max_input_tokens=480,max_new_tokens=48,evidence_threshold=.95)
    pipeline = Pipeline(corpus,FakeGenerator(),FakeEncoder(),cfg)
    cache = Cache(threshold=1.001)
    pipeline.request('capital?',p,'hybrid',cache,'f')
    result = pipeline.request('Please answer: capital?',p,'hybrid',cache,'f')
    assert result['hit']=='semantic' and result['reuse_stage']=='evidence'
    assert result['generation_calls']==1 and result['answer']=='Paris'

def test_actual_model_revision_changes_scope(fixture):
    corpus,p,_ = fixture
    generator,encoder = FakeGenerator(),FakeEncoder()
    generator.identity = 'same-alias:revision-one'
    encoder.identity = 'embedding:revision-one'
    cfg = dict(prompt_version='p',retrieval_k=1,max_input_tokens=480,max_new_tokens=48)
    pipeline = Pipeline(corpus,generator,encoder,cfg)
    cache = Cache()
    pipeline.request('capital?',p,'exact',cache,'f')
    generator.identity = 'same-alias:revision-two'
    assert pipeline.request('capital?',p,'exact',cache,'f')['hit']=='miss'
    encoder.identity = 'embedding:revision-two'
    assert pipeline.request('capital?',p,'exact',cache,'f')['hit']=='miss'
