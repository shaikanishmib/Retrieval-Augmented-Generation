"""Bounded exact, semantic and evidence reuse with checks before content access."""
from collections import OrderedDict
from threading import RLock
import time
import numpy as np
from .artifacts import digest
from .security import normalize, lexical_guard

class Cache:
    def __init__(self, capacity=256, ttl=3600, threshold=1.001, store=None, backend='numpy', clock=time.time, guard=True):
        if capacity<1 or ttl<=0:
            raise ValueError('Cache capacity and TTL must be positive.')
        self.capacity,self.ttl,self.threshold = capacity,ttl,threshold
        self.store,self.backend,self.clock,self.guard = store,backend,clock,guard
        self.entries = OrderedDict()
        self.lock = RLock()
        self.evictions = 0
        self.invalidations = 0
        self.index = None
        self.next_id = 0
        self.ids = {}
        if backend not in {'numpy','faiss'}:
            raise ValueError('Choose numpy or faiss for semantic search.')
        if store:
            for entry in sorted(store.load(),key=lambda e:e['accessed']):
                self.entries[entry['key']] = entry
                self._index_add(entry)
            self._prune()

    def _index_add(self, entry):
        if self.backend!='faiss' or entry['vector'] is None:
            return
        import faiss
        vector = np.asarray([entry['vector']],dtype='float32')
        if self.index is None:
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(vector.shape[1]))
        old = self.ids.pop(entry['key'],None)
        if old is not None:
            self.index.remove_ids(np.asarray([old],dtype='int64'))
        number = self.next_id
        self.next_id += 1
        self.ids[entry['key']] = number
        self.index.add_with_ids(vector,np.asarray([number],dtype='int64'))

    def _remove(self, key):
        del self.entries[key]
        number = self.ids.pop(key,None)
        if number is not None:
            self.index.remove_ids(np.asarray([number],dtype='int64'))
        if self.store:
            self.store.delete(key)

    def _prune(self):
        now = self.clock()
        for key in list(self.entries):
            if self.entries[key]['expires']<=now:
                self._remove(key)
                self.invalidations += 1
        while len(self.entries)>self.capacity:
            self._remove(next(iter(self.entries)))
            self.evictions += 1

    def put(self, question, scope, answer, sources, corpus, vector=None, family=None, evidence=None):
        if not sources or any(s not in corpus.documents for s in sources):
            raise ValueError('Cache entries need existing evidence sources.')
        if vector is not None:
            vector = np.asarray(vector,dtype='float32')
            if vector.ndim!=1 or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector),1,atol=1e-4):
                raise ValueError('Cache vectors must be finite normalized one-dimensional arrays.')
        key = digest([normalize(question),scope])
        now = self.clock()
        entry = dict(key=key,question=normalize(question),scope=scope,answer=answer,evidence=evidence,sources=list(sources),stamps={s:corpus.documents[s].stamp() for s in sources},vector=None if vector is None else vector.tolist(),family=family,created=now,accessed=now,expires=now+self.ttl)
        with self.lock:
            if self.store:
                self.store.put(entry)
            self.entries[key] = entry
            self.entries.move_to_end(key)
            self._index_add(entry)
            self._prune()

    def lookup(self, question, scope, principal, corpus, vector=None, semantic=False, threshold=None):
        with self.lock:
            self._prune()
            key = digest([normalize(question),scope])
            candidate = self.entries.get(key)
            kind,score = 'exact',1.0
            if candidate is None and semantic and vector is not None:
                candidates = [e for e in self.entries.values() if e['scope']==scope and e['vector'] is not None]
                if candidates:
                    if self.backend=='faiss':
                        scores,ids = self.index.search(np.asarray([vector],dtype='float32'),self.index.ntotal)
                        ranking = {int(i):float(s) for s,i in zip(scores[0],ids[0])}
                        scored = [(ranking[self.ids[e['key']]],e) for e in candidates]
                    else:
                        scored = [(float(np.dot(vector,e['vector'])),e) for e in candidates]
                    score,candidate = max(scored,key=lambda pair:pair[0])
                    if score<(self.threshold if threshold is None else threshold) or (self.guard and not lexical_guard(question,candidate['question'])):
                        candidate = None
                    kind = 'semantic'
            if candidate is None:
                return None
            # A matching scope never substitutes for current source checks.
            if not corpus.authorized(principal,candidate['sources']) or any(corpus.documents[s].stamp()!=stamp for s,stamp in candidate['stamps'].items()):
                self._remove(candidate['key'])
                self.invalidations += 1
                return None
            candidate['accessed'] = self.clock()
            if self.store:
                self.store.put(candidate)
            self.entries.move_to_end(candidate['key'])
            return dict(candidate,hit_kind=kind,similarity=score)
