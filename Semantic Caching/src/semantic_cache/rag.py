"""BM25 passage retrieval and measured cache workflows."""
from collections import Counter
import math
import re
import time
from .security import scope

def tokens(text):
    return re.findall(r'\w+',text.casefold())

class BM25:
    def __init__(self, corpus):
        self.passages = []
        for document in corpus.documents.values():
            words = document.text.split()
            for offset in range(0,len(words),96):
                self.passages.append((document.id,' '.join(words[offset:offset+128])))
        self.counts = [Counter(tokens(text)) for _,text in self.passages]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.avg = sum(self.lengths)/max(1,len(self.lengths))
        frequencies = Counter(t for c in self.counts for t in c)
        n = len(self.counts)
        self.idf = {t:math.log(1+(n-df+.5)/(df+.5)) for t,df in frequencies.items()}

    def search(self, question, principal, corpus, k=3):
        query = set(tokens(question))
        scored = []
        for i,counts in enumerate(self.counts):
            sid = self.passages[i][0]
            if not corpus.authorized(principal,[sid]):
                continue
            score = sum(self.idf.get(t,0)*counts[t]*2.5/(counts[t]+1.5*(.25+.75*self.lengths[i]/max(self.avg,1))) for t in query if counts[t])
            scored.append((score,i))
        return [self.passages[i] for _,i in sorted(scored,key=lambda v:(-v[0],v[1]))[:k]]

class Pipeline:
    def __init__(self, corpus, generator, encoder, cfg):
        self.corpus,self.generator,self.encoder,self.cfg = corpus,generator,encoder,cfg
        start = time.perf_counter()
        self.retriever = BM25(corpus)
        self.index_ms = (time.perf_counter()-start)*1000

    def request(self, question, principal, strategy='none', cache=None, family=None):
        start = time.perf_counter()
        cfg = self.cfg
        model_identity = [getattr(self.generator,'identity',self.generator.alias),getattr(self.encoder,'identity',None)]
        request_scope = scope(principal,self.corpus,model_identity,cfg['prompt_version'])
        embedding_ms,lookup_ms,retrieval_ms,insertion_ms = 0.,0.,0.,0.
        vector,hit = None,None
        evidence_reuse = False
        semantic = strategy in {'semantic','evidence','hybrid'}
        if cache and strategy!='none':
            tick = time.perf_counter()
            hit = cache.lookup(question,request_scope,principal,self.corpus)
            lookup_ms += (time.perf_counter()-tick)*1000
            if hit is None and semantic:
                vectors,embedding_ms = self.encoder.encode([question])
                vector = vectors[0]
                tick = time.perf_counter()
                hit = cache.lookup(question,request_scope,principal,self.corpus,vector,True)
                lookup_ms += (time.perf_counter()-tick)*1000
                if hit is None and strategy=='hybrid':
                    tick = time.perf_counter()
                    hit = cache.lookup(question,request_scope,principal,self.corpus,vector,True,threshold=cfg.get('evidence_threshold',.95))
                    evidence_reuse = hit is not None
                    lookup_ms += (time.perf_counter()-tick)*1000
        answer_reuse = hit is not None and strategy!='evidence' and not evidence_reuse
        generation = dict(answer=hit['answer'] if answer_reuse else '',generation_ms=0.,gpu_generation_ms=0.,input_tokens=0,output_tokens=0,prompt=None)
        if answer_reuse:
            sources = hit['sources']
        else:
            tick = time.perf_counter()
            if hit:
                # Store passage text in the entry answer payload for evidence caches.
                passages = hit['evidence'] if evidence_reuse else hit['answer']
            else:
                passages = self.retriever.search(question,principal,self.corpus,cfg['retrieval_k'])
            retrieval_ms = (time.perf_counter()-tick)*1000
            sources = sorted({sid for sid,_ in passages})
            if not sources:
                raise PermissionError('No authorized evidence is available.')
            generation = self.generator.generate(question,[f'[{sid}] {text}' for sid,text in passages],cfg['max_input_tokens'],cfg['max_new_tokens'])
        if cache and strategy!='none' and hit is None:
            tick = time.perf_counter()
            cache.put(question,request_scope,passages if strategy=='evidence' else generation['answer'],sources,self.corpus,vector,family,evidence=passages)
            insertion_ms = (time.perf_counter()-tick)*1000
        return dict(**generation,total_ms=(time.perf_counter()-start)*1000,embedding_ms=embedding_ms,lookup_ms=lookup_ms,retrieval_ms=retrieval_ms,insertion_ms=insertion_ms,generation_calls=int(not answer_reuse),reuse_stage='answer' if answer_reuse else 'evidence' if hit else 'none',hit=hit['hit_kind'] if hit else 'miss',cached_family=hit['family'] if hit else None,similarity=hit['similarity'] if hit else None,sources=sources,source_versions={s:self.corpus.documents[s].stamp() for s in sources},scope=request_scope,tenant=principal.tenant,user=principal.user,permission_epoch=principal.epoch)
