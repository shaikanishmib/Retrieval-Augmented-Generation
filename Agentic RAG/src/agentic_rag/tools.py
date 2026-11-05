"""Candidate-passage search and deterministic title-mention links with provenance."""
from dataclasses import dataclass,asdict
import re
import time
import numpy as np
from collections import Counter
from .artifacts import digest

def words(text):
    return re.findall(r"\w+",text.casefold())

class BM25Okapi:
    """BM25 with positive Robertson IDF, k1=1.5 and b=0.75."""
    def __init__(self,documents):
        self.tf=[Counter(d) for d in documents]
        self.length=np.array([len(d) for d in documents],dtype=float)
        self.average=max(float(self.length.mean()),1.)
        self.df=Counter(t for d in self.tf for t in d)

    def get_scores(self,query):
        scores=np.zeros(len(self.tf))
        for term in set(query):
            df=self.df[term]
            idf=np.log(1+(len(self.tf)-df+.5)/(df+.5))
            tf=np.array([d[term] for d in self.tf],dtype=float)
            scores+=idf*tf*2.5/(tf+1.5*(.25+.75*self.length/self.average))
        return scores

@dataclass(frozen=True)
class Evidence:
    id: str
    document: str
    text: str
    sentence_indices: tuple

class EvidenceStore:
    def __init__(self,limit=12,deduplicate=True):
        self.items = []
        self.limit = limit
        self.deduplicate = deduplicate

    def add(self,items):
        before = len(self.items)
        for e in items:
            if len(self.items)>=self.limit:
                break
            if not self.deduplicate or e.id not in {v.id for v in self.items}:
                self.items.append(e)
        return len(self.items)-before

class Index:
    def __init__(self,task,services=None):
        self.task,self.services = task,services
        self.passages = []
        for did,sentences in task.sentences.items():
            # Keep original sentence indexes; bounded chunks also cover long SQuAD paragraphs.
            for offset in range(0,len(sentences),2):
                text = " ".join(sentences[offset:offset+2]).strip()
                tokens = text.split()
                for pos in range(0,len(tokens),100):
                    body = " ".join(tokens[pos:pos+120])
                    if body:
                        self.passages.append(Evidence(digest([did,offset,pos,body])[:16],did,body,tuple(range(offset,min(offset+2,len(sentences))))))
        self.texts = [f"{e.document}: {e.text}" for e in self.passages]
        self.bm25 = BM25Okapi([words(t) for t in self.texts]) if self.texts else None
        self.vectors = None
        self.links = {d: sorted(t for t in task.documents if t!=d and t.casefold() in task.documents[d].casefold()) for d in task.documents}

    def execute(self,tool,query="",document="",k=2):
        if tool in ["fetch","expand"]:
            if document not in self.task.documents:
                raise ValueError("Unknown document ID")
            passages=[p for p in self.passages if p.document==document]
            if tool=="expand" and passages:
                anchor=max(range(len(passages)),key=lambda i:len(set(words(query))&set(words(passages[i].text))))
                start=max(0,anchor-1)
                return passages[start:start+k]
            return passages[:k]
        if tool=="graph":
            if document not in self.links:
                raise ValueError("Unknown graph node")
            return [p for p in self.passages if p.document in self.links[document]][:k]
        if tool not in ["bm25","dense","hybrid","rerank"]:
            raise ValueError("Unavailable tool")
        if not self.passages:
            return []
        sparse = np.asarray(self.bm25.get_scores(words(query)))
        if tool=="bm25":
            scores=sparse
        else:
            if self.services is None or not callable(getattr(self.services,"encode",None)):
                raise RuntimeError("Dense service unavailable")
            if self.vectors is None:
                self.vectors=self.services.encode(self.texts)
            dense=self.vectors @ self.services.encode([query],query=True)[0]
            if tool=="dense":
                scores=dense
            else:
                scores=np.zeros(len(self.passages))
                for values in [sparse,dense]:
                    for rank,i in enumerate(np.argsort(-values,kind="stable")):
                        scores[i]+=1/(60+rank+1)
                if tool=="rerank":
                    candidates=np.argsort(-scores,kind="stable")[:12]
                    reranked=self.services.rerank(query,[self.texts[i] for i in candidates])
                    return [self.passages[candidates[i]] for i in np.argsort(-reranked,kind="stable")[:k]]
        return [self.passages[i] for i in np.argsort(-scores,kind="stable")[:k]]

def citation_check(citations,context):
    valid={p.id for p in context}
    return bool(citations) and all(isinstance(c,str) and c in valid for c in citations)
