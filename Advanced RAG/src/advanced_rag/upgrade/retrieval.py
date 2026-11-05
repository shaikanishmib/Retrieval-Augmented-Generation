"""Retrieve windows and record every stage of evidence-dependent search."""
import time
import numpy as np
from ..pipeline import BM25, chunk, fuse
from .models import milliseconds

class Index:
    def __init__(self, documents, encoder):
        self.documents, self.mapping = chunk(documents)
        self.encoder = encoder
        self.ids = sorted(self.documents)
        start = time.perf_counter()
        self.sparse = BM25({d: self.mapping[d] + ": " + self.documents[d] for d in self.ids})
        self.vectors, self.encoding_ms = encoder.encode([self.mapping[d] + ": " + self.documents[d] for d in self.ids])
        self.index_ms = milliseconds(start) - self.encoding_ms
        self.vector_bytes = self.vectors.nbytes

    def search(self, question, method="hybrid", reranker=None):
        timings = {"query_encoding_ms": 0., "retrieval_ms": 0., "reranking_ms": 0.}
        start = time.perf_counter()
        lexical = self.sparse.search(question)
        timings["retrieval_ms"] += milliseconds(start)
        if method == "bm25":
            return lexical, timings
        vectors, elapsed = self.encoder.encode([question], query=True)
        timings["query_encoding_ms"] = elapsed
        start = time.perf_counter()
        dense = sorted(zip(self.ids, (self.vectors @ vectors[0]).tolist()), key=lambda r: (-r[1], r[0]))
        ranks = dense if method == "dense" else fuse(lexical[:20], dense[:20])
        timings["retrieval_ms"] += milliseconds(start)
        if method == "rerank":
            if reranker is None:
                raise ValueError("Reranked retrieval requires a reranker")
            candidates = ranks[:12]
            values, elapsed = reranker.predict([(question, self.mapping[d] + ": " + self.documents[d]) for d, _ in candidates])
            timings["reranking_ms"] = elapsed
            ranks = sorted([(d, float(v)) for (d, _), v in zip(candidates, values)], key=lambda r: (-r[1], r[0]))
        return ranks, timings

def diverse_order(ranking, mapping):
    first, rest, seen = [], [], set()
    for item in ranking:
        source = mapping[item[0]]
        (rest if source in seen else first).append(item)
        seen.add(source)
    return first + rest

def context(ranking, index, generator, budget=768, diversity=True, reverse=False):
    ranked = diverse_order(ranking, index.mapping) if diversity else list(ranking)
    evidence, count = {}, 0
    for cid, _ in ranked:
        item = {"chunk_id": cid, "document_id": index.mapping[cid], "text": index.documents[cid]}
        cost = len(generator.tokenizer.encode(item["document_id"] + "\n" + item["text"])) + 12
        if count + cost <= budget:
            evidence[f"S{len(evidence) + 1}"] = item
            count += cost
        if len(evidence) == 8:
            break
    if reverse:
        evidence = {f"S{i + 1}": item for i, item in enumerate(reversed(list(evidence.values())))}
    return evidence, count
