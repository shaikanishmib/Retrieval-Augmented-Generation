import math
import re
from collections import Counter, defaultdict

import numpy as np


def tokenize(text):
    return re.findall(r'\w+', text.lower(), flags=re.UNICODE)


class BM25:
    def __init__(self, documents, k1=0.9, b=0.4):
        if not documents or k1 <= 0 or not 0 <= b <= 1:
            raise ValueError('BM25 needs documents, k1 > 0, and 0 <= b <= 1')
        self.ids = sorted(documents)
        self.positions = {did: i for i, did in enumerate(self.ids)}
        lengths = []
        postings = defaultdict(list)
        self.k1, self.b = k1, b
        for position, did in enumerate(self.ids):
            terms = Counter(tokenize(documents[did]))
            lengths.append(sum(terms.values()))
            for term, count in terms.items():
                postings[term].append((position, count))
        self.lengths = np.asarray(lengths, dtype=np.float64)
        self.avg_length = float(self.lengths.mean())
        if not self.avg_length:
            raise ValueError('Corpus contains no tokens')
        self.postings = {}
        for term, entries in postings.items():
            values = np.asarray(entries, dtype=np.int64)
            positions, counts = values[:, 0], values[:, 1]
            idf = math.log(1 + (len(self.ids) - len(entries) + 0.5) / (len(entries) + 0.5))
            norm = k1 * (1 - b + b * self.lengths[positions] / self.avg_length)
            weights = idf * counts * (k1 + 1) / (counts + norm)
            self.postings[term] = positions.copy(), weights

    def search(self, query, k=20, candidates=None):
        if k < 1:
            raise ValueError('k must be positive')
        if candidates is None:
            allowed = np.arange(len(self.ids))
        else:
            identifiers = set(candidates)
            if not identifiers <= self.positions.keys():
                raise ValueError('Unknown candidate document')
            allowed = np.array(sorted(self.positions[did] for did in identifiers), dtype=np.int64)
        scores = np.zeros(len(self.ids), dtype=np.float64)
        for term, query_count in Counter(tokenize(query)).items():
            if term in self.postings:
                positions, weights = self.postings[term]
                scores[positions] += query_count * weights
        order = np.argsort(-scores[allowed], kind='stable')[:k]
        return [(self.ids[i], float(scores[i])) for i in allowed[order]]


def evaluate(qrels, rankings, corpus_ids):
    if not qrels:
        raise ValueError('No judged queries')
    rows = {}
    for qid, labels in qrels.items():
        if not labels or any(v < 0 for v in labels.values()) or not set(labels) <= corpus_ids:
            raise ValueError('Invalid qrels or missing targets')
        positive = {did for did, value in labels.items() if value > 0}
        if not positive:
            raise ValueError('Query has no positive judgments')
        seen = set()
        ranking = []
        for did in rankings.get(qid, []):
            if did not in corpus_ids:
                raise ValueError(f'Unknown retrieved document: {did}')
            if did not in seen:
                ranking.append(did)
                seen.add(did)
        gains = [labels.get(did, 0) for did in ranking[:10]]
        dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(gains))
        ideal = sum(rel / math.log2(i + 2) for i, rel in enumerate(sorted(labels.values(), reverse=True)[:10]))
        rows[qid] = {'ndcg@10': dcg / ideal,
                     'map@10': sum(sum(x in positive for x in ranking[:i + 1]) / (i + 1)
                                   for i, did in enumerate(ranking[:10]) if did in positive) / min(10, len(positive)),
                     'mrr@10': next((1 / (i + 1) for i, did in enumerate(ranking[:10]) if did in positive), 0.0),
                     **{f'recall@{k}': len(positive.intersection(ranking[:k])) / len(positive) for k in (1, 3, 5, 10, 20)}}
    return {key: sum(row[key] for row in rows.values()) / len(rows) for key in next(iter(rows.values()))}, rows
