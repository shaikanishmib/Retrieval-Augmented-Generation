"""Corpus-derived query expansion and reciprocal rank fusion."""

from collections import Counter, defaultdict
from .retrieval import tokenize


def rrf(rankings, constant=60, weights=None, k=100):
    if constant <= 0 or k < 1:
        raise ValueError("Fusion constant and result limit must be positive")
    weights = weights if weights is not None else [1.] * len(rankings)
    if len(weights) != len(rankings) or any(w < 0 for w in weights):
        raise ValueError("Fusion weights must match rankings and be nonnegative")
    scores = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        seen = set()
        rank = 0
        for did, _ in ranking:
            if did in seen:
                continue
            seen.add(did)
            rank += 1
            scores[did] += weight / (constant + rank)
    return sorted(scores.items(), key=lambda r: (-r[1], r[0]))[:k]


STOP = set("the a an of and or to in for is are i my can how what do does with on it that this be as at from have you your was if by not will would should".split())


def expand(query, ranking, documents, terms=3):
    original = set(tokenize(query))
    counts = Counter(t for did, _ in ranking[:3] for t in tokenize(documents[did]) if t not in STOP and t not in original and len(t) > 2)
    return query + " " + " ".join(t for t, _ in sorted(counts.items(), key=lambda r: (-r[1], r[0]))[:terms])


def rewrite(query):
    words = [t for t in tokenize(query) if t not in STOP]
    return " ".join(words) or query


def validation_ids(train_qrels, test_qrels, queries, limit=150):
    import hashlib
    test_texts = {queries[q]["text"].strip().casefold() for q in test_qrels}
    eligible = [q for q in train_qrels if q not in test_qrels and queries[q]["text"].strip().casefold() not in test_texts]
    return sorted(eligible, key=lambda q: hashlib.sha256(q.encode()).hexdigest())[:limit]
