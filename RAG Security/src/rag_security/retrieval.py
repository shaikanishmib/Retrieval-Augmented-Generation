"""BM25 with stable ordering and authorization before ranking."""
from collections import Counter
import math
import re

def tokens(text):
    return re.findall(r'\w+', text.casefold())

def retrieve(question, documents, registry, principal, k=3, enforce=True, integrity=False):
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 4:
        raise ValueError('Retrieval count must be between one and four.')
    if len(question) > 12000 or len(documents) > 64 or any(len(d.text) > 12000 for d in documents):
        raise ValueError('Local input budget exceeded.')
    candidates = [d for d in documents if (not enforce or registry.authorized(principal, d.id)) and (not integrity or registry.intact(d))]
    if not candidates:
        return []
    bags = [Counter(tokens(d.text + ' ' + d.title)) for d in candidates]
    average = sum(sum(c.values()) for c in bags) / len(bags)
    query = tokens(question)
    def score(bag):
        length = sum(bag.values())
        result = 0.0
        for term in set(query):
            frequency = sum(term in b for b in bags)
            idf = math.log(1 + (len(bags) - frequency + 0.5) / (frequency + 0.5))
            tf = bag[term]
            result += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * length / max(average, 1)))
        return result
    ranked = sorted(zip(candidates, bags), key=lambda x: (-score(x[1]), x[0].id))
    return [d for d, _ in ranked[:k]]
