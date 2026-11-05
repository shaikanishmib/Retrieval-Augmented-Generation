"""Offset-preserving sentence units and deterministic evidence windows."""
import re
from collections import Counter

def tokens(text):
    return re.findall(r'\w+', text.lower())

def claims(text):
    # Sentence units can contain multiple claims; no atomicity claim is made.
    return [{'text': m.group().strip(), 'start': m.start() + len(m.group()) - len(m.group().lstrip()), 'end': m.end() - len(m.group()) + len(m.group().rstrip())} for m in re.finditer(r'[^.!?]+(?:[.!?]+|$)', text) if m.group().strip()]

def windows(context, size=180, stride=120):
    matches = list(re.finditer(r'\S+', context))
    return [{'text': context[matches[i].start(): matches[min(i + size, len(matches)) - 1].end()], 'start': matches[i].start(), 'end': matches[min(i + size, len(matches)) - 1].end()} for i in range(0, len(matches), stride)]

def select_windows(context, claim, count=3):
    query = set(tokens(claim))
    candidates = windows(context)
    ranked = sorted(enumerate(candidates), key=lambda x: (-len(query & set(tokens(x[1]['text']))) / max(1, len(query)), x[0]))
    return [w for _, w in ranked[:count]]

def citation_ids(text):
    return re.findall(r'\[(\d+)\]', text)

def citation_validity(text, available):
    ids = citation_ids(text)
    return None if not ids else sum(x in set(available) for x in ids) / len(ids)

def token_f1(answer, reference):
    a, b = Counter(tokens(answer)), Counter(tokens(reference))
    overlap = sum((a & b).values())
    return 2 * overlap / (sum(a.values()) + sum(b.values())) if a or b else 1.0

def lexical_score(row):
    answer = set(tokens(row['response']))
    return 1 - len(answer & set(tokens(row['context']))) / max(1, len(answer))
