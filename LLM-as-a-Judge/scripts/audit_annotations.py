"""Validate original span text and preserve quality-label distributions."""
from collections import Counter
from llm_judge.artifacts import ROOT, write
from llm_judge.data import load_raw

rows = load_raw()
mismatches = sum(r['response'][s['start']:s['end']] != s['text'] for r in rows for s in r['labels'])
assert mismatches == 0
write(ROOT / 'reports/annotation-audit.json', {'span_text_offset_mismatches':mismatches,'quality':dict(Counter(r['quality'] for r in rows)),'implicit_true':dict(Counter(str(s.get('implicit_true')) for r in rows for s in r['labels'])),'quality_policy':'Retain all original quality flags. Incorrect refusals can have no hallucination span while still failing question relevance or completeness.','test_visibility':'Schema and aggregate test prevalence audited before freezing; individual test predictions and errors unavailable to selection.'})
print('Every annotation text matches its original character offsets.')
