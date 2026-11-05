"""Screening scores are fitted without protected evaluation examples."""
import re
import joblib
from .artifacts import ROOT, read

PATTERNS = [r'ignore.{0,30}(question|instruction)', r'system notice', r'reply only', r'entire answer', r'next (lookup|action)', r'encode your', r'reverse your', r'nonexistent source']

def lexical(text):
    return float(any(re.search(p,text,re.I|re.S) for p in PATTERNS))

class Screening:
    def __init__(self):
        self.classifier = joblib.load(ROOT / 'data/detectors/tfidf.joblib')
        self.config = read(ROOT / 'reports/detector-freeze.json')

    def score(self, text, method):
        if method == 'lexical':
            return lexical(text)
        if method == 'tfidf':
            return float(self.classifier.predict_proba([text])[0,1])
        raise ValueError(method)

    def flagged(self, text, method=None):
        method = method or self.config['selected_lightweight']
        return self.score(text,method) >= self.config['thresholds'][method]
