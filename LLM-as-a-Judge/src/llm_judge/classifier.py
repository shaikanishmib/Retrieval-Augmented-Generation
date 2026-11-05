"""Supervised response-style baseline trained on unused official training sources."""
import time
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from llm_judge.artifacts import ROOT, digest, read, write, source_identity
from llm_judge.data import load_raw
from llm_judge.claims import lexical_score, tokens

def scalar(rows):
    return csr_matrix([[lexical_score(r), np.log1p(len(tokens(r['response']))) / 10, len(r['response']) / max(1, len(r['context']))] for r in rows])

def predict(rows):
    path = ROOT / 'data/classifier.joblib'
    identity = source_identity()
    if path.exists():
        artifact = joblib.load(path)
        assert artifact['identity'] == identity, 'Stale classifier'
    else:
        protected = [r for split in ['development', 'validation', 'test'] for r in read(ROOT / f'data/partitions/{split}.json')]
        groups = {r['source_id'] for r in protected}
        evidence = {r['evidence_hash'] for r in protected}
        responses = {r['response'] for r in protected}
        training = [r for r in load_raw() if r['split'] == 'train' and r['source_id'] not in groups and r['evidence_hash'] not in evidence and r['response'] not in responses]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30000, sublinear_tf=True)
        start = time.perf_counter()
        x = hstack([vectorizer.fit_transform([r['response'] for r in training]), scalar(training)]).tocsr()
        model = LogisticRegression(C=1, max_iter=1000, random_state=58021)
        model.fit(x, [r['y'] for r in training])
        artifact = {'model': model, 'vectorizer': vectorizer, 'identity': identity}
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, path)
        write(ROOT / 'reports/classifier-training.json', {'training_responses': len(training), 'training_sources': len({r['source_id'] for r in training}), 'training_identity': digest(training), 'positive_labels': sum(r['y'] for r in training), 'seconds': time.perf_counter() - start, 'features': 'Response-only unigram/bigram TF-IDF plus lexical evidence overlap, response length and response/context length ratio. No generator model identity supplied.', 'limitations': 'May learn task and generator style priors. Does not verify claims; no neural fine-tuning performed.', 'protected_source_groups_excluded': len(groups), 'exact_protected_response_duplicates_excluded': True})
    start = time.perf_counter()
    x = hstack([artifact['vectorizer'].transform([r['response'] for r in rows]), scalar(rows)]).tocsr()
    p = artifact['model'].predict_proba(x)[:, 1].tolist()
    return p, time.perf_counter() - start
