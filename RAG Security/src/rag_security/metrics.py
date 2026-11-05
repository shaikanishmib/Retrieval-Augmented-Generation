"""Deterministic outcomes and paired group bootstrap intervals."""
from collections import Counter
import re
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

def normalize(text):
    text = re.sub(r'\b(a|an|the)\b', ' ', text.casefold())
    return ' '.join(re.findall(r'\w+', text))

def answer_metrics(answer, reference):
    a, b = normalize(answer).split(), normalize(reference).split()
    overlap = sum((Counter(a) & Counter(b)).values())
    return {'em': int(a==b), 'f1': 2*overlap/(len(a)+len(b)) if a or b else 1.0}

def interval(values, groups=None, repetitions=2000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return [None, None]
    groups = list(range(len(values))) if groups is None else groups
    unique = sorted(set(groups))
    indices = {g: np.flatnonzero(np.array(groups)==g) for g in unique}
    rng = np.random.default_rng(606)
    means = [float(values[np.concatenate([indices[g] for g in rng.choice(unique, size=len(unique), replace=True)])].mean()) for _ in range(repetitions)]
    return np.quantile(means,[.025,.975]).tolist()

def wilson(successes, n):
    if not n:
        return [None, None]
    z = 1.959963984540054
    p = successes/n
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/(1+z*z/n)
    return [max(0,center-half), min(1,center+half)]

def detection(y, scores, threshold):
    y, scores = np.array(y), np.array(scores)
    pred = scores>=threshold
    p,r,f,_ = precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    return {'n':len(y), 'precision':float(p),'recall':float(r),'f1':float(f),'pr_auc_average_precision':float(average_precision_score(y,scores)), 'false_positive_rate':float(pred[y==0].mean()), 'false_positive_interval':wilson(int(pred[y==0].sum()),int((y==0).sum())), 'threshold':float(threshold), 'tp':int(sum(pred & (y==1))), 'fp':int(sum(pred & (y==0))), 'tn':int(sum(~pred & (y==0))), 'fn':int(sum(~pred & (y==1)))}
