"""Positive-class detection metrics and source-cluster paired uncertainty."""
import numpy as np
from sklearn import metrics as m

def evaluate(y, p, threshold=0.5):
    y, p = np.asarray(y), np.asarray(p)
    assert len(y) and len(y) == len(p) and np.isfinite(p).all()
    pred = p >= threshold
    tn, fp, fn, tp = m.confusion_matrix(y, pred, labels=[0, 1]).ravel()
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(np.mean(bins == i) * abs(np.mean(y[bins == i]) - np.mean(p[bins == i])) for i in range(10) if np.any(bins == i))
    return {'n': len(y), 'accuracy': m.accuracy_score(y, pred), 'balanced_accuracy': m.balanced_accuracy_score(y, pred), 'precision': m.precision_score(y, pred, zero_division=0), 'recall': m.recall_score(y, pred, zero_division=0), 'f1': m.f1_score(y, pred, zero_division=0), 'macro_f1': m.f1_score(y, pred, average='macro', zero_division=0), 'kappa': float(np.nan_to_num(m.cohen_kappa_score(y, pred))), 'mcc': m.matthews_corrcoef(y, pred), 'roc_auc': m.roc_auc_score(y, p) if len(set(y)) == 2 else None, 'pr_auc': m.average_precision_score(y, p) if len(set(y)) == 2 else None, 'brier': m.brier_score_loss(y, p), 'ece_10_bins': ece, 'false_positive_rate': fp / max(1, fp + tn), 'false_negative_rate': fn / max(1, fn + tp), 'confusion': [[int(tn), int(fp)], [int(fn), int(tp)]]}

def choose_threshold(y, p, candidates):
    return max(candidates, key=lambda t: (evaluate(y, p, t)['macro_f1'], -abs(t - .5)))

def paired_interval(rows, p, q, tp=.5, tq=.5, repeats=2000):
    groups = sorted({r['source_id'] for r in rows})
    indexes = {g: [i for i, r in enumerate(rows) if r['source_id'] == g] for g in groups}
    y = np.array([r['y'] for r in rows])
    p, q = np.array(p), np.array(q)
    rng = np.random.default_rng(58021)
    diffs = []
    for _ in range(repeats):
        ix = np.concatenate([indexes[g] for g in rng.choice(groups, len(groups), replace=True)])
        diffs.append(m.f1_score(y[ix], p[ix] >= tp, average='macro', zero_division=0) - m.f1_score(y[ix], q[ix] >= tq, average='macro', zero_division=0))
    return {'metric': 'macro_f1_difference', 'resampling_unit': 'source_id', 'repeats': repeats, 'interval_95': np.quantile(diffs, [.025, .975]).tolist(), 'point': m.f1_score(y, p >= tp, average='macro', zero_division=0) - m.f1_score(y, q >= tq, average='macro', zero_division=0)}

def selective(y, p, threshold=.5):
    y, p = np.array(y), np.array(p)
    confidence = abs(p - threshold)
    return [{'minimum_margin': float(t), 'coverage': float(np.mean(confidence >= t)), 'accuracy': float(np.mean((p[confidence >= t] >= threshold) == y[confidence >= t])) if np.any(confidence >= t) else None} for t in np.linspace(0, .45, 10)]
