"""Describe original RAGBench evaluator scores on exactly matched available rows."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from llm_judge.artifacts import ROOT, read, write

frame = pd.concat([pd.read_parquet(p) for p in sorted((ROOT/'data/raw/ragbench/techqa').glob('*test*.parquet'))],ignore_index=True)
frame = frame.sort_values('id',key=lambda s:s.astype(str)).head(120)
rows = read(ROOT/'data/partitions/transfer.json')
assert list(frame['response']) == [r['response'] for r in rows]
scores = read(ROOT/'data/scores/transfer.json')['scores']
freeze = read(ROOT/'reports/selection-freeze.json')
y = np.array([r['y'] for r in rows])
results = {}
for column in ['ragas_faithfulness','trulens_groundedness','gpt3_adherence']:
    raw = frame[column].to_numpy(dtype=float)
    mask = np.isfinite(raw)&(raw>=0)&(raw<=1)
    comparators = {column:1-raw[mask],freeze['selected']:np.array(scores[freeze['selected']])[mask],'supervised-tfidf':np.array(scores['supervised-tfidf'])[mask],'nli-base/claims-mean':np.array(scores['nli-base/claims-mean'])[mask]}
    results[column] = {'matched_rows':int(mask.sum()),'excluded_missing_or_invalid':int((~mask).sum()),'metrics':{name:{'roc_auc':roc_auc_score(y[mask],values),'average_precision':average_precision_score(y[mask],values)} for name,values in comparators.items()}}
write(ROOT/'reports/original-comparators.json',{'scope':'Exploratory post-freeze comparison using original precomputed dataset columns. Ragas, TruLens and GPT-3 were not executed locally; original software versions and exact scorer prompts are not independently verified. No thresholds are selected from these results.','reference':'Original GPT-4-turbo adherence labels, inverted to positive=non-adherence. These are not human labels.','alignment':'Response strings checked in exact evaluated row order; each comparator uses the identical nonmissing subset.','score_orientation':'One minus original support score; AUROC and average precision only, without asserting calibrated probabilities or threshold equivalence.','results':results})
print('Compared original precomputed scores on matched rows, separately from local inference.')
