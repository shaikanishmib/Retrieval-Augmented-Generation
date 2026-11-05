from llm_judge.transfer import run
from collections import Counter
import pandas as pd
from llm_judge.artifacts import ROOT, read, write
run()
frame = pd.concat([pd.read_parquet(p) for p in sorted((ROOT/'data/raw/ragbench/techqa').glob('*test*.parquet'))],ignore_index=True)
rows = read(ROOT/'data/partitions/transfer.json')
protected = [r for split in ['development','validation','test'] for r in read(ROOT/f'data/partitions/{split}.json')]
write(ROOT/'reports/transfer-data-validation.json', {'official_test_rows_downloaded':len(frame),'evaluated_rows':len(rows),'unique_ids':len({r['id'] for r in rows}),'unique_evidence_groups':len({r['evidence_hash'] for r in rows}),'duplicate_response_texts':len(rows)-len({r['response'] for r in rows}),'original_test_missing_values':{k:int(v) for k,v in frame.isna().sum().items()},'evaluated_positive_labels':sum(r['y'] for r in rows),'exact_evidence_overlap_with_primary_partitions':len({r['evidence_hash'] for r in rows}&{r['evidence_hash'] for r in protected}),'annotation_models':dict(Counter(frame['annotating_model_name'])),'label_origin':'Original model-generated adherence_score, inverted to positive=non-adherence; not human labels.'})
