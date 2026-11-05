"""Check recorded answer metrics with a separate dataset-convention implementation."""
from collections import Counter
import re
import string
from agentic_rag.artifacts import ROOT,read,write

def reference_score(prediction,answers,hotpot):
    def tokens(text):
        cleaned=''.join(c for c in text.lower() if c not in string.punctuation)
        return re.sub(r'\b(a|an|the)\b',' ',cleaned).split()
    predicted=tokens(prediction)
    values=[]
    for answer in answers or ['']:
        expected=tokens(answer)
        exact=float(predicted==expected)
        if hotpot and (predicted in [['yes'],['no'],['noanswer']] or expected in [['yes'],['no'],['noanswer']]) and not exact:
            overlap=0.
        elif not predicted or not expected:
            overlap=exact
        else:
            shared=Counter(predicted)&Counter(expected)
            hits=sum(shared.values())
            precision=hits/len(predicted)
            recall=hits/len(expected)
            overlap=2*precision*recall/(precision+recall) if hits else 0.
        values.append((exact,overlap))
    return max(v[0] for v in values),max(v[1] for v in values)

if __name__=='__main__':
    checked=0
    mismatches=[]
    for path in (ROOT/'data/manifests').glob('*.json'):
        name,split=path.stem.rsplit('-',2)[-2:]
        annotations={r['id']:r for r in read(ROOT/f'data/partitions/{name}.json')[split]}
        recorded={r['id']:r for r in read(ROOT/f'data/scores/{path.name}')}
        for entry in read(path)['rows']:
            raw=read(ROOT/entry['path'])
            em,f1=reference_score(raw['answer'],annotations[raw['id']]['answers'],name=='hotpot')
            actual=recorded[raw['id']]
            checked+=1
            if abs(em-actual['em'])>1e-12 or abs(f1-actual['f1'])>1e-12:
                mismatches.append(dict(configuration=path.stem,id=raw['id'],reference_em=em,reference_f1=f1,recorded_em=actual['em'],recorded_f1=actual['f1']))
    write(ROOT/'data/metric-convention-mismatches.json',mismatches)
    write(ROOT/'reports/metric-convention-audit.json',dict(question_results_checked=checked,mismatches=len(mismatches),
         implementation='Separate standard-library normalizer and precision/recall implementation; normalized yes/no/noanswer guard only for HotpotQA. Tolerance 1e-12.',
         reference_sources=['https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py','https://github.com/rajpurkar/SQuAD-explorer/blob/master/evaluate-v2.0.py']))
    print('Metric convention check:',checked,'question results;',len(mismatches),'mismatches')
    assert not mismatches,'Inspect metric differences before publication'
