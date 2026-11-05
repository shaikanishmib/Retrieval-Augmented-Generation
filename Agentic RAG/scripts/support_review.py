"""Prepare real citation windows, then validate explicit assistant review judgments."""
import argparse
from collections import Counter
from agentic_rag.artifacts import ROOT,read,write,digest

def cases():
    selected=[]
    for name in ['hotpot','squad']:
        seen=set()
        annotations={r['id']:r for r in read(ROOT/f'data/partitions/{name}.json')['evaluation']}
        for strategy in ['fixed','agent','verify']:
            manifest=read(ROOT/f'data/manifests/{strategy}-{name}-evaluation.json')
            predictions=[read(ROOT/e['path']) for e in manifest['rows']]
            candidates=[r for r in predictions if r.get('rejected_answer')] if strategy=='verify' else [r for r in predictions if r['answer']]
            if len(candidates)<2:
                candidates=[r for r in predictions if r['answer'] or r.get('rejected_answer')]
            candidates=[r for r in candidates if r['id'] not in seen]
            for i,prediction in enumerate(candidates[:2]):
                seen.add(prediction['id'])
                proposal=prediction.get('rejected_answer') or prediction['answer']
                # Rejected proposals keep their packed windows even after final citations are cleared.
                original_citations=prediction['citations']
                if not original_citations and proposal:
                    from agentic_rag.tools import words
                    context=prediction['context']
                    original_citations=[max(context,key=lambda e:len(set(words(proposal))&set(words(e['text']))))['id']] if context else []
                windows=[e for e in prediction['context'] if e['id'] in original_citations]
                row=annotations[prediction['id']]
                case=dict(tag=f'{name}-{strategy}-{i+1}',dataset=name,strategy=strategy,id=prediction['id'],question=row['question'],reference_answers=row['answers'],
                          proposed_answer=proposal,accepted=bool(prediction['answer']),claim=prediction['claim'],windows=windows,nli=prediction['support'])
                case['case_hash']=digest(case)
                selected.append(case)
    return selected

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finalize',action='store_true')
    args=parser.parse_args()
    selected=cases()
    write(ROOT/'data/review/cases.json',selected)
    if not args.finalize:
        import json
        print(json.dumps(selected,ensure_ascii=False,indent=2))
    else:
        judgments=read(ROOT/'data/review/judgments.json')
        assert {j['tag'] for j in judgments}=={c['tag'] for c in selected}
        final=[]
        for case in selected:
            judgment=next(j for j in judgments if j['tag']==case['tag'])
            assert judgment['case_hash']==case['case_hash'],'Changed answer/evidence; re-review required'
            assert judgment['support'] in {'supported','partial','unsupported','unclear'}
            final.append(dict(tag=case['tag'],dataset=case['dataset'],strategy=case['strategy'],id=case['id'],case_hash=case['case_hash'],accepted=case['accepted'],
                              **{k:v for k,v in judgment.items() if k not in {'tag','case_hash'}}))
        write(ROOT/'reports/support-review.json',dict(annotator='Assistant review; no independent human rater.',
              selection='First two nonempty proposals per dataset and strategy, excluding questions already selected for another method; verification preferentially samples rejected proposals. Unblinded to reference answers. Qualitative, purposive sample.',
              scope='Whether the proposed answer answers the question and is supported by its actual assigned citation window; generated-claim wording is inspected separately.',
              criteria='Supported: the requested answer relation is established. Partial: relevant support leaves an identifying condition unresolved or material wording ambiguous. Unsupported: the answer relation is absent or contradicted. Claim-rewriting defects are also described separately. These are subjective assistant judgments, not independent human annotations.',
              unique_questions=len({c['id'] for c in final}),counts=dict(Counter(c['support'] for c in final)),accepted_counts=dict(Counter(c['support'] for c in final if c['accepted'])),cases=final))
        print('Validated',len(final),'case judgments against exact answer/evidence hashes')
