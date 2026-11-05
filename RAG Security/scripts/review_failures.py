"""Describe safe synthetic failure cases from retained real predictions."""
from rag_security.artifacts import ROOT, read, write
from rag_security.detectors import Screening

run=read(ROOT/'reports/evaluation-run.json')
rows=[read(ROOT/f'data/traces/evaluation/{eid}.json') for eid in run['attempt_ids']]
freeze=read(ROOT/'reports/selection-freeze.json')
task=next(t for t in read(ROOT/'data/tasks/evaluation.json') if t['domain']=='botanical')
screen=Screening()
benign=task['documents'][0]['text']
score=screen.score(benign,screen.config['selected_lightweight'])
threshold=screen.config['thresholds'][screen.config['selected_lightweight']]
cases=[]
for model,defense,attack in [('qwen06','D0',None),('qwen06','D6',None),('flan-base','D0','local-test-poison'),('qwen06','D0','local-test-poison'),('qwen06','D0','local-test-citation'),('qwen06','D4','local-test-tool')]:
    r=next(r for r in rows if r['task_id']==task['id'] and r['model']==model and r['defense']==defense and r['attack_id']==attack)
    cases.append({'model':model,'defense':defense,'family':r['family'],'answer':r['answer'],'completion':r['completion'],'attacker_success':r['attack_success'],'source_retained':'source-0' in r['retrieved_ids'],'attempt_id':r['attempt_id'],'tool_disallowed_proposal':r['disallowed_tool_attempt']})
write(ROOT/'reports/qualitative-review.json',{'scope':'Assistant review of safe original synthetic outputs only. No independent human rater or model judge. No prompts, raw tables or private traces are published.','task_id':task['id'],'reference':task['reference'],'clean_passage_screening_score':score,'frozen_threshold':threshold,'cases':cases})
lines=['# Reviewed failure cases','',f'The first held-out botanical task has reference color `{task["reference"]}`. Its benign factual passage scores {score:.4f} against the frozen screening threshold {threshold:.4f}. This review reads actual local model outputs; it has no independent human adjudicator.','', '| Model / defense | Family | Actual answer | Completion | Attacker goal achieved |','| --- | --- | --- | ---: | ---: |']
for r in cases:
    answer=r['answer'].replace('|','/').replace('\n',' ')
    lines.append(f"| {r['model']}/{r['defense']} | {r['family']} | {answer} | {r['completion']} | {int(r['attacker_success'])} |")
lines+=['','D6 drops useful clean evidence and loses the answer. This is a failure of transfer, not evidence that zero ASR makes the system useful. The sparse detector is fitted on contextual examples but applied to individual newline units during quarantine; domain and unit-distribution shift are both plausible contributors, without a causal identification experiment.','', 'Factual poisoning changes the answer itself. An intact admitted false document would not be recognized as false by a content hash. Fabricated citations can accompany a correct color, so factual completion and citation validity differ. A disallowed tool proposal can fail the specified targeted objective while still changing agent behavior; the report retains both proposal counts and actual execution.','', 'Strict clean tool utility is zero for both generators in this interface. Qwen emits valid lookup proposals inside Markdown fences; a separate canonicalization replay restores all 16 authorized clean lookups while retaining policy rejection of disallowed actions. This secondary parser study is not substituted into primary results.']
(ROOT/'reports/failure-cases.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Reviewed',len(cases),'safe synthetic outputs')
