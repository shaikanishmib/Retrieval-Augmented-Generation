"""Post-freeze lexical sensitivity diagnostic; never retune the detector."""
from rag_security.artifacts import ROOT, read, write
from rag_security.detectors import Screening

screen=Screening()
task=next(t for t in read(ROOT/'data/tasks/evaluation.json') if t['domain']=='botanical')
original=task['documents'][0]['text']
variants={'original':original,'neutral_name':original.replace('evaluation-station','research-station'),'development_name':original.replace('evaluation-station','development-station'),'development_domain':original.replace('botanical','observatory'),'both_development_terms':original.replace('evaluation-station','development-station').replace('botanical','observatory')}
method=screen.config['selected_lightweight']
threshold=screen.config['thresholds'][method]
scores={name:{'score':screen.score(text,method),'flagged':screen.flagged(text,method)} for name,text in variants.items()}
write(ROOT/'reports/counterfactual-diagnostics.json',{'task':task['id'],'method':method,'threshold':threshold,'variants':scores,'scope':'Post-freeze detector-only analysis of one synthetic factual passage. No generator is run, no protected task or threshold changes, and no result replaces the primary study. Administrative split prefixes are present in original synthetic facility names and can create lexical distribution artifacts. These variants test sensitivity, not general domain transfer or corrected security performance.'})
print(scores)
