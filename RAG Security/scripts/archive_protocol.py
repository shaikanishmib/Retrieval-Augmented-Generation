"""Preserve the superseded development protocol before corrected scoring."""
from rag_security.artifacts import ROOT, read, write

assert not (ROOT/'reports/evaluation-run.json').exists()
freeze=read(ROOT/'reports/selection-freeze.json')
run=read(ROOT/'reports/development-run.json')
destination=ROOT/'.local/protocol-v0'
destination.mkdir(parents=True,exist_ok=True)
for name in ['selection-freeze.json','development-run.json']:
    path=ROOT/'reports'/name
    assert not (destination/name).exists()
    path.rename(destination/name)
write(ROOT/'reports/protocol-history.json',{'superseded_development_attempts':run['attempts'],'superseded_identity':run['identity'],'reason':'Pilot review found full-sentence color answers and fenced JSON proposals were undercounted. Completion and poison/proposal oracles were corrected before held-out generation. Old source, manifests, freeze and per-attempt traces remain local. No held-out generator result was observed before correction.','initial_freeze':freeze['selection']})
print('Preserved superseded protocol; held-out generation has not started.')
