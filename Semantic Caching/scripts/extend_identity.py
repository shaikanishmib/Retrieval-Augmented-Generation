"""Extend the checkpoint identity to include experiment drivers before final evaluation."""
from semantic_cache.artifacts import ROOT,read,write,digest,sha,identity

freeze = read(ROOT/'reports/selection-freeze.json')
files = sorted([*ROOT.glob('src/**/*.py'),*ROOT.glob('configs/*.json'),ROOT/'models.lock.json',ROOT/'sources.lock.json'])
legacy = digest({p.relative_to(ROOT).as_posix():sha(p) for p in files if p.exists()})
assert freeze['identity']==legacy
assert not (ROOT/'data/traces').exists(),'No final trial may precede identity extension.'
write(ROOT/'data/selection-before-identity-extension.json',freeze)
write(ROOT/'reports/identity-extension.json',dict(previous_identity=legacy,extended_identity=identity(),reason='Include experiment-driver bytes in checkpoint identity. Core source hashes match the development selection; no final trials ran under the shorter identity.'))
freeze['identity'] = identity()
write(ROOT/'reports/selection-freeze.json',freeze)
