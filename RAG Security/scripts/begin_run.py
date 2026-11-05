"""Preserve published run reports before a fresh study in a new checkout."""
from datetime import datetime, timezone
import shutil
from rag_security.artifacts import ROOT

if (ROOT/'data/traces').exists():
    raise SystemExit('Use a separate checkout for a new study; this workspace already has traces.')
destination=ROOT/'.local'/('published-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
shutil.copytree(ROOT/'reports',destination/'reports')
for name in ['selection-freeze.json','detector-freeze.json']:
    path=ROOT/'reports'/name
    if path.exists():
        path.rename(destination/name)
print('Published reports preserved under',destination.name)
