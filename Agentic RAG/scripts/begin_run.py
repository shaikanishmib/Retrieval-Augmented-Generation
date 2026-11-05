"""Start a fresh experiment record while preserving published reports locally."""
import shutil
import time
from agentic_rag.artifacts import ROOT,write,sha

if __name__=='__main__':
    full=ROOT/'notebooks/01_complete_agentic_rag_research.ipynb'
    if full.exists():
        raise FileExistsError('Use a fresh checkout for a new study. The local complete notebook remains protected; use checkpoints to resume this one.')
    archive=ROOT/f'data/prior-publication/{time.time_ns()}'
    archive.mkdir(parents=True)
    shutil.copytree(ROOT/'reports',archive/'reports')
    moved=[]
    for name in ['selection-freeze.json','preservation.json']:
        path=ROOT/'reports'/name
        if path.exists():
            target=archive/name
            moved.append(dict(source=str(path.relative_to(ROOT)),destination=str(target.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha(path),reason='Preserve the earlier run and permit an independent new freeze/preservation snapshot.'))
            path.rename(target)
    write(archive/'preservation-manifest.json',dict(moved=moved,deletions=[],recovered_bytes=0))
    print('Prior publication retained at',archive)
    print('Run audit_environment.py, then the acquisition, sampling and experiment stages. New support judgments are required for changed predictions.')
