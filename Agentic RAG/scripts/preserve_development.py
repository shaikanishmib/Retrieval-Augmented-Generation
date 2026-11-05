"""Keep prior development summaries when the protocol changes before evaluation."""
import shutil
import time
from agentic_rag.artifacts import ROOT,write

if __name__=='__main__':
    destination=ROOT / f'data/development-history/revision-{time.time_ns()}'
    destination.mkdir(parents=True)
    for directory in ['src','configs','scripts']:
        shutil.copytree(ROOT/directory,destination/directory)
    for name in ['aggregates','resource-stages','planner-selection.json','generator-comparison.json']:
        source=ROOT/'reports'/name
        if source.is_dir():
            shutil.copytree(source,destination/name)
        elif source.exists():
            shutil.copyfile(source,destination/name)
    write(destination/'revision.json',dict(reason='Pre-freeze engineering corrections: reserve decoder-start token; count action attempts including plan retries; retain controller stop reason; seed Torch; validate cached row hash.',
                                         status='Development interrupted intentionally; partial checkpoints and summaries retained. No final evaluation had begun.'))
    print(destination)
