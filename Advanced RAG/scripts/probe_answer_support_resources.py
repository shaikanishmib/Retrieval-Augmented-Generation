"""Measure CPU verifier loading and memory on a small uncached replay workload."""
import torch
from advanced_rag.data import load
from advanced_rag.answer_support import check
from advanced_rag.upgrade.models import Generator, Verifier
from advanced_rag.upgrade.study import Resources
from advanced_rag.upgrade.artifacts import ROOT, LOCAL, REPORTS, read, save

torch.set_num_threads(4)
observer = Resources()
qa = Generator("flan-base",device="cpu")
verifier = Verifier()
results = []
for name in ("hotpot","squad"):
    manifest = read(LOCAL / "manifests" / f"selected-{name}-development.json")
    for row, item in list(zip(load(name,64,320)[0],manifest['rows']))[:4]:
        results.append(check(row['question'],read(ROOT / item['path'])['result'],qa,verifier))
save(REPORTS / "answer-support-resources.json",{**observer.finish(),"scope":"CPU QA/NLI verifier probe over eight development generations; not a full-study peak",
     "qa_loading_ms":qa.load_ms,"nli_loading_ms":verifier.load_ms,"questions":len(results),
     "qa_checked":sum(p['qa_checked'] for p in results),"mean_validation_ms":sum(p['validation_ms'] for p in results)/len(results)})
print('CPU verifier resource probe recorded')
