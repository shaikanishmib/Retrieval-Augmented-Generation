"""Record the packages actually imported and a reproducible package inventory."""
import importlib
import importlib.metadata
import subprocess
import sys
from llm_judge.artifacts import ROOT, write, sha, read, source_identity

versions = {}
for name in ['torch','transformers','numpy','scipy','sklearn','pandas','pyarrow','psutil','nbformat','nbclient','ipykernel','matplotlib','pytest','sentencepiece','joblib']:
    module = importlib.import_module(name)
    versions[name] = {'version': getattr(module,'__version__',None), 'file':str(module.__file__)}
write(ROOT / 'reports/runtime-environment.json', versions)
packages = {}
for distribution in importlib.metadata.distributions():
    packages.setdefault(distribution.metadata['Name'].lower(), distribution.version)
(ROOT / 'reports/requirements-lock.txt').write_text('\n'.join(f'{name}=={version}' for name,version in sorted(packages.items()))+'\n',encoding='utf-8')
print(versions)
classifier = ROOT/'data/classifier.joblib'
training = read(ROOT/'reports/classifier-training.json')
write(ROOT/'trained-classifier.lock.json', {'type':'scikit-learn LogisticRegression with TF-IDF and three scalar features','sha256':sha(classifier),'bytes':classifier.stat().st_size,'source_identity':source_identity(),'training_identity':training['training_identity'],'sklearn':versions['sklearn']['version'],'classes':[0,1],'positive_class':'At least one original RAGTruth hallucination span','artifact_policy':'Trained parameters remain local; retrain from pinned training data.'})
for split in ['development','validation','test','transfer']:
    path = ROOT/f'data/scores/{split}.json'
    if not path.exists(): continue
    artifact = read(path)
    for name,record in artifact['runtime'].items():
        if not name.startswith('qwen') and 'truncated_context_fraction' in record:
            record['truncated_context_fraction'] = None
            record['truncation_scope'] = 'Pair tokenization capped at 512; original pair lengths unmeasured.'
    write(path,artifact)
    if split in ['test','transfer']:
        report_path = ROOT/('reports/heldout.json' if split=='test' else 'reports/transfer.json')
        report = read(report_path)
        report['runtime'] = artifact['runtime']
        write(report_path,report)
subprocess.run([sys.executable,str(ROOT/'scripts/run.py'),'audit_perturbations.py'],cwd=ROOT,check=True)
subprocess.run([sys.executable,str(ROOT/'scripts/run.py'),'original_comparators.py'],cwd=ROOT,check=True)
