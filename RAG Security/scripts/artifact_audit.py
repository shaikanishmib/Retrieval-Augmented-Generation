"""Verify all acquired bytes and document the redistribution boundaries."""
from rag_security.artifacts import ROOT, read, write, file_hash

verified=[]
for name,lock in read(ROOT/'sources.lock.json').items():
    for path,meta in lock['files'].items():
        local=ROOT/'data/raw'/name/path
        assert file_hash(local)==meta['sha256'] and local.stat().st_size==meta['bytes']
        verified.append({'kind':'source','name':name,'path':path,'sha256':meta['sha256']})
for name,lock in read(ROOT/'models.lock.json').items():
    folder=ROOT.parent/lock['local_path']
    if not folder.exists():
        folder=ROOT/'data/models'/name
    for path,meta in lock['files'].items():
        local=folder/path
        assert file_hash(local)==meta['sha256'] and local.stat().st_size==meta['bytes']
        verified.append({'kind':'model','name':name,'path':path,'sha256':meta['sha256']})
write(ROOT/'reports/artifact-audit.json',{'verified_files':len(verified),'files':verified,'all_locked_hashes_match':True,'notes':'Qwen3-1.7B is verified cached inventory only, not an evaluated generator.'})
write(ROOT/'trained-detector.lock.json',{'method':'word/character TF-IDF logistic regression','artifact':'data/detectors/tfidf.joblib','sha256':file_hash(ROOT/'data/detectors/tfidf.joblib'),'training_identity':read(ROOT/'reports/detector-freeze.json')['training_identity'],'sklearn_version':read(ROOT/'reports/runtime-environment.json')['modules']['sklearn']['version'],'redistributed':False})
write(ROOT/'reports/license-audit.json',{'original_code_and_synthetic_fixtures':'MIT, see LICENSE','BIPIA_repository_and_attack_definitions':'MIT; pinned LICENSE hash in sources.lock.json','BIPIA_TableQA_contexts':'Derived from WikiTableQuestions/Wikipedia, CC BY-SA 4.0; raw contexts not redistributed','BIPIA_email':'Acquired for inspection only; not evaluated or redistributed','AgentDojo':'MIT metadata only; not executed or claimed as benchmark coverage','RAG-PIBench':'Paper only; dataset access and license unverified; excluded from scores','models':'Model-specific declared terms and hashes in models.lock.json; no weights redistributed','pretrained_detector':'Apache-2.0 declared by upstream; upstream data membership/rights are not independently established; no training data or weights redistributed','publication':'Original code, safe synthetic fixtures, aggregate results, locks, metadata and output-free notebook only'})
print('Verified',len(verified),'source and model files')
