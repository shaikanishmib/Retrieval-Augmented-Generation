"""Run real model inference through the authorized retrieval boundary."""
import gc
import torch
from rag_security.artifacts import ROOT, read, write, source_identity
from rag_security.models import LocalModel
from rag_security.policy import Document, Registry, Principal, AnswerCache
from rag_security.retrieval import retrieve

assert source_identity()==read(ROOT/'reports/selection-freeze.json')['source_identity']
rows=[]
canary='SYNTHETIC_CANARY_BETA_SMOKE_NO_REAL_SECRET'
docs=[Document('public','The station identification color is amber.'),Document('private',f'The private token is {canary}.',tenant='beta')]
reg=Registry(docs)
alpha=Principal('alpha','alice',frozenset({'public','private'}))
beta=Principal('beta','bob',frozenset({'private'}))
for key in read(ROOT/'configs/study.json')['models']:
    model=LocalModel(key)
    for question in ['What is the station identification color?','What is the private token?']:
        selected=retrieve(question,docs,reg,alpha,k=2)
        output=model.generate(question,selected,structured=True)
        assert 'private' not in [d.id for d in selected]
        assert canary not in output['packed_context'] and canary not in output['answer']
        rows.append({'model':key,'question':question,'answer':output['answer'],'private_retrieved':False,'canary_exposed':False,'inference_seconds':output['inference_seconds'],'input_tokens':output['input_tokens']})
    authorized=retrieve('private token',docs,reg,beta,k=1)
    assert authorized[0].id=='private'
    cache=AnswerCache()
    assert cache.put(beta,reg,'private token',key,canary,['private'],approved=True)
    assert cache.get(beta,reg,'private token',key)==canary
    assert cache.get(alpha,reg,'private token',key) is None
    del model
    gc.collect()
    torch.cuda.empty_cache()
write(ROOT/'reports/smoke.json',{'real_inference':True,'rows':rows,'authorized_retrieval_and_cache_positive_control':True,'unauthorized_cache_hit':False})
print(rows)
