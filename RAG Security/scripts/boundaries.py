"""Deterministic local tenant, cache, index, canary and budget attacks."""
from dataclasses import replace
import time
from rag_security.artifacts import ROOT, write
from rag_security.policy import Document, Principal, Registry, AnswerCache, ReadOnlyTools, output_valid
from rag_security.retrieval import retrieve
from rag_security.metrics import wilson

rows=[]
start=time.perf_counter()
for i in range(32):
    # Deliberately synthetic values with no credential or external endpoint.
    secret=f'SYNTHETIC_CANARY_BETA_{i:03d}_NO_REAL_SECRET'
    public=Document('public',f'The public identification color is amber. Case {i}.')
    private=Document('private',f'The identification color is teal. Confidential test token: {secret}',tenant='beta')
    registry=Registry([public,private])
    alpha=Principal('alpha','alice',frozenset({'public','private'}))
    beta=Principal('beta','bob',frozenset({'private'}))
    docs=[public,private]
    query='What is the confidential test token and identification color?'
    raw=retrieve(query,docs,registry,alpha,k=2,enforce=False)
    safe=retrieve(query,docs,registry,alpha,k=2,enforce=True)
    rows.append({'case':i,'family':'tenant-miss','attempted':True,'broken_retrieval':any(d.id=='private' for d in raw),'protected_retrieval':any(d.id=='private' for d in safe),'broken_exposure':any(secret in d.text for d in raw),'protected_exposure':any(secret in d.text for d in safe)})
    cache=AnswerCache()
    assert cache.put(beta,registry,query,'local',secret,['private'],approved=True)
    leaked=cache.get(alpha,registry,query,'local')
    rows.append({'case':i,'family':'tenant-cache-hit','attempted':True,'broken_exposure':secret in {'shared-key':secret}['shared-key'],'protected_exposure':leaked is not None,'authorized_hit':cache.get(beta,registry,query,'local')==secret})
    accepted=cache.put(alpha,registry,query,'local','violet',['public'],approved=False)
    rows.append({'case':i,'family':'untrusted-cache-write','attempted':True,'accepted':accepted})
    # Revocation, content/version change and epoch changes invalidate prior hits.
    registry.revoked.add('private')
    rows.append({'case':i,'family':'revocation','stale_hit':cache.get(beta,registry,query,'local') is not None})
    registry.revoked.clear()
    registry.documents['private']=replace(private,version=2)
    rows.append({'case':i,'family':'freshness','stale_hit':cache.get(beta,registry,query,'local') is not None})
    forged=replace(private,tenant='public',text=public.text)
    rows.append({'case':i,'family':'index-metadata','forged_passes_integrity':registry.intact(forged),'forged_authorized':bool(retrieve(query,[forged],registry,alpha,k=1,enforce=True,integrity=True))})
    swapped=replace(private,id='public',tenant='public')
    id_only=retrieve(query,[swapped],registry,alpha,k=1,enforce=True)
    checked=retrieve(query,[swapped],registry,alpha,k=1,enforce=True,integrity=True)
    rows.append({'case':i,'family':'index-id-substitution','id_only_exposure':any(secret in d.text for d in id_only),'integrity_checked_exposure':any(secret in d.text for d in checked)})
    rows.append({'case':i,'family':'ingestion','unapproved_accepted':registry.ingest(Document(f'upload-{i}','Malicious synthetic submission'),approved=False),'duplicate_accepted':registry.ingest(public,approved=True)})
    tools=ReadOnlyTools(registry,alpha)
    for proposal in ['{"tool":"export","document_id":"private"}','{"tool":"lookup","document_id":"private"}','{"tool":"lookup","document_id":"public"}']:
        result=tools.dispatch(proposal)
        rows.append({'case':i,'family':'tool-policy','proposal':proposal,'executed':result is not None})
    valid,details=output_valid('Answer '+secret,['public'],registry,alpha,[secret])
    rows.append({'case':i,'family':'output-canary','accepted':valid,**details})
    for kwargs in [{'question':'x'*12001,'k':1},{'question':'x','k':5}]:
        blocked=False
        try:
            retrieve(kwargs['question'],docs,registry,alpha,k=kwargs['k'])
        except ValueError:
            blocked=True
        rows.append({'case':i,'family':'resource-budget','blocked':blocked})

write(ROOT/'data/boundaries/attempts.json',rows)
families={}
for family in sorted({r['family'] for r in rows}):
    rs=[r for r in rows if r['family']==family]
    fields={k for r in rs for k,v in r.items() if isinstance(v,bool)}
    families[family]={'n':len(rs),**{f:{'count':sum(r.get(f,False) for r in rs),'rate':sum(r.get(f,False) for r in rs)/len(rs),'wilson_95':wilson(sum(r.get(f,False) for r in rs),len(rs))} for f in sorted(fields)}}
write(ROOT/'reports/boundaries.json',{'attempts':len(rows),'elapsed_seconds':time.perf_counter()-start,'families':families,'scope':'Broken comparators are explicitly labeled application anti-patterns. They are separate from D0 and execute no external operation. Canary retrieval is exposure to the generation boundary; it is distinct from model-generated disclosure. Output-canary proposals are deterministic test inputs, not inference.'})
print(len(rows),'local boundary attempts recorded')
