"""Measure duplicate candidate bodies, title reuse and split overlap without relabeling."""
from collections import Counter
from agentic_rag.artifacts import ROOT,read,write,digest

if __name__=='__main__':
    output={}
    for name in ['hotpot','squad']:
        parts=read(ROOT/f'data/partitions/{name}.json')
        docs={s:{k for r in rows for k in r['documents']} for s,rows in parts.items()}
        bodies=[body for rows in parts.values() for r in rows for body in r['documents'].values()]
        output[name]=dict(candidate_document_instances=len(bodies),unique_exact_bodies=len({digest(b) for b in bodies}),
                          unique_document_ids_by_split={s:len(v) for s,v in docs.items()},
                          document_id_overlap={f'{a}/{b}':len(docs[a]&docs[b]) for i,a in enumerate(docs) for j,b in enumerate(docs) if i<j})
    raw=read(ROOT/'data/raw/hotpot.json')
    output['hotpot']['questions_with_duplicate_context_titles']=sum(len(r['context'])!=len({t for t,s in r['context']}) for r in raw)
    write(ROOT/'reports/data-structure.json',output)
    print(output)
