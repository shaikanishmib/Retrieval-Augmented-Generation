from semantic_cache.artifacts import ROOT,read,digest
from semantic_cache.data import workload
from semantic_cache.metrics import aggregate,paired_interval

def test_partitions():
    all_ids = set()
    for dataset in ['hotpot','squad']:
        splits = read(ROOT/f'data/partitions/{dataset}.json')
        docs = []
        ids = []
        for split in ['development','evaluation']:
            rows = splits[split]
            docs.append(set().union(*(set(r['documents']) for r in rows)))
            ids.append({r['id'] for r in rows})
            assert len(ids[-1])==len(rows)
            assert not all_ids&ids[-1]
            all_ids.update(ids[-1])
        assert not docs[0]&docs[1]
        assert not ids[0]&ids[1]
    paws = read(ROOT/'data/partitions/paws.json')
    texts = [{r[k].strip().casefold() for r in rows for k in ['sentence1','sentence2']} for rows in paws.values()]
    assert not texts[0]&texts[1]

def test_ordered_workloads():
    rows = [dict(id=str(i),question=str(i)) for i in range(8)]
    mixed = workload(rows)
    assert len(mixed)==24
    assert len(workload(rows,'unique'))==8
    assert len(workload(rows,'unique-heavy'))==10
    assert digest(mixed)==digest(workload(rows))
    assert all(r['kind']=='synthetic_wrapper' for r in mixed[9::2])

def test_aggregate_denominators_and_paired_interval():
    rows = [dict(trial=0,request_id=i,group=str(i),hit=hit,reuse_equivalent=valid,total_ms=ms,em=1.,f1=1.,generation_calls=int(hit=='miss'),gpu_generation_ms=0.,lookup_ms=0.,embedding_ms=0.,retrieval_ms=0.,generation_ms=0.) for i,(hit,valid,ms) in enumerate([('miss',True,10.),('exact',True,1.),('semantic',False,1.)])]
    result = aggregate(rows)
    assert result['false_hits']==1
    assert result['false_hit_rate_per_hit']==.5
    assert result['false_hit_rate_per_request']==1/3
    assert result['generation_calls']==1
    assert abs(result['miss_rate']-1/3)<1e-10
    paired = paired_interval(rows,[dict(r,f1=r['f1']-.25) for r in rows],'f1')
    assert paired['mean_difference']==paired['low']==paired['high']==-.25
