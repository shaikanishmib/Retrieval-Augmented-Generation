"""Create frozen source-disjoint local partitions and ID-only public manifests."""
from semantic_cache.data import prepare,load_qa
from semantic_cache.artifacts import ROOT,read,write,digest
import pyarrow.parquet as pq

manifest = ROOT/'reports/data-validation.json'
if not manifest.exists():
    print(prepare())
else:
    validation = read(manifest)
    for dataset in ['hotpot','squad']:
        rows = {r['id']:r for r in load_qa(dataset)}
        partitions = {}
        for split,record in validation['datasets'][dataset]['partitions'].items():
            partitions[split] = [rows[i] for i in record['ids']]
            assert digest(partitions[split])==record['sha256']
        write(ROOT/f'data/partitions/{dataset}.json',partitions)
    partitions = {}
    for split,raw in [('development','dev'),('evaluation','test')]:
        rows = {r['id']:r for r in pq.read_table(ROOT/f'data/raw/paws-{raw}.parquet').to_pylist()}
        record = validation['datasets']['paws-'+split]
        partitions[split] = [rows[i] for i in record['ids']]
        assert digest(partitions[split])==record['sha256']
    write(ROOT/'data/partitions/paws.json',partitions)
    print('Published partition IDs and full row hashes reconstructed without previous project internals.')
