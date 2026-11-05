import math

import numpy as np
import pytest

from rag_data.chunking import aggregate, chunks
from rag_data.dense import Dense, normalize
from rag_data.retrieval import BM25, evaluate
from rag_data.transforms import expand, rewrite, rrf, validation_ids
from rag_data.validation import load_records, validate


@pytest.mark.parametrize("method", ["whole", "fixed", "sentence", "recursive"])
def test_chunk_coverage_and_original_ids(method):
    documents = {"a": "First sentence. Second sentence. Third sentence.", "b": "", "x::y": "one two three four five six"}
    rows = chunks(documents, method, 4, 1)
    assert {r.document_id for r in rows} == set(documents)
    assert len({r.id for r in rows}) == len(rows)
    for did, text in documents.items():
        assert set(text.split()) <= {t for r in rows if r.document_id == did for t in r.text.split()}
    if method != "whole":
        assert max(r.tokens for r in rows) <= 4


@pytest.mark.parametrize("size,overlap", [(0,0),(3,3),(3,-1)])
def test_invalid_chunk_parameters(size, overlap):
    with pytest.raises(ValueError):
        chunks({"a":"text"},"fixed",size,overlap)


def test_document_aggregation_keeps_first_score():
    assert aggregate([("x",.9),("y",.8),("z",.7)],{"x":"a","y":"a","z":"b"}) == [("a",.9),("b",.7)]
    with pytest.raises(KeyError):
        aggregate([("unknown",1)],{})


def test_metrics_hand_calculation_and_deduplication():
    metrics,per = evaluate({"q":{"a":1,"c":1},"missing":{"b":1}}, {"q":["a","a","b","c"]}, {"a","b","c"})
    assert per["q"]["ndcg@10"] == pytest.approx(1.5/(1+1/math.log2(3)))
    assert per["q"]["map@10"] == pytest.approx((1+2/3)/2)
    assert per["q"]["recall@1"] == .5
    assert per["q"]["recall@3"] == 1
    assert metrics["mrr@10"] == .5


def test_bad_qrels_fail():
    for labels in [{}, {"q":{"a":0}}, {"q":{"a":-1}}, {"q":{"missing":1}}]:
        with pytest.raises(ValueError):
            evaluate(labels,{}, {"a"})
    with pytest.raises(ValueError,match="Unknown retrieved"):
        evaluate({"q":{"a":1}},{"q":["missing","a"]},{"a"})


def test_bm25_formula_ties_and_filter():
    index=BM25({"a":"bond bond","b":"equity"})
    assert index.search("bond")[0][1] == pytest.approx(math.log(2)*2*1.9/(2+.9*(.6+.4*2/1.5)))
    assert index.search("unknown")==[("a",0),("b",0)]
    assert index.search("bond",candidates=["b"])==[("b",0)]
    with pytest.raises(ValueError):
        index.search("bond",candidates=["missing"])


def test_bm25_matches_document_loop_reference():
    from collections import Counter
    from random import Random
    from rag_data.retrieval import tokenize
    rng=Random(7)
    documents={str(i):" ".join(rng.choices(["bond","stock","green","blue"],k=rng.randrange(1,10))) for i in range(25)}
    documents["empty"]=""
    tokenized={did:Counter(tokenize(text)) for did,text in documents.items()}
    average=sum(sum(v.values()) for v in tokenized.values())/len(documents)
    index=BM25(documents)
    for query in ("bond bond stock","blue green","unknown"):
        reference=[]
        for did,frequencies in tokenized.items():
            score=0.
            for term,qtf in Counter(tokenize(query)).items():
                df=sum(term in row for row in tokenized.values())
                tf=frequencies[term]
                norm=.9*(.6+.4*sum(frequencies.values())/average)
                score+=qtf*math.log(1+(len(documents)-df+.5)/(df+.5))*tf*1.9/(tf+norm)
            reference.append((did,score))
        reference.sort(key=lambda row:(-row[1],row[0]))
        actual=index.search(query,k=len(documents))
        assert [d for d,_ in actual]==[d for d,_ in reference]
        assert [s for _,s in actual]==pytest.approx([s for _,s in reference],abs=1e-12)


@pytest.mark.parametrize("backend",["exact","faiss","qdrant"])
def test_backend_scores_and_persistence(backend,tmp_path):
    index=Dense(["a","b","c"],[[1,0],[0,1],[-1,0]],backend,tmp_path/"db")
    assert index.search([1,0],3)==[("a",1.),("b",0.),("c",-1.)]
    index.close()


def test_embedding_validation():
    assert normalize([[3,4]]).tolist()[0] == pytest.approx([.6,.8])
    for matrix in [[[float("nan"),1]],[1,2]]:
        with pytest.raises(ValueError):
            normalize(matrix)
    with pytest.raises(ValueError):
        Dense(["a","a"],[[1,0],[0,1]])
    with pytest.raises(ValueError):
        Dense(["a"],[[1,0]]).search([1,2,3])


def test_fusion_deduplicates_and_weights():
    ranks=[[('a',1),('a',.9),('b',.8)],[('b',1),('a',.5)]]
    fused=dict(rrf(ranks,constant=10))
    assert fused['a']==pytest.approx(1/11+1/12)
    assert fused['b']==pytest.approx(1/12+1/11)
    assert rrf(ranks,weights=[1,0])[0][0]=='a'


def test_validation_excludes_test_ids_and_duplicate_text():
    queries={"a":{"text":"Same query"},"b":{"text":" same query "},"c":{"text":"Other"}}
    assert validation_ids({"b":{},"c":{}},{"a":{}},queries)==["c"]
    assert validation_ids({"c":{}},{"a":{}},queries)==validation_ids({"c":{}},{"a":{}},queries)


def test_transforms_without_generated_observations():
    assert rewrite("How do I invest in bonds?")=="invest bonds"
    assert expand("bonds",[("a",1)],{"a":"interest interest rates"})=="bonds interest rates"


def test_empty_original_records_are_reported(tmp_path):
    p=tmp_path/"corpus.jsonl"
    p.write_text('{"_id":"a","title":"","text":""}\n')
    with pytest.raises(ValueError,match="Empty text"):
        load_records(p,corpus=True)
    rows,stats=load_records(p,corpus=True,allow_empty=True)
    assert rows["a"]["text"]==""
    assert stats["empty_fields"]["text"]==1


def test_id_namespaces_and_split_leakage(tmp_path):
    (tmp_path/"corpus.jsonl").write_text('{"_id":"same","title":"","text":"A document."}\n')
    (tmp_path/"queries.jsonl").write_text('{"_id":"same","text":"A question?"}\n')
    (tmp_path/"qrels").mkdir()
    row="query-id\tcorpus-id\tscore\nsame\tsame\t1\n"
    (tmp_path/"qrels/train.tsv").write_text(row)
    report=validate(tmp_path)
    assert report["valid"]
    assert report["query_corpus_id_overlap"]==1
    (tmp_path/"qrels/test.tsv").write_text(row)
    assert not validate(tmp_path)["valid"]
    (tmp_path/"qrels/test.tsv").unlink()
    (tmp_path/"qrels/train.tsv").write_text(row+"same\tsame\t1\n")
    with pytest.raises(ValueError,match="Duplicate qrels"):
        validate(tmp_path)
    (tmp_path/"qrels/train.tsv").write_text("query-id\tcorpus-id\tscore\nsame\tmissing\t1\n")
    assert not validate(tmp_path)["valid"]


@pytest.mark.parametrize("name,documents,queries",[("scifact",5183,1109),("fiqa",57638,6648)])
def test_real_dataset_integrity(name,documents,queries):
    from pathlib import Path
    raw=Path(__file__).resolve().parents[1]/"data/raw"/name
    if not raw.exists():
        pytest.skip("Acquire the original BEIR dataset to run the integration check")
    report=validate(raw,allow_empty_corpus=True)
    assert report["valid"]
    assert report["corpus"]["count"]==documents
    assert report["queries"]["count"]==queries
    assert not any(report["split_query_overlap"].values())
