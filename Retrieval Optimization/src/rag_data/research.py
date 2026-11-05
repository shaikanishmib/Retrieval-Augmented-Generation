"""Run a staged retrieval study using original BEIR splits."""

import csv
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .chunking import aggregate, chunks
from .dense import Dense
from .retrieval import BM25, evaluate, tokenize
from .transforms import expand, rewrite, rrf, validation_ids
from .validation import load_records, validate

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/research"
LOCAL = ROOT / "data/experiments"
REQUIRED_RUNS = {"bm25", "bm25_chunked", "dense_0", "dense_1", "faiss", "qdrant", "hybrid",
                 "expansion", "rewrite", "multiquery", "length_filter", "threshold_0.2",
                 "threshold_0.4", "threshold_0.6", "rerank_20", "rerank_50"}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def qrels(path):
    result = {}
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            result.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    return result


def measure(search, queries, labels, corpus_ids):
    rankings, scores, timings = {}, {}, []
    for qid in sorted(labels):
        start = time.perf_counter()
        ranking = search(qid, queries[qid]["text"])
        timings.append((time.perf_counter() - start) * 1000)
        rankings[qid] = [did for did, _ in ranking]
        scores[qid] = ranking
    metrics, per_query = evaluate(labels, rankings, corpus_ids)
    metrics.update({"p50_ms": float(np.percentile(timings, 50)), "p95_ms": float(np.percentile(timings, 95)), "query_count": len(labels)})
    return metrics, per_query, scores


def encode(model, texts, destination):
    lock_hash = hashlib.sha256((ROOT / "models.lock.json").read_bytes()).hexdigest()
    fingerprint = hashlib.sha256((str(model.model_card_data.model_id) + lock_hash + "\0" + "\0".join(texts)).encode()).hexdigest()
    meta = destination.with_suffix(".json")
    if destination.exists() and meta.exists() and json.loads(meta.read_text())["fingerprint"] == fingerprint:
        return np.load(destination), json.loads(meta.read_text())
    start = time.perf_counter()
    matrix = model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
    import torch
    info = {"fingerprint": fingerprint, "seconds": time.perf_counter() - start, "shape": list(matrix.shape), "bytes": matrix.nbytes, "torch_threads": torch.get_num_threads(), "model_lock_sha256": lock_hash}
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, matrix)
    save(meta, info)
    return matrix, info


def run_dataset(name):
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer
    torch.set_num_threads(min(12, os.cpu_count() or 1))
    torch.manual_seed(42)
    raw = ROOT / "data/raw" / name
    records, _ = load_records(raw / "corpus.jsonl", corpus=True, allow_empty=True)
    queries, _ = load_records(raw / "queries.jsonl", corpus=False)
    documents = {did: (r["title"] + " " + r["text"]).strip() for did, r in records.items()}
    test = qrels(raw / "qrels/test.tsv")
    train = qrels(raw / "qrels/train.tsv")
    dev = qrels(raw / "qrels/dev.tsv") if (raw / "qrels/dev.tsv").exists() else train
    dev_ids = validation_ids(dev, test, queries, 150)
    development = {qid: dev[qid] for qid in dev_ids}
    save(LOCAL / name / "validation-selection.json", {"query_ids": dev_ids, "source": "dev" if name == "fiqa" else "train", "selection": "SHA-256 order, at most 150, excluding test text duplicates"})
    corpus_ids = set(documents)
    summary = {"dataset": name, "validation": validate(raw, allow_empty_corpus=True), "selection": {"validation_queries": len(development), "test_queries": len(test)}, "runs": {}, "chunking": [], "encoding": {}, "backend_checks": {}}
    for kind, values in [("documents", documents), ("queries", {q:r["text"] for q,r in queries.items()})]:
        lengths = [len(tokenize(v)) for v in values.values()]
        summary[kind + "_lengths"] = {"min": min(lengths), "p25": float(np.percentile(lengths,25)), "median": float(np.median(lengths)), "p75": float(np.percentile(lengths,75)), "p95": float(np.percentile(lengths,95)), "max": max(lengths), "mean": float(np.mean(lengths))}
        save(LOCAL / name / f"{kind}-lengths.json", lengths)
    frequency = Counter(t for text in documents.values() for t in tokenize(text) if len(t) > 3)
    save(LOCAL / name / "terminology.json", frequency.most_common(60))
    selected = None
    best = -1
    for method, size, overlap in [("whole",128,24),("fixed",128,0),("fixed",128,24),("fixed",256,32),("recursive",128,24),("sentence",128,24)]:
        items = chunks(documents, method, size, overlap)
        mapping = {c.id:c.document_id for c in items}
        started = time.perf_counter()
        index = BM25({c.id:c.text for c in items})
        build = time.perf_counter() - started
        search = lambda q,t: aggregate(index.search(t,k=100),mapping,20)
        metrics, _, _ = measure(search, queries, development, corpus_ids)
        row = {"method":method,"size":size,"overlap":overlap,"chunks":len(items),"covered_documents":len(set(mapping.values())),"build_seconds":build,**metrics}
        summary["chunking"].append(row)
        if metrics["ndcg@10"] > best:
            best, selected = metrics["ndcg@10"], (method,size,overlap)
        del index, search
        print(name, "chunking", row, flush=True)
    summary["selected_chunking"] = list(selected)
    items = chunks(documents,*selected)
    mapping = {c.id:c.document_id for c in items}
    sparse_chunk = BM25({c.id:c.text for c in items})
    sparse = BM25(documents)
    sparse_search = lambda q,t: sparse.search(t,k=100)
    all_scores = {}
    def record(label, search, labels=test, details=None):
        metrics, per, scores = measure(search,queries,labels,corpus_ids)
        summary["runs"][label] = {**metrics, **(details or {})}
        all_scores[label] = scores
        save(LOCAL / name / f"{label}-per-query.json",per)
        save(LOCAL / name / f"{label}-rankings.json",scores)
        save(OUT / f"{name}.json",summary)
        print(name,label,metrics,flush=True)
        return metrics,per,scores
    record("bm25",lambda q,t:sparse.search(t,k=20))
    record("bm25_chunked",lambda q,t:aggregate(sparse_chunk.search(t,k=100),mapping,20))
    models = json.loads((ROOT / "models.lock.json").read_text())
    dense_searches = []
    for mi, spec in enumerate(m for m in models if m["role"] == "embedding"):
        model_name = spec["id"].split("/")[-1]
        model = SentenceTransformer(str(ROOT / "data/models" / model_name), device="cpu", local_files_only=True)
        model.max_seq_length = 256
        ids = sorted(documents)
        qids = sorted(set(test) | set(development))
        corpus_matrix, corpus_info = encode(model,[documents[d] for d in ids],LOCAL/name/f"{model_name}-corpus.npy")
        query_matrix, query_info = encode(model,[queries[q]["text"] for q in qids],LOCAL/name/f"{model_name}-queries.npy")
        qmap = dict(zip(qids,query_matrix))
        summary["encoding"][model_name] = {"corpus":corpus_info,"queries":query_info,"max_seq_length":256,"normalization":"L2","revision":spec["revision"]}
        exact = Dense(ids,corpus_matrix,"exact")
        search = lambda q,t,idx=exact,qm=qmap:idx.search(qm[q],100)
        dense_searches.append(search)
        record(f"dense_{mi}",lambda q,t,s=search:s(q,t)[:20],details={"index_seconds":exact.build_seconds,"embedding_bytes":corpus_matrix.nbytes,"latency_scope":"Search only; precomputed query embeddings"})
        if mi == 0:
            primary_model, primary_exact, primary_qmap = model, exact, qmap
            for backend in ["faiss","qdrant"]:
                index = Dense(ids,corpus_matrix,backend,LOCAL/name/"qdrant")
                record(backend,lambda q,t,idx=index,qm=qmap:idx.search(qm[q],20),details={"index_seconds":index.build_seconds,"latency_scope":"Search only; precomputed query embeddings"})
                agreement=[]
                for q in sorted(test)[:30]:
                    a={d for d,_ in exact.search(qmap[q],20)}
                    b={d for d,_ in index.search(qmap[q],20)}
                    agreement.append(len(a & b)/20)
                summary["backend_checks"][backend]={"mean_top20_overlap":float(np.mean(agreement)),"queries":len(agreement)}
                index.close()
    # Fusion choices are selected using development judgments only.
    dense_search = dense_searches[0]
    fusion_dev=[]
    for constant in (10,60):
        for depth in (20,100):
            search=lambda q,t,c=constant,d=depth:rrf([sparse_search(q,t)[:d],dense_search(q,t)[:d]],constant=c,k=20)
            metrics,_,_=measure(search,queries,development,corpus_ids)
            fusion_dev.append({"constant":constant,"depth":depth,**metrics})
    chosen=max(fusion_dev,key=lambda x:x["ndcg@10"])
    summary["fusion_validation"]=fusion_dev
    hybrid=lambda q,t:rrf([sparse_search(q,t)[:chosen["depth"]],dense_search(q,t)[:chosen["depth"]]],constant=chosen["constant"],k=100)
    record("hybrid",lambda q,t:hybrid(q,t)[:20],details={"selected_constant":chosen["constant"],"selected_depth":chosen["depth"],"latency_scope":"BM25, cached dense query vector, and fusion"})
    record("expansion",lambda q,t:sparse.search(expand(t,sparse.search(t,3),documents),20))
    record("rewrite",lambda q,t:sparse.search(rewrite(t),20))
    record("multiquery",lambda q,t:rrf([sparse.search(t,100),sparse.search(rewrite(t),100),sparse.search(expand(t,sparse.search(t,3),documents),100)],k=20))
    # This filter is an ablation of document length, without an inferred topic label.
    allowed=[d for d,text in documents.items() if len(tokenize(text)) >= 20]
    record("length_filter",lambda q,t:sparse.search(t,20,candidates=allowed),details={"min_document_tokens":20,"eligible_documents":len(allowed)})
    for threshold in (.2,.4,.6):
        record(f"threshold_{threshold}",lambda q,t,s=threshold:[r for r in dense_search(q,t)[:20] if r[1]>=s],details={"threshold":threshold})
    spec=next(m for m in models if m["role"]=="reranking")
    cross=CrossEncoder(str(ROOT/"data/models"/spec["id"].split("/")[-1]),device="cpu",local_files_only=True,max_length=256)
    for depth in (20,50):
        def rerank(q,t,depth=depth):
            candidates=hybrid(q,t)[:depth]
            scores=cross.predict([(t,documents[d]) for d,_ in candidates],batch_size=64,show_progress_bar=False)
            return sorted([(d,float(s)) for (d,_),s in zip(candidates,scores)],key=lambda r:(-r[1],r[0]))[:20]
        record(f"rerank_{depth}",rerank,details={"candidate_depth":depth,"model_revision":spec["revision"],"latency_scope":"Candidate retrieval and cross-encoder scoring; cached dense query vector"})
    # Chunked dense retrieval is limited to a development-selected segmentation.
    if selected[0] != "whole":
        vectors,info=encode(primary_model,[c.text for c in items],LOCAL/name/"selected-chunks.npy")
        chunk_index=Dense([c.id for c in items],vectors,"faiss")
        record("dense_chunked",lambda q,t:aggregate(chunk_index.search(primary_qmap[q],100),mapping,20),details={"encoding":info})
    else:
        summary["dense_chunking_note"]="Development selected whole documents; dense segmentation equals the whole-document baseline."
    summary["storage_bytes"] = sum(p.stat().st_size for p in (LOCAL/name).rglob("*") if p.is_file())
    save(OUT/f"{name}.json",summary)
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["scifact", "fiqa"], help="Run one dataset; the default runs both.")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"]="1"
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    OUT.mkdir(parents=True,exist_ok=True)
    source_hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "src").rglob("*.py")}
    save(OUT/"environment.json",{"python":sys.version,"platform":platform.platform(),"processor":platform.processor(),"logical_cpus":os.cpu_count(),"seed":42,"dense_threads":min(12,os.cpu_count() or 1),"thread_selection":"Corpus-only 64-document throughput probe; no relevance judgments","source_sha256":source_hashes})
    for name in ([args.dataset] if args.dataset else ("scifact","fiqa")):
        run_dataset(name)


if __name__=="__main__":
    main()
