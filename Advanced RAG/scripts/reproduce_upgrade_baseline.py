"""Reproduce old answers and extend unchanged baseline methods to fresh questions."""
import time
from advanced_rag.data import load
from advanced_rag.pipeline import Models, Pipeline
from advanced_rag.evaluation import score, aggregate
from advanced_rag.upgrade.artifacts import ROOT, LOCAL, REPORTS, fingerprint, read, save, preserved

preserved()
models = Models.load(ROOT)
results = {}
for name in ("hotpot", "squad"):
    _, rows, _ = load(name, development=64, heldout=320)
    original = read(ROOT / f"data/experiments/{name}-traces.json")["heldout"]
    agreement = {method: [] for method in ("hybrid", "rerank")}
    expanded = {method: [] for method in ("bm25", "dense", "hybrid", "rerank")}
    for i, row in enumerate(rows):
        key = fingerprint({"row": row, "stage": "baseline_cpu_reproduction_v1"})
        path = LOCAL / "baseline" / f"{key}.json"
        if path.exists():
            record = read(path)
        else:
            started = time.perf_counter()
            pipeline = Pipeline(models, row["docs"])
            build = (time.perf_counter() - started) * 1000
            methods = ("hybrid", "rerank") if i < 64 else tuple(expanded)
            record = {"id": row["id"], "indexing_ms": build, "predictions": {m: pipeline.run(row["question"], m) for m in methods}}
            save(path, record)
        if i < 64:
            assert row["id"] in original
            for method in agreement:
                agreement[method].append(record["predictions"][method]["answer"] == original[row["id"]]["predictions"][method]["answer"])
        else:
            for method in expanded:
                expanded[method].append(score(row, record["predictions"][method]))
        if (i + 1) % 32 == 0:
            print(name, i + 1, len(rows), flush=True)
    results[name] = {"original_overlap_questions": 64, "fresh_questions": 256,
                     "answer_exact_reproduction": {method: {"matching": sum(v), "total": len(v)} for method, v in agreement.items()},
                     "fresh_metrics": {method: aggregate(values) for method, values in expanded.items()}}
    save(REPORTS / "baseline-reproduction.json", results)
preserved()
