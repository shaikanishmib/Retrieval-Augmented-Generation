"""Repeat a small uncached workload with fixed seeds after model selection."""
import numpy as np
from advanced_rag.data import load
from advanced_rag.upgrade.artifacts import LOCAL, REPORTS, read, save
from advanced_rag.upgrade.models import Generator, Encoder, Reranker, Verifier
from advanced_rag.upgrade.retrieval import Index
from advanced_rag.upgrade.engine import Engine

selection = read(REPORTS / "selection.json")
cfg = dict(selection["selected_configuration"])
generator = Generator(cfg["generator"])
encoder = Encoder(cfg["encoder"])
verifier = Verifier()
reranker = Reranker(cfg["reranker"]) if cfg.get("reranker") else None
rows = [dict(row, dataset=name) for name in ("hotpot", "squad") for row in load(name, 64, 320)[0][:4]]
engines = {row["id"]: Engine(Index(row["docs"], encoder), generator, verifier, reranker) for row in rows}
passes = [[engines[row["id"]].answer(row["question"], cfg) for row in rows] for _ in range(2)]
matches = [all(a[k] == b[k] for k in ("answer", "claims", "citations", "valid")) for a, b in zip(*passes)]
save(LOCAL / "repeatability-predictions.json", passes)
save(REPORTS / "repeatability.json", {"scope": "Two uncached passes over eight development questions with fixed per-question seeds",
     "matching_predictions": sum(matches), "questions": len(rows),
     "p50_ms": [float(np.median([p["total_ms"] for p in values])) for values in passes],
     "p95_ms": [float(np.percentile([p["total_ms"] for p in values],95)) for values in passes],
     "model_load_ms": generator.load_ms, "cached_predictions_used": False})
print(sum(matches), "of", len(rows), "predictions match across fresh inference passes")
