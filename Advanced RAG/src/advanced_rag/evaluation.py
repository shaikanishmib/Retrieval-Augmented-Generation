"""Compute answer, evidence, and selective prediction diagnostics."""
import numpy as np
from .pipeline import answer_scores

def score(row, prediction):
    em, f1 = answer_scores(prediction["answer"], row["answers"])
    gold = set(row["gold"])
    citations = set(prediction["citations"])
    selected = set(prediction["selected"])
    return {"em": em, "f1": f1, "recall3": len(gold & set(prediction["ranking"][:3])) / len(gold) if gold else None,
            "context_recall": len(gold & selected) / len(gold) if gold else None,
            "citation_precision": len(citations & gold) / len(citations) if citations else None,
            "citation_recall": len(citations & gold) / len(gold) if gold else None,
            "abstained": float(not prediction["answer"]), "answerable": float(bool(row["answers"])),
            "abstention_correct": float(bool(prediction["answer"]) == bool(row["answers"])),
            "support_overlap": prediction["support_overlap"], "total_ms": prediction["total_ms"],
            "retrieval_ms": prediction["retrieval_ms"], "input_tokens": prediction["input_tokens"], "output_tokens": prediction["output_tokens"]}

def aggregate(rows):
    result = {k: float(np.mean([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else None for k in rows[0]}
    result.update(n=len(rows), p50_ms=float(np.median([r["total_ms"] for r in rows])), p95_ms=float(np.percentile([r["total_ms"] for r in rows], 95)))
    answered = [r for r in rows if not r["abstained"]]
    result["selective_em"] = float(np.mean([r["em"] for r in answered])) if answered else None
    negatives = [r for r in rows if not r["answerable"]]
    result["unanswerable_recall"] = float(np.mean([r["abstained"] for r in negatives])) if negatives else None
    return result

def paired(a, b, replicates=2000):
    delta = np.asarray(b) - np.asarray(a)
    rng = np.random.default_rng(42)
    samples = delta[rng.integers(0, len(delta), size=(replicates, len(delta)))].mean(axis=1)
    return {"delta": float(delta.mean()), "low": float(np.quantile(samples, .025)), "high": float(np.quantile(samples, .975)), "replicates": replicates}
