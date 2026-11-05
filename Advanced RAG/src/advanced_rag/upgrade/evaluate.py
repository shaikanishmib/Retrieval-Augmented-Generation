"""Evaluate grounding diagnostics, abstention, calibration, and paired changes."""
import numpy as np
from ..pipeline import answer_scores, tokens
from .grounding import gate

def per_question(row, result, threshold=0.):
    p = gate(result, threshold)
    em, f1 = answer_scores(p["final_answer"], row["answers"])
    raw_em, raw_f1 = answer_scores(p["answer"], row["answers"])
    gold = set(row["gold"])
    cited = {p["evidence"][s]["document_id"] for s in p["final_citations"]}
    answerable = bool(row["answers"])
    answered = bool(p["final_answer"])
    return {"id": row["id"], "group": row["group"], "kind": row["kind"], "em": em, "f1": f1,
            "raw_em": raw_em, "raw_f1": raw_f1, "answerable": answerable, "answered": answered,
            "retrieval_recall3": len(gold & set(p["ranking"][:3])) / len(gold) if gold else None,
            "evidence_recall": len(gold & set(p["selected"])) / len(gold) if gold else None,
            "citation_source_precision": len(gold & cited) / len(cited) if cited else None,
            "citation_source_recall": len(gold & cited) / len(gold) if gold else None,
            "citation_count": len(cited), "valid": p["valid"], "support": p["support"],
            "nli_claim_support_mean": float(np.mean(p["claim_scores"])) if p["claim_scores"] else None,
            "nli_citation_support_mean": float(np.mean(p["citation_scores"])) if p["citation_scores"] else None,
            "claims": len(p["claims"]), "retried": p["retried"], "retrieval_steps": len(p["retrieval_steps"]),
            "input_tokens": p["input_tokens"], "output_tokens": p["output_tokens"], "total_ms": p["total_ms"], **p["timings"]}

def mean(values):
    present = [v for v in values if v is not None]
    return float(np.mean(present)) if present else None

def aggregate(rows):
    numeric = [k for k in rows[0] if k not in {"id", "group", "kind"}]
    report = {k: mean([r[k] for r in rows]) for k in numeric}
    positives = [r for r in rows if r["answerable"]]
    negatives = [r for r in rows if not r["answerable"]]
    abstained = [r for r in rows if not r["answered"]]
    answered = [r for r in rows if r["answered"]]
    report.update(n=len(rows), groups=len({r["group"] for r in rows}), coverage=len(answered) / len(rows),
                  answerable_em=mean([r["em"] for r in positives]), answerable_f1=mean([r["f1"] for r in positives]),
                  unanswerable_recall=mean([not r["answered"] for r in negatives]),
                  abstention_precision=mean([not r["answerable"] for r in abstained]),
                  selective_risk=mean([1 - r["em"] for r in answered]),
                  p50_ms=float(np.median([r["total_ms"] for r in rows])), p95_ms=float(np.percentile([r["total_ms"] for r in rows], 95)),
                  questions_per_second=1000 * len(rows) / sum(r["total_ms"] for r in rows),
                  generated_tokens_per_second=1000 * sum(r["output_tokens"] for r in rows) / max(1e-9, sum(r["generation_ms"] + r["planning_ms"] for r in rows)))
    return report

def cluster_interval(a, b, groups, replicates=2000):
    a, b = np.asarray(a), np.asarray(b)
    if not len(a) or len(a) != len(b) or len(a) != len(groups):
        raise ValueError("Paired values and groups must have equal nonzero length")
    labels = sorted(set(groups))
    delta = b - a
    sums = np.asarray([delta[np.asarray(groups) == g].sum() for g in labels])
    counts = np.asarray([sum(x == g for x in groups) for g in labels])
    sampled = np.random.default_rng(42).integers(0, len(labels), size=(replicates, len(labels)))
    values = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return {"delta": float(delta.mean()), "low": float(np.quantile(values, .025)), "high": float(np.quantile(values, .975)),
            "clusters": len(labels), "replicates": replicates, "unit": "article for SQuAD; question for HotpotQA"}

def calibration(probabilities, outcomes):
    probabilities, outcomes = np.asarray(probabilities), np.asarray(outcomes)
    if len(probabilities) != len(outcomes) or not len(outcomes):
        raise ValueError("Calibration requires paired nonempty probabilities and outcomes")
    bins, ece = [], 0.
    for i in range(10):
        mask = (probabilities >= i / 10) & ((probabilities < (i + 1) / 10) if i < 9 else (probabilities <= 1))
        if mask.any():
            confidence, accuracy = probabilities[mask].mean(), outcomes[mask].mean()
            ece += mask.mean() * abs(confidence - accuracy)
            bins.append({"lower": i / 10, "n": int(mask.sum()), "confidence": float(confidence), "accuracy": float(accuracy)})
    return {"brier": float(np.mean((probabilities - outcomes) ** 2)), "ece10": float(ece), "bins": bins,
            "target": "raw answer exact correctness, not entailment"}
