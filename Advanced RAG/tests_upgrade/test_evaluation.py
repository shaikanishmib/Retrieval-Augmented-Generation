import pytest
from advanced_rag.upgrade.artifacts import fingerprint
from advanced_rag.upgrade.evaluate import calibration, cluster_interval
from advanced_rag.upgrade.evaluate import aggregate, per_question
from advanced_rag.upgrade.retrieval import diverse_order

def test_fingerprint_is_order_independent_and_content_sensitive():
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})

def test_diversity_orders_sources_before_repeated_windows():
    rows = [("a0", 3), ("a1", 2), ("b0", 1)]
    assert diverse_order(rows, {"a0": "a", "a1": "a", "b0": "b"}) == [rows[0], rows[2], rows[1]]

def test_cluster_interval_keeps_within_group_dependence():
    result = cluster_interval([0] * 6, [1] * 6, ["a"] * 4 + ["b"] * 2, 100)
    assert result["clusters"] == 2
    assert result["delta"] == result["low"] == result["high"] == 1
    with pytest.raises(ValueError):
        cluster_interval([], [], [])

def test_brier_and_calibration_hand_calculation():
    result = calibration([.2, .8], [0, 1])
    assert result["brier"] == pytest.approx(.04)
    assert result["ece10"] == pytest.approx(.2)
    assert sum(b["n"] for b in result["bins"]) == 2

def test_abstention_metrics_keep_answerable_accuracy_and_coverage_separate():
    prediction = {"answer": "Paris", "valid": True, "reason": "parsed", "support": .9,
                  "claims": [{"text": "Paris is the capital of France.", "sources": ["S1"]}], "citations": ["S1"],
                  "evidence": {"S1": {"document_id": "France", "text": "Paris is the capital of France."}},
                  "ranking": ["France"], "selected": ["France"], "claim_scores": [.9], "citation_scores": [.9],
                  "retried": False, "retrieval_steps": [{}], "input_tokens": 10, "output_tokens": 5, "total_ms": 4.,
                  "timings": {"generation_ms": 2., "planning_ms": 0.}}
    positive = {"id": "a", "group": "article-a", "kind": "answerable", "answers": ["Paris"], "gold": ["France"]}
    negative = {"id": "b", "group": "article-b", "kind": "unanswerable", "answers": [], "gold": []}
    rows = [per_question(positive, prediction, .5), per_question(negative, dict(prediction, support=.2), .5)]
    metrics = aggregate(rows)
    assert metrics["coverage"] == .5
    assert metrics["answerable_f1"] == 1
    assert metrics["unanswerable_recall"] == metrics["abstention_precision"] == 1
    assert metrics["selective_risk"] == 0
