import json
import pytest
from advanced_rag.upgrade.grounding import parse, verify, gate

EVIDENCE = {"S1": {"text": "France has Paris as its capital.", "document_id": "France"},
            "S2": {"text": "Germany has Berlin as its capital.", "document_id": "Germany"}}

def response(sources=None):
    return json.dumps({"answer": "Paris", "claims": [{"text": "The capital of France is Paris.", "sources": sources or ["S1"]}]})

def test_explicit_citation_parser():
    result = parse(response(), EVIDENCE)
    assert result["valid"] and result["citations"] == ["S1"]
    assert parse("```json\n" + response() + "\n```", EVIDENCE)["valid"]

@pytest.mark.parametrize("text", ["Paris [S1]", "[]", "null", '{"answer":3,"claims":[]}', '{"answer":"Paris","claims":[]}',
                                '{"answer":"Paris","claims":[{"text":"Paris","sources":[]}]}'])
def test_malformed_outputs_are_rejected(text):
    result = parse(text, EVIDENCE)
    assert not result["valid"]
    assert not gate(dict(result, support=1.), 0.)["final_answer"]

def test_fabricated_citation_invalidates_whole_answer():
    result = parse(response(["S1", "S99"]), EVIDENCE)
    assert result["reason"] == "fabricated_citation"
    assert not result["answer"]

def test_answer_must_be_in_generated_claims():
    result = parse(response().replace('"answer": "Paris"', '"answer": "Berlin"'), EVIDENCE)
    assert not result["valid"] and result["reason"] == "answer_not_in_claims"

def test_model_abstention_is_valid_empty_response():
    result = parse('{"answer":"","claims":[]}', EVIDENCE)
    assert result["valid"] and result["reason"] == "model_abstention"

class FakeVerifier:
    def predict(self, pairs):
        self.pairs = pairs
        return [.9, .8, .2, .7, .6], 12.

def test_multisource_support_checks_joint_and_each_citation():
    parsed = {"answer": "Paris", "valid": True, "claims": [
        {"text": "Paris and Berlin are capitals.", "sources": ["S1", "S2"]},
        {"text": "Paris is in France.", "sources": ["S1"]}]}
    verifier = FakeVerifier()
    result = verify(parsed, EVIDENCE, verifier)
    assert len(verifier.pairs) == 5
    assert "Germany" in verifier.pairs[0][0]
    assert result["support"] == .7
    assert result["claim_scores"] == [.9, .7]
    assert result["citation_scores"] == [.8, .2, .6]
    assert result["verification_ms"] == 12

def test_unsupported_output_is_flagged_without_fabricating_a_source():
    parsed = parse(response(), EVIDENCE)
    result = gate(dict(parsed, support=.2), .5)
    assert result["answer"] == "Paris"
    assert result["final_answer"] == ""
    assert result["final_citations"] == []
    assert result["flag"] == "insufficient_support"

def test_raw_outputs_keep_a_low_support_flag_at_zero_cutoff():
    result = gate(dict(parse(response(), EVIDENCE), support=.2), 0.)
    assert result["final_answer"] == "Paris"
    assert result["flag"] == "low_model_support"
