import json
from advanced_rag.answer_support import check, accept

def raw(sources=None):
    return {"evidence": {"S1": {"text":"Both drinks contain alcohol.","document_id":"drinks"}},
            "calls": [{"kind":"answer","text":json.dumps({"answer":"yes","claims":[{"text":"Both drinks contain alcohol.","sources":sources or ["S1"]}]})}]}

class QA:
    limit = 512
    def __init__(self, answer="yes", length=100):
        self.answer, self.length, self.calls = answer, length, 0
    def count(self, text):
        return self.length
    def generate(self, text, limit=32):
        self.calls += 1
        return {"text":self.answer,"generation_ms":1.,"input_tokens":100,"output_tokens":2}

class NLI:
    def predict(self, pairs):
        return [.99] * len(pairs), 1.

def test_yes_answer_can_be_bound_through_question_conditioned_evidence():
    result = check("Do both drinks contain alcohol?",raw(),QA(),NLI())
    assert result["reason"] == "requires_question_check"
    assert result["qa_agreement"] == 1
    assert accept(result)["final_answer"] == "yes"

def test_question_disagreement_rejects_an_otherwise_entailed_claim():
    result = check("Do both drinks contain alcohol?",raw(),QA("no"),NLI())
    assert result["support"] == .99
    assert accept(result)["flag"] == "qa_disagreement"
    assert not accept(result)["final_answer"]

def test_fabricated_citation_never_reaches_qa_model():
    qa = QA()
    result = check("Question",raw(["S99"]),qa,NLI())
    assert qa.calls == 0
    assert accept(result)["flag"] == "fabricated_citation"

def test_complete_cited_context_must_fit_verifier_budget():
    qa = QA(length=1000)
    result = check("Question",raw(),qa,NLI())
    assert qa.calls == 0
    assert accept(result)["flag"] == "qa_context_limit"

def test_qa_unknown_can_reject_high_nli_support():
    result = check("Question",raw(),QA("unknown"),NLI())
    assert result["support"] == .99
    assert accept(result)["flag"] == "qa_evidence_insufficient"
