import json
from advanced_rag.upgrade.engine import Engine

class Tokenizer:
    def encode(self, text):
        return text.split()

class Generator:
    alias = "test-generator"
    limit = 2048
    tokenizer = Tokenizer()
    def count(self, text):
        return len(text.split())
    def generate(self, text, seed=42, limit=128):
        if '"queries"' in text:
            result = {"queries": ["What is the capital of France?"]}
        else:
            result = {"answer": "Paris", "claims": [{"text": "The capital of France is Paris.", "sources": ["S1"]}]}
        return {"text": json.dumps(result), "input_tokens": 30, "output_tokens": 20, "generation_ms": 1., "truncated": False}

class Index:
    documents = {"a0": "France has Paris as its capital.", "b0": "France is in Europe."}
    mapping = {"a0": "a", "b0": "b"}
    def __init__(self):
        self.queries = []
    def search(self, question, method, reranker):
        self.queries.append(question)
        return [("a0", 2.), ("b0", 1.)], {"retrieval_ms": 1., "query_encoding_ms": 1., "reranking_ms": 0.}

class Verifier:
    def predict(self, pairs):
        return [.2] * len(pairs), 2.

def test_multistep_generates_and_executes_a_new_evidence_based_query():
    index = Index()
    result = Engine(index, Generator(), Verifier()).answer("Where is the capital of the country described?", {"reasoning": "multistep"})
    assert len(result["retrieval_steps"]) == 2
    assert index.queries[1] == "What is the capital of France?"
    assert index.queries[0] != index.queries[1]
    assert result["calls"][0]["kind"] == "plan"

def test_corrective_retry_uses_low_support_and_accounts_for_both_attempts():
    index = Index()
    result = Engine(index, Generator(), Verifier()).answer("Where is the capital of the country described?", {"reasoning": "corrective", "retry_threshold": .5})
    assert result["retried"]
    assert [c["kind"] for c in result["calls"]] == ["answer", "plan", "answer"]
    assert result["output_tokens"] == 60
    assert result["timings"]["generation_ms"] == 2.
    assert result["timings"]["planning_ms"] == 1.
    assert result["timings"]["verification_ms"] == 4.
