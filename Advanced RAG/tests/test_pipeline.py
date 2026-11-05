import math

import numpy as np
import pytest

from advanced_rag.pipeline import BM25, answer_scores, chunk, fuse, select_context
from advanced_rag.evaluation import aggregate, paired, score

def test_bm25_formula():
    index = BM25({"a": "bond bond", "b": "stock"})
    assert index.search("bond")[0][1] == pytest.approx(math.log(2) * 2 * 1.9 / (2 + .9 * (.6 + .4 * 2 / 1.5)))
    assert index.search("absent") == [("a", 0), ("b", 0)]

def test_chunk_source_and_coverage():
    docs = {"a::0": " ".join(str(i) for i in range(23)), "empty": ""}
    chunks, mapping = chunk(docs, 7, 2)
    assert set(mapping.values()) == set(docs)
    assert max(len(t.split()) for t in chunks.values()) <= 7
    assert set(docs["a::0"].split()) == {w for cid, text in chunks.items() if mapping[cid] == "a::0" for w in text.split()}

@pytest.mark.parametrize("size,overlap", [(0, 0), (3, 3), (3, -1)])
def test_bad_chunk_settings(size, overlap):
    with pytest.raises(ValueError):
        chunk({"a": "text"}, size, overlap)

def test_fusion_deduplicates():
    assert dict(fuse([("a", 1), ("a", .5)], [("a", 2)]))["a"] == pytest.approx(2 / 61)

def test_official_style_answer_normalization():
    assert answer_scores("The United States", ["United States"]) == (1, 1)
    assert answer_scores("red blue", ["red green"]) == (.0, .5)
    assert answer_scores("", []) == (1, 1)

class Tokenizer:
    def encode(self, text, **kwargs):
        return text.split()

def test_context_budget_includes_labels_and_skips_long_chunks():
    selected, used = select_context([("a", 3), ("b", 2), ("c", 1)], {"a": "too " * 20, "b": "short", "c": "next"}, Tokenizer(), 4)
    assert [d for d, _ in selected] == ["b", "c"]
    assert used == 4

def test_paired_bootstrap_direction():
    result = paired([0, 0, 0], [1, 1, 1], 100)
    assert result["low"] == result["high"] == result["delta"] == 1

def test_citation_metrics_do_not_treat_overlap_as_entailment():
    row = {"answers": ["red"], "gold": ["a", "b"]}
    p = dict(answer="red", citations=["a"], selected=["a"], ranking=["b", "a"], support_overlap=1., total_ms=2, retrieval_ms=1, input_tokens=10, output_tokens=2)
    values = score(row, p)
    assert values["citation_precision"] == 1
    assert values["citation_recall"] == .5
    assert "entailment" not in values
    assert aggregate([values])["selective_em"] == 1
