"""Check evidence traces against original windows without trusting report scores."""
import hashlib
import json
from pathlib import Path
import pytest

from advanced_rag.data import load
from advanced_rag.pipeline import answer_scores, chunk
from advanced_rag.evaluation import score

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize("name", ["hotpot", "squad"])
def test_generated_trace_integrity(name):
    path = ROOT / f"data/experiments/{name}-scores.json"
    if not path.exists():
        pytest.skip("Run the full local matrix for generated-trace checks")
    config = json.loads((ROOT / "configs/study.json").read_text())
    _, test, _ = load(name, config["development_questions"], config["heldout_questions"])
    traces = json.loads((ROOT / f"data/experiments/{name}-traces.json").read_text())["heldout"]
    scores = json.loads(path.read_text())
    assert set(traces) == {r["id"] for r in test}
    for i, row in enumerate(test):
        documents, mapping = chunk(row["docs"])
        for method, prediction in traces[row["id"]]["predictions"].items():
            assert set(prediction["citations"]) <= set(prediction["selected"])
            assert set(prediction["selected"]) <= set(row["docs"])
            assert prediction["input_tokens"] <= (1024 if method == "corrective" else 512)
            for cid, text in zip(prediction["citation_chunks"], prediction["citation_text"]):
                assert documents[cid] == text
                assert mapping[cid] in prediction["citations"]
            assert score(row, prediction) == scores[method][i]

def test_executed_source_matches_current_package():
    path = ROOT / "reports/environment.json"
    if not path.exists():
        pytest.skip("Run experiments before checking source identity")
    for name, expected in json.loads(path.read_text())["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
