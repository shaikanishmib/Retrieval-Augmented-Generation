import hashlib
import json
import re
from pathlib import Path

import pytest

from rag_data.research import REQUIRED_RUNS

ROOT = Path(__file__).resolve().parents[1]


def test_complete_experiment_artifacts():
    paths=[ROOT/"reports/research"/f"{name}.json" for name in ("scifact","fiqa")]
    if not all(p.exists() for p in paths):
        pytest.skip("Run both dataset experiments to check the complete matrix")
    for name,path,count in zip(("scifact","fiqa"),paths,(300,648)):
        result=json.loads(path.read_text())
        assert REQUIRED_RUNS <= set(result["runs"])
        assert len(result["chunking"])==6
        assert len(result["encoding"])==2
        assert len(result["fusion_validation"])==4
        for row in result["runs"].values():
            assert row["query_count"]==count
            assert 0 <= row["ndcg@10"] <= 1
            assert row["p95_ms"]>=row["p50_ms"]>=0
        selected=max(result["chunking"],key=lambda r:r["ndcg@10"])
        assert result["selected_chunking"]==[selected["method"],selected["size"],selected["overlap"]]
        fusion=max(result["fusion_validation"],key=lambda r:r["ndcg@10"])
        assert result["runs"]["hybrid"]["selected_constant"]==fusion["constant"]
        assert result["runs"]["hybrid"]["selected_depth"]==fusion["depth"]


def test_model_revision_and_local_file_pins():
    lock=ROOT/"models.lock.json"
    if not lock.exists():
        pytest.skip("Acquire the pinned model snapshots")
    models=json.loads(lock.read_text())
    assert sum(m["role"]=="embedding" for m in models)==2
    assert sum(m["role"]=="reranking" for m in models)==1
    for model in models:
        assert re.fullmatch(r"[0-9a-f]{40}",model["revision"])
        for relative,expected in model["files"].items():
            path=ROOT/"data/models"/model["id"].split("/")[-1]/relative
            if path.exists():
                assert hashlib.file_digest(path.open("rb"),"sha256").hexdigest()==expected
