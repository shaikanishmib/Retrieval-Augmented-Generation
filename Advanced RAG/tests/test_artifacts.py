import hashlib
import csv
import json
from pathlib import Path

import nbformat
import pytest

from advanced_rag.data import load
from advanced_rag.evaluation import aggregate

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize("name", ["hotpot", "squad"])
def test_original_dataset_integrity_and_split_separation(name):
    if not (ROOT / f"data/raw/{name}.json").exists():
        pytest.skip("Acquire original data for the integration check")
    dev, test, info = load(name)
    assert info["original_questions"] == (7405 if name == "hotpot" else 11873)
    assert not {r["id"] for r in dev} & {r["id"] for r in test}
    assert not {r["group"] for r in dev} & {r["group"] for r in test}
    assert not {r["question"].strip().casefold() for r in dev} & {r["question"].strip().casefold() for r in test}

def test_download_pins():
    lock = ROOT / "sources.lock.json"
    if not lock.exists():
        pytest.skip("Acquire datasets before checking pins")
    for spec in json.loads(lock.read_text()):
        path = ROOT / f"data/raw/{spec['name']}.json"
        assert path.stat().st_size == spec["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == spec["sha256"]

def test_report_recalculation():
    path = ROOT / "reports/metrics.json"
    if not path.exists():
        pytest.skip("Execute the study before checking metrics")
    metrics = json.loads(path.read_text())
    for dataset, methods in metrics.items():
        scores = json.loads((ROOT / f"data/experiments/{dataset}-scores.json").read_text())
        for method, values in methods.items():
            assert values == aggregate(scores[method])

def test_notebook_identity_and_outputs():
    public = ROOT / "notebooks/01_public_advanced_rag_research.ipynb"
    if not public.exists():
        pytest.skip("Publish the notebook before checking equivalence")
    a = nbformat.read(ROOT / "notebooks/01_complete_advanced_rag_research.ipynb", as_version=4)
    b = nbformat.read(public, as_version=4)
    assert len(a.cells) >= 76
    assert [(c.cell_type, c.source) for c in a.cells] == [(c.cell_type, c.source) for c in b.cells]
    codes = [c for c in a.cells if c.cell_type == "code"]
    assert [c.execution_count for c in codes] == list(range(1, len(codes) + 1))
    assert not any(o.output_type == "error" for c in codes for o in c.outputs)
    assert all(c.execution_count is None and not c.outputs for c in b.cells if c.cell_type == "code")

def test_reference_unchanged():
    baseline = json.loads((ROOT / "reports/reference-audit.json").read_text())
    for name, info in baseline["files"].items():
        path = ROOT.parent / "Retrieval Optimization" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"]

def test_benchmark_table_matches_measured_values():
    if not (ROOT / "reports/metrics.csv").exists():
        pytest.skip("Build the benchmark report before reconciling tables")
    metrics = json.loads((ROOT / "reports/metrics.json").read_text())
    prose = (ROOT / "reports/benchmark.md").read_text()
    with (ROOT / "reports/metrics.csv").open() as stream:
        for row in csv.DictReader(stream):
            values = metrics[row["dataset"]][row["method"]]
            for key in ("em", "f1", "context_recall", "p50_ms"):
                assert float(row[key]) == values[key]
            line = f"| {row['dataset']} | {row['method']} | {values['em']:.4f} | {values['f1']:.4f} | {values['context_recall']:.4f} | {values['p50_ms']:.2f} |"
            assert line in prose

def test_model_file_hashes():
    path = ROOT / "models.lock.json"
    if not path.exists():
        pytest.skip("Acquire model snapshots before checking hashes")
    for spec in json.loads(path.read_text()):
        assert len(spec["revision"]) == 40
        for relative, expected in spec["files"].items():
            with (ROOT / "data/models" / spec["role"] / relative).open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == expected

def test_manual_review_trace_hashes():
    path = ROOT / "reports/manual-review-provenance.json"
    if not path.exists():
        pytest.skip("Complete the qualitative review before checking its provenance")
    for item in json.loads(path.read_text()):
        traces = json.loads((ROOT / f"data/experiments/{item['dataset']}-traces.json").read_text())
        p = traces["heldout"][item["id"]]["predictions"]["hybrid"]
        content = json.dumps({k: p[k] for k in ("answer", "citations", "citation_chunks", "citation_text")}, sort_keys=True, ensure_ascii=False).encode()
        assert hashlib.sha256(content).hexdigest() == item["prediction_evidence_sha256"]
