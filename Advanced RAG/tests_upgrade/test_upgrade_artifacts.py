import csv
import hashlib
import json
from pathlib import Path
import nbformat
import pytest

from advanced_rag.data import load
from advanced_rag.upgrade.artifacts import ROOT, LOCAL, REPORTS, read, sources, preserved
from advanced_rag.upgrade.evaluate import aggregate, per_question

def test_complete_baseline_preservation():
    preserved()

def test_fresh_sample_excludes_original_and_development_groups():
    for name in ("hotpot", "squad"):
        dev, test, info = load(name, 64, 320)
        assert len(dev) == 64 and len(test) == 320
        assert not {r["id"] for r in test[:64]} & {r["id"] for r in test[64:]}
        assert not {r["group"] for r in dev} & {r["group"] for r in test}
        assert not {r["question"].strip().casefold() for r in dev} & {r["question"].strip().casefold() for r in test}

def test_development_pilot_did_not_evaluate_fresh_questions():
    assert set(read(REPORTS / "development-pilot/metrics.json")) == {"development"}

def test_final_source_identity():
    assert read(REPORTS / "experiment-identity.json")["source_sha256"] == sources()

def test_final_generated_trace_metrics_and_citations():
    path = REPORTS / "resources.json"
    if not path.exists():
        pytest.skip("Complete the study before checking every final trace")
    metrics = read(REPORTS / "metrics.json")
    for split, datasets in metrics.items():
        for name, methods in datasets.items():
            dev, all_test, _ = load(name, 64, 320)
            lookup = {r["id"]: r for r in dev + all_test}
            for label, expected in methods.items():
                manifest_path = LOCAL / "manifests" / f"{label}-{name}-{split}.json"
                if not manifest_path.exists():
                    assert label in {"selected_verified", "selected_lexical"}
                    continue
                manifest = read(manifest_path)
                scores = []
                for item in manifest["rows"]:
                    record = read(ROOT / item["path"])
                    p = record["result"]
                    row = lookup[item["id"]]
                    assert set(p["citations"]) <= p["evidence"].keys()
                    assert all(s in p["evidence"] for c in p["claims"] for s in c["sources"])
                    assert set(p["selected"]) <= row["docs"].keys()
                    for evidence in p["evidence"].values():
                        did, start = json.loads(evidence["chunk_id"])
                        assert did == evidence["document_id"]
                        assert evidence["text"] == " ".join(row["docs"][did].split()[start:start + 100])
                    assert p["input_tokens"] > 0 and p["output_tokens"] > 0
                    assert p["total_ms"] >= 0
                    assert p["timings"]["generation_ms"] + p["timings"]["planning_ms"] <= p["total_ms"] + 1
                    scores.append(per_question(row, p))
                assert aggregate(scores) == expected

def test_calibration_groups_and_fixed_thresholds():
    selection = read(REPORTS / "selection.json")
    if not selection.get("thresholds"):
        pytest.skip("Select thresholds on development data before this check")
    for name, values in selection["thresholds"].items():
        dev, test, _ = load(name, 64, 320)
        lookup = {r["id"]: r for r in dev}
        fit = {lookup[q]["group"] for q in values["fit_ids"]}
        tune = {lookup[q]["group"] for q in values["threshold_ids"]}
        assert not fit & tune
        assert set(values["fit_ids"] + values["threshold_ids"]) <= lookup.keys()
        assert values["selected"] == max(values["trials"], key=lambda r:r["f1"])["threshold"]

def test_upgrade_notebook_equivalence_and_preservation():
    public = ROOT / "notebooks/02_public_grounding_upgrade.ipynb"
    if not public.exists():
        pytest.skip("Execute and publish the upgrade notebook before this check")
    local = ROOT / "notebooks/02_complete_grounding_upgrade.ipynb"
    a, b = [nbformat.read(path, as_version=4) for path in (local, public)]
    assert len(a.cells) >= 90
    assert [(c.cell_type, c.source) for c in a.cells] == [(c.cell_type, c.source) for c in b.cells]
    code = [c for c in a.cells if c.cell_type == "code"]
    assert [c.execution_count for c in code] == list(range(1, len(code) + 1))
    assert not any(o.output_type == "error" for c in code for o in c.outputs)
    assert all(not c.outputs and c.execution_count is None for c in b.cells if c.cell_type == "code")
    assert read(REPORTS / "notebook-verification.json")["local_sha256"] == hashlib.sha256(local.read_bytes()).hexdigest()
    preserved()

def test_upgrade_model_pins():
    for alias, spec in read(ROOT / "models.upgrade.lock.json").items():
        assert len(spec["revision"]) == 40
        for relative, value in spec["files"].items():
            path = LOCAL / "models" / alias / relative
            assert path.stat().st_size == value["bytes"]
            with path.open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == value["sha256"]

def test_published_csv_reconciles():
    path = REPORTS / "metrics.csv"
    if not path.exists():
        pytest.skip("Analyze the completed study before checking publication tables")
    metrics = read(REPORTS / "metrics.json")
    with path.open() as stream:
        for row in csv.DictReader(stream):
            expected = metrics[row["split"]][row["dataset"]][row["method"]]
            for key, value in expected.items():
                if value is not None:
                    assert float(row[key]) == pytest.approx(value, abs=1e-12)

def test_practical_model_was_selected_on_development():
    path = REPORTS / "practical-baseline.json"
    if not path.exists():
        pytest.skip("Complete the practical baseline comparison before this check")
    report = read(path)
    assert report["generator"] == max(read(REPORTS / "selection.json")["plain_generators"], key=lambda r:r["f1"])["generator"]
    from advanced_rag.evaluation import aggregate as legacy_aggregate
    for name, splits in report["datasets"].items():
        scores = read(LOCAL / "practical" / f"{name}-scores.json")
        for split, methods in splits.items():
            for method, expected in methods.items():
                assert expected["n"] == (256 if split == "fresh" else 64)
                assert legacy_aggregate(scores[split][method]) == expected

def test_question_conditioned_results_reconcile():
    path = REPORTS / "answer-support-results.json"
    if not path.exists():
        pytest.skip("Complete the question-conditioned check before this audit")
    import importlib.util
    spec = importlib.util.spec_from_file_location("support_evaluation", ROOT / "scripts/evaluate_answer_support.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    selection = read(REPORTS / "answer-support-selection.json")
    assert hashlib.sha256((ROOT / "src/advanced_rag/answer_support.py").read_bytes()).hexdigest() == selection["source_sha256"]
    from advanced_rag.pipeline import answer_scores
    for name, splits in read(path)["results"].items():
        _, test, _ = load(name,64,320)
        cfg = selection["selection"][name]
        for split, rows in (("fresh",test[64:]),("original",test[:64])):
            manifest = read(LOCAL / "answer-support" / f"{name}-{split}-manifest.json")
            checked = [read(ROOT / r["path"]) for r in manifest["rows"]]
            assert [r["id"] for r in manifest["rows"]] == [r["id"] for r in rows]
            for result in checked:
                if result["qa_checked"]:
                    expected = answer_scores(result["answer"],[result["qa_answer"]])[1] if result["qa_answer"] else 0.
                    assert result["qa_agreement"] == expected
            actual, _ = module.quality(rows,checked,cfg["agreement"],cfg["support"])
            for key, value in actual.items():
                assert splits[split][key] == value

def test_question_check_adoption_uses_only_development():
    path = REPORTS / "answer-support-adoption.json"
    if not path.exists():
        pytest.skip("Select the optional question check before auditing adoption")
    for name, values in read(path)["decisions"].items():
        assert values["enabled"] == (values["question_check_development_f1"] > values["original_development_f1"] + 1e-12)

def test_support_review_is_sealed_to_actual_evidence():
    path = REPORTS / "support-review.json"
    if not path.exists():
        pytest.skip("Finish the case review before checking its hashes")
    from advanced_rag.upgrade.artifacts import fingerprint
    review = read(path)
    assert len(review["cases"]) == 20
    for name in ("hotpot","squad"):
        manifest = read(LOCAL / "manifests" / f"selected_raw-{name}-original.json")
        paths = {r["id"]: ROOT / r["path"] for r in manifest["rows"]}
        for case in [c for c in review["cases"] if c["dataset"]==name]:
            raw = read(paths[case["id"]])["result"]
            semantic = {"answer_text":[c["text"] for c in raw["calls"] if c["kind"]=="answer"],"answer":raw["answer"],
                        "claims":raw["claims"],"reason":raw["reason"],"cited_evidence":{s:raw["evidence"][s] for s in raw["citations"]}}
            assert fingerprint(semantic) == case["reviewed_prediction_sha256"] == case["prediction_sha256"]
