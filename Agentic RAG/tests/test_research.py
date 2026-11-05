"""Research integrity checks against both fixtures and actual local artifacts."""
import copy
import json
from pathlib import Path
import pytest
from agentic_rag.artifacts import ROOT,read,digest,sha
from agentic_rag.evaluation import score,aggregate
from agentic_rag.tools import Index,Evidence
from agentic_rag.data import Task,task
from test_controller import controller,TASK

def test_unknown_tool_never_dispatches():
    index=Index(TASK)
    with pytest.raises(ValueError):
        index.execute("exec",query="malicious code")

def test_bm25_exact_term_ranking():
    assert Index(TASK).execute("bm25",query="England",k=1)[0].document=="London"

def test_provenance_stable_after_index_rebuild():
    assert Index(TASK).passages==Index(TASK).passages

def test_no_source_deployable_path_abstains():
    result=controller(['{"tool":"answer"}']).run(TASK,"agent")
    assert result["answer"]=="" and result["reason"]=="no_valid_source"

def test_no_retrieval_control_is_explicit():
    result=controller().run(TASK,"none")
    assert result["answer"]=="London" and result["state"]["calls"]==0
    assert not result["citation_valid"]

def test_invalid_plan_retry_bound():
    result=controller(["not json"]*100).run(TASK,"plan")
    assert result["state"]["invalid_actions"]==2
    assert result["reason"]=="no_valid_source" and not result["answer"]

def test_oracle_interface_requires_labels():
    with pytest.raises(ValueError):
        controller().run(TASK,"oracle")

def test_terminal_abstention_does_not_generate():
    result=controller(['{"tool":"abstain"}']).run(TASK,"agent")
    assert not result["answer"]
    assert len(result["state"]["generations"])==1

def test_tool_restriction_is_enforced():
    result=controller(['{"tool":"dense","query":"Ada"}']*5).run(TASK,"agent",allowed=["bm25"])
    assert result["state"]["calls"]==0

def test_empty_registry_has_no_search_capabilities():
    result=controller(['{"tool":"bm25","query":"Ada"}']*5).run(TASK,"agent",allowed=[])
    assert result["state"]["calls"]==0

def test_decoder_start_token_reserved_in_budget():
    from agentic_rag.controller import BudgetError,State
    engine=controller(max_tokens=60)
    with pytest.raises(BudgetError):
        engine.infer(engine.generator,'short',60,'generation',State())

def test_control_reason_survives_source_rejection():
    result=controller(['{"tool":"fetch","document":"absent"}']*4).run(TASK,"agent")
    assert result['control_reason']=='failure_limit' and result['reason']=='no_valid_source'

def test_plan_failures_are_counted_as_attempts():
    result=controller(['invalid']*4).run(TASK,'plan')
    assert result['state']['action_attempts']==2
    assert result['state']['invalid_actions']<=result['state']['action_attempts']

def test_factual_support_metric_uses_complete_sentence():
    row=dict(id="q",group="q",question=TASK.question,documents=TASK.documents,sentences=TASK.sentences,answers=["London"],gold=["Ada"],facts=[["Ada",0]],kind="bridge")
    result=controller().run(TASK,"fixed",retrieval="bm25")
    assert score(result,row)["supporting_fact_recall"]==1.
    for passage in result["context"]:
        passage["text"]="Ada is based"
    assert score(result,row)["supporting_fact_recall"]==0.

def test_original_annotations_never_enter_task():
    row=dict(id="q",question="q?",documents={"a":"b"},sentences={"a":["b"]},answers=["secret"],gold=["a"],facts=[["a",0]])
    assert "secret" not in json.dumps(task(row).__dict__)

def test_sampling_lock_matches_actual_partitions():
    path=ROOT / "reports/data-validation.json"
    if not path.exists():
        pytest.skip("Data stage required")
    for name,report in read(path).items():
        actual=read(ROOT / f"data/partitions/{name}.json")
        for split,locked in report["partitions"].items():
            assert digest(actual[split])==locked["hash"]
            assert [r["id"] for r in actual[split]]==locked["ids"]

def test_prior_project_preservation():
    path=ROOT / "reports/preservation.json"
    if not path.exists():
        pytest.skip("Audit required")
    for name,expected in read(path).items():
        assert sha(ROOT.parent / name)==expected,name

def test_public_notebook_identity_if_published():
    import nbformat
    path=ROOT / "notebooks/01_public_agentic_rag_research.ipynb"
    if not path.exists():
        pytest.skip("Publication stage required")
    full=nbformat.read(ROOT / "notebooks/01_complete_agentic_rag_research.ipynb",as_version=4)
    public=nbformat.read(path,as_version=4)
    assert [(c.cell_type,c.source) for c in full.cells]==[(c.cell_type,c.source) for c in public.cells]
    assert all(not c.get("outputs") and c.get("execution_count") is None for c in public.cells)
    assert all(c.execution_count is not None for c in full.cells if c.cell_type=="code")

def test_final_metrics_reconstruct_if_present():
    path=ROOT / "reports/metrics.json"
    if not path.exists():
        pytest.skip("Final report required")
    metrics=read(path)
    for name in ["hotpot","squad"]:
        rows={r["id"]:r for r in read(ROOT / f"data/partitions/{name}.json")["evaluation"]}
        manifest=read(ROOT / f"data/manifests/fixed-{name}-evaluation.json")
        actual=[score(read(ROOT / e["path"]),rows[e["id"]]) for e in manifest["rows"]]
        assert aggregate(actual)==metrics[f"fixed-{name}-evaluation"]
