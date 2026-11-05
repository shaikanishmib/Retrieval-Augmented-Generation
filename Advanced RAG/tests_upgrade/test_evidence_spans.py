import importlib.util
import json
from advanced_rag.upgrade.artifacts import ROOT

def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

def prediction(evidence):
    return {"answer": "w105", "claims": [], "valid": True, "reason": "parsed", "support": .9,
            "citations": ["S1"], "evidence": evidence}

def test_answer_span_can_be_selected_but_missing_from_cited_window():
    words = [f"w{i}" for i in range(120)]
    context = " ".join(words)
    evidence = {"S1": {"chunk_id": json.dumps(["doc",0]), "text": " ".join(words[:100]), "document_id":"doc"},
                "S2": {"chunk_id": json.dumps(["doc",80]), "text": " ".join(words[80:]), "document_id":"doc"}}
    answers = [{"text": "w105", "answer_start": context.index("w105")}]
    result = module("analyze_squad_spans").span_coverage(context, answers, prediction(evidence))
    assert result == {"selected_span_coverage": 1., "cited_span_coverage": 0.}

def test_sentence_diagnostic_records_invalid_annotation_without_repair():
    original = {"context": [["doc", ["First fact.", " Second fact."]]], "supporting_facts": [["doc",0],["doc",1],["doc",902]]}
    evidence = {"S1": {"text": "First fact.", "document_id":"doc"}}
    result = module("analyze_sentence_evidence").containment(original, prediction(evidence))
    assert result["gold_sentences"] == 2
    assert result["invalid_annotations"] == 1
    assert result["selected_sentence_recall"] == .5
