"""Link the assistant's case judgments to the exact retained predictions."""
from collections import Counter
from advanced_rag.upgrade.artifacts import ROOT, LOCAL, REPORTS, read, save, fingerprint
from advanced_rag.upgrade.grounding import gate
from advanced_rag.answer_support import accept

cases = []
selection = read(REPORTS / "selection.json")
qa_selection = read(REPORTS / "answer-support-selection.json")["selection"]
adoption = read(REPORTS / "answer-support-adoption.json")["decisions"]
for name in ("hotpot", "squad"):
    original = read(LOCAL / "manifests" / f"selected_raw-{name}-original.json")
    paths = {r["id"]: ROOT / r["path"] for r in original["rows"]}
    question = read(LOCAL / "answer-support" / f"{name}-original-manifest.json")
    qa_paths = {r["id"]: ROOT / r["path"] for r in question["rows"]}
    for judgment in read(REPORTS / f"support-review-{name}.json")["cases"]:
        case = dict(judgment)
        raw = read(paths[case["id"]])["result"]
        final = gate(raw, selection["thresholds"][name]["selected"])
        assert bool(final["final_answer"]) == (case["final_label"] != "abstained")
        checked = read(qa_paths[case["id"]])
        qa = accept(checked, qa_selection[name]["agreement"], qa_selection[name]["support"])
        case["question_label"] = case["raw_label"] if qa["final_answer"] else "abstained"
        assert case["question_label"] != "invalid"
        case["default_label"] = case["question_label"] if adoption[name]["enabled"] else case["final_label"]
        reference = gate(raw, .5)
        case["reference_0.5_label"] = case["raw_label"] if reference["final_answer"] else "abstained"
        case["prediction_sha256"] = fingerprint({"answer_text": [c["text"] for c in raw["calls"] if c["kind"] == "answer"],
               "answer": raw["answer"], "claims": raw["claims"], "reason": raw["reason"],
               "cited_evidence": {s:raw["evidence"][s] for s in raw["citations"]}})
        if case["prediction_sha256"] != case["reviewed_prediction_sha256"]:
            raise ValueError(f"Reassess the changed prediction before reusing its review: {case['id']}")
        case["question_validation_sha256"] = fingerprint(checked)
        cases.append(case)
def summary(field):
    counts = dict(Counter(c[field] for c in cases))
    answered = sum(counts.get(k, 0) for k in ("supported", "partial", "unsupported"))
    return {"counts": counts, "answered": answered, "coverage": answered / len(cases),
            "strict_support_among_answered": counts.get("supported",0) / answered if answered else None}
report = {"reviewer": "Coding assistant; no independent human rater", "references_visible": True,
          "selection": "First ten original questions per dataset in deterministic study order, matched to the baseline review",
          "criterion": "Cited text must support the complete answer to the question. Extraneous citations and entity/answer-type errors are noted separately.",
          "scope": "Twenty exploratory matched cases, not a population faithfulness estimate",
          "baseline_counts": read(ROOT / "reports/manual-support-review.json")["counts"],
          "raw_counts": dict(Counter(c["raw_label"] for c in cases)), "final_counts": summary("final_label")["counts"],
          "frozen_claims_path": summary("final_label"), "question_check_path": summary("question_label"),
          "development_selected_default": summary("default_label"), "fixed_0.5_reference": summary("reference_0.5_label"), "cases": cases}
save(REPORTS / "support-review.json", report)
print({k:v for k,v in report.items() if k != "cases"})
