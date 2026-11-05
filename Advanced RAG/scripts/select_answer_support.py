"""Choose the optional question check from development results alone."""
from advanced_rag.upgrade.artifacts import REPORTS, read, save

question = read(REPORTS / "answer-support-selection.json")
original = read(REPORTS / "selection.json")
decisions = {}
for name, value in question["selection"].items():
    previous = original["thresholds"][name]
    baseline = next(r for r in previous["trials"] if r["threshold"] == previous["selected"])
    selected = next(r for r in value["trials"] if r["agreement"] == value["agreement"] and r["support"] == value["support"])
    assert set(value["threshold_ids"]) == set(previous["threshold_ids"])
    decisions[name] = {"enabled": selected["f1"] > baseline["f1"] + 1e-12, "original_development_f1": baseline["f1"],
                       "question_check_development_f1": selected["f1"], "agreement": value["agreement"], "support": value["support"]}
save(REPORTS / "answer-support-adoption.json", {"criterion": "Strictly higher development threshold-group F1; no held-out result used",
     "validation_identity": question["identity"], "decisions": decisions})
print(decisions)
