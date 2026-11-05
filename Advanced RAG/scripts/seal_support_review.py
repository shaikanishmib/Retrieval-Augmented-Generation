"""Bind the completed assistant judgments to the reviewed answer and cited text."""
from advanced_rag.upgrade.artifacts import ROOT, LOCAL, REPORTS, read, save, fingerprint

for name in ("hotpot", "squad"):
    manifest = read(LOCAL / "manifests" / f"selected_raw-{name}-original.json")
    paths = {row["id"]: ROOT / row["path"] for row in manifest["rows"]}
    path = REPORTS / f"support-review-{name}.json"
    review = read(path)
    for case in review["cases"]:
        raw = read(paths[case["id"]])["result"]
        semantic = {"answer_text": [c["text"] for c in raw["calls"] if c["kind"] == "answer"], "answer": raw["answer"],
                    "claims": raw["claims"], "reason": raw["reason"], "cited_evidence": {s:raw["evidence"][s] for s in raw["citations"]}}
        observed = fingerprint(semantic)
        if case.get("reviewed_prediction_sha256", observed) != observed:
            raise ValueError(f"Review belongs to a changed prediction: {case['id']}")
        case["reviewed_prediction_sha256"] = observed
    save(path, review)
print("Twenty judgments sealed to their reviewed predictions")
