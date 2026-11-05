"""Export matched original questions and cited windows for an explicit assistant case review."""
import argparse
from advanced_rag.data import load
from advanced_rag.upgrade.artifacts import ROOT, LOCAL, REPORTS, read, save
from advanced_rag.upgrade.grounding import gate

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", choices=["hotpot", "squad"])
args = parser.parse_args()
selection = read(REPORTS / "selection.json")
cases = []
for name in ([args.dataset] if args.dataset else ("hotpot", "squad")):
    _, rows, _ = load(name, 64, 320)
    manifest = read(LOCAL / "manifests" / f"selected_raw-{name}-original.json")
    for row, item in list(zip(rows[:64], manifest["rows"]))[:10]:
        assert row["id"] == item["id"]
        raw = read(ROOT / item["path"])["result"]
        final = gate(raw, selection["thresholds"][name]["selected"])
        cases.append({"dataset": name, "id": row["id"], "question": row["question"], "references": row["answers"], "gold_sources": row["gold"],
                      "raw_answer": raw["answer"], "final_answer": final["final_answer"], "flag": final["flag"],
                      "support": raw["support"], "claims": raw["claims"], "cited_windows": {s: raw["evidence"][s] for s in raw["citations"]},
                      "answer_calls": [c["text"] for c in raw["calls"] if c["kind"] == "answer"]})
save(LOCAL / ("review-cases" + ("-" + args.dataset if args.dataset else "") + ".json"), cases)
import json
print(json.dumps(cases, indent=2, ensure_ascii=False))
