"""Export deterministic citation cases for a separate support review."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cases = []
for name in ("hotpot", "squad"):
    path = ROOT / f"data/experiments/{name}-traces.json"
    if not path.exists():
        continue
    trace = json.loads(path.read_text())
    for qid, row in list(trace["heldout"].items())[:10]:
        p = row["predictions"]["hybrid"]
        cases.append({"dataset": name, "id": qid, "question": row["question"], "answer": p["answer"],
                      "citations": p["citations"], "citation_text": p.get("citation_text", []), "reference": row["references"]})
output = ROOT / "data/experiments/manual-review-cases.json"
output.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(cases, indent=2, ensure_ascii=False))
