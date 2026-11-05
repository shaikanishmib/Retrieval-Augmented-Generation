"""Check annotated answer-span availability separately from source-document recall."""
import json
import re
import numpy as np
from advanced_rag.upgrade.artifacts import ROOT, LOCAL, REPORTS, read, save
from advanced_rag.upgrade.grounding import gate

def span_coverage(context, answers, result, threshold=0.):
    words = list(re.finditer(r"\S+", context))
    accepted = gate(result, threshold)
    def covers(evidence):
        if not answers:
            return None
        for item in evidence:
            _, start = json.loads(item["chunk_id"])
            end = min(start + 100, len(words))
            if start >= end:
                continue
            a, b = words[start].start(), words[end - 1].end()
            if any(a <= answer["answer_start"] and answer["answer_start"] + len(answer["text"]) <= b for answer in answers):
                return 1.
        return 0.
    return {"selected_span_coverage": covers(result["evidence"].values()),
            "cited_span_coverage": covers(result["evidence"][s] for s in accepted["final_citations"])}

def main():
    annotations = {}
    for article in read(ROOT / "data/raw/squad.json")["data"]:
        for paragraph in article["paragraphs"]:
            for question in paragraph["qas"]:
                annotations[question["id"]] = (paragraph["context"], question["answers"])
    metrics = read(REPORTS / "metrics.json")
    threshold = read(REPORTS / "selection.json")["thresholds"]["squad"]["selected"]
    output = {"definition": "Fraction of answerable questions for which at least one original annotated character span lies wholly inside a selected or cited window. This measures evidence availability, not generated-claim entailment.", "results": {}}
    for split in ("fresh", "controlled", "original"):
        for label in metrics[split]["squad"]:
            path = LOCAL / "manifests" / f"{label}-squad-{split}.json"
            if not path.exists():
                continue
            values, verified = [], []
            for item in read(path)["rows"]:
                context, answers = annotations[item["id"]]
                result = read(ROOT / item["path"])["result"]
                values.append(span_coverage(context, answers, result))
                verified.append(span_coverage(context, answers, result, threshold))
            selected = [v['selected_span_coverage'] for v in values if v['selected_span_coverage'] is not None]
            cited = [v['cited_span_coverage'] for v in values if v['cited_span_coverage'] is not None]
            item = {"n": len(values), "answerable_n": len(selected), "selected_span_coverage": float(np.mean(selected)), "raw_cited_span_coverage": float(np.mean(cited))}
            if label == "selected_raw":
                item["verified_cited_span_coverage"] = float(np.mean([v['cited_span_coverage'] for v in verified if v['cited_span_coverage'] is not None]))
            output["results"][f"{split}/{label}"] = item
    save(REPORTS / "squad-answer-evidence.json", output)
    print(output)

if __name__ == "__main__":
    main()
