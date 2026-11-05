"""Measure exact supporting-sentence containment in selected HotpotQA windows."""
import numpy as np
from advanced_rag.upgrade.artifacts import ROOT, LOCAL, REPORTS, read, save
from advanced_rag.upgrade.grounding import gate

def containment(original, result, threshold=0.):
    sources = dict(original["context"])
    accepted = gate(result, threshold)
    selected = list(result["evidence"].values())
    cited = [result["evidence"][s] for s in accepted["final_citations"]]
    valid, selected_hits, cited_hits, invalid = 0, 0, 0, 0
    for title, index in original["supporting_facts"]:
        if not 0 <= index < len(sources[title]):
            invalid += 1
            continue
        sentence = " ".join(sources[title][index].split())
        if not sentence:
            invalid += 1
            continue
        valid += 1
        selected_hits += any(e["document_id"] == title and sentence in e["text"] for e in selected)
        cited_hits += any(e["document_id"] == title and sentence in e["text"] for e in cited)
    return {"gold_sentences": valid, "invalid_annotations": invalid, "selected_hits": selected_hits, "cited_hits": cited_hits,
            "selected_sentence_recall": selected_hits / valid if valid else None, "cited_sentence_recall": cited_hits / valid if valid else None}

def main():
    original = {r["_id"]: r for r in read(ROOT / "data/raw/hotpot.json")}
    metrics = read(REPORTS / "metrics.json")
    output = {"definition": "Fraction of valid annotated supporting sentences wholly contained in at least one selected or accepted cited window, after whitespace normalization. Split-sentence reconstruction is not counted.", "results": {}}
    for split in ("fresh", "controlled", "original"):
        for label in metrics[split]["hotpot"]:
            manifest_path = LOCAL / "manifests" / f"{label}-hotpot-{split}.json"
            if not manifest_path.exists():
                continue
            rows = []
            for item in read(manifest_path)["rows"]:
                result = read(ROOT / item["path"])["result"]
                rows.append(containment(original[item["id"]], result))
            output["results"][f"{split}/{label}"] = {"n": len(rows),
                "selected_sentence_recall": float(np.mean([r['selected_sentence_recall'] for r in rows if r['selected_sentence_recall'] is not None])),
                "cited_sentence_recall": float(np.mean([r['cited_sentence_recall'] for r in rows if r['cited_sentence_recall'] is not None])),
                "invalid_annotations": sum(r['invalid_annotations'] for r in rows)}
    save(REPORTS / "sentence-evidence.json", output)
    print(output)

if __name__ == "__main__":
    main()
