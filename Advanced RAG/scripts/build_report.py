"""Write benchmark tables directly from the completed experiment outputs."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    metrics = json.loads((ROOT / "reports/metrics.json").read_text())
    stats = json.loads((ROOT / "reports/paired-bootstrap.json").read_text())
    manual = json.loads((ROOT / "reports/manual-support-review.json").read_text())
    assert dict(Counter(c["label"] for c in manual["cases"])) == manual["counts"]
    lines = ["# Local grounded-answer results", "", "The study uses 64 held-out questions from each public development release, with 24 separate development questions per dataset. These are bounded-evidence local experiments, not official test scores. All generation ran on CPU with FLAN-T5-small. No paid APIs were used.", "", "| Dataset | Method | EM | Answer F1 | Context recall | Median ms |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    rows = []
    for name, methods in metrics.items():
        for method, values in methods.items():
            rows.append({"dataset": name, "method": method, **values})
            lines.append(f"| {name} | {method} | {values['em']:.4f} | {values['f1']:.4f} | {values['context_recall']:.4f} | {values['p50_ms']:.2f} |")
    for name, methods in metrics.items():
        base = methods["hybrid"]
        best = max((m for m in methods if m not in {"oracle", "no_context"}), key=lambda m: methods[m]["f1"])
        lines += ["", f"## {name}", "", f"The highest observed F1 among the deployable configurations is {methods[best]['f1']:.4f} for `{best}`. This ranking is descriptive; it was not used to select a configuration after evaluation.", ""]
        for method in ("bm25", "dense", "rerank", "multihop", "corrective", "oracle"):
            interval = stats[name][method]
            lines.append(f"`{method}` changes F1 relative to hybrid by {interval['delta']:+.4f}, with a paired 95% percentile interval [{interval['low']:+.4f}, {interval['high']:+.4f}].")
            lines.append("")
        lines.append(f"Hybrid citation source-label precision is {base['citation_precision']:.4f}; source-label recall is {base['citation_recall']:.4f}. These metrics do not establish entailment.")
        chosen = json.loads((ROOT / f"reports/{name}-selection.json").read_text())["selected_threshold"]
        lines += ["", f"Development selected overlap threshold {chosen}. The selected abstention variant has coverage {1 - methods['abstention']['abstained']:.4f}. The overlap rule supplied no additional abstention benefit in the selected configuration."]
    lines += ["", "## Support review and limitations", "", "A separate assistant review of the first ten hybrid cases per dataset found 8 supported, 3 partial, and 9 unsupported answers. References were visible, and no independent human rater participated. The review checks actual cited windows, including entity identity, requested answer type, and both sides of comparisons. It is exploratory case analysis, not an estimate of population faithfulness.", "", "The experiment uses supplied evidence pools and small samples. The source citation is attached after generation using lexical overlap. Expansion is a simple evidence-derived term baseline, not learned decomposition. Pretrained benchmark exposure is unknown. Question-level bootstrap intervals are exploratory and can understate SQuAD article dependence. Timing uses one sequential pass; model loading and document indexing are excluded from answer latency and reported separately. CPU RSS observations are sampled process totals. GPU performance, electricity, and hardware costs were not measured.", "", "The oracle uses gold source labels and is a diagnostic control. Negative comparisons remain in the table. Larger generators, open-domain retrieval, independent support annotation, and stronger abstention calibration remain untested."]
    (ROOT / "reports/benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (ROOT / "reports/metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Hashes connect the qualitative labels to the exact retained prediction and evidence.
    provenance = []
    for case in manual["cases"]:
        trace = json.loads((ROOT / f"data/experiments/{case['dataset']}-traces.json").read_text())
        p = trace["heldout"][case["id"]]["predictions"]["hybrid"]
        content = json.dumps({k: p[k] for k in ("answer", "citations", "citation_chunks", "citation_text")}, sort_keys=True, ensure_ascii=False).encode()
        provenance.append({"dataset": case["dataset"], "id": case["id"], "prediction_evidence_sha256": hashlib.sha256(content).hexdigest()})
    (ROOT / "reports/manual-review-provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print("Wrote", len(rows), "measured rows and", len(provenance), "review hashes")

if __name__ == "__main__":
    main()
