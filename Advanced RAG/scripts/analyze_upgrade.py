"""Recalculate upgrade tables and paired comparisons from original prediction traces."""
import csv
import json
from pathlib import Path
from advanced_rag.data import load
from advanced_rag.pipeline import answer_scores
from advanced_rag.upgrade.artifacts import ROOT, LOCAL, REPORTS, read, save, fingerprint, preserved
from advanced_rag.upgrade.evaluate import per_question, aggregate, cluster_interval, calibration

def manifest(label, name, split):
    value = read(LOCAL / "manifests" / f"{label}-{name}-{split}.json")
    return value, [read(ROOT / item["path"])["result"] for item in value["rows"]]

def main():
    preserved()
    metrics = read(REPORTS / "metrics.json")
    selection = read(REPORTS / "selection.json")
    before_after, fresh_comparisons, controls, priors, practical_intervals, rows = {}, {}, {}, {}, {}, []
    original_metrics = read(ROOT / "reports/metrics.json")
    for name in ("hotpot", "squad"):
        _, all_test, _ = load(name, 64, 320)
        threshold = selection["thresholds"][name]["selected"]
        _, original_results = manifest("selected_raw", name, "original")
        original_scores = [per_question(row, result, threshold) for row, result in zip(all_test[:64], original_results)]
        baseline_scores = read(ROOT / f"data/experiments/{name}-scores.json")
        before_after[name] = {"n": 64, "original_hybrid": original_metrics[name]["hybrid"], "original_rerank": original_metrics[name]["rerank"],
                              "upgraded_raw": metrics["original"][name]["selected_raw"], "upgraded_verified": aggregate(original_scores),
                              "paired_f1_vs_hybrid": cluster_interval([s['f1'] for s in baseline_scores['hybrid']], [s['f1'] for s in original_scores], [r['group'] for r in all_test[:64]]),
                              "paired_f1_vs_rerank": cluster_interval([s['f1'] for s in baseline_scores['rerank']], [s['f1'] for s in original_scores], [r['group'] for r in all_test[:64]])}
        fresh = all_test[64:]
        _, results = manifest("selected_raw", name, "fresh")
        recalculated = [per_question(row, result) for row, result in zip(fresh, results)]
        assert aggregate(recalculated) == metrics["fresh"][name]["selected_raw"]
        verified = [per_question(row, result, threshold) for row, result in zip(fresh, results)]
        assert aggregate(verified) == metrics["fresh"][name]["selected_verified"]
        constant_scores = [answer_scores("", row["answers"])[1] for row in fresh]
        controls[name] = {"n": len(fresh), "always_abstain_em": sum(constant_scores) / len(fresh),
                          "always_abstain_f1": sum(constant_scores) / len(fresh), "coverage": 0., "answerable_f1": 0.,
                          "unanswerable_recall": 1. if any(not r["answers"] for r in fresh) else None,
                          "selected_f1_minus_always_abstain": metrics["fresh"][name]["selected_verified"]["f1"] - sum(constant_scores) / len(fresh)}
        dev = load(name, 64, 320)[0]
        _, dev_results = manifest("selected", name, "development")
        fit_ids = set(selection["thresholds"][name]["fit_ids"])
        outcomes = [answer_scores(p["answer"], row["answers"])[0] for row, p in zip(dev, dev_results) if row["id"] in fit_ids]
        prior = (sum(outcomes) + 1) / (len(outcomes) + 2)
        priors[name] = {"development_fit_n": len(outcomes), "smoothed_prior": prior,
                       "constant_prior": calibration([prior] * len(fresh), [v["raw_em"] for v in recalculated])}
        comparisons = {}
        for method in ("bm25", "dense", "hybrid", "rerank"):
            baseline = []
            for row in fresh:
                key = fingerprint({"row": row, "stage": "baseline_cpu_reproduction_v1"})
                prediction = read(LOCAL / "baseline" / f"{key}.json")["predictions"][method]
                baseline.append(answer_scores(prediction["answer"], row["answers"])[1])
            comparisons[method] = cluster_interval(baseline, [s['f1'] for s in verified], [r['group'] for r in fresh])
        fresh_comparisons[name] = comparisons
        practical_scores = read(LOCAL / "practical" / f"{name}-scores.json")
        practical_intervals[name] = {}
        for split, part in (("original", all_test[:64]), ("fresh", fresh)):
            practical_intervals[name][split] = {}
            for method in ("hybrid", "rerank"):
                if split == "original":
                    reference = [r["f1"] for r in baseline_scores[method]]
                else:
                    reference = []
                    for row in part:
                        cache = fingerprint({"row": row, "stage": "baseline_cpu_reproduction_v1"})
                        prediction = read(LOCAL / "baseline" / f"{cache}.json")["predictions"][method]
                        reference.append(answer_scores(prediction["answer"], row["answers"])[1])
                practical_intervals[name][split][method] = cluster_interval(reference, [r['f1'] for r in practical_scores[split][method]], [r['group'] for r in part])
    families = {}
    for name, methods in metrics["controlled"].items():
        def paired(reference, candidate):
            a = read(LOCAL / "scores" / f"{reference}-{name}-controlled.json")
            b = read(LOCAL / "scores" / f"{candidate}-{name}-controlled.json")
            assert [r["id"] for r in a] == [r["id"] for r in b]
            return cluster_interval([r["f1"] for r in a], [r["f1"] for r in b], [r["group"] for r in a])
        families[name] = {
            "plain_generators_vs_flan_small": {m: paired("generator_plain_flan-small", m) for m in methods if m.startswith("generator_plain_")},
            "retrieval_vs_bm25": {m: paired("retrieval_bm25", m) for m in methods if m.startswith("retrieval_")},
            "context_vs_768_diverse": {m: paired("context_768_1_0", m) for m in methods if m.startswith("context_")},
            "bge_vs_minilm_hybrid": paired("retrieval_hybrid_minilm", "retrieval_hybrid_bge"),
            "minilm_vs_tinybert_reranking": paired("retrieval_tinybert", "retrieval_minilm_rerank"),
        }
    save(REPORTS / "controlled-family-intervals.json", {"scope": "Matched controlled questions; candidate minus named reference; exploratory 95% cluster bootstrap intervals without multiple-comparison correction", "datasets": families})
    for split, datasets in metrics.items():
        for name, methods in datasets.items():
            for method, values in methods.items():
                rows.append({"split": split, "dataset": name, "method": method, **values})
    with (REPORTS / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save(REPORTS / "before-after.json", before_after)
    save(REPORTS / "fresh-baseline-intervals.json", fresh_comparisons)
    save(REPORTS / "abstention-controls.json", controls)
    save(REPORTS / "calibration-prior-baseline.json", priors)
    save(REPORTS / "practical-intervals.json", practical_intervals)
    lines = ["# Model and grounding results", "", "The original reports at commit `92a85e1` are preserved. The tables distinguish the original 64-question samples, 256 fresh held-out questions per dataset, and 64-question controlled subsets. No held-out score selected a model, retrieval setting, context policy, or threshold.", "", "## Fresh held-out comparison", "", "| Dataset | Configuration | Questions | EM | F1 | Coverage | Answerable F1 | Unanswerable recall | Median ms |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    baseline = read(REPORTS / "baseline-reproduction.json")
    for name in ("hotpot", "squad"):
        for method, values in baseline[name]["fresh_metrics"].items():
            unanswerable = values["unanswerable_recall"]
            ua = "n/a" if unanswerable is None else f"{unanswerable:.4f}"
            lines.append(f"| {name} | original_{method} | 256 | {values['em']:.4f} | {values['f1']:.4f} | {1-values['abstained']:.4f} | n/a | {ua} | {values['p50_ms']:.2f} |")
        for method in ("selected_raw", "selected_verified", "selected_lexical", "balanced", "strongest"):
            values = metrics["fresh"][name][method]
            ua = "n/a" if values["unanswerable_recall"] is None else f"{values['unanswerable_recall']:.4f}"
            lines.append(f"| {name} | {method} | {values['n']} | {values['em']:.4f} | {values['f1']:.4f} | {values['coverage']:.4f} | {values['answerable_f1']:.4f} | {ua} | {values['p50_ms']:.2f} |")
    practical = read(REPORTS / "practical-baseline.json")
    lines += ["", "## Practical generator reference", "", "The plain-answer development winner is evaluated through the preserved post hoc pipeline. This separate full-sample extension keeps the original retrieval and generation settings. It retains lexical citations, whose support remains unverified. Both this reference and the original baseline use CPU float32 inference. The practical pass ran while the separate GPU study was active, so host load was uncontrolled.", "", "| Dataset | Method | Fresh questions | EM | F1 | Median ms |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for name, parts in practical["datasets"].items():
        for method, values in parts["fresh"].items():
            lines.append(f"| {name} | {practical['generator']} {method} | {values['n']} | {values['em']:.4f} | {values['f1']:.4f} | {values['p50_ms']:.2f} |")
    lines += ["", "## Matched original questions", "", "| Dataset | Original hybrid F1 | Original rerank F1 | Upgraded raw F1 | Upgraded verified F1 |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, values in before_after.items():
        lines.append(f"| {name} | {values['original_hybrid']['f1']:.4f} | {values['original_rerank']['f1']:.4f} | {values['upgraded_raw']['f1']:.4f} | {values['upgraded_verified']['f1']:.4f} |")
    lines += ["", "## Question-conditioned validation", "", "This exploratory follow-up checks the proposed answer against a FLAN-T5-base answer generated from cited windows. It reuses the recorded Qwen generations, selects cutoffs on development groups, and measures additional CPU validation. It does not claim a newly timed end-to-end pass."]
    question = read(REPORTS / "answer-support-results.json")
    question_analysis = read(REPORTS / "answer-support-analysis.json")
    adoption = read(REPORTS / "answer-support-adoption.json")["decisions"]
    for name in ("hotpot", "squad"):
        qa = question["results"][name]["fresh"]
        ci = question_analysis[name]["fresh"]["question_check_vs_nli_f1"]
        lines += ["", f"For {name}, question-conditioned validation has F1 {qa['f1']:.4f}, compared with {metrics['fresh'][name]['selected_verified']['f1']:.4f} for the frozen NLI rule. The paired difference is {ci['delta']:+.4f}, with a cluster interval [{ci['low']:+.4f}, {ci['high']:+.4f}]. Development selection {'enables' if adoption[name]['enabled'] else 'disables'} this check by default. Mean additional validation is {qa['mean_additional_validation_ms']:.2f} ms per question."]
    review = read(REPORTS / "support-review.json")
    default_review = review["development_selected_default"]
    lines += ["", "## Matched support review", "", f"The coding assistant reviewed the first ten original questions per dataset with references and cited text visible. The original review had 8 supported, 3 partial, and 9 unsupported answers. The development-selected path has {default_review['counts'].get('supported',0)} supported, {default_review['counts'].get('partial',0)} partial, {default_review['counts'].get('unsupported',0)} unsupported, and {default_review['counts'].get('abstained',0)} abstained outputs. Strict support among its accepted answers is {default_review['strict_support_among_answered']:.4f}, at coverage {default_review['coverage']:.4f}. Fewer unsupported accepted answers therefore accompany lower coverage, not a universal improvement in answered-question quality. These are twenty exploratory assistant judgments, not independent human annotations."]
    lines += ["", "## Development choices", "", "The citation-capable generator was selected from the two Qwen models after all four generators were compared with fixed retrieval in plain-answer mode. FLAN models remain in the generator comparison even when they score better. The selected configuration is:", "", "```json", json.dumps(selection["selected_configuration"], indent=2), "```", "", "NLI support thresholds and lexical thresholds were selected on development groups. Their held-out risk curves are descriptive. Logistic calibration predicts raw answer correctness, not entailment.", "", "## Interpretation limits", "", "NLI scores are model-based support diagnostics. They do not establish claim truth or independently annotated citation accuracy. Source precision and recall measure agreement with original document labels. Case review by the coding assistant is explicitly disclosed separately. Structured output can improve source traceability while reducing coverage or answer F1; those effects must be assessed together.", "", "Original baseline inference uses CPU, while the new models use CUDA. The latency comparison therefore includes a backend change and cannot isolate a model-size effect. Timings include active query stages, with indexing and model loading recorded separately. Cached results retain original measured inference times. Memory peaks are whole-process observations.", "", "The controlled study uses 64 fresh questions per dataset. Paired intervals resample SQuAD articles or HotpotQA questions and are exploratory without multiple-comparison correction. The larger frozen comparison uses 256 questions per dataset. All candidate evidence is supplied locally; open-domain search, quantized 4B inference, independent human support labels, and production injection resistance remain untested."]
    lines += ["", "## Controlled comparisons", "", "Each row uses the same 64 fresh questions within its dataset. Structured rows require valid citations and use a zero NLI cutoff. Plain-generator rows share retrieval and prompt format, but tokenizer-specific packing can change the evidence. Family-matched paired intervals are in controlled-family-intervals.json; paired-intervals.json uses selected_raw as its reference. All intervals are exploratory without multiple-comparison correction.", "", "| Dataset | Configuration | EM | F1 | Evidence-document recall | Median ms |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for name, methods in metrics["controlled"].items():
        for method, values in methods.items():
            lines.append(f"| {name} | {method} | {values['em']:.4f} | {values['f1']:.4f} | {values['evidence_recall']:.4f} | {values['p50_ms']:.2f} |")
    (REPORTS / "benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Reconciled", len(rows), "aggregate rows")
    print(json.dumps(before_after, indent=2))

if __name__ == "__main__":
    main()
