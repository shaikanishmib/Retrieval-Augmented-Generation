"""Write the project README and aggregate tables from reconciled measurements."""
import csv
import io
from llm_judge.artifacts import ROOT, read

h = read(ROOT / 'reports/heldout.json')
f = read(ROOT / 'reports/selection-freeze.json')
b = read(ROOT / 'reports/bias-robustness.json')
t = read(ROOT / 'reports/transfer.json')
n = read(ROOT / 'reports/notebook-audit.json')
selected_runtime = read(ROOT/'reports/selected-runtime.json')
transfer_audit = read(ROOT/'reports/transfer-data-validation.json')
selected = h['results'][f['selected']]
lexical_comparison = h['paired_comparisons']['lexical']
classifier_comparison = h['paired_comparisons']['supervised-tfidf']
models = read(ROOT / 'models.lock.json')
def table(results):
    header = '| Approach | Accuracy | Precision | Recall | F1 | Macro-F1 |\n| --- | ---: | ---: | ---: | ---: | ---: |\n'
    return header + '\n'.join('| '+name+' | '+' | '.join(f"{values[key]:.4f}" for key in ['accuracy','precision','recall','f1','macro_f1'])+' |' for name,values in results.items())

def runtime_table():
    text = '| Comparator | p50 s | p95 s | Evaluations/s | Peak GPU GiB |\n| --- | ---: | ---: | ---: | ---: |\n'
    for name,record in h['runtime'].items():
        median = f"{record['p50_seconds']:.3f}" if 'p50_seconds' in record else 'batch only'
        tail = f"{record['p95_seconds']:.3f}" if 'p95_seconds' in record else 'batch only'
        gpu = f"{record['peak_gpu_allocated']/2**30:.3f}" if 'peak_gpu_allocated' in record else 'not measured'
        text += f"| {name} | {median} | {tail} | {record['evaluations_per_second']:.2f} | {gpu} |\n"
    return text

text = f'''# LLM-as-a-Judge and RAG evaluation

This project tests whether local judges detect unsupported RAG answers against independent human annotations. It compares generative judges, specialized NLI cross-encoders, lexical and embedding proxies, a supervised linear classifier, calibration and an ensemble. It measures failures and resource costs alongside agreement.

The development-selected configuration is **`{f['selected']}`**, with threshold `{f['thresholds'][f['selected']]:.8g}`. On 540 responses from 90 source groups sampled from RAGTruth's official test split, its accuracy is **{selected['accuracy']:.4f}**, hallucination precision **{selected['precision']:.4f}**, recall **{selected['recall']:.4f}**, and F1 **{selected['f1']:.4f}**. This is response-level detection on a sampled test study. It is not span localization or a full official-test leaderboard result.

The [public notebook](notebooks/01_public_llm_judge_research.ipynb), [research design](docs/research-design.md), [metric definitions](docs/metrics.md), [selection freeze](reports/selection-freeze.json) and [full results](reports/heldout.json) provide the experimental record. The executed notebook and individual source passages remain local.

## Data and reference labels

[RAGTruth](https://github.com/ParticleMedia/RAGTruth), revision `c103204b9ce28d6bbad859304bf30de72b8ed8fe`, contains 17,790 responses from 2,965 sources. Its official split has 15,090 training and 2,700 test responses. Human hallucination spans supply the reference labels. Any span makes a response positive, including `implicit_true`: an assertion can be true yet unsupported by context. Conflict and baseless-information metadata are preserved locally. Absence of a span is not an independent factual-correctness, relevance or citation-support annotation.

All six responses from a source stay together. Exact duplicate evidence groups are excluded before sampling. Development uses 216 responses from 36 groups; validation uses 144 from 24; final evaluation uses 540 from 90. Each partition has equal source counts across QA, summarization and data-to-text. The [data audit](reports/data-validation.json) records counts, annotation types, duplicate checks and partition identities. Raw files, model files and local traces are ignored by Git.

[RAGBench](https://huggingface.co/datasets/galileo-ai/ragbench) provides complementary TechQA transfer labels. Its explainable adherence annotations are model-generated, so transfer agreement is reported separately from human-label agreement. The first {t['n']} official test response rows sorted lexically by question ID are evaluated without retuning. They contain {transfer_audit['unique_ids']} original question IDs and {transfer_audit['unique_evidence_groups']} evidence groups, with two generated responses per group. Its reference annotators are `{', '.join(t['annotation_models'])}`. [Dataset locks](sources.lock.json) and [RAGBench lock](ragbench.lock.json) record URLs, revisions, licenses, hashes and sizes. Repository licenses do not supersede original source-content rights.

## Frozen selection and measured results

The development pilot exposed over-rejection by default-threshold Qwen and maximum-error sentence aggregation. It motivated a stronger NLI candidate, an affirmative rubric, mean aggregation and fit-score quantile thresholds. Those pilot results remain in [the record](reports/pilot.json).

Twenty-four development source groups fit thresholds and logistic calibration. Twelve other development groups select macro-F1. Validation is diagnostic and does not change the selection. An ensemble is eligible only after a selection gain greater than 0.01 over the best single approach. The freeze precedes held-out inference and includes source, configuration, model and partition identities. A comparator that performs better on the test set does not become the selected system retrospectively.

{table(h['results'])}

The selected system has Cohen's kappa {selected['kappa']:.4f}, MCC {selected['mcc']:.4f}, Brier score {selected['brier']:.4f}, and 10-bin calibration error {selected['ece_10_bins']:.4f}. Raw conditional logits and NLI support scores are not calibrated human-correctness probabilities. ROC-AUC and average precision, task breakdowns, confusion matrices, false-positive and false-negative rates are in the complete results.

The macro-F1 difference from lexical overlap is {lexical_comparison['point']:.4f}, with a paired 95% interval [{lexical_comparison['interval_95'][0]:.4f}, {lexical_comparison['interval_95'][1]:.4f}]. The difference from TF-IDF is {classifier_comparison['point']:.4f}, with interval [{classifier_comparison['interval_95'][0]:.4f}, {classifier_comparison['interval_95'][1]:.4f}]. The latter interval includes zero, so held-out evidence does not establish that ensemble complexity improves on the trained classifier. The classifier is a useful inexpensive comparator, with the same limitations about task and writing-style priors.

The global threshold generalizes poorly across task types. Hallucination recall is {h['by_task']['Data2txt'][f['selected']]['recall']:.1%} for data-to-text, {h['by_task']['QA'][f['selected']]['recall']:.1%} for QA, and only {h['by_task']['Summary'][f['selected']]['recall']:.1%} for summaries. Overall accuracy therefore does not establish a reliable general-purpose hallucination filter. Thresholds remain unchanged after observing this gap.

{table({task:values[f['selected']] for task,values in h['by_task'].items()})}

![Hallucination precision and recall](reports/figures/precision-recall.png)

Paired macro-F1 intervals resample complete source groups with 2,000 replicates. The [comparison intervals](reports/heldout.json) are pointwise and have no multiplicity adjustment. Claims of improvement require a positive held-out difference with uncertainty considered. Public benchmark scores are not placed beside these results because splits and scoring scopes differ.

## Model and architecture choices

'''
text += '| Model | Exact revision | License |\n| --- | --- | --- |\n'
text += '\n'.join(f"| {m['id']} | `{m['revision']}` | {m['license']} |" for m in models.values())
text += f'''

Qwen3 models use float16, their non-thinking chat template, and single-token Yes/No conditional scoring. Direct, detailed rubric and affirmative-support prompts are compared. Evidence is capped at 2,400 tokenizer tokens; the full answer and instructions remain. This protocol avoids generated JSON parsing but provides no explanatory rationale. It is not a multi-pass reasoning judge.

NLI models use float32 and 512-token pairs. Sentence units retain offsets and the exact lexical-ranked evidence windows used. Whole-answer, one-window, maximum-error and mean-error aggregations are compared. Sentence splitting does not ensure atomic factual claims. Original span overlap supplies sentence-unit diagnostics, not independent atomic-claim labels. At the response-selected threshold, the stronger NLI sentence units have precision {b['sentence_units']['metrics_at_response_threshold']['precision']:.4f}, recall {b['sentence_units']['metrics_at_response_threshold']['recall']:.4f}, and F1 {b['sentence_units']['metrics_at_response_threshold']['f1']:.4f}.

The trained TF-IDF baseline uses otherwise unused official training sources and excludes protected source groups, exact evidence and exact protected responses. It can learn style or task priors and is not a factual verifier. BGE CLS similarity, lexical coverage, exact substring matching and token F1 against context are weak support proxies. They are not reference-answer correctness scores.

The [trained-classifier lock](trained-classifier.lock.json) records its artifact hash, storage, training identity and scikit-learn version. Its parameters remain local and can be retrained from the pinned annotations.

Prometheus, 4B generation and neural fine-tuning are not tested. A larger unquantized generator leaves little working room in 8 GB VRAM; quantization would introduce another runtime to validate. The specialized approaches here are NLI cross-encoders. Ragas is not executed, and no numerical equivalence to it is claimed.

## Bias, calibration and transfer

Pairwise evaluation covers {b['pairwise']['n_pairs']} source-matched pairs in both orders. Derived support-order accuracy is {b['pairwise']['accuracy']:.4f}; reversal consistency is {b['pairwise']['reversal_consistency']:.4f}. These labels distinguish zero-span from positive-span responses, rather than human style preference. Self-preference is not established.

The [bias report](reports/bias-robustness.json) records sensitivity to repetition, model attribution, injected instructions, irrelevant context, evidence order, truncation and rubric paraphrase. Perturbations receive no fabricated human labels. Repeated checks use two uncached passes. These bounded probes do not establish general security or a causal explanation for each change.

An additional [selected-system diagnostic](reports/selected-bias.json) applies the frozen configuration to one response from each of 18 test source groups. It was added after freezing, using source IDs without inspecting test errors. It reports decision sensitivity, without assigning transformed human labels.

![Bias diagnostics](reports/figures/bias-diagnostics.png)

Selective evaluation reports coverage together with observed accuracy. It does not provide a guaranteed risk bound. The [transfer report](reports/transfer.json) uses the original frozen thresholds and coefficients against RAGBench's different annotation source.

{table(t['results'])}

At its frozen threshold, the ensemble's TechQA agreement accuracy is {t['results'][f['selected']]['accuracy']:.1%}, non-adherence recall is {t['results'][f['selected']]['recall']:.1%}, and F1 is {t['results'][f['selected']]['f1']:.4f}. This poor threshold transfer reinforces the cross-task limitation. The annotation source also changes, so these numbers cannot be interpreted as a change in independent human accuracy. The stronger NLI mean-sentence configuration reaches F1 {t['results']['nli-base/claims-mean']['f1']:.4f} against the same model annotator, without becoming a newly selected primary system.

The [original-comparator analysis](reports/original-comparators.json) uses the dataset's precomputed Ragas, TruLens and GPT-3 support scores on exactly matched nonmissing rows. It is an exploratory post-freeze ranking comparison against the same model annotator. Those evaluators were not executed locally, and their original software versions and prompts are not independently verified. AUROC and average precision use inverted support scores; no new thresholds are fitted.

## Runtime and hardware

The recorded machine has an RTX 4060 Laptop GPU with 8,188 MiB VRAM, 34.0 GB RAM, 16 physical cores and 24 logical processors. About 662 GB disk space was free during the initial audit. Cached models were verified before reuse, and only one additional NLI model and one complementary dataset subset were downloaded. No paid APIs were used.

[Runtime measurements](reports/heldout.json) include loading, tokenization and evidence preparation within total inference time, p50/p95 latency, throughput, input tokens, process RSS and peak PyTorch GPU allocation. [The bounded runtime profile](reports/runtime-profile.json) separately measures tokenization, CPU/GPU execution and batching consistency. Cached traces retain original inference timings. PyTorch allocation is not total device memory, and process timing includes warm-up and uncontrolled background load.

[The selected CLI profile](reports/selected-runtime.json) measures three task-balanced development inputs in separate cold processes and verifies agreement with retained predictions. It includes imports and component loading. Each required model loads once per request and supplies its related features. Cold latency is not a persistent-service benchmark. Small-sample latency quantiles are descriptive.

The selected CLI's cold-process p50 is {selected_runtime['p50_cold_process_seconds']:.2f} seconds and p95 is {selected_runtime['p95_cold_process_seconds']:.2f} seconds, at {selected_runtime['evaluations_per_second']:.3f} evaluations/second. The three-input probe peaks at {selected_runtime['peak_gpu_allocated']/2**30:.2f} GiB PyTorch GPU allocation and {selected_runtime['peak_process_working_set']/2**30:.2f} GiB process working set. These cold-process observations have a different scope from the warmed per-response comparator times below.

{runtime_table()}

![Agreement and latency](reports/figures/performance-latency.png)

## Reproduce

Use Python 3.12 and an isolated environment. Install the matching CUDA Torch wheel from the official PyTorch index, then the recorded scientific packages and this project. [Imported versions](reports/runtime-environment.json) and [package inventory](reports/requirements-lock.txt) identify the observed environment. The package inventory includes incidental packages; the project dependencies define the required subset.

```powershell
cd LLM-as-a-Judge
python -m venv data/venv
data/venv/Scripts/python.exe -m pip install torch==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128
data/venv/Scripts/python.exe -m pip install -e .
data/venv/Scripts/python.exe scripts/run.py begin_run.py
data/venv/Scripts/python.exe scripts/run.py acquire.py
data/venv/Scripts/python.exe scripts/run.py acquire_models.py
data/venv/Scripts/python.exe scripts/run.py audit.py
data/venv/Scripts/python.exe scripts/run.py acquire_complementary.py
data/venv/Scripts/python.exe scripts/run.py prepare.py
data/venv/Scripts/python.exe scripts/run.py pilot.py
data/venv/Scripts/python.exe scripts/run.py experiment.py development
data/venv/Scripts/python.exe scripts/run.py experiment.py heldout
data/venv/Scripts/python.exe scripts/run.py diagnostics.py
data/venv/Scripts/python.exe scripts/run.py transfer.py
data/venv/Scripts/python.exe scripts/run.py benchmark_runtime.py
data/venv/Scripts/python.exe scripts/run.py runtime_versions.py
data/venv/Scripts/python.exe scripts/run.py build_notebook.py
data/venv/Scripts/python.exe scripts/run.py execute_notebook.py
data/venv/Scripts/python.exe scripts/run.py publish_notebook.py
data/venv/Scripts/python.exe scripts/run.py build_report.py
data/venv/Scripts/python.exe scripts/run.py test -q
```

On this recorded workspace, `scripts/run.py` can read earlier projects' package overlays and pinned model files without changing them. A fresh checkout downloads the exact locked files into this project's ignored data directory. Prediction caches include source/configuration/model and input identities. Changed identities create separate trace directories; failures are recorded separately and never become successful scores. The freeze rejects source changes during held-out evaluation. To resume, run the same stage again. To conduct a new study, preserve existing reports, freeze and traces in a separate copy before changing the design.

Model variants and adapter limits are fixed in `src/llm_judge/study.py` and `models.py`. JSON controls sampling, seeds and threshold candidates; context-limit changes require corresponding implementation changes. Qwen records evidence-prefix truncation. Original NLI and embedding pair lengths were not instrumented, so their truncation fractions are reported as unknown rather than zero.

## Notebook and publication checks

The local notebook contains {n['cells']} cells, including {n['code_cells']} executed code cells and {n['local_outputs']} saved outputs. It produces {n['figures']} aggregate figures. The public notebook has identical source cells, types and order, zero saved outputs, and null execution counts. Publication preserves the complete local notebook. [Notebook checks](reports/notebook-audit.json) verify these properties.

Tests cover span-to-response mapping, metrics, source/evidence separation, offsets, citation parsing, calibration leakage, perturbations, deterministic identities, actual NLI inference, failures and notebook equivalence. [Publication checks](reports/publication-audit.json) reconcile metrics with predictions, verify dataset/model hashes, compare previous tracked file bytes, scan staged secrets and excluded artifacts, and record Git-normalized hashes.

## Evaluate a local answer

Save evidence and an answer in UTF-8 files, then run:

```powershell
python scripts/run.py judge.py --evidence data/evidence.txt --answer data/answer.txt --question "Your question"
```

The CLI loads the development-selected configuration and prints a structured score, frozen threshold and decision. It checks input presence and frozen source identity. The distance from threshold is descriptive; it is not a calibrated review guarantee. Keep the evidence available when reviewing a flagged answer.

## Interpretation and limitations

This study evaluates evidence support against a specific annotation protocol. It does not establish independent correctness, relevance, context recall or citation-support accuracy. An answer can be correct without source support and faithful to an incorrect source. Separate annotation is required for those questions.

Development selection is uncertain with few source groups. Exact deduplication cannot remove all related passages, and pretraining contamination cannot be ruled out. Sentence units are not atomic claims. TF-IDF may exploit response style. Truncation and lexical evidence selection can hide necessary evidence. Calibration may shift across tasks. RAGBench agreement uses a model annotator, and the robustness subset is small. Larger specialized generative judges and quantized models remain open comparisons.

Earlier projects motivate the failure cases but supply no independent human labels. Projects 1–4 remain unchanged. Recommendations should follow the selected system's held-out errors, coverage and task breakdown rather than a larger model's name or the highest exploratory score.

## References and attribution

RAGTruth: Cheng Niu et al., 2024, [RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models](https://arxiv.org/abs/2401.00396). Dataset files come from ParticleMedia's pinned repository release under its recorded repository license, with upstream source rights preserved.

RAGBench: Robert Friel, Masha Belyi and Atindriyo Sanyal, 2024, [RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems](https://arxiv.org/abs/2407.11005). The pinned Galileo dataset release declares CC BY 4.0. [Methodology notes](docs/related-work.md) and [BibTeX entries](docs/references.bib) provide the benchmark references. Exact model cards, licenses and file hashes are in the model lock.
'''
(ROOT / 'README.md').write_text(text, encoding='utf-8')
fields = ['approach','accuracy','balanced_accuracy','precision','recall','f1','macro_f1','kappa','mcc','roc_auc','pr_auc','brier','ece_10_bins']
with (ROOT / 'reports/metrics.csv').open('w',newline='',encoding='utf-8') as out:
    writer = csv.DictWriter(out,fieldnames=fields)
    writer.writeheader()
    for name,values in h['results'].items():
        writer.writerow({'approach':name,**{k:values[k] for k in fields[1:]}})
print('Wrote README and aggregate metric table.')
