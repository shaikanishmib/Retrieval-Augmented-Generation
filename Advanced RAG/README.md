# Advanced RAG

This project evaluates local answer generation from retrieved evidence. It compares BM25, MiniLM cosine retrieval, reciprocal rank fusion, TinyBERT reranking, evidence expansion, and conditional retrieval retries. FLAN-T5-small generates every answer. Each prediction retains selected sources, citation text, token counts, and latency in local traces.

The primary study uses HotpotQA distractor candidate passages. SQuAD 2.0 adds original unanswerable questions. The local study reserves 24 development questions and 64 held-out questions per dataset. These are partitions of public development releases, not official test results. See [research design](docs/research-design.md) for dataset decisions, task boundaries, and metric definitions.

| Method | HotpotQA answer F1 | SQuAD 2.0 answer F1 |
| --- | ---: | ---: |
| BM25 | 0.3934 | 0.3958 |
| Dense | 0.3372 | 0.3958 |
| Hybrid | 0.3370 | 0.4114 |
| Reranking | 0.5020 | 0.3958 |

Reranking improved HotpotQA answer F1 in this sample, while hybrid did best on SQuAD. Development selection retained no additional overlap-based abstention on either dataset. The [complete measured comparison](reports/benchmark.md) includes intervals, failed comparisons, controls, and resource limits. A 20-case assistant review of actual citation windows found 8 supported answers, 3 partial answers, and 9 unsupported answers; no independent human rater participated.

## Reproduce

Use Python 3.12. From `Advanced RAG`, create an environment and install the project:

```powershell
python -m venv data/venv
data/venv/Scripts/python.exe -m pip install -e ".[dev]"
data/venv/Scripts/python.exe scripts/acquire.py
data/venv/Scripts/python.exe scripts/run_research.py
data/venv/Scripts/python.exe scripts/build_report.py
data/venv/Scripts/python.exe scripts/build_notebook.py
data/venv/Scripts/python.exe scripts/execute_notebook.py
data/venv/Scripts/python.exe scripts/publish_notebook.py
data/venv/Scripts/python.exe scripts/checks.py
data/venv/Scripts/python.exe scripts/audit_publication.py
```

Acquisition checks existing files against [dataset](sources.lock.json) and [model](models.lock.json) locks. `--verify-only` makes no network requests. For the recorded build, extra packages live in ignored `data/packages`; `python scripts/bootstrap.py <script.py>` selects those packages. The exact executed package versions are recorded in `reports/requirements-lock.txt` after validation.

For the recorded package versions, install `reports/requirements-lock.txt` into the new environment before installing the project with `--no-deps`. The support annotations describe the recorded run. A new run needs a fresh review of its retained citation windows; the old qualitative labels must not be reused as judgments of changed answers.

The [public notebook](notebooks/01_public_advanced_rag_research.ipynb) contains the same source cells as the executed local notebook, `notebooks/01_complete_advanced_rag_research.ipynb`. Only the public copy has outputs and execution counts removed. Both the builder and executor refuse to replace an executed notebook. Re-experiments should use a separate checkout or preserve the existing notebook under `.local/` first.

## Answer a local question

Create a local JSON object mapping source IDs to document text, then run:

```powershell
data/venv/Scripts/python.exe scripts/answer.py --documents data/my-documents.json --question "Your question" --method hybrid
```

Citation IDs point to selected source documents. The result also contains cited chunk IDs and text so a reader can inspect support. The default overlap threshold is a conservative engineering setting, not a calibrated confidence score. Retrieved text enters a prompt as untrusted data; the generator cannot execute commands or invoke tools. Prompt injection resistance is not established by prompt wording.

## Interpret the results

[Aggregate measurements](reports/metrics.json) report answer exact match and F1, source recall, citation source-label agreement, abstention behavior, token use, and latency. [Paired intervals](reports/paired-bootstrap.json) compare per-question F1 with hybrid retrieval. Source-label agreement and lexical overlap do not establish answer entailment; the separate manual review inspects actual cited text.

The study uses bounded candidate pools, a small generator, one timing pass, and small held-out samples. It cannot establish open-domain search quality, general multi-hop reasoning, or production reliability. Oracle evidence uses annotations and is a diagnostic control. Per-answer latency excludes model loading and document indexing, which have separate resource observations. There are no paid API calls; local electricity and hardware costs are unmeasured.

Raw data, weights, caches, indexes, traces, package installations, and the executed notebook stay outside Git. Project 1 is unchanged. [Reference findings](docs/reference-findings.md) describes the methods carried into this study.

## Limitations and Future Improvements

The original study used FLAN-T5-small; the later Phase 1 grounding study tested FLAN-T5-base and Qwen3 models that were practical in the recorded local setup. Bounded context budgets and evidence packing can omit links needed for multi-hop answers. The structured Qwen path had parsing and answer-generation failures, while abstention and verification reduced answer coverage in some settings. These results do not show that replacing a model with a larger one would guarantee better F1.

A valid source ID or citation window does not establish that the cited text supports every claim. The recorded 20-case citation review was assistant-led, not an independent human evaluation. NLI agreement is a diagnostic and does not prove factual support. Structured-output reliability and multi-hop reasoning were tested on limited samples from HotpotQA and SQuAD 2.0.

Future work can improve question-focused evidence selection and grounded extraction, then evaluate citation claims with independently annotated evidence and a fixed adjudication rubric. More reliable structured output and abstention need matched answer-quality and coverage measurements. These approaches have not been shown to resolve the observed errors.

## Phase 1 results
The original study used FLAN-T5-small, 24 development and 64 held-out questions per dataset from HotpotQA distractor and SQuAD 2.0. Its best observed deployable scores were HotpotQA reranking F1 0.5020 and SQuAD hybrid F1 0.4114. These are descriptive rankings, not retrospective selection. The [original report](reports/benchmark.md) remains the baseline record.

The previously completed [grounding/model upgrade](README.upgrade.md) is also historical Phase 1 work. On its 256 fresh questions per dataset, practical FLAN-T5-base with original reranking reached HotpotQA F1 0.5074 and SQuAD F1 0.6017. Structured Qwen generation with NLI reached 0.2832 / 0.5330. The development-selected question-conditioned path reached SQuAD 0.5838, with an uncertain gain over NLI; it was disabled for HotpotQA. See the [historical upgrade report](reports/upgrade/benchmark.md) and [public upgrade notebook](notebooks/02_public_grounding_upgrade.ipynb).

The practical FLAN-T5-base configuration is the stronger answer-quality reference in that study, while its lexical citations remain unverified. Structured generation had parsing failures, and verification reduced coverage. Citation support was not independently established: assistant case reviews and NLI diagnostics are not independent human annotation. Bounded evidence pools, small samples, tokenizer-dependent packing, and different CPU/GPU backends limit causal and runtime comparisons. The original and historical upgrade notebooks, model locks, and results remain separate.
