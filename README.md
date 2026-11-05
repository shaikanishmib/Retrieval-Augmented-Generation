# Six independent RAG research projects

This repository contains six empirical projects on retrieval-augmented generation. Each studies a separate part of the system: finding evidence, generating grounded answers, choosing retrieval actions, reusing cached work, evaluating hallucinations, and securing retrieval pipelines. The six studies evaluate separate components; no combined system was tested.

## Projects and Phase 1 findings

| Project | Objective and methods | Main finding |
| --- | --- | --- |
| [Retrieval Optimization](Retrieval%20Optimization/README.md) | Compare BM25, MiniLM dense retrieval, rank fusion, chunking, and TinyBERT reranking on BEIR SciFact and FiQA. | Selected fusion reached nDCG@10 of **0.7068 / 0.3817** on SciFact / FiQA. TinyBERT reranking fell below fusion. [Metrics](Retrieval%20Optimization/reports/research/metrics.csv) · [Notebook](Retrieval%20Optimization/notebooks/01_public_retrieval_research.ipynb) |
| [Advanced RAG](Advanced%20RAG/README.md) | Measure answer quality, evidence recall, citation behavior, abstention, and retrieval variants on HotpotQA and SQuAD 2.0. | The original FLAN-T5-small study reached F1 **0.5020** with reranking on HotpotQA and **0.4114** with hybrid retrieval on SQuAD. A separate historical Phase 1 grounding study tested FLAN-T5-base and structured Qwen generation; citation support remained unverified. [Original report](Advanced%20RAG/reports/benchmark.md) · [Historical grounding report](Advanced%20RAG/reports/upgrade/benchmark.md) · [Notebook](Advanced%20RAG/notebooks/01_public_advanced_rag_research.ipynb) |
| [Agentic RAG](Agentic%20RAG/README.md) | Compare fixed and adaptive retrieval with planning, tool selection, and verification-guided control. | On 128 held-out questions per dataset, the tool-selecting agent scored F1 **0.431 / 0.748**, below fixed RAG at **0.493 / 0.770**. It was about 9–10 times slower at median latency. [Report](Agentic%20RAG/reports/benchmark.md) · [Notebook](Agentic%20RAG/notebooks/01_public_agentic_rag_research.ipynb) |
| [Semantic Caching](Semantic%20Caching/README.md) | Compare exact answer caching, semantic answer reuse, evidence caching, and hybrid policies on QA and PAWS-Wiki. | Exact caching produced a **33.3%** hit rate and reduced mixed HotpotQA median latency from **210.36 ms to 192.25 ms**. Strict semantic answer reuse accepted no held-out pairs because useful thresholds admitted false matches during development. [Report](Semantic%20Caching/reports/benchmark.md) · [Notebook](Semantic%20Caching/notebooks/01_public_semantic_caching_research.ipynb) |
| [LLM-as-a-Judge](LLM-as-a-Judge/README.md) | Evaluate response-level hallucination detection against human-labeled RAGTruth and separately assess transfer to model-labeled RAGBench. | The selected judge reached hallucination F1 **0.6708** and macro-F1 **0.7664** on 540 human-labeled responses. QA recall was **36.0%** and summarization recall **7.5%**. [Held-out results](LLM-as-a-Judge/reports/heldout.json) · [Notebook](LLM-as-a-Judge/notebooks/01_public_llm_judge_research.ipynb) |
| [RAG Security](RAG%20Security/README.md) | Test injection, poisoning, citation and tool attacks alongside legitimate task completion and deterministic boundaries. | The frozen Qwen D6 policy had **0.0%** observed attack success and **0.0%** clean and attacked completion, so it failed the utility requirement. Authorization, tenant and cache probes passed in bounded fixtures, but ID-only index checks exposed relabeled secret-bearing bytes. Factual poisoning remained unresolved. [Metrics](RAG%20Security/reports/metrics.json) · [Failure cases](RAG%20Security/reports/failure-cases.md) · [Notebook](RAG%20Security/notebooks/01_public_rag_security_research.ipynb) |

The Advanced RAG project contains two distinct historical Phase 1 studies. The grounding study's practical FLAN-T5-base reference reached F1 **0.5074 / 0.6017** on 256 fresh HotpotQA / SQuAD questions. Question-conditioned verification reached **0.5838** on SQuAD; its advantage over the NLI comparison was uncertain. See its [study README](Advanced%20RAG/README.upgrade.md) and [public notebook](Advanced%20RAG/notebooks/02_public_grounding_upgrade.ipynb).

Scores across projects are not directly comparable. Datasets, labels, query sets, models, and metrics differ. The reported results describe tested configurations and do not establish the performance of a combined system.

## Shared research practice

Each project records source and model identities, data partitions, configuration choices, and measured outcomes. Selection uses development data where the design permits it; held-out results are kept distinct from development metrics. Paired uncertainty and per-task outcomes are reported when available. Failed comparisons and known provenance limits remain part of each project's record.

The six projects measure different properties. Document relevance does not establish answer correctness. Citation validity does not prove that cited text supports a claim. A cache hit does not establish question equivalence. Low attack success is useful only when legitimate work still completes. Consult each README and report for metric definitions and evaluation boundaries.

## Environment and compute

The recorded Windows workstation has an NVIDIA RTX 4060 Laptop GPU with 8 GB VRAM, an Intel Core i7-13700HX, and approximately 32 GB RAM. Project records show that device use varied: some runs used CUDA, while the original Advanced RAG benchmark ran in a CPU-only PyTorch environment. GPU memory constrained feasible model sizes and precision, but experiments did not all use the GPU. The studies report no paid API calls; energy and hardware costs were not measured.

## Reproduction and navigation

Start with the README in the project you want to reproduce. It documents its Python environment, acquisition and verification commands, data and model locks, tests, and publication checks. Public notebooks omit saved outputs; executed notebooks, datasets, model weights, traces, and indexes remain local or ignored. Check each project's data terms and unresolved reuse conditions before redistributing source material.

The root and project reports are the authoritative records for the numbers summarized here. Future work listed in project limitations is proposed work, not a measured result.