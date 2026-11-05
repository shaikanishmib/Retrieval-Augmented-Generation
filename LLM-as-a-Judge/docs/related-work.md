# Related methodology

[RAGTruth](https://arxiv.org/abs/2401.00396) studies naturally generated RAG responses with human hallucination-span annotation. Its response detection and span localization endpoints differ. This project maps the released spans to response labels, preserving the distinction between absent evidence and factual contradiction. It reports no span-localization score.

[RAGBench](https://arxiv.org/html/2407.11005v1) defines TRACe relevance, utilization, completeness and adherence. Relevance measures the retrieved context fraction useful to the question; utilization measures the fraction used in the answer; completeness measures how much relevant context was used. Adherence requires the entire answer to be grounded. Its LLM annotations and task definitions make it a complementary reference, rather than a replacement for human RAGTruth labels. The released annotation-model field is retained because the annotator in a specific release may differ from the paper's original setup.

[The stronger NLI model card](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) identifies its MultiNLI, FEVER-NLI and ANLI training and label order: entailment, neutral, contradiction. That order differs from the cached small cross-encoder. The adapter reads named label mappings, and actual inference verifies the loaded configuration.

[Prometheus 2](https://huggingface.co/prometheus-eval/prometheus-7b-v2.0) is an Apache-2.0 specialized generative evaluator with absolute and pairwise rubric formats. It is a relevant future comparator. Its 7B unquantized footprint exceeds this machine's comfortable GPU budget. This study uses specialized NLI evaluation and does not present it as an empirical test of Prometheus or all evaluator architectures.

No published number is treated as directly comparable to this sampled, source-group-aware response-detection study. Different labels, splits, annotation sources and localization objectives can change apparent model rankings.
