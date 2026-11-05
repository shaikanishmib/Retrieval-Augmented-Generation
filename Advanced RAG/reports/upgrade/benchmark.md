# Model and grounding results

The original reports at commit `92a85e1` are preserved. The tables distinguish the original 64-question samples, 256 fresh held-out questions per dataset, and 64-question controlled subsets. No held-out score selected a model, retrieval setting, context policy, or threshold.

## Fresh held-out comparison

| Dataset | Configuration | Questions | EM | F1 | Coverage | Answerable F1 | Unanswerable recall | Median ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hotpot | original_bm25 | 256 | 0.3438 | 0.4478 | 1.0000 | n/a | n/a | 115.83 |
| hotpot | original_dense | 256 | 0.2969 | 0.3786 | 1.0000 | n/a | n/a | 123.07 |
| hotpot | original_hybrid | 256 | 0.3242 | 0.4325 | 1.0000 | n/a | n/a | 124.84 |
| hotpot | original_rerank | 256 | 0.3320 | 0.4401 | 1.0000 | n/a | n/a | 136.01 |
| hotpot | selected_raw | 256 | 0.2109 | 0.2832 | 0.5547 | 0.2832 | n/a | 2397.58 |
| hotpot | selected_verified | 256 | 0.2109 | 0.2832 | 0.5547 | 0.2832 | n/a | 2397.58 |
| hotpot | selected_lexical | 256 | 0.2109 | 0.2832 | 0.5547 | 0.2832 | n/a | 2397.58 |
| hotpot | balanced | 256 | 0.0312 | 0.0511 | 0.1250 | 0.0511 | n/a | 3562.89 |
| hotpot | strongest | 256 | 0.2109 | 0.2832 | 0.5547 | 0.2832 | n/a | 2397.58 |
| squad | original_bm25 | 256 | 0.3477 | 0.3905 | 0.9688 | n/a | 0.0519 | 94.42 |
| squad | original_dense | 256 | 0.3438 | 0.3892 | 0.9688 | n/a | 0.0519 | 102.62 |
| squad | original_hybrid | 256 | 0.3477 | 0.3931 | 0.9688 | n/a | 0.0519 | 101.80 |
| squad | original_rerank | 256 | 0.3477 | 0.3905 | 0.9688 | n/a | 0.0519 | 105.71 |
| squad | selected_raw | 256 | 0.4023 | 0.4272 | 0.5352 | 0.3914 | 0.4593 | 1929.25 |
| squad | selected_verified | 256 | 0.5117 | 0.5330 | 0.2188 | 0.1937 | 0.8370 | 1929.25 |
| squad | selected_lexical | 256 | 0.4336 | 0.4555 | 0.4922 | 0.3852 | 0.5185 | 1929.25 |
| squad | balanced | 256 | 0.4609 | 0.4706 | 0.1836 | 0.1032 | 0.8000 | 2162.31 |
| squad | strongest | 256 | 0.4023 | 0.4272 | 0.5352 | 0.3914 | 0.4593 | 1929.25 |

## Practical generator reference

The plain-answer development winner is evaluated through the preserved post hoc pipeline. This separate full-sample extension keeps the original retrieval and generation settings. It retains lexical citations, whose support remains unverified. Both this reference and the original baseline use CPU float32 inference. The practical pass ran while the separate GPU study was active, so host load was uncontrolled.

| Dataset | Method | Fresh questions | EM | F1 | Median ms |
| --- | --- | ---: | ---: | ---: | ---: |
| hotpot | flan-base hybrid | 256 | 0.3711 | 0.4829 | 492.13 |
| hotpot | flan-base rerank | 256 | 0.4023 | 0.5074 | 497.74 |
| squad | flan-base hybrid | 256 | 0.5625 | 0.5997 | 385.85 |
| squad | flan-base rerank | 256 | 0.5625 | 0.6017 | 400.92 |

## Matched original questions

| Dataset | Original hybrid F1 | Original rerank F1 | Upgraded raw F1 | Upgraded verified F1 |
| --- | ---: | ---: | ---: | ---: |
| hotpot | 0.3370 | 0.5020 | 0.2435 | 0.2435 |
| squad | 0.4114 | 0.3958 | 0.4245 | 0.4531 |

## Question-conditioned validation

This exploratory follow-up checks the proposed answer against a FLAN-T5-base answer generated from cited windows. It reuses the recorded Qwen generations, selects cutoffs on development groups, and measures additional CPU validation. It does not claim a newly timed end-to-end pass.

For hotpot, question-conditioned validation has F1 0.2471, compared with 0.2832 for the frozen NLI rule. The paired difference is -0.0361, with a cluster interval [-0.0704, -0.0040]. Development selection disables this check by default. Mean additional validation is 366.91 ms per question.

For squad, question-conditioned validation has F1 0.5838, compared with 0.5330 for the frozen NLI rule. The paired difference is +0.0508, with a cluster interval [-0.0001, +0.0990]. Development selection enables this check by default. Mean additional validation is 315.32 ms per question.

## Matched support review

The coding assistant reviewed the first ten original questions per dataset with references and cited text visible. The original review had 8 supported, 3 partial, and 9 unsupported answers. The development-selected path has 6 supported, 1 partial, 3 unsupported, and 10 abstained outputs. Strict support among its accepted answers is 0.6000, at coverage 0.5000. Fewer unsupported accepted answers therefore accompany lower coverage, not a universal improvement in answered-question quality. These are twenty exploratory assistant judgments, not independent human annotations.

## Development choices

The citation-capable generator was selected from the two Qwen models after all four generators were compared with fixed retrieval in plain-answer mode. FLAN models remain in the generator comparison even when they score better. The selected configuration is:

```json
{
  "name": "minilm_rerank",
  "encoder": "bge-small",
  "retrieval": "rerank",
  "reranker": "minilm-rerank",
  "budget": 1280,
  "generator": "qwen17",
  "diversity": true,
  "reverse": false
}
```

NLI support thresholds and lexical thresholds were selected on development groups. Their held-out risk curves are descriptive. Logistic calibration predicts raw answer correctness, not entailment.

## Interpretation limits

NLI scores are model-based support diagnostics. They do not establish claim truth or independently annotated citation accuracy. Source precision and recall measure agreement with original document labels. Case review by the coding assistant is explicitly disclosed separately. Structured output can improve source traceability while reducing coverage or answer F1; those effects must be assessed together.

Original baseline inference uses CPU, while the new models use CUDA. The latency comparison therefore includes a backend change and cannot isolate a model-size effect. Timings include active query stages, with indexing and model loading recorded separately. Cached results retain original measured inference times. Memory peaks are whole-process observations.

The controlled study uses 64 fresh questions per dataset. Paired intervals resample SQuAD articles or HotpotQA questions and are exploratory without multiple-comparison correction. The larger frozen comparison uses 256 questions per dataset. All candidate evidence is supplied locally; open-domain search, quantized 4B inference, independent human support labels, and production injection resistance remain untested.

## Controlled comparisons

Each row uses the same 64 fresh questions within its dataset. Structured rows require valid citations and use a zero NLI cutoff. Plain-generator rows share retrieval and prompt format, but tokenizer-specific packing can change the evidence. Family-matched paired intervals are in controlled-family-intervals.json; paired-intervals.json uses selected_raw as its reference. All intervals are exploratory without multiple-comparison correction.

| Dataset | Configuration | EM | F1 | Evidence-document recall | Median ms |
| --- | --- | ---: | ---: | ---: | ---: |
| hotpot | generator_plain_flan-small | 0.3594 | 0.4613 | 0.6797 | 91.74 |
| hotpot | generator_plain_flan-base | 0.3594 | 0.4570 | 0.6797 | 125.68 |
| hotpot | generator_plain_qwen06 | 0.0312 | 0.1040 | 0.7734 | 126.58 |
| hotpot | generator_plain_qwen17 | 0.1562 | 0.2462 | 0.7734 | 417.40 |
| hotpot | retrieval_bm25 | 0.2656 | 0.2808 | 0.8828 | 2559.32 |
| hotpot | retrieval_dense_minilm | 0.1719 | 0.2085 | 0.8594 | 2456.60 |
| hotpot | retrieval_hybrid_minilm | 0.2344 | 0.2699 | 0.9062 | 2275.62 |
| hotpot | retrieval_hybrid_bge | 0.2188 | 0.2413 | 0.8906 | 2289.46 |
| hotpot | retrieval_tinybert | 0.2188 | 0.2538 | 0.8516 | 2401.99 |
| hotpot | retrieval_minilm_rerank | 0.2500 | 0.2963 | 0.9141 | 2443.65 |
| hotpot | context_384_1_0 | 0.2188 | 0.2469 | 0.7969 | 2452.21 |
| hotpot | context_768_1_0 | 0.2500 | 0.2963 | 0.9141 | 2443.65 |
| hotpot | context_1280_1_0 | 0.2188 | 0.2572 | 0.9219 | 2372.38 |
| hotpot | context_768_0_0 | 0.2188 | 0.2538 | 0.8984 | 2460.87 |
| hotpot | context_768_1_1 | 0.3750 | 0.4240 | 0.9141 | 6543.41 |
| hotpot | rewrite | 0.1719 | 0.2099 | 0.9141 | 13433.40 |
| hotpot | multistep | 0.1562 | 0.2027 | 0.9453 | 4327.27 |
| hotpot | corrective | 0.2656 | 0.2971 | 0.9375 | 7743.12 |
| squad | generator_plain_flan-small | 0.3594 | 0.4436 | 1.0000 | 83.19 |
| squad | generator_plain_flan-base | 0.4844 | 0.5565 | 1.0000 | 113.45 |
| squad | generator_plain_qwen06 | 0.2500 | 0.3590 | 1.0000 | 650.17 |
| squad | generator_plain_qwen17 | 0.3906 | 0.5023 | 1.0000 | 459.07 |
| squad | retrieval_bm25 | 0.3906 | 0.4391 | 1.0000 | 1895.82 |
| squad | retrieval_dense_minilm | 0.4062 | 0.4516 | 1.0000 | 1921.77 |
| squad | retrieval_hybrid_minilm | 0.3906 | 0.4391 | 1.0000 | 1919.73 |
| squad | retrieval_hybrid_bge | 0.3906 | 0.4391 | 1.0000 | 2036.16 |
| squad | retrieval_tinybert | 0.3906 | 0.4315 | 1.0000 | 1922.67 |
| squad | retrieval_minilm_rerank | 0.3906 | 0.4315 | 1.0000 | 2055.57 |
| squad | context_384_1_0 | 0.4062 | 0.4420 | 1.0000 | 2044.16 |
| squad | context_768_1_0 | 0.3906 | 0.4315 | 1.0000 | 2055.57 |
| squad | context_1280_1_0 | 0.3906 | 0.4315 | 1.0000 | 1897.58 |
| squad | context_768_0_0 | 0.3906 | 0.4315 | 1.0000 | 4584.00 |
| squad | context_768_1_1 | 0.3438 | 0.4027 | 1.0000 | 5912.85 |
| squad | rewrite | 0.3906 | 0.4315 | 1.0000 | 3573.61 |
| squad | multistep | 0.3594 | 0.4055 | 1.0000 | 3843.64 |
| squad | corrective | 0.3281 | 0.3871 | 1.0000 | 7204.85 |
