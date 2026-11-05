# Measured comparison

All values below were recomputed from retained local traces. These are local held-out partitions of public releases, not official test results.

| Dataset | Strategy | n | EM | F1 | Evidence recall | Coverage | Calls | p50 s | p95 s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hotpot | fixed | 128 | 0.383 | 0.493 | 0.602 | 0.938 | 1.00 | 0.74 | 1.39 |
| hotpot | adaptive | 128 | 0.383 | 0.496 | 0.637 | 0.945 | 1.28 | 0.80 | 1.52 |
| hotpot | plan | 128 | 0.281 | 0.390 | 0.602 | 0.922 | 1.93 | 2.62 | 3.75 |
| hotpot | agent | 128 | 0.344 | 0.431 | 0.660 | 0.914 | 2.35 | 6.93 | 16.86 |
| hotpot | verify | 128 | 0.062 | 0.080 | 0.629 | 0.188 | 2.12 | 5.30 | 8.78 |
| hotpot | none | 128 | 0.047 | 0.096 | 0.000 | 0.742 | 0.00 | 0.12 | 0.19 |
| hotpot | oracle | 128 | 0.531 | 0.653 | 0.984 | 0.977 | 0.00 | 0.58 | 0.94 |
| squad | fixed | 128 | 0.688 | 0.770 | 1.000 | 0.805 | 1.00 | 0.42 | 0.83 |
| squad | adaptive | 128 | 0.688 | 0.770 | 1.000 | 0.805 | 1.14 | 0.42 | 0.84 |
| squad | plan | 128 | 0.664 | 0.741 | 0.988 | 0.805 | 1.53 | 2.01 | 3.01 |
| squad | agent | 128 | 0.664 | 0.748 | 0.988 | 0.797 | 1.97 | 4.16 | 5.81 |
| squad | verify | 128 | 0.461 | 0.500 | 0.988 | 0.219 | 1.76 | 4.60 | 6.65 |
| squad | none | 128 | 0.195 | 0.195 | 0.000 | 0.516 | 0.00 | 0.15 | 0.25 |
| squad | oracle | 128 | 0.828 | 0.905 | 1.000 | 0.656 | 0.00 | 0.49 | 1.13 |

## Matched differences from fixed RAG

Intervals resample questions for HotpotQA and complete articles for SQuAD (2,000 replicates). They are pointwise, unadjusted intervals.

| Dataset | Strategy | F1 difference | 95% interval |
| --- | --- | ---: | --- |
| hotpot | adaptive | +0.003 | [-0.020, +0.026] |
| hotpot | plan | -0.104 | [-0.202, -0.006] |
| hotpot | agent | -0.062 | [-0.132, +0.014] |
| hotpot | verify | -0.413 | [-0.493, -0.329] |
| squad | adaptive | +0.000 | [+0.000, +0.000] |
| squad | plan | -0.029 | [-0.061, -0.006] |
| squad | agent | -0.022 | [-0.048, -0.001] |
| squad | verify | -0.270 | [-0.390, -0.149] |

## Interpretation limits

Source-label citation agreement is distinct from NLI support. Citations are assigned after generation by lexical overlap; generated claims can alter answer meaning. Verification may trade answer coverage for fewer unsupported proposals. The no-retrieval control may answer without citations by design; deployable paths reject answers with no valid source.

Oracle supplies gold source documents but still uses bounded context and the same generator. It is a diagnostic gold-document reference, not a guaranteed upper bound or an answer oracle. Additional calls, context changes and generation retries are confounded in verification-guided control; gains cannot be attributed solely to new evidence.

Full development, validation, model checks, ablations, reliability and resource measurements are in the adjacent JSON/CSV reports and the public notebook. Timing is a single sequential pass with model loading excluded from per-question latency. Fixed strategy includes the same claim-generation and NLI diagnostics as the other methods, so it costs more than a deployment without diagnostics.
