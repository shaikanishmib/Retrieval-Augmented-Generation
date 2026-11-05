# Measured benchmark

All values below are reconstructed from actual timed inference traces. Model loading and one discarded warm-up are separate. Three sequential trials use a fresh cache for each strategy, workload and dataset. Candidate-pool indexing is charged to each first question.

| Dataset | Workload | Strategy | F1 | Exact hit | Semantic hit | p50 ms | p95 ms | Calls avoided |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hotpot | mixed | none | 0.522 | 0.000 | 0.000 | 210.36 | 604.58 | 0 |
| hotpot | mixed | exact | 0.522 | 0.333 | 0.000 | 192.25 | 602.29 | 192 |
| hotpot | mixed | semantic | 0.522 | 0.333 | 0.000 | 215.72 | 578.71 | 192 |
| hotpot | mixed | evidence | 0.522 | 0.333 | 0.000 | 296.47 | 705.66 | 0 |
| hotpot | mixed | hybrid | 0.527 | 0.333 | 0.302 | 187.98 | 584.25 | 192 |
| hotpot | unique-heavy | none | 0.523 | 0.000 | 0.000 | 341.17 | 741.46 | 0 |
| hotpot | unique-heavy | exact | 0.523 | 0.100 | 0.000 | 335.64 | 727.99 | 24 |
| hotpot | unique-heavy | semantic | 0.523 | 0.100 | 0.000 | 362.20 | 827.98 | 24 |
| hotpot | unique-heavy | evidence | 0.523 | 0.100 | 0.000 | 370.56 | 809.28 | 0 |
| hotpot | unique-heavy | hybrid | 0.527 | 0.100 | 0.087 | 358.39 | 776.62 | 24 |
| hotpot | unique | none | 0.517 | 0.000 | 0.000 | 348.15 | 771.84 | 0 |
| hotpot | unique | exact | 0.517 | 0.000 | 0.000 | 345.18 | 785.10 | 0 |
| hotpot | unique | semantic | 0.517 | 0.000 | 0.000 | 371.51 | 814.95 | 0 |
| hotpot | unique | evidence | 0.517 | 0.000 | 0.000 | 371.19 | 816.06 | 0 |
| hotpot | unique | hybrid | 0.517 | 0.000 | 0.000 | 366.90 | 811.45 | 0 |
| squad | mixed | none | 0.613 | 0.000 | 0.000 | 203.36 | 551.65 | 0 |
| squad | mixed | exact | 0.613 | 0.333 | 0.000 | 187.54 | 570.29 | 192 |
| squad | mixed | semantic | 0.613 | 0.333 | 0.000 | 162.48 | 553.88 | 192 |
| squad | mixed | evidence | 0.613 | 0.333 | 0.000 | 211.41 | 577.82 | 0 |
| squad | mixed | hybrid | 0.618 | 0.333 | 0.307 | 154.92 | 539.55 | 192 |
| squad | unique-heavy | none | 0.606 | 0.000 | 0.000 | 425.15 | 709.81 | 0 |
| squad | unique-heavy | exact | 0.606 | 0.100 | 0.000 | 415.47 | 732.85 | 24 |
| squad | unique-heavy | semantic | 0.606 | 0.100 | 0.000 | 436.71 | 712.49 | 24 |
| squad | unique-heavy | evidence | 0.606 | 0.100 | 0.000 | 434.64 | 719.48 | 0 |
| squad | unique-heavy | hybrid | 0.615 | 0.100 | 0.096 | 433.77 | 705.82 | 24 |
| squad | unique | none | 0.607 | 0.000 | 0.000 | 414.78 | 734.39 | 0 |
| squad | unique | exact | 0.607 | 0.000 | 0.000 | 410.43 | 677.11 | 0 |
| squad | unique | semantic | 0.607 | 0.000 | 0.000 | 435.79 | 716.51 | 0 |
| squad | unique | evidence | 0.607 | 0.000 | 0.000 | 439.17 | 716.16 | 0 |
| squad | unique | hybrid | 0.607 | 0.000 | 0.000 | 435.11 | 697.44 | 0 |

Reuse correctness is based on request-family equivalence, separate from baseline answer correctness. Synthetic wrappers preserve the question and do not represent independent human paraphrases. Different corpus scopes prevent semantic cross-task hits on HotpotQA. PAWS supplies the harder annotated equivalence diagnostic; it is not an end-to-end QA result.

The hybrid policy tries exact answers, strict semantic answers, then evidence reuse at an experimental threshold of 0.95 with fresh generation. Evidence caching regenerates answers after both exact and semantic evidence hits. Backend microbenchmarks use synthetic normalized vectors and exclude embedding time.

Confidence intervals resample HotpotQA question families and SQuAD articles. HotpotQA evaluation families can share source pages, so their intervals may understate document-level dependence. Repeated timings share one host, and three trials do not establish production tail latency. Avoided CUDA elapsed time is neither energy nor monetary savings.
