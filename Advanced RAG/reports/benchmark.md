# Local grounded-answer results

The study uses 64 held-out questions from each public development release, with 24 separate development questions per dataset. These are bounded-evidence local experiments, not official test scores. All generation ran on CPU with FLAN-T5-small. No paid APIs were used.

| Dataset | Method | EM | Answer F1 | Context recall | Median ms |
| --- | --- | ---: | ---: | ---: | ---: |
| hotpot | no_context | 0.0625 | 0.1017 | 0.0000 | 66.85 |
| hotpot | bm25 | 0.2344 | 0.3934 | 0.6484 | 109.52 |
| hotpot | dense | 0.2031 | 0.3372 | 0.6875 | 118.13 |
| hotpot | hybrid | 0.2031 | 0.3370 | 0.7109 | 120.90 |
| hotpot | rerank | 0.3594 | 0.5020 | 0.7188 | 124.32 |
| hotpot | multihop | 0.2188 | 0.3548 | 0.7188 | 122.98 |
| hotpot | corrective | 0.2188 | 0.3526 | 0.7109 | 127.35 |
| hotpot | oracle | 0.3125 | 0.4679 | 0.9766 | 116.32 |
| hotpot | abstention | 0.2031 | 0.3370 | 0.7109 | 120.90 |
| hotpot | budget180 | 0.2188 | 0.3292 | 0.5234 | 85.39 |
| squad | no_context | 0.0312 | 0.0427 | 0.0000 | 62.01 |
| squad | bm25 | 0.3438 | 0.3958 | 1.0000 | 88.74 |
| squad | dense | 0.3438 | 0.3958 | 1.0000 | 94.12 |
| squad | hybrid | 0.3594 | 0.4114 | 1.0000 | 92.98 |
| squad | rerank | 0.3438 | 0.3958 | 1.0000 | 98.60 |
| squad | multihop | 0.3594 | 0.4114 | 1.0000 | 94.75 |
| squad | corrective | 0.3594 | 0.4114 | 1.0000 | 93.13 |
| squad | oracle | 0.3594 | 0.4114 | 1.0000 | 93.61 |
| squad | abstention | 0.3594 | 0.4114 | 1.0000 | 92.98 |
| squad | budget180 | 0.3281 | 0.3905 | 1.0000 | 81.99 |

## hotpot

The highest observed F1 among the deployable configurations is 0.5020 for `rerank`. This ranking is descriptive; it was not used to select a configuration after evaluation.

`bm25` changes F1 relative to hybrid by +0.0564, with a paired 95% percentile interval [-0.0347, +0.1498].

`dense` changes F1 relative to hybrid by +0.0002, with a paired 95% percentile interval [-0.0800, +0.0841].

`rerank` changes F1 relative to hybrid by +0.1650, with a paired 95% percentile interval [+0.0573, +0.2740].

`multihop` changes F1 relative to hybrid by +0.0178, with a paired 95% percentile interval [-0.0793, +0.1180].

`corrective` changes F1 relative to hybrid by +0.0156, with a paired 95% percentile interval [+0.0000, +0.0469].

`oracle` changes F1 relative to hybrid by +0.1309, with a paired 95% percentile interval [+0.0212, +0.2358].

Hybrid citation source-label precision is 0.7188; source-label recall is 0.3594. These metrics do not establish entailment.

Development selected overlap threshold -1. The selected abstention variant has coverage 1.0000. The overlap rule supplied no additional abstention benefit in the selected configuration.

## squad

The highest observed F1 among the deployable configurations is 0.4114 for `hybrid`. This ranking is descriptive; it was not used to select a configuration after evaluation.

`bm25` changes F1 relative to hybrid by -0.0156, with a paired 95% percentile interval [-0.0469, +0.0000].

`dense` changes F1 relative to hybrid by -0.0156, with a paired 95% percentile interval [-0.0469, +0.0000].

`rerank` changes F1 relative to hybrid by -0.0156, with a paired 95% percentile interval [-0.0469, +0.0000].

`multihop` changes F1 relative to hybrid by +0.0000, with a paired 95% percentile interval [+0.0000, +0.0000].

`corrective` changes F1 relative to hybrid by +0.0000, with a paired 95% percentile interval [+0.0000, +0.0000].

`oracle` changes F1 relative to hybrid by +0.0000, with a paired 95% percentile interval [+0.0000, +0.0000].

Hybrid citation source-label precision is 0.5873; source-label recall is 1.0000. These metrics do not establish entailment.

Development selected overlap threshold -1. The selected abstention variant has coverage 0.9844. The overlap rule supplied no additional abstention benefit in the selected configuration.

## Support review and limitations

A separate assistant review of the first ten hybrid cases per dataset found 8 supported, 3 partial, and 9 unsupported answers. References were visible, and no independent human rater participated. The review checks actual cited windows, including entity identity, requested answer type, and both sides of comparisons. It is exploratory case analysis, not an estimate of population faithfulness.

The experiment uses supplied evidence pools and small samples. The source citation is attached after generation using lexical overlap. Expansion is a simple evidence-derived term baseline, not learned decomposition. Pretrained benchmark exposure is unknown. Question-level bootstrap intervals are exploratory and can understate SQuAD article dependence. Timing uses one sequential pass; model loading and document indexing are excluded from answer latency and reported separately. CPU RSS observations are sampled process totals. GPU performance, electricity, and hardware costs were not measured.

The oracle uses gold source labels and is a diagnostic control. Negative comparisons remain in the table. Larger generators, open-domain retrieval, independent support annotation, and stronger abstention calibration remain untested.
