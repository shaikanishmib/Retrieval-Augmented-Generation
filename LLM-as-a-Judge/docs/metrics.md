# Evaluation dimensions

| Dimension | Required inputs and interpretation | Study coverage |
| --- | --- | --- |
| Evidence faithfulness | Answer claims and supplied evidence; fraction supported, with an explicit claim aggregation rule | NLI sentence support estimates and human response hallucination detection |
| Factual answer correctness | Answer, independent factual reference and scoring rubric | Not labeled by RAGTruth; no correctness claim |
| Answer relevance | Question and answer, with relevance judgments | Prompt context only; no independently labeled relevance score |
| Context relevance | Question and retrieved passages with relevance labels | Not scored against RAGTruth hallucination spans |
| Context recall | Reference answer facts or gold evidence and retrieved context | Not available as a human endpoint in this study |
| Citation validity | Citation IDs and available source-ID registry; valid IDs divided by cited IDs | Tested parser; dataset has no uniform citation registry |
| Citation support | Each cited claim and its cited evidence, independently labeled | NLI windows retained; no human citation-quality benchmark claim |
| Claim-supported fraction | Number of supported claim units divided by evaluated units | Sentence-unit support estimates only; sentence atomicity is not assumed |
| Noise sensitivity | Change after adding controlled irrelevant context | Score and decision changes, separately from human-label agreement |

Exact matching, lexical overlap, token F1 against evidence and embedding similarity are transparent baselines. They require answer and context and output support proxies in [0,1]. They cannot establish correctness, and a high similarity score can miss a changed number or negation. Missing citations yield no citation-validity score, rather than a perfect score. Hallucinated source IDs fail validity even if their accompanying prose sounds plausible.

NLI scores distinguish entailment, neutral and contradiction at the model level. Neutral includes insufficient evidence. A response may be factually correct while neutral against context; it may also faithfully repeat factually wrong context. Response-level hallucination detection is the primary human-labeled endpoint. Correlations for ordinal scores and general human preference accuracy are omitted because suitable independent labels are absent.

The machine-readable field `pr_auc` stores scikit-learn's `average_precision_score`, using its stepwise precision-recall definition. It is not a trapezoidal interpolation of the plotted curve. Calibration error uses ten equal-width bins over [0,1], weighted by bin counts; empty bins contribute zero. Brier score averages squared error against the binary response label. Selective coverage is the fraction whose absolute distance from the frozen decision threshold exceeds a margin, and selective accuracy scores only those retained responses.
