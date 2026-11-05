# Research design

The primary experiment uses HotpotQA distractor questions, which supply ten candidate passages and annotated supporting sentences. Retrieval operates within each question's candidate set. This measures evidence selection in a supplied candidate pool; it does not estimate open-domain search performance. Paragraphs are split into overlapping 100-word windows with a 20-word overlap. Every window retains its source identifier.

SQuAD 2.0 supplies a complementary abstention check with adversarial unanswerable questions. Its supplied paragraph is the evidence universe. A label of unanswerable applies to that paragraph, so adding unrelated corpus passages would change the task. Retrieval comparisons on this dataset therefore mainly test window selection and generation sensitivity.

The public development releases are partitioned into local development and held-out groups using SHA-256 order. HotpotQA groups are question IDs. SQuAD groups are article titles, preventing paragraphs from one article from entering both sets. Exact normalized question matches are excluded from development if they occur in the held-out selection. These partitions are local research subsets, not official test sets. Pretrained models may have encountered benchmark material; this study cannot establish contamination-free evaluation.

The fixed configuration selects 24 development questions and 64 held-out questions per dataset. All methods use the same held-out questions. Abstention chooses among three fixed thresholds by development answer F1. No test result selects a configuration. Sample sizes are suitable for a local engineering study but support only broad conclusions.

## Comparisons

BM25 is the lexical baseline. MiniLM supplies normalized embeddings with exact cosine search. Hybrid retrieval uses reciprocal rank fusion with constant 60 and at most 20 candidates from each ranking. TinyBERT reranks at most ten hybrid candidates. All answer generation uses the same greedy FLAN-T5-small model with at most 32 new tokens.

The multi-hop variant appends four frequent terms from the highest-ranked chunk and fuses a second lexical search with the original ranking. It is a simple evidence expansion baseline. It does not implement learned decomposition or prove multi-hop reasoning. Corrective retrieval retries that expansion when the initial answer is empty or has token overlap below 0.5. A retry's latency and token totals include the initial attempt.

The no-context control measures the generator's behavior with no retrieved evidence. The oracle control moves annotated source documents first, retaining the same chunk budget. It uses gold labels deliberately and diagnoses evidence-selection errors. It is not a guaranteed answer-quality ceiling: ordering, selected windows, and generation can still change the answer. The 180-token context ablation tests sensitivity to a smaller evidence budget. Index construction and model loading are recorded separately from answer latency. Query embedding, ranking, context assembly, generation, and citation attachment are included in answer latency.

## Evaluation

Answer exact match and token F1 follow the SQuAD normalization convention. Retrieval recall uses source documents, deduplicating windows. Context recall uses only documents actually included in the prompt. Citation precision and recall compare attributed sources with gold source labels. A single citation is attached after generation to the selected source with greatest answer-token overlap; the model does not generate these citations itself.

Source-label agreement is not entailment. Answer-token overlap is an explicitly weak diagnostic: it misses negation, attribution, comparison, and paraphrase. Manual review must separately ask whether the cited text supports the complete answer to the question. Review records belong in a distinct report, with unsupported, partial, supported, and unclear labels. The project must not report overlap as faithfulness accuracy.

Paired percentile bootstrap intervals use 2,000 resamples of question-level F1 differences with seed 42. They are exploratory intervals, with no correction for multiple comparisons. SQuAD questions share articles, so question-level intervals can understate dependence. Latencies come from one sequential pass on shared hardware; background load, warm caches, and method order can affect them. Process RSS sampled after questions is not a per-method memory peak. API charges are zero; electricity and hardware depreciation are unmeasured.

## Dataset candidates

| Candidate | Evidence and reason for decision |
| --- | --- |
| [HotpotQA](https://hotpotqa.github.io/) | Selected for supporting facts, comparison questions, and a manageable distractor release. CC BY-SA 4.0. |
| [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) | Selected for natural unanswerable questions and original answer spans. CC BY-SA 4.0. |
| [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop) | Explicit reasoning paths would add a useful later check. Its templated construction overlaps the primary multi-hop objective; not downloaded. Terms require review before reuse. |
| [MuSiQue](https://github.com/StonyBrookNLP/musique) | Composed questions and disconnected variants would test harder reasoning and abstention. Deferred to keep the first study's data and runtime manageable. Terms require review before reuse. |
| [Natural Questions](https://ai.google.com/research/NaturalQuestions/download) | Real search queries and long-page annotations would broaden external validity. Full-page acquisition and preprocessing add substantial scope; not downloaded. |

Raw text stays local. Dataset and model lockfiles record observed file hashes and immutable model revisions. A hash recorded on first acquisition identifies the bytes acquired; it is not an independent authenticity certificate. Dataset provenance follows maintainers' releases or a named mirror when an original host is unavailable.
