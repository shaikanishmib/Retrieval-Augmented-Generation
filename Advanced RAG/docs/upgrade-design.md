# Model and answer-grounding study

The original study at commit `92a85e1` remains intact, including source files, reports, locks, traces, and the executed local notebook. `reports/upgrade/preservation.json` records their hashes. The notebook also has a local backup under `.local/upgrade-baseline`. New code lives in `src/advanced_rag/upgrade`, new reports in `reports/upgrade`, and raw predictions under ignored `data/upgrade`.

## What the audit found

The original generator is FLAN-T5-small. Its pipeline chooses one citation after generation by matching answer tokens to retrieved windows. Its multihop and corrective paths append frequent words from the first window. Threshold selection uses three lexical-overlap cutoffs, and the selected threshold is -1 for both datasets. The published evaluation contains 64 held-out questions per dataset. The original assistant review labels 9 of 20 answers unsupported. These observations come from the implementation and recorded artifacts, not inferred model quality.

## Feasibility and candidates

The initial hardware audit found 16 physical CPU cores, 24 logical cores, 34.0 GB RAM, 17.2 GB available RAM, 676.9 GB free disk, and an RTX 4060 Laptop GPU with 8,188 MiB VRAM. Available capacity changes during execution. The baseline environment has CPU-only PyTorch and no Ollama or llama.cpp executable. An eight-question CPU probe preceded new model acquisition. A separate CUDA 12.8 PyTorch 2.10 wheel enabled GPU experiments without changing the baseline package directory.

| Candidate | Terms and instructions | Decision |
| --- | --- | --- |
| [FLAN-T5-small](https://huggingface.co/google/flan-t5-small) | Apache 2.0; sequence-to-sequence generation | Preserve and reproduce the original baseline. |
| [FLAN-T5-base](https://huggingface.co/google/flan-t5-base) | Apache 2.0; same adapter family | Evaluate a stronger sequence-to-sequence baseline. |
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | Apache 2.0; Transformers 4.51 or newer; chat template | Evaluate a smaller chat generator. |
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | Apache 2.0; same chat interface | Evaluate after the 0.6B probe left GPU capacity. |
| [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | Apache 2.0; same interface | Do not download. Unquantized half-precision weights alone approach 8 GB, leaving insufficient space for the verifier, retrieval models, cache, and activations. Quantized 4B inference remains untested. |
| [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Apache 2.0; normalized sentence embeddings | Retain the original encoder. |
| [BGE-small](https://huggingface.co/BAAI/bge-small-en-v1.5) | MIT; use the documented retrieval instruction on queries | Evaluate as a stronger retrieval candidate. |
| [BGE-base](https://huggingface.co/BAAI/bge-base-en-v1.5) | MIT; same query instruction | Defer. The first study already tests an encoder change; an additional base-sized encoder is not required to establish its effect. |
| [TinyBERT](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L2-v2) | Apache 2.0; query-passage cross-encoder | Retain the original reranker. |
| [MiniLM reranker](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) | Apache 2.0; query-passage cross-encoder | Evaluate as the stronger reranker. |
| [BGE reranker](https://huggingface.co/BAAI/bge-reranker-base) | MIT; multilingual cross-encoder | Defer the larger multilingual model while testing the smaller English reranker. |
| [DeBERTa NLI](https://huggingface.co/cross-encoder/nli-deberta-v3-small) | Apache 2.0; contradiction, entailment, neutral labels | Evaluate claim-support diagnostics separately from source-label agreement. |

The first GPU probe used about 2.1 GB allocated memory with Qwen3-0.6B and supporting models. The 1.7B probe used about 4.4 GB. These are probe observations, not isolated weight sizes. Initial and later feasibility files remain separate because the structured prompt was shortened between probes. Formal model comparisons use one fixed study prompt.

Qwen uses `apply_chat_template` with `enable_thinking=False`. Non-thinking generation samples with temperature 0.7, top-p 0.8, and top-k 20, following its model card. FLAN uses greedy sequence-to-sequence decoding. This adapter difference is intentional and disclosed. Seeds derive from the question alone, so controlled configurations share random seeds and resumed stages reproduce the same sampling decisions within the recorded environment. Cross-device bitwise identity is not promised.

Generator comparisons fix the retrieval model and ranking method. Context packing still uses each generator's tokenizer and input limit, so selected windows can differ. These are system-level generator comparisons, not experiments with identical token sequences or guaranteed identical evidence sets. The context manifests expose that difference.

## Sampling and stage order

The existing dataset loader retains the original SHA-256 group assignment. Development expands to 64 questions per dataset. The original 64 held-out questions are excluded from the fresh evaluation, which uses the next 256 questions in the same deterministic order. The original questions are evaluated separately for direct before-and-after comparisons. SQuAD development and evaluation article groups remain disjoint. Uncertainty resamples complete SQuAD articles; HotpotQA resamples questions.

Controlled generator, retrieval, context, and reasoning comparisons use the same first 64 fresh questions per dataset. The larger 256-question sets evaluate the frozen selected, balanced, and strongest configurations. Results from different sample sizes must not be treated as paired comparisons. The baseline's unchanged BM25, dense, hybrid, and reranking methods also run on all 256 fresh questions per dataset.

Controlled paired intervals use the first 64 `selected_raw` scores as their reference. Structured outputs still pass schema and citation validation, but the NLI cutoff is zero. These intervals compare recorded configurations before the separately selected abstention rule; they do not compare every ablation with the CLI's final verification path. Plain-generator rows also differ from this reference in prompt format and context policy. Compare the four plain-generator rows with each other when assessing generator effects.

The plain-answer development winner is extended to the full 256-question samples using the unchanged baseline hybrid and TinyBERT retrieval methods and post hoc citation assignment. This practical reference was identified by the earlier development comparison; its model is selected from those development scores alone. The full-sample extension is reported separately from the frozen structured-claim protocol. It adds no model download and does not imply that lexical citations have become faithful.

When the selected generator is Qwen3-1.7B, the selected raw configuration and the strongest resource tier are identical. They share cached predictions and are not independent repetitions. NLI and lexical gating also reuse the same generated answers, changing only the acceptance rule.

Stages select a generator on development answers with fixed retrieval, compare retrieval configurations with that generator fixed, and select context size and ordering on development data. Threshold fitting then uses development groups only. One half of development groups fits a logistic calibration mapping; the other half selects support thresholds by answer F1. The calibration target is raw answer exact correctness. It is not an entailment label. All selections are saved before fresh evaluation begins. Held-out risk curves describe sensitivity after that choice and do not select a new threshold.

The first development-only pilot ranked structured generators by pooled F1. It favored the smaller Qwen mainly through abstention on unanswerable SQuAD questions while its HotpotQA answer F1 was zero. Those pilot results are retained in `reports/upgrade/development-pilot`. Before any fresh evaluation, generator selection changed to answerable-question F1, with overall F1 as a tie-breaker. The final study reruns under a new source fingerprint. This change addresses the observed development failure; no held-out outcome informed it.

## Citation and support controls

The generator returns a short answer and complete claims with explicit source IDs. Each ID must refer to a window actually included in the prompt. Malformed JSON, missing claims, fabricated sources, and an answer absent from the claims invalidate the answer. Invalid and low-support responses produce an empty final answer with a reason flag. The original raw response remains local for diagnosis.

Each claim is scored against its combined cited windows and against each citation individually. The minimum combined-claim entailment probability is the support score. This is an NLI model judgment, not verified truth. Concatenated premises can be truncated to 512 NLI tokens; the report counts truncated pairs. An answer-string anchor cannot prove that the generated claims answer the question. Negation, entity confusion, omitted comparison evidence, and irrelevant but entailed claims remain possible failures.

Raw structured results can retain low-scoring answers for comparison. An accepted raw answer with NLI support below 0.5 carries `low_model_support`, even if development selected a zero cutoff. The threshold experiment reports this choice explicitly instead of silently treating weak model support as grounded truth.

The matched case review exposes both false positives and false negatives. An entailed fact about a different person can receive a high NLI score while answering the wrong question. Correct comparative claims can fail the literal answer-string guard, and irrelevant additional citations can disrupt NLI judgments. Alternative valid support can also occur outside the originally annotated document IDs. Formatting rejection, model abstention, and support filtering are therefore reported separately; an empty answer on an unanswerable item does not by itself establish evidence understanding.

An exploratory follow-up preserves the frozen experiment and replays its actual generated answers through a question-conditioned check. The existing FLAN-T5-base model answers from cited windows only. Agreement with the proposal supplies another consistency signal. Literal-anchor failures may become candidates for this check, but malformed JSON and fabricated citations remain rejected. Every cited window must fit the verifier's 512-token input limit; overflow causes a flag rather than silent context removal. Agreement and NLI cutoffs are selected on development threshold groups before the new check scores held-out outputs. The CLI adopts it per dataset only when development F1 increases. Neither QA-model agreement nor NLI proves faithfulness, and the models can share entity errors. Additional verification times are replay observations, not a newly measured combined wall-clock latency.

This follow-up arose after inspecting matched original case failures, when frozen-study scores were already available. Its cutoffs and adoption decision use development data, but its reused evaluation results remain exploratory. A new untouched confirmation sample is needed before treating the QA-check result as confirmatory evidence.

Multiple claims may cite different sources, or one claim may cite several. Source precision and recall compare document IDs with original annotations. NLI claim and citation scores are reported separately. Assistant case review is explicitly identified as such and is never described as independent human annotation.

## Multi-step retrieval

The planner reads the first retrieved windows and generates up to two standalone follow-up questions about a missing bridge or comparison fact. Each follow-up executes a new retrieval call. The resulting rankings are fused before context selection and answer generation. A rewrite ablation generates a search question without evidence. Corrective retrieval first generates and verifies an answer, then plans another retrieval if parsing fails, the answer is empty, or support is below a fixed development-defined trigger. It records both attempts, planning tokens, follow-up queries, and stage times. Invalid planner outputs trigger no invented query or term-expansion fallback.

The candidate universe remains HotpotQA's supplied distractor passages or SQuAD's original paragraph windows. This tests sequential evidence selection, not web search or general open-domain reasoning. Gold answers and supporting-fact labels enter evaluation only. The deployable engine has no oracle interface.

Corrective retrieval also makes another answer attempt with a different deterministic seed. The study has no generation-only retry control, so a corrective gain cannot be attributed entirely to the follow-up evidence search. Recorded retrieval steps distinguish executed follow-up queries from planner failures.

The main evidence-recall metric measures annotated source documents in the prompt. A separate HotpotQA diagnostic measures annotated supporting sentences wholly contained in selected or accepted cited windows after whitespace normalization. It does not reconstruct a sentence split across windows. This stricter sentence-containment measure is descriptive and does not select a configuration.

## Timing and limitations

Model loading, document encoding, index construction, query encoding, retrieval, reranking, context assembly, planning, generation, and verification are timed separately. CUDA synchronization bounds GPU stages. Total answer latency includes all active attempts. Cached predictions retain the original measured stage times and are marked by manifests; cache lookup speed is not inference throughput.

The resource observer samples process RSS every 50 ms and retains Windows process peak memory and CUDA allocation/reservation peaks. These are whole-process observations, not isolated per-method costs. The study is sequential and method order is not randomized. Background load, warm caches, pretrained benchmark exposure, shared pages across HotpotQA questions, small development samples, and model-based support errors limit interpretation. There are no paid API calls. Electricity and hardware depreciation remain unmeasured.

One host snapshot taken during slower inference is retained separately in `reports/upgrade/host-load.json`. It records available RAM, CPU utilization, AC status, and NVIDIA clock and power observations. It is not a controlled explanation for the slowdown and does not replace the benchmark timings.

The shared candidate index builds lexical and dense representations even for a BM25 comparison, so its encoding cost is an experiment-harness observation, not a minimum deployment cost for BM25. Loading times identify actual observed loads and reloads; warm filesystem caches are not controlled. The adapter's `truncated` field marks an output reaching its token limit, which can occasionally still contain a complete JSON object. Parsing determines whether that response is usable.

The main resource report retains the most recent observed loading time for each model alias, rather than a cumulative sum of every reload. Feasibility probes retain their own loading observations. Neither report establishes cold-disk loading performance.

Output-token counts measure returned token sequences before text decoding. FLAN counts include its decoder start token and any returned end token. Qwen counts exclude the prompt and include any returned end token. Counts therefore describe each adapter's work; they are not normalized counts of visible answer words.

An always-abstain control is scored on the same held-out questions. Calibration also includes a constant correctness probability estimated from development fit groups. These comparators distinguish useful evidence recognition from gains caused mainly by class prevalence. Cache keys include source, model, configuration, question content, and Torch version; hardware is recorded separately. Use a fresh cache when changing hardware or runtime settings outside this recorded setup.
