# Data, models and tooling

## Benchmark selection

[BIPIA](https://github.com/microsoft/BIPIA) is the primary public benchmark source. `sources.lock.json` pins revision `a004b69ec0dd446e0afd461d98cb5e96e120a5d0` and SHA-256 hashes of every acquired file. The upstream project provides separate training and test contexts and attack families, with start, middle and end insertion. This study takes a reduced TableQA sample and four unchanged public attack strings across development and evaluation. The local task, prompt, retrieval and deterministic outcome definitions differ from the upstream evaluation. No official score is claimed.

BIPIA code and attack definitions carry MIT terms. Table contexts derive from [WikiTableQuestions](https://github.com/ppasupat/WikiTableQuestions), by Panupong Pasupat and Percy Liang, under CC BY-SA 4.0 with Wikipedia attribution obligations. Table data stays local. Public reports include row IDs and hashes, not raw tables. The acquisition script also inspected BIPIA's available email fixtures; they are excluded from all scored experiments and publication. WebQA needs separately obtained NewsQA data, and summarization needs XSum. Neither is reproduced. CodeQA is outside the factual QA and read-only tool scope of this study.

[AgentDojo](https://github.com/ethz-spylab/agentdojo), pinned at `089ed468cf3ed0322acc66b0211f26d9d90dbf60`, provides an established separation between legitimate utility and attacker success in tool-response injection. Its MIT-licensed metadata was inspected. This project does not execute AgentDojo suites, reproduce their state-changing tools or claim its scores. The complementary read-only controller is an original local fixture inspired by that threat boundary. It avoids external accounts and paid models.

[RAG-PIBench](https://arxiv.org/abs/2610.08571), dated 6 October 2026, reports a leakage-aware detection task and frozen splits. Inspection of the paper found no usable dataset repository link. Dataset access, independent partitions, labels and data license could not be verified, so it supplies methodological context only. The paper's own license is not a license to an unavailable dataset. Its published results are not copied into this study's measurements.

The complementary benchmark is explicitly original synthetic data. Development uses observatory colors; evaluation uses botanical facility colors. Tenant tests use synthetic identities and obvious noncredential canary strings. These tasks support deterministic oracles but lack the complexity and linguistic diversity of real organizations. No public-benchmark count includes them.

## Model selection and rights

FLAN-T5-base and Qwen3-0.6B are the two measured generators. Existing Qwen3-1.7B weights were also verified, but that generator is not evaluated; a cached model is not counted as an experimental comparison. BGE-small-en-v1.5 supplies the embedding detector. All reused files are read from earlier project caches without modification. `models.lock.json` records exact revisions, file hashes, sizes and model licenses. FLAN and Qwen use Apache-2.0; BGE uses MIT. CPU fallback uses float32; the measured CUDA generators use float16, greedy decoding and Qwen's non-thinking chat template. This differs from Qwen's recommended sampling settings.

One additional detector, [Protect AI DeBERTa-v3-base prompt-injection-v2](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2), was acquired under its declared Apache-2.0 license. Its model card describes an English binary injection classifier and discloses a mixture of public and crafted training data. The project is archived. Training membership is insufficient to establish independence from BIPIA, so its scores are exploratory transfer diagnostics. The card's listed upstream dataset licenses include uncertain entries; no training data or weights are redistributed here. Its memory footprint fits this GPU when loaded sequentially. No unsupported claim of checkpoint accuracy is adopted from the card.

The sparse classifier is trained locally on explicitly constructed labels using other training contexts, with protected contexts excluded. It is a lexical classification baseline, not a truth verifier. Transformer fine-tuning is not undertaken because no independently adjudicated injection training set was verified within this study. The public pipeline can retrain the local sparse artifact; its parameters stay out of Git.

## Package decisions

The environment audit records whether garak, AgentDojo and LLM Guard are installed. They were absent at inspection. [garak](https://github.com/NVIDIA/garak) is an Apache-2.0 vulnerability scanner, but its broad probe collection does not directly supply the paired tenant, provenance and task oracles needed here. AgentDojo is MIT-licensed and useful for a full agent benchmark, but this study avoids its larger stateful surface and hosted-model adapters. BIPIA's package depends on Linux-oriented DeepSpeed, vLLM and several unnecessary training/API packages. Installing those packages on this Windows host would not improve these controlled experiments. The project uses pinned BIPIA data directly and the existing Transformers runtime instead.

The DeBERTa checkpoint is invoked directly through Transformers, without installing a guardrail framework. No paid API, telemetry client, external attack tool or remote generator participates in inference. Acquisition uses ordinary HTTPS downloads from GitHub and Hugging Face. Runtime version inventory and hardware observations are recorded in `reports/environment.json`; final imported versions are recorded separately.

## Related security guidance

[OWASP's LLM application security project](https://owasp.org/projects/top-10-for-large-language-model-applications) motivates prompt injection, sensitive information disclosure, poisoning, vector/embedding weaknesses, excessive agency and bounded resource consumption. These categories guide coverage; they are not a certification standard for this implementation. The tested enforcement points are described in the threat model. There is no compliance claim.

Earlier project findings supplied design cautions: source-label agreement did not establish entailment in Advanced RAG; Agentic RAG required deterministic tool schemas and budgets; Semantic Caching bound reuse to current permissions and corpus state; LLM-as-a-Judge showed weak transfer and threshold sensitivity. These observations are not reused as independent security labels.

## References

- Jingwei Yi et al. *Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models*. KDD, 2025. [arXiv:2312.14197](https://arxiv.org/abs/2312.14197).
- Edoardo Debenedetti et al. *AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents*. NeurIPS Datasets and Benchmarks, 2024. [arXiv:2406.13352](https://arxiv.org/abs/2406.13352).
- Panupong Pasupat and Percy Liang. *Compositional Semantic Parsing on Semi-Structured Tables*. ACL, 2015. [Dataset repository](https://github.com/ppasupat/WikiTableQuestions).
- Niveen O. Jaffal, Ahmet Yuksel and David Mohaisen. *RAG-PIBench: A Leakage-Aware Benchmark for Prompt-Injection Detection in Trustworthy RAG Systems*. [arXiv:2610.08571](https://arxiv.org/abs/2610.08571).
